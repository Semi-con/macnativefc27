// Mush8 - msl.rs
// Constantin Clerc (c22dev) - 0.1
use crate::inst::*;
use crate::program::{Inst, Program, branch_target};
use crate::sph::{Imap, Sph, Stage};
use crate::table::{InstName, Op};
use anyhow::{Result, bail};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write;
#[path="msl_compute.rs"]
mod compute_control;
#[path="msl_regions.rs"]
mod graphics_regions;
#[path="msl_structured.rs"]
mod graphics_structured;
#[path="msl_memory.rs"]
mod memory_loads;
#[path="msl_write_footprints.rs"]
mod write_footprints;
#[path="msl_dynamic_vertex.rs"]
mod dynamic_vertex;
pub use dynamic_vertex::DynamicVertexSources;
pub use write_footprints::{GridWriteFootprint,DescriptorWriteFootprint,WriteWindowFootprint};

#[derive(Clone,Copy,PartialEq,Eq,Default)]
pub enum GraphicsDispatch {#[default] Switch,Forward,Regions,Structured}

#[derive(Clone,Copy,Default)]
pub struct TranslationConfig {
    pub compute_memory:Option<(u32,u32)>,
    pub graphics_dispatch:GraphicsDispatch,
    pub vector_global_loads:bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TexKind {
    T2D,
    T2DArray,
    T3D,
    Cube,
    CubeArray,
    Depth2D,
    Depth2DArray,
    DepthCube,
}

impl TexKind {
    pub fn decl(self) -> &'static str {
        match self {
            TexKind::T2D => "texture2d<float>",
            TexKind::T2DArray => "texture2d_array<float>",
            TexKind::T3D => "texture3d<float>",
            TexKind::Cube => "texturecube<float>",
            TexKind::CubeArray => "texturecube_array<float>",
            TexKind::Depth2D => "depth2d<float>",
            TexKind::Depth2DArray => "depth2d_array<float>",
            TexKind::DepthCube => "depthcube<float>",
        }
    }
    pub fn is_depth(self) -> bool {
        matches!(self, TexKind::Depth2D | TexKind::Depth2DArray | TexKind::DepthCube)
    }
    fn coords(self) -> usize {
        match self {
            TexKind::T3D | TexKind::Cube | TexKind::CubeArray | TexKind::DepthCube => 3,
            _ => 2,
        }
    }
    pub fn is_array(self) -> bool {
        matches!(self, TexKind::T2DArray | TexKind::CubeArray | TexKind::Depth2DArray)
    }
}

#[derive(Debug, Clone, Default)]
pub struct Interface {

    pub textures: BTreeMap<u32, TexKind>,

    pub samplers: BTreeMap<u32, u32>,
    pub texture_slots: BTreeMap<u32,u32>,
    // More than sixteen statically resolved samplers use a typed argument
    // buffer; Metal's direct sampler table only has sixteen entries.
    pub argument_textures:bool,
    pub images: BTreeMap<u32,ImageBinding>,
    pub atomic_images:u32,
    pub image_writes:u32,
    pub buffer_metadata:bool,
    // Runtime descriptor handles, bit 0: float 2D, bit 1: depth 2D.
    pub dynamic_textures:u32,

    pub attributes: u32,

    pub outputs: u32,

    pub color_targets: u32,
    pub writes_depth: bool,
}

#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub struct ImageBinding {pub kind:TexKind,pub raw_bytes:u32}

pub struct Translated {
    pub stage: Stage,
    pub msl: String,
    pub entry: String,
    pub interface: Interface,
    pub grid_write_footprint:Option<GridWriteFootprint>,
    pub descriptor_write_footprint:Option<Vec<DescriptorWriteFootprint>>,
    pub write_window_footprint:Option<Vec<WriteWindowFootprint>>,
}

impl Translated {

    pub fn named(&self, suffix: &str) -> (String, String) {
        let mut src = self.msl[PRELUDE.len()..].to_string();
        for name in ["VOut", "FIn", "FOut", "StaticTextures", "DynamicTexturesWide", self.entry.as_str()] {
            src = rename(&src, name, &format!("{name}_{suffix}"));
        }
        (src, format!("{}_{suffix}", self.entry))
    }
}

pub fn prelude() -> &'static str {
    PRELUDE
}

pub const DYNAMIC_TEXTURE_CAPACITY:usize=8192;
pub const DYNAMIC_SAMPLER_ARGUMENT:usize=DYNAMIC_TEXTURE_CAPACITY*2;
pub const DYNAMIC_LOOKUP_ARGUMENT:usize=DYNAMIC_SAMPLER_ARGUMENT+256;
pub const DYNAMIC_COUNTS_ARGUMENT:usize=DYNAMIC_LOOKUP_ARGUMENT+3;

// Append only to shaders that actually use runtime descriptor selection, so
// unrelated shader source identities and compiled libraries remain reusable.
pub fn dynamic_texture_layout()->String{
    format!("struct DynamicTexturesWide {{\n    array<texture2d<float>,{n}> color [[id(0)]];\n    array<depth2d<float>,{n}> depth [[id({n})]];\n    array<sampler,256> samplers [[id({s})]];\n    device const uint2* color_lookup [[id({l})]];\n    device const uint2* depth_lookup [[id({d})]];\n    device const uint2* sampler_lookup [[id({t})]];\n    uint4 counts [[id({c})]];\n}};\n",n=DYNAMIC_TEXTURE_CAPACITY,s=DYNAMIC_SAMPLER_ARGUMENT,l=DYNAMIC_LOOKUP_ARGUMENT,d=DYNAMIC_LOOKUP_ARGUMENT+1,t=DYNAMIC_LOOKUP_ARGUMENT+2,c=DYNAMIC_COUNTS_ARGUMENT)
}

fn rename(src: &str, from: &str, to: &str) -> String {
    let word = |c: Option<char>| c.is_some_and(|c| c.is_ascii_alphanumeric() || c == '_');
    let mut out = String::with_capacity(src.len());
    let mut rest = src;
    while let Some(i) = rest.find(from) {
        let (before, after) = (rest[..i].chars().next_back(), rest[i + from.len()..].chars().next());
        out.push_str(&rest[..i]);
        out.push_str(if word(before) || word(after) { from } else { to });
        rest = &rest[i + from.len()..];
    }
    out.push_str(rest);
    out
}

const ATTR_POSITION: i32 = 0x70;
const ATTR_GENERIC: i32 = 0x80;
const ATTR_GENERIC_END: i32 = 0x280;
const ATTR_POINT_SIZE: i32 = 0x6c;
const ATTR_POINT_COORD: i32 = 0x2e0;
const ATTR_INSTANCE_ID: i32 = 0x2f8;
const ATTR_VERTEX_ID: i32 = 0x2fc;
const ATTR_FRONT_FACING: i32 = 0x3fc;

struct Gen<'a> {
    simt:bool,
    vector_global_loads:bool,
    embedded_constants:Option<&'a [u8]>,
    shared_memory:u32,
    sph: &'a Sph,
    prog: &'a Program,
    body: String,
    instruction_writes:Vec<Vec<(i32,String)>>,
    regs: BTreeSet<i32>,
    preds: BTreeSet<i32>,
    uses_cc: bool,
    uses_quad_lane: bool,
    uses_lmem: bool,
    uses_point_coord: bool,
    iface: Interface,
    warnings: Vec<String>,

    handle_regs: BTreeMap<i32, u32>,
    sampler_regs: BTreeMap<i32,u32>,
    query_only_textures:BTreeSet<u32>,
}

const RZ_: i32 = 255;

fn fimm(bits: u32) -> String {
    format!("as_type<float>({bits:#x}u)")
}

impl<'a> Gen<'a> {
    fn line(&mut self, s: impl AsRef<str>) {
        self.body.push_str("        ");
        self.body.push_str(s.as_ref());
        self.body.push('\n');
    }

    fn r(&mut self, n: i32) -> String {
        if n == RZ_ {
            "0u".into()
        } else {
            self.regs.insert(n);
            format!("r{n}")
        }
    }
    fn fr(&mut self, n: i32) -> String {
        if n == RZ_ { "0.0f".into() } else { format!("as_type<float>({})", self.r(n)) }
    }
    fn cb(&mut self, slot: i32, word: i32) -> String {
        // C1 belongs to this original program. The host's specialized shader
        // identity already includes every byte of a supplied embedded bank.
        // Keep live UBOs and dynamic C1 addressing as actual buffer reads.
        if slot==1 && word>=0 {
            if let Some(bytes)=self.embedded_constants.and_then(|c|c.get(word as usize*4..word as usize*4+4)) {
                return format!("{:#x}u",u32::from_le_bytes(bytes.try_into().unwrap()));
            }
        }
        format!("res.cb[{slot}][{word}]")
    }
    fn fcb(&mut self, slot: i32, word: i32) -> String {
        format!("as_type<float>({})", self.cb(slot, word))
    }
    fn half_src(raw:String,swizzle:u32,absolute:bool,negate:bool) -> String {
        let value=match swizzle {
            0=>format!("float2(as_type<half2>({raw}))"),
            1=>format!("float2(as_type<float>({raw}))"),
            2=>format!("float2(float(as_type<half2>({raw}).x))"),
            _=>format!("float2(float(as_type<half2>({raw}).y))"),
        };
        let value=if absolute {format!("abs({value})")} else {value};
        if negate {format!("(-{value})")} else {value}
    }
    fn half_pack(value:&str,old:&str,fmt:u32) -> String {
        let packed=format!("as_type<uint>(half2({value}))");
        match fmt {
            0=>packed,
            1=>format!("as_type<uint>({value}.x)"),
            2=>format!("(({old} & 0xffff0000u) | ({packed} & 0xffffu))"),
            _=>format!("(({old} & 0xffffu) | ({packed} & 0xffff0000u))"),
        }
    }
    fn set(&mut self, d: i32, expr: String) {
        if d != RZ_ {
            self.handle_regs.remove(&d);self.sampler_regs.remove(&d);
            self.regs.insert(d);
            if let Some(writes)=self.instruction_writes.last_mut() {writes.push((d,expr.clone()));}
            self.line(format!("r{d} = {expr};"));
        }
    }
    fn setf(&mut self, d: i32, fexpr: String) {
        self.set(d, format!("as_type<uint>({fexpr})"));
    }
    fn p(&mut self, n: i32, inv: bool) -> String {
        let base = if n == 7 {
            "true".to_string()
        } else {
            self.preds.insert(n);
            format!("p{n}")
        };
        if inv { format!("(!{base})") } else { base }
    }
    fn setp(&mut self, n: i32, expr: String) {
        if n != 7 {
            self.preds.insert(n);
            self.line(format!("p{n} = {expr};"));
        }
    }
    fn fmod(x: String, abs: bool, neg: bool) -> String {
        let x = if abs { format!("fabs({x})") } else { x };
        if neg { format!("(-{x})") } else { x }
    }
    fn sat(x: String, sat: bool) -> String {
        if sat { format!("saturate({x})") } else { x }
    }
    fn set_zn(&mut self, d: String) {
        self.uses_cc = true;
        self.line(format!("zf = ({d}) == 0u; nf = int({d}) < 0;"));
    }

    fn video_source(&mut self,reg:i32,select:VectorSelect)->Result<String> {
        let value=self.r(reg);let signed=select.0&8!=0;let mode=select.0&7;
        let (offset,bits)=match mode {0..=3=>(8*mode,8),4..=5=>(16*(mode-4),16),6=>(0,32),_=>bail!("invalid video component selector")};
        let extracted=if bits==32 {value} else {format!("bfe({value},{}u,{signed})",offset|(bits<<8))};
        Ok(if signed {format!("long(int({extracted}))")} else {format!("long(uint({extracted}))")})
    }

    fn condition(&mut self, c: Ccc) -> String {
        self.uses_cc=true;
        match c.0 {
            0=>"false".into(),2|10=>"zf".into(),5|13=>"!zf".into(),15=>"true".into(),
            _=>format!("ccc({}u, zf, nf, cf, vf)",c.0),
        }
    }

    fn label_target(&mut self, addr: u32) {
        self.line(format!("pc = {addr}u; continue;"));
    }

    fn attr_load(&mut self, addr: i32, output: bool) -> String {
        match (self.sph.stage, addr) {
            (Stage::Vertex, a) if output => self.vs_output_ref(a).map_or("0u".into(), |f| format!("as_type<uint>({f})")),
            (Stage::Vertex, ATTR_GENERIC..ATTR_GENERIC_END) => {
                let (i, c) = ((addr - ATTR_GENERIC) / 16, (addr - ATTR_GENERIC) / 4 % 4);
                self.iface.attributes |= 1 << i;
                format!("a{i}[{c}]")
            }
            (Stage::Vertex, ATTR_INSTANCE_ID) => "iid".into(),
            (Stage::Vertex, ATTR_VERTEX_ID) => "vid".into(),
            (Stage::Fragment, a) => self.fs_input(a, false),
            _ => {
                self.warnings.push(format!("attribute load {addr:#x}"));
                "0u".into()
            }
        }
    }

    fn vs_output_ref(&mut self, addr: i32) -> Option<String> {
        let c = ["x", "y", "z", "w"];
        Some(match addr {
            ATTR_POSITION..=0x7c => format!("o.pos.{}", c[((addr - ATTR_POSITION) / 4) as usize]),
            ATTR_GENERIC..ATTR_GENERIC_END => {
                let (i, k) = ((addr - ATTR_GENERIC) / 16, (addr - ATTR_GENERIC) / 4 % 4);
                self.iface.outputs |= 1 << i;
                format!("o.v{i}.{}", c[k as usize])
            }
            ATTR_POINT_SIZE => "o.psize".into(),
            _ => return None,
        })
    }

    fn fs_input(&mut self, addr: i32, perspective_scale: bool) -> String {
        let c = ["x", "y", "z", "w"];
        match addr {
            ATTR_POSITION..=0x7c => format!("as_type<uint>(in.pos.{})", c[((addr - ATTR_POSITION) / 4) as usize]),
            ATTR_GENERIC..ATTR_GENERIC_END => {
                let (i, k) = ((addr - ATTR_GENERIC) / 16, (addr - ATTR_GENERIC) / 4 % 4);
                self.iface.attributes |= 1 << i;
                let v = format!("in.v{i}.{}", c[k as usize]);

                if perspective_scale && self.sph.imap[i as usize] == Imap::Perspective {
                    format!("as_type<uint>({v} * in.pos.w)")
                } else {
                    format!("as_type<uint>({v})")
                }
            }
            ATTR_FRONT_FACING => "(front ? 0xffffffffu : 0u)".into(),
            ATTR_POINT_COORD | 0x2e4 => {
                self.uses_point_coord = true;
                format!("as_type<uint>(pcoord.{})", if addr == ATTR_POINT_COORD { "x" } else { "y" })
            }
            _ => {
                self.warnings.push(format!("fragment input {addr:#x}"));
                "0u".into()
            }
        }
    }

    fn texture_binding_id(binding:i32)->u32 {
        let offset=binding*4;
        if (0x168..0x568).contains(&offset) {0x1000+((offset-0x168)/8) as u32*128} else {((binding - 8) / 2).max(0) as u32}
    }
    fn tex(&mut self, binding: i32, kind: TexKind) -> u32 {
        self.tex_id(Self::texture_binding_id(binding),kind)
    }
    fn tex_id(&mut self,b:u32,kind:TexKind) -> u32 {
        if self.query_only_textures.remove(&b) {self.iface.textures.insert(b,kind);}
        let e = self.iface.textures.entry(b).or_insert(kind);
        if *e != kind {
            self.warnings.push(format!("texture binding {b} used as {:?} and {kind:?}", *e));
        }
        b
    }

    fn cbuf_handle(word:i32) -> (Option<u32>,Option<u32>) {
        let off=word*4;
        if off%8!=0 {return (None,None);}
        match off {
            0x20..0x120=>(Some(((off-0x20)/8) as u32),None),
            0x120..0x160=>(Some(0x200+((off-0x120)/8) as u32),None),
            0x168..0x568=>(Some(0x100+((off-0x168)/8) as u32),None),
            0x568..0x968=>(None,Some(((off-0x568)/8) as u32)),
            _=>(None,None),
        }
    }

    fn bindless_tex(&mut self,reg:i32,kind:TexKind) -> Result<u32> {
        let id=*self.handle_regs.get(&reg).ok_or_else(||anyhow::anyhow!("unresolved bindless texture handle in R{reg}"))?;
        let b=if (0x100..0x180).contains(&id) {
            let sampler=*self.sampler_regs.get(&reg).ok_or_else(||anyhow::anyhow!("separate texture R{reg} has no resolved sampler"))?;
            0x1000+(id-0x100)*128+sampler
        } else if id<32 {id} else {bail!("image handle used for texture sampling in R{reg}")};
        Ok(self.tex_id(b,kind))
    }

    fn bindless_refs(&mut self,reg:i32,kind:TexKind)->Result<(String,String,Option<String>)> {
        let resolved=self.handle_regs.get(&reg).is_some_and(|id| *id<32 || ((0x100..0x180).contains(id) && self.sampler_regs.contains_key(&reg)));
        if resolved {let b=self.bindless_tex(reg,kind)?;return Ok((format!("t{b}"),format!("s{b}"),None));}
        let (bit,bank)=match kind {TexKind::T2D=>(1,"color"),TexKind::Depth2D=>(2,"depth"),_=>bail!("runtime bindless texture kind {kind:?} is unsupported")};
        self.iface.dynamic_textures|=bit;
        Ok((format!("dyntex.{bank}[dynamic_texture]"),"dyntex.samplers[dynamic_sampler]".into(),Some(self.r(reg))))
    }

    fn bindless_open(&mut self,handle:Option<&str>,kind:TexKind) {
        self.line("{float4 sampled=float4(0.0f);");
        if let Some(handle)=handle {
            let (table,count)=if kind.is_depth() {("depth_lookup","y")} else {("color_lookup","x")};
            self.line(format!("uint dynamic_handle={handle};uint dynamic_texture=bindless_find(dyntex.{table},dyntex.counts.{count},dynamic_handle&0xfffffu);uint dynamic_sampler=bindless_find(dyntex.sampler_lookup,dyntex.counts.z,dynamic_handle>>20u);"));
            self.line("if(dynamic_texture==0xffffffffu || dynamic_sampler==0xffffffffu) {shader_fault(res,32u,pc);} else {");
        }
    }

    fn write_components(&mut self, value: &str, mask: u32, dests: &[i32]) {
        let mut k = 0;
        for c in 0..4 {
            if mask >> c & 1 == 0 {
                continue;
            }
            if let Some(&d) = dests.get(k) {
                self.set(d, format!("as_type<uint>({value}[{c}])"));
            }
            k += 1;
        }
    }

    fn coord(&mut self, regs: &[i32], n: usize) -> String {
        let v: Vec<String> = regs.iter().take(n).map(|&r| self.fr(r)).collect();
        match n {
            1 => v[0].clone(),
            2 => format!("float2({}, {})", v[0], v[1]),
            _ => format!("float3({}, {}, {})", v[0], v[1], v[2]),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn sample(&mut self, b: u32, kind: TexKind, coords: &[i32], array: Option<i32>, lod: Option<String>, bias: Option<String>, depth_ref: Option<i32>) -> String {
        self.sample_offset(b,kind,coords,array,lod,bias,depth_ref,None)
    }

    fn texture_offset(&mut self,reg:i32,dim:u32)->Result<String> {
        let count=match dim {0|1=>1,2|3=>2,4=>3,_=>bail!("texture offsets on cube/unsupported dimensions")};
        let packed=self.r(reg);let mut values=(0..count).map(|n|format!("(int({packed}<<{}u)>>28)",28-n*4)).collect::<Vec<_>>();
        if count==1 {values.push("0".into());}
        Ok(format!("int{}({})",values.len(),values.join(",")))
    }

    #[allow(clippy::too_many_arguments)]
    fn sample_offset(&mut self, b:u32,kind:TexKind,coords:&[i32],array:Option<i32>,lod:Option<String>,bias:Option<String>,depth_ref:Option<i32>,offset:Option<String>)->String {
        let n = kind.coords();
        let c = self.coord(coords, n);

        let arr = array.map(|a| format!(", ({} & 0xffffu)", self.r(a)));
        let level = match (&lod, &bias) {
            (Some(l), _) => format!(", level({l})"),
            (None, Some(bb)) => format!(", bias({bb})"),
            _ => String::new(),
        };
        let arr = arr.unwrap_or_default();
        let offset=offset.map(|v|format!(", {v}")).unwrap_or_default();
        if let Some(rf) = depth_ref {
            let rv = self.fr(rf);
            format!("float4(t{b}.sample_compare(s{b}, {c}{arr}, {rv}{level}{offset}))")
        } else if kind.is_depth() {
            format!("float4(t{b}.sample(s{b}, {c}{arr}{level}{offset}))")
        } else {
            format!("t{b}.sample(s{b}, {c}{arr}{level}{offset})")
        }
    }

    fn inst(&mut self, i: &Inst) -> Result<()> {
        let w = i.raw;
        macro_rules! op {
            ($t:ident) => {
                $t(w)
            };
        }

        let fb_r = |g: &mut Self, r: i32| g.fr(r);
        match i.op {
            Op::Invalid => bail!("invalid instruction {w:016x} at {:#x}", i.addr),
            Op::Nop | Op::Depbar => {}
            Op::Ssy|Op::Pbk => {
                if self.simt {
                    let target=branch_target(i);let kind=u32::from(i.op==Op::Pbk);
                    self.line(format!("if(token_sp>=32u) {{shader_fault(res,1u,{}u);live=false;continue;}}",i.addr));
                    self.line(format!("token_id[token_sp]={}u;token_target[token_sp]={target}u;token_kind[token_sp]={kind}u;token_mask[token_sp]=uint((simd_vote::vote_t)simd_ballot(true));token_sp++;",i.addr));
                }
            }
            Op::Membar => {
                if self.sph.stage!=Stage::Compute {bail!("graphics-stage memory barrier unsupported");}
                self.line("simdgroup_barrier(mem_flags::mem_device|mem_flags::mem_threadgroup);");
            }
            Op::Bar => {
                let o=op!(InstBar);
                if self.sph.stage!=Stage::Compute || o.bar_op().0!=0 || !o.a_fix_bar() || !o.b_fix_bar() || o.src_a()!=0 || o.imm12()!=0 {bail!("unsupported named/counted/reduction barrier form");}
                self.line("threadgroup_barrier(mem_flags::mem_threadgroup|mem_flags::mem_device);");
            }

            Op::FaddR | Op::FaddI | Op::FaddC | Op::Fadd32i => {
                let (d, a, b, abs_a, abs_b, neg_a, neg_b, s) = match i.op {
                    Op::FaddR => {
                        let o = op!(InstFaddR);
                        (o.dest(), self.fr(o.src_a()), fb_r(self, o.src_b()), o.abs_a(), o.abs_b(), o.neg_a(), o.neg_b(), o.sat())
                    }
                    Op::FaddI => {
                        let o = op!(InstFaddI);
                        (o.dest(), self.fr(o.src_a()), fimm((o.imm20() as u32) << 12), o.abs_a(), o.abs_b(), o.neg_a(), o.neg_b(), o.sat())
                    }
                    Op::FaddC => {
                        let o = op!(InstFaddC);
                        (o.dest(), self.fr(o.src_a()), self.fcb(o.cbuf_slot(), o.cbuf_offset()), o.abs_a(), o.abs_b(), o.neg_a(), o.neg_b(), o.sat())
                    }
                    _ => {
                        let o = op!(InstFadd32i);
                        (o.dest(), self.fr(o.src_a()), fimm(o.imm32() as u32), o.abs_a(), o.abs_b(), o.neg_a(), o.neg_b(), false)
                    }
                };
                let e = Self::sat(format!("({} + {})", Self::fmod(a, abs_a, neg_a), Self::fmod(b, abs_b, neg_b)), s);
                self.setf(d, e);
            }
            Op::FmulR | Op::FmulI | Op::FmulC | Op::Fmul32i => {
                let (d, a, b, neg_a, scale, s) = match i.op {
                    Op::FmulR => {
                        let o = op!(InstFmulR);
                        (o.dest(), self.fr(o.src_a()), fb_r(self, o.src_b()), o.neg_a(), o.scale().0, o.sat())
                    }
                    Op::FmulI => {
                        let o = op!(InstFmulI);
                        (o.dest(), self.fr(o.src_a()), fimm((o.imm20() as u32) << 12), o.neg_a(), o.scale().0, o.sat())
                    }
                    Op::FmulC => {
                        let o = op!(InstFmulC);
                        (o.dest(), self.fr(o.src_a()), self.fcb(o.cbuf_slot(), o.cbuf_offset()), o.neg_a(), o.scale().0, o.sat())
                    }
                    _ => {
                        let o = op!(InstFmul32i);
                        (o.dest(), self.fr(o.src_a()), fimm(o.imm32() as u32), false, 0, o.sat())
                    }
                };
                let k = ["", " * 0.5f", " * 0.25f", " * 0.125f", " * 8.0f", " * 4.0f", " * 2.0f"].get(scale as usize).copied().unwrap_or("");
                let e = Self::sat(format!("({} * {}{k})", Self::fmod(a, false, neg_a), b), s);
                self.setf(d, e);
            }
            Op::FfmaR | Op::FfmaI | Op::FfmaC | Op::FfmaRc | Op::Ffma32i => {
                let (d, a, b, c, neg_a, neg_c, s) = match i.op {
                    Op::FfmaR => {
                        let o = op!(InstFfmaR);
                        (o.dest(), self.fr(o.src_a()), self.fr(o.src_b()), self.fr(o.src_c()), o.neg_a(), o.neg_c(), o.sat())
                    }
                    Op::FfmaI => {
                        let o = op!(InstFfmaI);
                        (o.dest(), self.fr(o.src_a()), fimm((o.imm20() as u32) << 12), self.fr(o.src_c()), o.neg_a(), o.neg_c(), o.sat())
                    }
                    Op::FfmaC => {
                        let o = op!(InstFfmaC);
                        (o.dest(), self.fr(o.src_a()), self.fcb(o.cbuf_slot(), o.cbuf_offset()), self.fr(o.src_c()), o.neg_a(), o.neg_c(), o.sat())
                    }
                    Op::FfmaRc => {
                        let o = op!(InstFfmaRc);
                        (o.dest(), self.fr(o.src_a()), self.fr(o.src_c()), self.fcb(o.cbuf_slot(), o.cbuf_offset()), o.neg_a(), o.neg_c(), o.sat())
                    }
                    _ => {
                        let o = op!(InstFfma32i);
                        (o.dest(), self.fr(o.src_a()), fimm(o.imm32() as u32), self.fr(o.dest()), o.neg_a(), o.neg_c(), o.sat())
                    }
                };
                let e = Self::sat(format!("fma({}, {b}, {})", Self::fmod(a, false, neg_a), Self::fmod(c, false, neg_c)), s);
                self.setf(d, e);
            }
            Op::FmnmxR | Op::FmnmxI | Op::FmnmxC => {
                let (d, a, b, abs_a, abs_b, neg_a, neg_b, sp, spi) = match i.op {
                    Op::FmnmxR => {
                        let o = op!(InstFmnmxR);
                        (o.dest(), self.fr(o.src_a()), self.fr(o.src_b()), o.abs_a(), o.abs_b(), o.neg_a(), o.neg_b(), o.src_pred(), o.src_pred_inv())
                    }
                    Op::FmnmxI => {
                        let o = op!(InstFmnmxI);
                        (o.dest(), self.fr(o.src_a()), fimm((o.imm20() as u32) << 12), o.abs_a(), o.abs_b(), o.neg_a(), o.neg_b(), o.src_pred(), o.src_pred_inv())
                    }
                    _ => {
                        let o = op!(InstFmnmxC);
                        (o.dest(), self.fr(o.src_a()), self.fcb(o.cbuf_slot(), o.cbuf_offset()), o.abs_a(), o.abs_b(), o.neg_a(), o.neg_b(), o.src_pred(), o.src_pred_inv())
                    }
                };
                let (a, b) = (Self::fmod(a, abs_a, neg_a), Self::fmod(b, abs_b, neg_b));
                let p = self.p(sp, spi);
                self.setf(d, format!("({p} ? fmin({a}, {b}) : fmax({a}, {b}))"));
            }
            Op::Mufu => {
                let o = op!(InstMufu);
                let a = Self::fmod(self.fr(o.src_a()), o.abs_a(), o.neg_a());
                let e = match o.mufu_op() {
                    MufuOp::Cos => format!("cos({a})"),
                    MufuOp::Sin => format!("sin({a})"),
                    MufuOp::Ex2 => format!("exp2({a})"),
                    MufuOp::Lg2 => format!("log2({a})"),
                    // Compute kernels also pack geometry data; even a one-ULP
                    // change can alter its integer output. Keep their original
                    // Safe Metal arithmetic and gate only graphics estimates.
                    MufuOp::Rcp if self.sph.stage==Stage::Compute => format!("(1.0f / {a})"),
                    MufuOp::Rcp => format!("nx_mufu_rcp({a})"),
                    MufuOp::Rcp64h => format!("(1.0f / {a})"),
                    MufuOp::Rsq if self.sph.stage==Stage::Compute => format!("rsqrt({a})"),
                    MufuOp::Rsq => format!("nx_mufu_rsqrt({a})"),
                    MufuOp::Rsq64h => format!("rsqrt({a})"),
                    MufuOp::Sqrt => format!("sqrt({a})"),
                    m => bail!("mufu op {m:?}"),
                };
                self.setf(o.dest(), Self::sat(e, o.sat()));
            }
            Op::RroR | Op::RroC => {
                let (d, b, abs_b, neg_b) = if i.op == Op::RroR {
                    let o = op!(InstRroR);
                    (o.dest(), self.fr(o.src_b()), o.abs_b(), o.neg_b())
                } else {
                    let o = op!(InstRroC);
                    (o.dest(), self.fcb(o.cbuf_slot(), o.cbuf_offset()), o.abs_b(), o.neg_b())
                };
                self.setf(d, Self::fmod(b, abs_b, neg_b));
            }
            Op::FsetpR | Op::FsetpI | Op::FsetpC => {
                let (dp, dpi, a, b, aa, ab, na, nb, cmp, sp, spi, bop) = match i.op {
                    Op::FsetpR => {
                        let o = op!(InstFsetpR);
                        (o.dest_pred(), o.dest_pred_inv(), self.fr(o.src_a()), self.fr(o.src_b()), o.abs_a(), o.abs_b(), o.neg_a(), o.neg_b(), o.f_comp().0, o.src_pred(), o.src_pred_inv(), o.bop().0)
                    }
                    Op::FsetpI => {
                        let o = op!(InstFsetpI);
                        (o.dest_pred(), o.dest_pred_inv(), self.fr(o.src_a()), fimm((o.imm20() as u32) << 12), o.abs_a(), o.abs_b(), o.neg_a(), o.neg_b(), o.f_comp().0, o.src_pred(), o.src_pred_inv(), o.bop().0)
                    }
                    _ => {
                        let o = op!(InstFsetpC);
                        (o.dest_pred(), o.dest_pred_inv(), self.fr(o.src_a()), self.fcb(o.cbuf_slot(), o.cbuf_offset()), o.abs_a(), o.abs_b(), o.neg_a(), o.neg_b(), o.f_comp().0, o.src_pred(), o.src_pred_inv(), o.bop().0)
                    }
                };
                let c = format!("fcmp({cmp}u, {}, {})", Self::fmod(a, aa, na), Self::fmod(b, ab, nb));
                let p = self.p(sp, spi);
                self.line(format!("{{ bool c = {c};"));
                self.setp(dp, format!("bop({bop}u, c, {p})"));
                self.setp(dpi, format!("bop({bop}u, !c, {p})"));
                self.line("}");
            }
            Op::FsetR | Op::FsetI | Op::FsetC => {
                let (d, a, b, aa, ab, na, nb, cmp, sp, spi, bop) = match i.op {
                    Op::FsetR => {
                        let o = op!(InstFsetR);
                        (o.dest(), self.fr(o.src_a()), self.fr(o.src_b()), o.abs_a(), o.abs_b(), o.neg_a(), o.neg_b(), o.f_comp().0, o.src_pred(), o.src_pred_inv(), o.bop().0)
                    }
                    Op::FsetI => {
                        let o = op!(InstFsetI);
                        (o.dest(), self.fr(o.src_a()), fimm((o.imm20() as u32) << 12), o.abs_a(), o.abs_b(), o.neg_a(), o.neg_b(), o.f_comp().0, o.src_pred(), o.src_pred_inv(), o.bop().0)
                    }
                    _ => {
                        let o = op!(InstFsetC);
                        (o.dest(), self.fr(o.src_a()), self.fcb(o.cbuf_slot(), o.cbuf_offset()), o.abs_a(), o.abs_b(), o.neg_a(), o.neg_b(), o.f_comp().0, o.src_pred(), o.src_pred_inv(), o.bop().0)
                    }
                };
                let bf = (w >> 52) & 1 != 0;
                let p = self.p(sp, spi);
                let c = format!("bop({bop}u, fcmp({cmp}u, {}, {}), {p})", Self::fmod(a, aa, na), Self::fmod(b, ab, nb));
                let v = if bf { format!("({c} ? 0x3f800000u : 0u)") } else { format!("({c} ? 0xffffffffu : 0u)") };
                self.line(format!("{{uint comparison={v};"));self.set(d,"comparison".into());
                if w&(1<<47)!=0 {self.set_zn("comparison".into());}
                self.line("}");
            }
            Op::Fswzadd => {
                let o = op!(InstFswzadd);
                let (a, b) = (self.fr(o.src_a()), self.fr(o.src_b()));
                self.setf(o.dest(), format!("({a} + {b})"));
            }

            Op::Hadd2R | Op::Hadd2I => {
                let (d,a,b,fmt,sat)=if i.op==Op::Hadd2R {
                    let o=op!(InstHadd2R);
                    let a=Self::half_src(self.r(o.src_a()),o.a_swizzle().0,o.abs_a(),o.neg_a());
                    let b=Self::half_src(self.r(o.src_b()),o.b_swizzle().0,o.abs_b(),o.neg_b());
                    (o.dest(),a,b,o.o_fmt().0,o.sat())
                } else {
                    let o=op!(InstHadd2I);
                    let a=Self::half_src(self.r(o.src_a()),o.a_swizzle().0,o.abs_a(),o.neg_a());
                    let imm=(((o.bimm_h1() as u32)<<6)<<16)|((o.bimm_h0() as u32)<<6);
                    (o.dest(),a,Self::half_src(format!("{imm:#x}u"),0,false,false),o.o_fmt().0,o.sat())
                };
                let old=self.r(d);
                self.line(format!("{{ float2 hv = {a} + {b};"));
                if sat {self.line("hv=clamp(hv,float2(0.0f),float2(1.0f));");}
                self.set(d,Self::half_pack("hv",&old,fmt));self.line("}");
            }
            Op::Hsetp2R => {
                let o = op!(InstHsetp2R);
                let a=Self::half_src(self.r(o.src_a()),o.a_swizzle().0,o.abs_a(),o.neg_a());
                let b=Self::half_src(self.r(o.src_b()),o.b_swizzle().0,o.abs_b(),o.neg_b());
                let p = self.p(o.src_pred(), o.src_pred_inv());
                let cmp = o.f_comp2().0;
                self.line(format!("{{ float2 ha = {a}, hb = {b}; bool c0 = fcmp({cmp}u, ha.x, hb.x), c1 = fcmp({cmp}u, ha.y, hb.y);"));
                let bop = o.bop().0;
                if o.h_and() {
                    self.setp(o.dest_pred(), format!("bop({bop}u, c0 && c1, {p})"));
                    self.setp(o.dest_pred_inv(), format!("bop({bop}u, !(c0 && c1), {p})"));
                } else {
                    self.setp(o.dest_pred(), format!("bop({bop}u, c0, {p})"));
                    self.setp(o.dest_pred_inv(), format!("bop({bop}u, c1, {p})"));
                }
                self.line("}");
            }

            Op::LeaHiR|Op::LeaHiC => {
                let (d,a,b,c,shift,neg,extended,cc,pred)=if i.op==Op::LeaHiR {
                    let o=op!(InstLeaHiR);(o.dest(),self.r(o.src_a()),self.r(o.src_b()),self.r(o.src_c()),o.imm_u5(),o.neg_a(),o.x(),o.write_cc(),o.dest_pred())
                } else {
                    let o=op!(InstLeaHiC);(o.dest(),self.r(o.src_a()),self.cb(o.cbuf_slot(),o.cbuf_offset()),self.r(o.src_c()),o.imm_u5(),o.neg_a(),o.x(),o.write_cc(),o.dest_pred())
                };
                if extended || cc || pred!=7 {bail!("LEA.HI extended carry/predicate flags are not implemented");}
                self.line(format!("{{ ulong address=((ulong({c})<<32)|ulong({a})) << {shift}u;"));
                if neg {self.line("address=0ul-address;");}
                self.line(format!("ulong sum=(address>>32)+ulong({b});"));
                self.set(d,"uint(sum)".into());self.line("}");
            }
            Op::IaddR | Op::IaddI | Op::IaddC | Op::Iadd32i => {
                let (d, a, b, avg, x, cc) = match i.op {
                    Op::IaddR => {
                        let o = op!(InstIaddR);
                        (o.dest(), self.r(o.src_a()), self.r(o.src_b()), o.avg_mode().0, o.x(), o.write_cc())
                    }
                    Op::IaddI => {
                        let o = op!(InstIaddI);
                        (o.dest(), self.r(o.src_a()), format!("{:#x}u", ((o.imm20() << 12) >> 12) as u32), o.avg_mode().0, o.x(), o.write_cc())
                    }
                    Op::IaddC => {
                        let o = op!(InstIaddC);
                        (o.dest(), self.r(o.src_a()), self.cb(o.cbuf_slot(), o.cbuf_offset()), o.avg_mode().0, o.x(), o.write_cc())
                    }
                    _ => {
                        let o = op!(InstIadd32i);
                        (o.dest(), self.r(o.src_a()), format!("{:#x}u", o.imm32() as u32), o.avg_mode().0, o.x(), o.write_cc())
                    }
                };
                let a = if avg == 2 { format!("(0u - {a})") } else { a };
                let b = if avg == 1 { format!("(0u - {b})") } else { b };
                let one = if avg == 3 { " + 1u" } else { "" };
                let carry = if x {
                    self.uses_cc = true;
                    " + uint(cf)"
                } else {
                    ""
                };
                if cc {
                    self.uses_cc = true;
                    self.line(format!("{{ ulong t = ulong({a}) + ulong({b}){one}{carry}; cf = (t >> 32) != 0ul; uint rr = uint(t); vf = (int((rr ^ {a}) & ~({a} ^ {b}))) < 0; zf = rr == 0u; nf = int(rr) < 0;"));
                    self.set(d, "rr".into());
                    self.line("}");
                } else {
                    self.set(d, format!("({a} + {b}{one}{carry})"));
                }
            }
            Op::Iadd3R | Op::Iadd3I | Op::Iadd3C => {
                let (d, a, b, c, na, nb, nc, lrs, ap, bp, cp) = match i.op {
                    Op::Iadd3R => {
                        let o = op!(InstIadd3R);
                        (o.dest(), self.r(o.src_a()), self.r(o.src_b()), self.r(o.src_c()), o.neg_a(), o.neg_b(), o.neg_c(), o.lrs().0, o.apart().0, o.bpart().0, o.cpart().0)
                    }
                    Op::Iadd3I => {
                        let o = op!(InstIadd3I);
                        (o.dest(), self.r(o.src_a()), format!("{:#x}u", ((o.imm20() << 12) >> 12) as u32), self.r(o.src_c()), o.neg_a(), o.neg_b(), o.neg_c(), 0, 0, 0, 0)
                    }
                    _ => {
                        let o = op!(InstIadd3C);
                        (o.dest(), self.r(o.src_a()), self.cb(o.cbuf_slot(), o.cbuf_offset()), self.r(o.src_c()), o.neg_a(), o.neg_b(), o.neg_c(), 0, 0, 0, 0)
                    }
                };
                let part = |x: String, p: u32| match p {
                    1 => format!("({x} & 0xffffu)"),
                    2 => format!("({x} >> 16)"),
                    _ => x,
                };
                let ng = |x: String, n: bool| if n { format!("(0u - {x})") } else { x };
                let (a, b, c) = (ng(part(a, ap), na), ng(part(b, bp), nb), ng(part(c, cp), nc));
                let ab = match lrs {
                    1 => format!("(({a} + {b}) >> 16)"),
                    2 => format!("(({a} + {b}) << 16)"),
                    _ => format!("({a} + {b})"),
                };
                self.set(d, format!("({ab} + {c})"));
            }
            Op::IscaddR | Op::IscaddI | Op::IscaddC => {
                let (d, a, b, sh, avg) = match i.op {
                    Op::IscaddR => {
                        let o = op!(InstIscaddR);
                        (o.dest(), self.r(o.src_a()), self.r(o.src_b()), o.imm5(), o.avg_mode().0)
                    }
                    Op::IscaddI => {
                        let o = op!(InstIscaddI);
                        (o.dest(), self.r(o.src_a()), format!("{:#x}u", ((o.imm20() << 12) >> 12) as u32), o.imm5(), o.avg_mode().0)
                    }
                    _ => {
                        let o = op!(InstIscaddC);
                        (o.dest(), self.r(o.src_a()), self.cb(o.cbuf_slot(), o.cbuf_offset()), o.imm5(), o.avg_mode().0)
                    }
                };
                let a = if avg == 2 { format!("(0u - {a})") } else { a };
                let b = if avg == 1 { format!("(0u - {b})") } else { b };
                self.set(d, format!("(({a} << {sh}u) + {b})"));
            }
            Op::XmadR | Op::XmadI | Op::XmadC | Op::XmadRc => {
                let (d, a, b, c, cop, sa, sb, ha, hb, psl, mrg, x, write_cc) = match i.op {
                    Op::XmadR => {
                        let o = op!(InstXmadR);
                        (o.dest(), self.r(o.src_a()), self.r(o.src_b()), self.r(o.src_c()), o.xmad_cop().0, o.a_signed(), o.b_signed(), o.hilo_a(), o.hilo_b(), o.psl(), o.mrg(), o.x(), o.write_cc())
                    }
                    Op::XmadI => {
                        let o = op!(InstXmadI);
                        (o.dest(), self.r(o.src_a()), format!("{:#x}u", o.imm16() as u32 & 0xffff), self.r(o.src_c()), o.xmad_cop().0, o.a_signed(), o.b_signed(), o.hilo_a(), false, o.psl(), o.mrg(), o.x(), o.write_cc())
                    }
                    Op::XmadC => {
                        let o = op!(InstXmadC);
                        (o.dest(), self.r(o.src_a()), self.cb(o.cbuf_slot(), o.cbuf_offset()), self.r(o.src_c()), o.xmad_cop().0, o.a_signed(), o.b_signed(), o.hilo_a(), o.hilo_b(), o.psl(), o.mrg(), o.x(), o.write_cc())
                    }
                    _ => {
                        let o = op!(InstXmadRc);
                        (o.dest(), self.r(o.src_a()), self.r(o.src_c()), self.cb(o.cbuf_slot(), o.cbuf_offset()), o.xmad_cop().0, o.a_signed(), o.b_signed(), o.hilo_a(), o.hilo_b(), false, false, o.x(), o.write_cc())
                    }
                };
                let carry = if x {
                    self.uses_cc = true;
                    "uint(cf)"
                } else {
                    "0xffffffffu"
                };
                let v = format!("xmad({a}, {b}, {c}, {cop}u, {sa}, {sb}, {ha}, {hb}, {psl}, {mrg}, {carry})");
                if write_cc {

                    self.uses_cc = true;
                    let p = format!("xmad({a}, {b}, 0u, 0u, {sa}, {sb}, {ha}, {hb}, {psl}, false, 0xffffffffu)");
                    self.line(format!("{{ uint xr = {v}; uint xp = {p}; uint xc = {c}; cf = xr < xp; vf = int((xr ^ xp) & ~(xp ^ xc)) < 0; zf = xr == 0u; nf = int(xr) < 0;"));
                    self.set(d, "xr".into());
                    self.line("}");
                } else {
                    self.set(d, v);
                }
            }
            Op::ImnmxR | Op::ImnmxI | Op::ImnmxC => {
                let (d, a, b, s, sp, spi) = match i.op {
                    Op::ImnmxR => {
                        let o = op!(InstImnmxR);
                        (o.dest(), self.r(o.src_a()), self.r(o.src_b()), o.signed(), o.src_pred(), o.src_pred_inv())
                    }
                    Op::ImnmxI => {
                        let o = op!(InstImnmxI);
                        (o.dest(), self.r(o.src_a()), format!("{:#x}u", ((o.imm20() << 12) >> 12) as u32), o.signed(), o.src_pred(), o.src_pred_inv())
                    }
                    _ => {
                        let o = op!(InstImnmxC);
                        (o.dest(), self.r(o.src_a()), self.cb(o.cbuf_slot(), o.cbuf_offset()), o.signed(), o.src_pred(), o.src_pred_inv())
                    }
                };
                let p = self.p(sp, spi);
                let e = if s {
                    format!("({p} ? uint(min(int({a}), int({b}))) : uint(max(int({a}), int({b}))))")
                } else {
                    format!("({p} ? min({a}, {b}) : max({a}, {b}))")
                };
                self.set(d, e);
            }
            Op::IsetpR | Op::IsetpI | Op::IsetpC => {
                let (a, b, cmp, s, bop, sp, spi, dp, dpi) = match i.op {
                    Op::IsetpR => {
                        let o = op!(InstIsetpR);
                        (self.r(o.src_a()), self.r(o.src_b()), o.i_comp().0, o.signed(), o.bop().0, o.src_pred(), o.src_pred_inv(), o.dest_pred(), o.dest_pred_inv())
                    }
                    Op::IsetpI => {
                        let o = op!(InstIsetpI);
                        (self.r(o.src_a()), format!("{:#x}u", ((o.imm20() << 12) >> 12) as u32), o.i_comp().0, o.signed(), o.bop().0, o.src_pred(), o.src_pred_inv(), o.dest_pred(), o.dest_pred_inv())
                    }
                    _ => {
                        let o = op!(InstIsetpC);
                        (self.r(o.src_a()), self.cb(o.cbuf_slot(), o.cbuf_offset()), o.i_comp().0, o.signed(), o.bop().0, o.src_pred(), o.src_pred_inv(), o.dest_pred(), o.dest_pred_inv())
                    }
                };
                let p = self.p(sp, spi);
                self.line(format!("{{ bool c = icmp({cmp}u, {a}, {b}, {s});"));
                self.setp(dp, format!("bop({bop}u, c, {p})"));
                self.setp(dpi, format!("bop({bop}u, !c, {p})"));
                self.line("}");
            }
            Op::IsetR | Op::IsetI | Op::IsetC => {
                let (d, a, b, cmp, s, bop, sp, spi, bv,cc,extended) = match i.op {
                    Op::IsetR => {
                        let o = op!(InstIsetR);
                        (o.dest(), self.r(o.src_a()), self.r(o.src_b()), o.i_comp().0, o.signed(), o.bop().0, o.src_pred(), o.src_pred_inv(), o.b_val(),o.write_cc(),o.x())
                    }
                    Op::IsetI => {
                        let o = op!(InstIsetI);
                        (o.dest(), self.r(o.src_a()), format!("{:#x}u", ((o.imm20() << 12) >> 12) as u32), o.i_comp().0, o.signed(), o.bop().0, o.src_pred(), o.src_pred_inv(), o.b_val(),o.write_cc(),o.x())
                    }
                    _ => {
                        let o = op!(InstIsetC);
                        (o.dest(), self.r(o.src_a()), self.cb(o.cbuf_slot(), o.cbuf_offset()), o.i_comp().0, o.signed(), o.bop().0, o.src_pred(), o.src_pred_inv(), o.b_val(),o.write_cc(),o.x())
                    }
                };
                let p = self.p(sp, spi);
                let c = format!("bop({bop}u, icmp({cmp}u, {a}, {b}, {s}), {p})");
                if extended {bail!("extended integer comparison requires multiword condition handling");}
                self.line(format!("{{uint comparison={c}?{}:0u;",if bv {"0x3f800000u"} else {"0xffffffffu"}));
                self.set(d,"comparison".into());
                if cc {self.set_zn("comparison".into());}
                self.line("}");
            }
            Op::SelR | Op::SelI | Op::SelC => {
                let (d, a, b, sp, spi) = match i.op {
                    Op::SelR => {
                        let o = op!(InstSelR);
                        (o.dest(), self.r(o.src_a()), self.r(o.src_b()), o.src_pred(), o.src_pred_inv())
                    }
                    Op::SelI => {
                        let o = op!(InstSelI);
                        (o.dest(), self.r(o.src_a()), format!("{:#x}u", ((o.imm20() << 12) >> 12) as u32), o.src_pred(), o.src_pred_inv())
                    }
                    _ => {
                        let o = op!(InstSelC);
                        (o.dest(), self.r(o.src_a()), self.cb(o.cbuf_slot(), o.cbuf_offset()), o.src_pred(), o.src_pred_inv())
                    }
                };
                let p = self.p(sp, spi);
                self.set(d, format!("({p} ? {a} : {b})"));
            }

            Op::LopR | Op::LopI | Op::LopC | Op::Lop32i => {
                let (d, a, b, lop, na, nb, dp, pop, cc) = match i.op {
                    Op::LopR => {
                        let o = op!(InstLopR);
                        (o.dest(), self.r(o.src_a()), self.r(o.src_b()), o.lop().0, o.neg_a(), o.neg_b(), o.dest_pred(), o.predicate_op().0, o.write_cc())
                    }
                    Op::LopI => {
                        let o = op!(InstLopI);
                        (o.dest(), self.r(o.src_a()), format!("{:#x}u", ((o.imm20() << 12) >> 12) as u32), o.logic_op().0, o.neg_a(), o.neg_b(), o.dest_pred(), o.predicate_op().0, o.write_cc())
                    }
                    Op::LopC => {
                        let o = op!(InstLopC);
                        (o.dest(), self.r(o.src_a()), self.cb(o.cbuf_slot(), o.cbuf_offset()), o.logic_op().0, o.neg_a(), o.neg_b(), o.dest_pred(), o.predicate_op().0, o.write_cc())
                    }
                    _ => {
                        let o = op!(InstLop32i);
                        (o.dest(), self.r(o.src_a()), format!("{:#x}u", o.imm32() as u32), o.logic_op().0, o.neg_a(), o.neg_b(), 7, 0, o.write_cc())
                    }
                };
                let mut provenance=(None,None);
                if lop==1 && !na && !nb {
                    let (left,right)=match i.op {
                        Op::LopR=>{let o=op!(InstLopR);((self.handle_regs.get(&o.src_a()).copied(),self.sampler_regs.get(&o.src_a()).copied()),(self.handle_regs.get(&o.src_b()).copied(),self.sampler_regs.get(&o.src_b()).copied()))},
                        Op::LopC=>{let o=op!(InstLopC);((self.handle_regs.get(&o.src_a()).copied(),self.sampler_regs.get(&o.src_a()).copied()),if o.cbuf_slot()==2 {Self::cbuf_handle(o.cbuf_offset())} else {(None,None)})},
                        _=>((None,None),(None,None)),
                    };
                    provenance=(left.0.or(right.0),left.1.or(right.1));
                }
                let a = if na { format!("(~{a})") } else { a };
                let b = if nb { format!("(~{b})") } else { b };
                let e = match lop {
                    0 => format!("({a} & {b})"),
                    1 => format!("({a} | {b})"),
                    2 => format!("({a} ^ {b})"),
                    _ => b,
                };
                self.line(format!("{{ uint v = {e};"));
                self.setp(dp, match pop {
                    0 => "false".into(),
                    1 => "true".into(),
                    2 => "v == 0u".into(),
                    _ => "v != 0u".into(),
                });
                if cc {
                    self.set_zn("v".into());
                }
                self.set(d, "v".into());
                if let Some(t)=provenance.0 {self.handle_regs.insert(d,t);}
                if let Some(s)=provenance.1 {self.sampler_regs.insert(d,s);}
                self.line("}");
            }
            Op::Lop3R | Op::Lop3I | Op::Lop3C => {
                let (d, a, b, c, lut, dp, pop) = match i.op {
                    Op::Lop3R => {
                        let o = op!(InstLop3R);
                        (o.dest(), self.r(o.src_a()), self.r(o.src_b()), self.r(o.src_c()), o.imm(), o.dest_pred(), o.predicate_op().0)
                    }
                    Op::Lop3I => {
                        let o = op!(InstLop3I);
                        (o.dest(), self.r(o.src_a()), format!("{:#x}u", ((o.imm20() << 12) >> 12) as u32), self.r(o.src_c()), o.imm(), 7, 0)
                    }
                    _ => {
                        let o = op!(InstLop3C);
                        (o.dest(), self.r(o.src_a()), self.cb(o.cbuf_slot(), o.cbuf_offset()), self.r(o.src_c()), o.imm(), 7, 0)
                    }
                };
                self.line(format!("{{ uint v = lop3({a}, {b}, {c}, {lut}u);"));
                self.setp(dp, match pop {
                    0 => "false".into(),
                    1 => "true".into(),
                    2 => "v == 0u".into(),
                    _ => "v != 0u".into(),
                });
                self.set(d, "v".into());
                self.line("}");
            }
            Op::ShlR | Op::ShlI | Op::ShlC => {
                let (d, a, b, m) = match i.op {
                    Op::ShlR => {
                        let o = op!(InstShlR);
                        (o.dest(), self.r(o.src_a()), self.r(o.src_b()), o.m())
                    }
                    Op::ShlI => {
                        let o = op!(InstShlI);
                        (o.dest(), self.r(o.src_a()), format!("{:#x}u", ((o.imm20() << 12) >> 12) as u32), o.m())
                    }
                    _ => {
                        let o = op!(InstShlC);
                        (o.dest(), self.r(o.src_a()), self.cb(o.cbuf_slot(), o.cbuf_offset()), o.m())
                    }
                };
                let e = if m { format!("({a} << ({b} & 31u))") } else { format!("({b} >= 32u ? 0u : {a} << {b})") };
                self.set(d, e);
            }
            Op::ShrR | Op::ShrI | Op::ShrC => {
                let (d, a, b, s, m, brev) = match i.op {
                    Op::ShrR => {
                        let o = op!(InstShrR);
                        (o.dest(), self.r(o.src_a()), self.r(o.src_b()), o.signed(), o.m(), o.brev())
                    }
                    Op::ShrI => {
                        let o = op!(InstShrI);
                        (o.dest(), self.r(o.src_a()), format!("{:#x}u", ((o.imm20() << 12) >> 12) as u32), o.signed(), o.m(), o.brev())
                    }
                    _ => {
                        let o = op!(InstShrC);
                        (o.dest(), self.r(o.src_a()), self.cb(o.cbuf_slot(), o.cbuf_offset()), o.signed(), o.m(), o.brev())
                    }
                };
                let a = if brev { format!("reverse_bits({a})") } else { a };
                let sh = if m { format!("({b} & 31u)") } else { format!("min({b}, 31u)") };
                let e = if s {
                    format!("uint(int({a}) >> {sh})")
                } else if m {
                    format!("({a} >> {sh})")
                } else {
                    format!("({b} >= 32u ? 0u : {a} >> {b})")
                };
                self.set(d, e);
            }
            Op::BfeR | Op::BfeI | Op::BfeC => {
                let (d, a, b, s) = match i.op {
                    Op::BfeR => {
                        let o = op!(InstBfeR);
                        (o.dest(), self.r(o.src_a()), self.r(o.src_b()), o.signed())
                    }
                    Op::BfeI => {
                        let o = op!(InstBfeI);
                        (o.dest(), self.r(o.src_a()), format!("{:#x}u", ((o.imm20() << 12) >> 12) as u32), o.signed())
                    }
                    _ => {
                        let o = op!(InstBfeC);
                        (o.dest(), self.r(o.src_a()), self.cb(o.cbuf_slot(), o.cbuf_offset()), o.signed())
                    }
                };
                self.set(d, format!("bfe({a}, {b}, {s})"));
            }

            Op::I2fR | Op::I2fI | Op::I2fC => {
                let (d, b, fmt, sel, abs, neg, dst, rnd, cc) = match i.op {
                    Op::I2fR => {
                        let o = op!(InstI2fR);
                        (o.dest(), self.r(o.src_b()), o.i_src_fmt().0, o.byte_sel().0, o.abs_b(), o.neg_b(), o.dst_fmt().0, o.round_mode().0, o.write_cc())
                    }
                    Op::I2fI => {
                        let o = op!(InstI2fI);
                        (o.dest(), format!("{:#x}u", ((o.imm20() << 12) >> 12) as u32), o.i_src_fmt().0, o.byte_sel().0, o.abs_b(), o.neg_b(), o.dst_fmt().0, o.round_mode().0, o.write_cc())
                    }
                    _ => {
                        let o = op!(InstI2fC);
                        (o.dest(), self.cb(o.cbuf_slot(), o.cbuf_offset()), o.i_src_fmt().0, o.byte_sel().0, o.abs_b(), o.neg_b(), o.dst_fmt().0, o.round_mode().0, o.write_cc())
                    }
                };
                if cc || !matches!(dst,1|2) || !matches!(fmt,0|1|2|4|5|6) {bail!("unsupported I2F destination, integer width, or condition-code output");}
                if rnd==0 {
                    let value=Self::fmod(format!("i2f({b}, {fmt}u, {sel}u)"),abs,neg);
                    if dst==1 {self.set(d,format!("as_type<uint>(half2(half({value}),0.0h))"));}
                    else {self.setf(d,value);}
                } else {
                    // Preserve the exact integer while deciding whether the
                    // nearest float must move one ULP for directed rounding.
                    let exact=match fmt {
                        0=>format!("long(({b}>>{}u)&255u)",sel*8),
                        1=>format!("long(({b}>>{}u)&65535u)",(sel>>1)*16),
                        2=>format!("long({b})"),
                        4=>format!("long(int({b}<<{}u)>>24)",24-sel*8),
                        5=>format!("long(int({b}<<{}u)>>16)",16-(sel>>1)*16),
                        _=>format!("long(int({b}))"),
                    };
                    self.line(format!("{{long integer={exact};"));
                    if abs {self.line("if(integer<0l) integer=-integer;");}
                    if neg {self.line("integer=-integer;");}
                    self.line("float converted=float(integer);");
                    let (rounded,comparison)=if dst==1 {
                        self.line("uint bits=uint(as_type<ushort>(half(converted)));float nearest=float(as_type<half>(ushort(bits)));");
                        ("nearest","converted")
                    } else {
                        self.line("uint bits=as_type<uint>(converted);");
                        ("long(converted)","integer")
                    };
                    let condition=match rnd {
                        1=>format!("{rounded}>{comparison}"),
                        2=>format!("{rounded}<{comparison}"),
                        _=>format!("(integer>0l && {rounded}>{comparison}) || (integer<0l && {rounded}<{comparison})"),
                    };
                    let delta=match rnd {1=>"converted>0.0f?-1:1",2=>"converted>0.0f?1:-1",_=>"-1"};
                    self.line(format!("if({condition}) bits+=uint({delta});"));
                    self.set(d,"bits".into());self.line("}");
                }
            }
            Op::F2iR | Op::F2iI | Op::F2iC => {
                let (d, b, abs, neg, sh, dst, src, rnd) = match i.op {
                    Op::F2iR => {
                        let o = op!(InstF2iR);
                        (o.dest(), self.r(o.src_b()), o.abs_b(), o.neg_b(), o.sh(), o.i_dst_fmt().0, o.src_fmt().0, o.round_mode().0)
                    }
                    Op::F2iI => {
                        let o = op!(InstF2iI);
                        (o.dest(), format!("{:#x}u", (o.imm20() as u32) << 12), o.abs_b(), o.neg_b(), o.sh(), o.i_dst_fmt().0, o.src_fmt().0, o.round_mode().0)
                    }
                    _ => {
                        let o = op!(InstF2iC);
                        (o.dest(), self.cb(o.cbuf_slot(), o.cbuf_offset()), o.abs_b(), o.neg_b(), o.sh(), o.i_dst_fmt().0, o.src_fmt().0, o.round_mode().0)
                    }
                };
                let x = Self::fmod(format!("tof32({b}, {src}u, {sh})"), abs, neg);
                self.set(d, format!("f2i({x}, {rnd}u, {dst}u)"));
            }
            Op::F2fR | Op::F2fI | Op::F2fC => {
                let (d, b, abs, neg, sh, dst, src, rnd, s) = match i.op {
                    Op::F2fR => {
                        let o = op!(InstF2fR);
                        (o.dest(), self.r(o.src_b()), o.abs_b(), o.neg_b(), o.sh(), o.dst_fmt().0, o.src_fmt().0, o.round_mode().0, o.sat())
                    }
                    Op::F2fI => {
                        let o = op!(InstF2fI);
                        (o.dest(), format!("{:#x}u", (o.imm20() as u32) << 12), o.abs_b(), o.neg_b(), o.sh(), o.dst_fmt().0, o.src_fmt().0, o.round_mode().0, o.sat())
                    }
                    _ => {
                        let o = op!(InstF2fC);
                        (o.dest(), self.cb(o.cbuf_slot(), o.cbuf_offset()), o.abs_b(), o.neg_b(), o.sh(), o.dst_fmt().0, o.src_fmt().0, o.round_mode().0, o.sat())
                    }
                };
                let x = Self::sat(Self::fmod(format!("fround(tof32({b}, {src}u, {sh}), {rnd}u)"), abs, neg), s);
                if dst == 1 {
                    self.set(d, format!("as_type<uint>(half2(half({x}), 0.0h))"));
                } else {
                    self.setf(d, x);
                }
            }
            Op::I2iR | Op::I2iI | Op::I2iC => {
                let (d, b, sel, src, dst, s, abs, neg) = match i.op {
                    Op::I2iR => {
                        let o = op!(InstI2iR);
                        (o.dest(), self.r(o.src_b()), o.byte_sel().0, o.i_src_fmt().0, o.i_dst_fmt().0, o.sat(), o.abs_b(), o.neg_b())
                    }
                    Op::I2iI => {
                        let o = op!(InstI2iI);
                        (o.dest(), format!("{:#x}u", ((o.imm20() << 12) >> 12) as u32), o.byte_sel().0, o.i_src_fmt().0, o.i_dst_fmt().0, o.sat(), o.abs_b(), o.neg_b())
                    }
                    _ => {
                        let o = op!(InstI2iC);
                        (o.dest(), self.cb(o.cbuf_slot(), o.cbuf_offset()), o.byte_sel().0, o.i_src_fmt().0, o.i_dst_fmt().0, o.sat(), o.abs_b(), o.neg_b())
                    }
                };
                self.set(d, format!("i2i({b}, {sel}u, {src}u, {dst}u, {s}, {abs}, {neg})"));
            }

            Op::MovR | Op::MovI | Op::MovC | Op::Mov32i => {
                let provenance=match i.op {
                    Op::MovR=>{let o=op!(InstMovR);(self.handle_regs.get(&o.src_a()).copied(),self.sampler_regs.get(&o.src_a()).copied())},
                    Op::MovC=>{let o=op!(InstMovC);if o.cbuf_slot()==2 {Self::cbuf_handle(o.cbuf_offset())} else {(None,None)}},
                    _=>(None,None),
                };
                let (d, v) = match i.op {
                    Op::MovR => {
                        let o = op!(InstMovR);
                        (o.dest(), self.r(o.src_a()))
                    }
                    Op::MovI => {
                        let o = op!(InstMovI);
                        (o.dest(), format!("{:#x}u", ((o.imm20() << 12) >> 12) as u32))
                    }
                    Op::MovC => {
                        let o = op!(InstMovC);
                        (o.dest(), self.cb(o.cbuf_slot(), o.cbuf_offset()))
                    }
                    _ => {
                        let o = op!(InstMov32i);
                        (o.dest(), format!("{:#x}u", o.imm32() as u32))
                    }
                };
                self.set(d, v);
                if let Some(t)=provenance.0 {self.handle_regs.insert(d,t);}
                if let Some(s)=provenance.1 {self.sampler_regs.insert(d,s);}
            }
            Op::Ldc => {
                let o = op!(InstLdc);
                let mut a = self.r(o.src_a());

                let slot = if matches!(o.address_mode().0, 2 | 3) {
                    let s = format!("min({}u + ({a} >> 16), 17u)", o.cbuf_slot());
                    a = format!("({a} & 0xffffu)");
                    s
                } else {
                    o.cbuf_slot().to_string()
                };

                let off = (o.cbuf_offset() as u16 as i16) as i32;
                let addr = format!("({a} + {:#x}u)", off as u32);
                let bank=if o.cbuf_slot()==1 && o.address_mode().0<2 && self.prog.indirect_constant_bytes.is_some() && self.embedded_constants.is_some() {"embedded_c1".into()} else {format!("res.cb[{slot}]")};
                if o.cbuf_slot()==1 && o.address_mode().0<2 {
                    if let Some(size)=self.prog.indirect_constant_bytes {
                        let bytes=match o.ls_size().0 {0|1=>1,2|3=>2,4=>4,5=>8,_=>bail!("unknown specialized C1 load width")};
                        let start=if bytes>=4 {format!("({addr}&~3u)")} else {addr.clone()};
                        let stop=if self.simt {"live=false;continue;"} else {"return epilogue();"};
                        self.line(format!("if({start}>{size}u || {bytes}u>{size}u-{start}) {{shader_fault(res,256u,{}u);{stop}}}",i.addr));
                    }
                }
                match o.ls_size().0 {
                    5 => {

                        self.line(format!("{{ uint2 cv = uint2({bank}[{addr} >> 2], {bank}[({addr} >> 2) + 1u]);"));
                        self.set(o.dest(), "cv.x".into());
                        if o.dest() != RZ_ {
                            self.set(o.dest() + 1, "cv.y".into());
                        }
                        self.line("}");
                    }
                    4 => self.set(o.dest(), format!("{bank}[{addr} >> 2]")),
                    sz => {
                        let (bits, signed) = match sz {
                            0 => (8, false),
                            1 => (8, true),
                            2 => (16, false),
                            _ => (16, true),
                        };
                        self.set(o.dest(), format!("bfe({bank}[{addr} >> 2], ((({addr} & 3u) * 8u) | ({bits}u << 8)), {signed})"));
                    }
                }
            }
            Op::Lds|Op::Sts => {
                if self.sph.stage!=Stage::Compute || self.shared_memory==0 {bail!("shared-memory instruction without compute shared-memory metadata");}
                let (d,a,offset,size)=if i.op==Op::Lds {let o=op!(InstLds);(o.dest(),self.r(o.src_a()),o.imm24(),o.ls_size().0)} else {let o=op!(InstSts);(o.dest(),self.r(o.src_a()),o.imm24(),o.ls_size().0)};
                let bytes=match size {0|1=>1,2|3=>2,4=>4,5=>8,6=>16,_=>bail!("unsupported shared memory access width")};
                let offset=((offset<<8)>>8) as u32;
                self.line(format!("{{uint sa={a}+{offset}u;if(sa>{}u) {{shader_fault(res,8u,{}u);live=false;continue;}}",self.shared_memory.saturating_sub(bytes),i.addr));
                if bytes>self.shared_memory {bail!("shared access larger than allocation");}
                for k in 0..bytes.div_ceil(4) {
                    let n=(bytes-4*k).min(4);let reg=(d+k as i32).min(RZ_);
                    if i.op==Op::Lds {
                        let word=(0..n).map(|b|format!("(uint(shared_mem[sa+{}u])<<{}u)",4*k+b,8*b)).collect::<Vec<_>>().join("|");
                        self.line(format!("uint shared_v{k}={word};"));
                    } else {
                        let value=self.r(reg);self.line(format!("uint shared_v{k}={value};"));
                        for b in 0..n {self.line(format!("shared_mem[sa+{}u]=uchar(shared_v{k}>>{}u);",4*k+b,8*b));}
                    }
                }
                if i.op==Op::Lds {for k in 0..bytes.div_ceil(4) {
                    let value=if size==1 {format!("uint(int(shared_v{k}<<24u)>>24)")} else if size==3 {format!("uint(int(shared_v{k}<<16u)>>16)")} else {format!("shared_v{k}")};
                    self.set((d+k as i32).min(RZ_),value);
                }}
                let next=if (i.addr+8)%32==0 {i.addr+16} else {i.addr+8};
                self.line(format!("pc={next}u;continue;}}"));
            }
            Op::Atoms => {
                let o=op!(InstAtoms);
                let operation=o.atom_op().0;
                if self.sph.stage!=Stage::Compute || self.shared_memory<4 || o.atoms_size().0>1 || !matches!(operation,0|1|2|5|6|7|8) {bail!("unsupported shared atomic form: compute32 ADD/MIN/MAX/AND/OR/XOR/EXCH required");}
                let address=self.r(o.src_a());let value=self.r(o.src_b());let offset=(((o.imm22()<<10)>>10) as u32).wrapping_mul(4);
                let function=match operation{0=>"atomic_fetch_add",1=>"atomic_fetch_min",2=>"atomic_fetch_max",5=>"atomic_fetch_and",6=>"atomic_fetch_or",7=>"atomic_fetch_xor",8=>"atomic_exchange",_=>unreachable!()};
                let ty=if o.atoms_size().0==1&&matches!(operation,1|2){"int"}else{"uint"};
                self.line(format!("{{uint sa=({address}&~3u)+{offset}u,operand={value};if(sa>{}u) {{shader_fault(res,8u,{}u);live=false;continue;}} uint previous=uint({function}_explicit(reinterpret_cast<threadgroup atomic_{ty}*>(shared_mem+sa),{ty}(operand),memory_order_relaxed));",self.shared_memory-4,i.addr));
                self.set(o.dest(),"previous".into());self.line("}");
            }
            Op::Atom => {
                let o=op!(InstAtom);
                if self.sph.stage!=Stage::Compute || o.size().0>1 || o.op().0!=0 {bail!("unsupported global atomic form: compute32 ADD required");}
                let lo=self.r(o.src_a());let hi=if o.e() {self.r((o.src_a()+1).min(RZ_))} else {"0u".into()};let value=self.r(o.src_b());let offset=((o.imm20()<<12)>>12) as u32;
                self.line(format!("{{ulong address=(ulong({lo})|(ulong({hi}&255u)<<32u))+ulong(long(int({offset}u)));uint operand={value};ulong pointer=gmem_gpu(res,address);if(pointer==0ul || (address&3ul)!=0ul || gmem_gpu(res,address+3ul)!=pointer+3ul) {{shader_fault(res,16u,{}u);live=false;continue;}}uint previous=atomic_fetch_add_explicit(reinterpret_cast<device atomic_uint*>(pointer),operand,memory_order_relaxed);",i.addr));
                self.set(o.dest(),"previous".into());self.line("}");
            }
            Op::Red => {
                let o=op!(InstRed);let size=o.red_size().0;let operation=o.red_op().0;
                if size>1 {bail!("unsupported wide or floating-point global reduction");}
                let lo=self.r(o.src_a());let hi=if o.e() {self.r((o.src_a()+1).min(RZ_))} else {"0u".into()};let value=self.r(o.src_b());let offset=((o.imm20()<<12)>>12) as u32;
                self.line(format!("{{ulong address=(ulong({lo})|(ulong({hi}&255u)<<32u))+ulong(long(int({offset}u)));ulong pointer=gmem_gpu(res,address);if(pointer==0ul || (address&3ul)!=0ul) {{shader_fault(res,16u,{}u);}} else {{",i.addr));
                if matches!(operation,3|4) {
                    let update=if operation==3 {format!("(old>={value}?0u:old+1u)")} else {format!("(old==0u || old>{value}?{value}:old-1u)")};
                    self.line(format!("device atomic_uint* atomic=reinterpret_cast<device atomic_uint*>(pointer);uint old=atomic_load_explicit(atomic,memory_order_relaxed);for(;;) {{uint next={update};if(atomic_compare_exchange_weak_explicit(atomic,&old,next,memory_order_relaxed,memory_order_relaxed)) break;}}"));
                } else {
                    let function=match operation {0=>"add",1=>"min",2=>"max",5=>"and",6=>"or",7=>"xor",_=>unreachable!()};
                    let signed=size==1&&matches!(operation,1|2);let ty=if signed {"int"} else {"uint"};
                    self.line(format!("atomic_fetch_{function}_explicit(reinterpret_cast<device atomic_{ty}*>(pointer),{ty}({value}),memory_order_relaxed);"));
                }
                self.line("}}");
            }
            Op::Ldl | Op::Stl => {
                self.uses_lmem = true;
                let (d, a, off, size) = if i.op == Op::Ldl {
                    let o = op!(InstLdl);
                    (o.dest(), self.r(o.src_a()), o.imm24(), o.ls_size().0)
                } else {
                    let o = op!(InstStl);
                    (o.dest(), self.r(o.src_a()), o.imm24(), o.ls_size().0)
                };
                let bytes=match size {0|1=>1u32,2|3=>2,4=>4,5=>8,6=>16,_=>bail!("unsupported local memory width")};
                if bytes>self.sph.local_memory {bail!("local access exceeds declared private-memory allocation");}
                let off=((off<<8)>>8) as u32;let words=bytes.div_ceil(4);
                let fault=if self.simt {"live=false;continue;"} else {"return epilogue();"};
                self.line(format!("{{uint la={a}+{off}u;if(la>{}u) {{shader_fault(res,32u,{}u);{fault}}}",self.sph.local_memory-bytes,i.addr));
                for k in 0..words {
                    let n=(bytes-4*k).min(4);
                    if i.op == Op::Ldl {
                        let word=(0..n).map(|b|format!("(uint(lmem[la+{}u])<<{}u)",4*k+b,8*b)).collect::<Vec<_>>().join("|");
                        self.line(format!("uint lv{k}={word};"));
                    } else {
                        let v=self.r((d+k as i32).min(RZ_));self.line(format!("uint lv{k}={v};"));
                        for b in 0..n {self.line(format!("lmem[la+{}u]=uchar(lv{k}>>{}u);",4*k+b,8*b));}
                    }
                }
                if i.op == Op::Ldl && d != RZ_ {
                    for k in 0..words {
                        let value=if size==1 {format!("uint(int(lv{k}<<24u)>>24)")} else if size==3 {format!("uint(int(lv{k}<<16u)>>16)")} else {format!("lv{k}")};
                        self.set((d+k as i32).min(RZ_),value);
                    }
                }
                self.line("}");
            }
            Op::Ldg | Op::Stg => {

                let (d, ra, imm, size, e) = if i.op == Op::Ldg {
                    let o = op!(InstLdg);
                    (o.dest(), o.src_a(), o.imm24(), o.ls_size().0, o.e())
                } else {
                    let o = op!(InstStg);
                    (o.dest(), o.src_a(), o.imm24(), o.ls_size().0, o.e())
                };
                let imm = ((imm << 8) >> 8) as i64;
                let lo = self.r(ra);
                let hi = if e && ra != RZ_ { self.r(ra + 1) } else { "0u".into() };
                let words = match size {
                    5 => 2,
                    6 | 7 => 4,
                    _ => 1,
                };
                let wide=self.vector_global_loads && i.op==Op::Ldg && words>1;
                if wide {self.line(format!("{{ ulong ga = ((ulong({hi}) << 32) | ulong({lo})) + ulong({imm}l);"));}
                else {self.line(format!("{{ ulong ga = ((ulong({hi}) << 32) | ulong({lo})) + ulong({imm}l); device uint* gp = gmem(res, ga & ~3ul);"));}
                if i.op == Op::Ldg {
                    if size < 4 {
                        let (bits, signed) = match size {
                            0 => (8, false),
                            1 => (8, true),
                            2 => (16, false),
                            _ => (16, true),
                        };
                        self.set(d, format!("bfe(gp[0], ((uint(ga & 3ul) * 8u) | ({bits}u << 8)), {signed})"));
                    } else {

                        if wide {self.line(format!("uint{words} gv=nx_gmem_ld{words}(res,ga&~3ul);"));}
                        else {
                            self.line(format!("uint gv[{words}];"));
                            for k in 0..words {self.line(format!("gv[{k}] = gmem(res, (ga & ~3ul) + {k}ul * 4ul)[0];"));}
                        }
                        for k in 0..words {
                            if d != RZ_ {
                                self.set(d + k, format!("gv[{k}]"));
                            }
                        }
                    }
                } else if size >= 4 {
                    for k in 0..words {
                        let v = self.r(if d == RZ_ { RZ_ } else { d + k });
                        self.line(format!("if (gmem_gpu(res, (ga & ~3ul) + {k}ul * 4ul) != 0ul) gmem(res, (ga & ~3ul) + {k}ul * 4ul)[0] = {v};"));
                    }
                } else {
                    self.warnings.push("sub-word global stores ignored".into());
                }
                self.line("}");
            }
            Op::S2r => {
                let o = op!(InstS2r);
                let compute = self.sph.stage == Stage::Compute;
                let v = match o.s_reg().0 {
                    0 => "simd_lane",
                    // Metal framebuffer derivatives use a downward Y axis.
                    // The guest's lower-left window origin reverses that axis;
                    // expose the same sign used by the vertex-position flip.
                    0x12 if !compute=>"((res.flags&1u)!=0u?0xbf800000u:0x3f800000u)",

                    0x20 if compute => "(tid.x | tid.y << 16 | tid.z << 26)",
                    0x21 if compute => "tid.x",
                    0x22 if compute => "tid.y",
                    0x23 if compute => "tid.z",

                    0x25 if compute => "ctaid.x",
                    0x26 if compute => "ctaid.y",
                    0x27 if compute => "ctaid.z",
                    0x38 if compute=>"(1u<<simd_lane)",
                    0x39 if compute=>"((1u<<simd_lane)-1u)",
                    0x3a if compute=>"(0xffffffffu>>(31u-simd_lane))",
                    0x3b if compute=>"~(0xffffffffu>>(31u-simd_lane))",
                    0x3c if compute=>"~((1u<<simd_lane)-1u)",
                    reg=>bail!("unsupported special-register read {reg:#x} for {:?}",self.sph.stage),
                };
                self.set(o.dest(), v.into());
            }

            Op::Ald => {
                let o = op!(InstAld);
                let n = o.al_size().0 as i32 + 1;
                if o.src_a() != RZ_ {
                    self.warnings.push("indexed attribute load".into());
                }
                for k in 0..n {
                    let v = self.attr_load(o.imm11() + 4 * k, o.o());
                    self.set(o.dest() + k, v);
                }
            }
            Op::Ast => {
                let o = op!(InstAst);
                let n = o.al_size().0 as i32 + 1;
                if o.src_a() != RZ_ {
                    self.warnings.push("indexed attribute store".into());
                }
                for k in 0..n {
                    let v = self.fr(o.src_b() + k);
                    match self.vs_output_ref(o.imm11() + 4 * k) {
                        Some(lv) => self.line(format!("{lv} = {v};")),
                        None => self.warnings.push(format!("attribute store {:#x}", o.imm11() + 4 * k)),
                    }
                }
            }
            Op::Ipa => {
                let o = op!(InstIpa);
                let mut v = if o.idx() {
                    self.warnings.push("indexed ipa".into());
                    "0u".into()
                } else {
                    self.fs_input(o.imm10(), true)
                };
                let fixed = matches!(o.imm10(), ATTR_POSITION..=0x7c | ATTR_FRONT_FACING);
                if o.ipa_op() == IpaOp::Multiply && !fixed {
                    let b = self.fr(o.src_b());
                    v = format!("as_type<uint>(as_type<float>({v}) * {b})");
                }
                if o.sat() {
                    v = format!("as_type<uint>(saturate(as_type<float>({v})))");
                }
                self.set(o.dest(), v);
            }
            Op::Isberd => {
                let o = op!(InstIsberd);
                let a = self.r(o.src_a());
                self.set(o.dest(), a);
            }
            Op::OutR | Op::OutI | Op::OutC => {
                self.warnings.push("geometry OUT".into());
            }

            Op::Texs => {
                let o = op!(InstTexs);
                self.texs(o)?;
            }
            Op::Tex => {
                let o = op!(InstTex);
                self.tex_full(o)?;
            }
            Op::TexB=>self.tex_bindless(op!(InstTexB))?,
            Op::TxdB => {
                let o=op!(InstTxdB);
                if o.dim().0!=2 || o.toff() || o.dest_pred()!=7 || o.lc() {bail!("unsupported derivative texture sample shape/offset/residency");}
                let a=o.src_a();let b=o.src_b();let (texture,sampler,handle)=self.bindless_refs(a,TexKind::T2D)?;
                let coords=self.coord(&[(a+1).min(RZ_),(a+2).min(RZ_)],2);
                let dx=self.coord(&[b,(b+1).min(RZ_)],2);let dy=self.coord(&[(b+2).min(RZ_),(b+3).min(RZ_)],2);
                self.bindless_open(handle.as_deref(),TexKind::T2D);
                self.line(format!("sampled={texture}.sample({sampler},{coords},gradient2d({dx},{dy}));"));
                if handle.is_some() {self.line("}");}
                let dests=(0..4).map(|n|(o.dest()+n).min(RZ_)).collect::<Vec<_>>();self.write_components("sampled",o.w_mask() as u32,&dests);self.line("}");
            }
            Op::Tld4B => {
                let o=op!(InstTld4B);
                if !matches!(o.dim().0,2|3) || o.toff().0>1 || o.dest_pred()!=7 || o.lc() {bail!("unsupported bindless gather shape/offset/residency");}
                let array=o.dim().0==3;
                let kind=match (array,o.dc()) {(false,false)=>TexKind::T2D,(false,true)=>TexKind::Depth2D,(true,false)=>TexKind::T2DArray,(true,true)=>TexKind::Depth2DArray};let (texture,sampler,handle)=self.bindless_refs(o.src_b(),kind)?;
                let a=(o.src_a()+i32::from(array)).min(RZ_);
                let coords=self.coord(&[a,(a+1).min(RZ_)],2);
                let layer=if array {format!(",({}&0xffffu)",self.r(o.src_a()))} else {String::new()};
                let offset=if o.toff().0==1 {let packed=self.r((o.src_b()+1).min(RZ_));format!("int2(int({packed}<<26u)>>26,int({packed}<<18u)>>26)")} else {"int2(0)".into()};
                let gather=if o.dc() {
                    let depth=self.fr((o.src_b()+1+i32::from(o.toff().0!=0)).min(RZ_));format!("{texture}.gather_compare({sampler},{coords}{layer},{depth},{offset})")
                } else {let component=["x","y","z","w"][o.tex_comp().0 as usize];format!("{texture}.gather({sampler},{coords}{layer},{offset},component::{component})")};
                self.bindless_open(handle.as_deref(),kind);self.line(format!("sampled={gather};"));if handle.is_some() {self.line("}");}
                let dests=(0..4).map(|n|(o.dest()+n).min(RZ_)).collect::<Vec<_>>();self.write_components("sampled",o.w_mask() as u32,&dests);self.line("}");
            }
            Op::SuldDB | Op::SustB => self.surface(i)?,
            Op::SuatomB => self.surface_atomic(InstSuatomB(i.raw))?,
            Op::Tld => self.tld(op!(InstTld))?,
            Op::Tlds => {
                let o = op!(InstTlds);
                self.tlds(o)?;
            }
            Op::Tld4 | Op::Tld4s => self.tld4(i)?,
            Op::TmmlB | Op::Tmml => {
                let (d, a, mask, dim, tid) = if i.op == Op::Tmml {
                    let o = op!(InstTmml);
                    (o.dest(), o.src_a(), o.w_mask() as u32, o.dim().0, o.tid_b())
                } else {

                    let o = op!(InstTmmlB);
                    let Some(&binding) = self.handle_regs.get(&o.src_b()) else {
                        self.warnings.push("unresolved bindless texture".into());
                        let dd = o.dest();
                        self.set(dd, "0u".into());
                        return Ok(());
                    };
                    (o.dest(), o.src_a(), o.w_mask() as u32, o.dim().0, 8 + 2 * binding as i32)
                };
                let kind = match dim {
                    3 => TexKind::T2DArray,
                    4 => TexKind::T3D,
                    6 => TexKind::Cube,
                    _ => TexKind::T2D,
                };
                let b = self.tex(tid, kind);
                let ra: Vec<i32> = (0..3).map(|k| if a == RZ_ { RZ_ } else { a + k }).collect();
                let coords: Vec<i32> = if kind.is_array() { ra[1..].to_vec() } else { ra };
                let c = self.coord(&coords, kind.coords());

                self.line(format!("{{ float l = t{b}.calculate_clamped_lod(s{b}, {c}); float4 v = float4(l, l, 0.0f, 0.0f);"));
                let dests: Vec<i32> = (0..4).map(|k| if d == RZ_ { RZ_ } else { d + k }).collect();
                self.write_components("v", mask, &dests);
                self.line("}");
            }
            Op::Txq | Op::TxqB => {
                let (d, a, mask, tid, query) = if i.op == Op::Txq {
                    let o = op!(InstTxq);
                    (o.dest(), o.src_a(), o.w_mask() as u32, o.tid_b(), o.tex_query().0)
                } else {
                    let o = op!(InstTxqB);
                    (o.dest(), o.src_a(), o.w_mask() as u32, 8, o.tex_query().0)
                };
                if i.op==Op::TxqB {bail!("runtime bindless texture queries are unsupported");}
                if query!=1 {bail!("unsupported texture query {query}");}
                let b=Self::texture_binding_id(tid);
                if let std::collections::btree_map::Entry::Vacant(entry)=self.iface.textures.entry(b) {
                    entry.insert(TexKind::T2D);self.query_only_textures.insert(b);
                }
                let lod = self.r(a);
                if query == 1 {

                    self.iface.buffer_metadata=true;
                    self.line(format!("{{ uint l = {lod}; uint4 v = buffer_tex{b}.w!=0u?uint4(buffer_tex{b}.z,1u,1u,1u):uint4(t{b}.get_width(l), t{b}.get_height(l), 1u, t{b}.get_num_mip_levels());"));
                } else {
                    self.line("{ uint4 v = uint4(0u);");
                }
                let dests: Vec<i32> = (0..4).map(|k| if d == RZ_ { RZ_ } else { d + k }).collect();
                let mut k = 0;
                for c in 0..4 {
                    if mask >> c & 1 != 0 {
                        if let Some(&dd) = dests.get(k) {
                            self.set(dd, format!("v[{c}]"));
                        }
                        k += 1;
                    }
                }
                self.line("}");
            }

            Op::FcmpR|Op::FcmpI|Op::FcmpC|Op::FcmpRc|Op::IcmpR|Op::IcmpI|Op::IcmpC|Op::IcmpRc => {
                let d=(w&255) as i32;let a=self.r(((w>>8)&255) as i32);let mut c=self.r(((w>>39)&255) as i32);
                let floating=matches!(i.op,Op::FcmpR|Op::FcmpI|Op::FcmpC|Op::FcmpRc);
                let immediate=matches!(i.op,Op::FcmpI|Op::IcmpI);
                let constant=matches!(i.op,Op::FcmpC|Op::IcmpC|Op::FcmpRc|Op::IcmpRc);
                let reverse=matches!(i.op,Op::FcmpRc|Op::IcmpRc);
                let mut b=if immediate {
                    let imm=(((w>>20)&0x7ffff)|((w>>37)&0x80000)) as u32;
                    let bits=if floating {imm<<12} else {(((imm<<12) as i32)>>12) as u32};format!("{bits:#x}u")
                } else if constant {self.cb(((w>>34)&31) as i32,((w>>20)&0x3fff) as i32)} else {self.r(((w>>20)&255) as i32)};
                if reverse {std::mem::swap(&mut b,&mut c);}
                let condition=if floating {
                    if w&(1<<47)!=0 {c=format!("(({c}&0x7f800000u)==0u?({c}&0x80000000u):{c})");}
                    format!("fcmp({}u,as_type<float>({c}),0.0f)",(w>>48)&15)
                } else {format!("icmp({}u,{c},0u,{})",(w>>49)&7,w&(1<<48)!=0)};
                self.set(d,format!("({condition}?{a}:{b})"));
            }
            Op::FloR => {
                let o=op!(InstFloR);let b=self.r(o.src_b());
                let b=if o.neg_b() {format!("(~{b})")} else {b};
                let b=if o.signed()&&!o.sh() {format!("(int({b})<0?~{b}:{b})")} else {b};
                self.line(format!("{{uint bits={b};uint found=bits==0u?0xffffffffu:{};",if o.sh() {"clz(bits)"} else {"31u-clz(bits)"}));
                self.set(o.dest(),"found".into());if o.write_cc() {self.set_zn("found".into());}self.line("}");
            }
            Op::PopcR => {
                let o=op!(InstPopcR);let b=self.r(o.src_b());let b=if o.neg_b() {format!("(~{b})")} else {b};
                self.set(o.dest(),format!("popcount({b})"));
            }
            Op::BfiI => {
                let o=op!(InstBfiI);let control=o.imm20() as u32;let pos=control&255;let bits=((control>>8)&255).min(32u32.saturating_sub(pos));
                let a=self.r(o.src_a());let c=self.r(o.src_c());
                let value=if bits==0 {c} else {let mask=if bits==32 {u32::MAX} else {((1u32<<bits)-1)<<pos};format!("(({c}&{:#x}u)|(({a}<<{pos}u)&{mask:#x}u))",!mask)};
                self.line(format!("{{uint inserted={value};"));self.set(o.dest(),"inserted".into());if o.write_cc() {self.set_zn("inserted".into());}self.line("}");
            }
            Op::R2pR|Op::R2pI|Op::R2pC => {
                if w&(1<<40)!=0 {bail!("R2P condition-code restore is unsupported");}
                let a=self.r(((w>>8)&255) as i32);let shift=((w>>41)&3)*8;
                let mask=match i.op {
                    Op::R2pI=>format!("{:#x}u",((w>>20)&0x7ffff)|((w>>37)&0x80000)),
                    Op::R2pC=>self.cb(((w>>34)&31) as i32,((w>>20)&0x3fff) as i32),
                    _=>self.r(((w>>20)&255) as i32),
                };
                // Snapshot register operands before writing any predicates.
                self.line(format!("{{uint predicate_value={a};uint predicate_mask={mask};"));
                for bit in 0..7 {let old=self.p(bit,false);self.setp(bit,format!("(predicate_mask&{}u)!=0u?((predicate_value&{}u)!=0u):{old}",1u32<<bit,1u32<<(bit as u64+shift)));}
                self.line("}");
            }
            Op::ShfLR|Op::ShfRR|Op::ShfLI|Op::ShfRI => {
                if (w>>37)&3>=2 || (w>>48)&3!=0 {bail!("unimplemented extended funnel-shift form");}
                let a=self.r(((w>>8)&255) as i32);let c=self.r(((w>>39)&255) as i32);
                let shift=if matches!(i.op,Op::ShfLI|Op::ShfRI) {format!("{}u",(w>>20)&63)} else {self.r(((w>>20)&255) as i32)};
                let value=if matches!(i.op,Op::ShfLR|Op::ShfLI) {format!("uint((((ulong({c})<<32u)|ulong({a}))<<({shift}&31u))>>32u)")} else {format!("uint(((ulong({c})<<32u)|ulong({a}))>>({shift}&31u))")};
                let value=if w&(1<<50)==0 {format!("({shift}<32u?{value}:0u)")} else {value};
                self.line(format!("{{uint shifted={value};"));self.set((w&255) as i32,"shifted".into());if w&(1<<47)!=0 {self.set_zn("shifted".into());}self.line("}");
            }
            Op::PrmtI|Op::PrmtR|Op::PrmtC => {
                let (d,a,c,selector,mode)=match i.op {
                    Op::PrmtI=>{let o=op!(InstPrmtI);(o.dest(),o.src_a(),o.src_c(),format!("{}u",o.imm20() as u32),o.p_mode().0)},
                    Op::PrmtR=>{let o=op!(InstPrmtR);let s=self.r(o.src_b());(o.dest(),o.src_a(),o.src_c(),s,o.p_mode().0)},
                    _=>{let o=op!(InstPrmtC);let s=self.cb(o.cbuf_slot(),o.cbuf_offset());(o.dest(),o.src_a(),o.src_c(),s,o.p_mode().0)},
                };
                if mode!=0 {bail!("special PRMT selector mode {mode} is unsupported");}
                let a=self.r(a);let c=self.r(c);
                self.line(format!("{{ulong bytes=ulong({a})|(ulong({c})<<32u);uint selectors={selector},permuted=0u;for(uint n=0u;n<4u;n++) {{uint pick=(selectors>>(n*4u))&15u;uint b=uint(bytes>>((pick&7u)*8u))&255u;if((pick&8u)!=0u) b=(b&128u)!=0u?255u:0u;permuted|=b<<(n*8u);}}"));
                self.set(d,"permuted".into());self.line("}");
            }
            Op::Psetp => {
                let o = op!(InstPsetp);
                let (a, b, c) = (self.p(o.src_pred(), o.src_pred_inv()), self.p(o.src2_pred(), o.src2_pred_inv()), self.p(o.src1_pred(), o.src1_pred_inv()));
                let _ = c;
                let (ab, cop) = (o.bool_op_ab().0, o.bool_op_c().0);
                let src_c = self.p(o.src_pred(), o.src_pred_inv());
                let (pa, pb) = (self.p(o.src1_pred(), o.src1_pred_inv()), self.p(o.src2_pred(), o.src2_pred_inv()));
                let _ = (a, b);
                self.line(format!("{{ bool t = bop({ab}u, {pa}, {pb});"));
                self.setp(o.dest_pred(), format!("bop({cop}u, t, {src_c})"));
                self.setp(o.dest_pred_inv(), format!("bop({cop}u, !t, {src_c})"));
                self.line("}");
            }
            Op::Pset => {
                let o = op!(InstPset);
                let (pa, pb) = (self.p(o.src1_pred(), o.src1_pred_inv()), self.p(o.src2_pred(), o.src2_pred_inv()));
                let pc = self.p(o.src_pred(), o.src_pred_inv());
                let c = format!("bop({}u, bop({}u, {pa}, {pb}), {pc})", o.bool_op_c().0, o.bool_op_ab().0);
                self.set(o.dest(), if o.b_val() { format!("({c} ? 0x3f800000u : 0u)") } else { format!("({c} ? 0xffffffffu : 0u)") });
            }
            Op::Csetp => {
                let o = op!(InstCsetp);
                self.uses_cc = true;
                let p = self.p(o.src_pred(), o.src_pred_inv());
                let c = self.condition(o.ccc());
                self.setp(o.dest_pred(), format!("bop({}u, {c}, {p})", o.bop().0));
                self.setp(o.dest_pred_inv(), format!("bop({}u, !{c}, {p})", o.bop().0));
            }
            Op::Vsetp => {
                let o = op!(InstVsetp);
                self.setp(o.dest_pred(), "false".into());
                self.setp(o.dest_pred_inv(), "true".into());
            }
            Op::Vmad => {
                let o = op!(InstVmad);
                let (a, b, c) = (self.r(o.src_a()), self.r(o.src_b()), self.r(o.src_c()));
                self.set(o.dest(), format!("(({a} & 0xffffu) * ({b} & 0xffffu) + {c})"));
            }
            Op::Vmnmx => {
                let o=op!(InstVmnmx);
                if o.write_cc() {bail!("VMNMX condition-code writes are not implemented");}
                let a=self.video_source(o.src_a(),o.a_select())?;
                let b=if o.b_video() {self.video_source(o.src_b(),o.b_select())?} else {
                    let immediate=if o.b_select().0&8!=0 {o.imm16() as u16 as i16 as i64} else {o.imm16() as i64};format!("{immediate}l")
                };
                let c=self.r(o.src_c());let minmax=if o.mn() {"max"} else {"min"};
                self.line(format!("{{long va={a},vb={b};long result={minmax}(va,vb);uint addend={c};"));
                if o.sat() {self.line(if o.d_format() {"result=min(result,2147483647l);"} else {"result=max(result,0l);"});}
                self.line("uint bits=uint(result);");
                let result=match o.video_op().0 {
                    0=>"(addend&0x0000ffffu)|(bits<<16u)",1=>"(addend&0xffff0000u)|(bits&0xffffu)",
                    2=>"(addend&0xffffff00u)|(bits&255u)",3=>"(addend&0xff00ffffu)|((bits&255u)<<16u)",4=>"bits+addend",
                    5 if o.d_format()=>"uint(min(int(bits),int(addend)))",5=>"min(bits,addend)",
                    6 if o.d_format()=>"uint(max(int(bits),int(addend)))",6=>"max(bits,addend)",_=>bail!("invalid VMNMX merge operation"),
                };
                self.set(o.dest(),result.into());self.line("}");
            }
            Op::Shfl => {
                let o = op!(InstShfl);
                let quad=self.sph.stage==Stage::Fragment;
                if !self.simt && !(quad && o.c_fix_shfl() && o.src_c_imm()==0x1c03) {bail!("unsupported graphics-stage non-quad SHFL");}
                if quad {self.uses_quad_lane=true;}
                let lane_id=if quad {"quad_lane"} else {"simd_lane"};
                let a = self.r(o.src_a());
                let lane = if o.b_fix_shfl() { format!("{}u", o.src_b_imm()) } else { self.r(o.src_b()) };
                let control=if o.c_fix_shfl() {format!("{}u",o.src_c_imm())} else {self.r(o.src_c())};
                self.line(format!("{{uint input={a},delta=({lane})&31u,control={control};uint segment=(control>>8u)&31u;uint edge=({lane_id}&segment)|((control&31u)&~segment);"));
                let mode=o.shfl_mode().0;
                let index=match mode {0=>format!("({lane_id}&segment)|(delta&~segment)"),1=>format!("{lane_id}-delta"),2=>format!("{lane_id}+delta"),3=>format!("{lane_id}^delta"),_=>unreachable!()};
                let shuffle=if quad {"quad_shuffle"} else {"simd_shuffle"};
                self.line(format!("uint source={index};bool valid={};uint shuffled={shuffle}(input,ushort(valid?source:{lane_id}));",if mode==1 {"source>=edge && source<32u"} else {"source<=edge"}));
                self.set(o.dest(),"shuffled".into());self.setp(o.dest_pred(),"valid".into());self.line("}");
            }
            Op::Vote => {
                if !self.simt {bail!("graphics-stage VOTE is not implemented");}
                let o=op!(InstVote);let value=self.p(o.src_pred(),o.src_pred_inv());
                let reduce=match o.vote_mode().0 {0=>"simd_all(vote_input)",1=>"simd_any(vote_input)",2=>"(simd_all(vote_input)||!simd_any(vote_input))",_=>bail!("unknown warp vote mode")};
                self.line(format!("{{bool vote_input={value};uint ballot=uint((simd_vote::vote_t)simd_ballot(vote_input));bool reduced={reduce};"));
                self.set(o.dest(),"ballot".into());self.setp(o.vp_dest(),"reduced".into());self.line("}");
            }

            Op::Bra => {
                let o = op!(InstBra);
                let t = branch_target(i);
                if o.ccc() != Ccc::T {
                    self.uses_cc = true;
                    let c=self.condition(o.ccc());
                    self.line(format!("if ({c}) {{ pc = {t}u; continue; }}"));
                } else {
                    self.label_target(t);
                }
            }
            Op::Brx => {
                let o=op!(InstBrx);let a=self.r(o.src_a());
                if o.ca()||o.lmt() {bail!("unsupported absolute or limited indirect branch");}
                let base=i.addr.wrapping_add(8).wrapping_add(((o.imm24()<<8)>>8) as u32);
                let jump=format!("{{uint target={a}+{base}u;pc=target%32u==0u?target+8u:target;continue;}}");
                if o.ccc()!=Ccc::T {let c=self.condition(o.ccc());self.line(format!("if({c}) {jump}"));} else {self.line(jump);}
            }
            Op::Sync|Op::Brk if self.simt => {
                let kind=u32::from(i.op==Op::Brk);
                self.line(format!("{{uint found=token_sp;while(found>0u && token_kind[found-1u]!={kind}u) found--;if(found==0u) {{shader_fault(res,4u,{}u);live=false;continue;}}token_sp=found;pc=token_target[found-1u];waiting=true;continue;}}",i.addr));
            }
            Op::Sync | Op::Brk => match self.prog.converge.get(&i.addr) {
                Some(&t) => self.label_target(t),
                None => {
                    self.warnings.push(format!("unresolved {:?} at {:#x}", i.name, i.addr));
                    self.line("return epilogue();");
                }
            },
            Op::Exit => {
                let o = op!(InstExit);
                let exit=if self.simt {"{live=false;continue;}"} else {"return epilogue();"};
                if o.ccc() != Ccc::T {
                    self.uses_cc = true;
                    let c=self.condition(o.ccc());
                    self.line(format!("if ({c}) {exit}"));
                } else {
                    self.line(exit);
                }
            }
            Op::Kil => {
                let o = op!(InstKil);
                if o.ccc() != Ccc::T {
                    self.uses_cc = true;
                    let c=self.condition(o.ccc());
                    self.line(format!("if ({c}) discard_fragment();"));
                } else {
                    self.line("discard_fragment();");
                }
            }
            other => bail!("unimplemented instruction {other:?} ({:?}) at {:#x}", i.name, i.addr),
        }
        Ok(())
    }

    fn texs(&mut self, o: InstTexs) -> Result<()> {
        let (dest, dest2) = (o.dest(), o.dest2());
        if dest == RZ_ && dest2 == RZ_ {
            return Ok(());
        }
        let mut ra = o.src_a();
        let mut rb = o.src_b();
        let mut next_a = |g: &mut Self| {
            let r = ra;
            if ra != RZ_ {
                ra += 1;
            }
            let _ = g;
            r
        };
        let mut a_regs = Vec::new();
        let mut b_regs = Vec::new();
        let t = o.target().0;
        let kind = match t {
            4..=6 => TexKind::Depth2D,
            9 => TexKind::Depth2DArray,
            7 | 8 => TexKind::T2DArray,
            10 | 11 => TexKind::T3D,
            12 | 13 => TexKind::Cube,
            _ => TexKind::T2D,
        };
        let mut lod = None;
        let mut depth_ref = None;
        let mut array = None;
        let take_b = |rb: &mut i32| {
            let r = *rb;
            if *rb != RZ_ {
                *rb += 1;
            }
            r
        };
        match t {
            7 | 8 | 9 => {

                array = Some(next_a(self));
                a_regs.push(next_a(self));
                b_regs.push(take_b(&mut rb));
                if t == 9 {
                    depth_ref = Some(take_b(&mut rb));
                }
                if t != 7 {
                    lod = Some("0.0f".to_string());
                }
            }
            0 => {
                a_regs.push(next_a(self));
                lod = Some("0.0f".into());
            }
            1 => {
                a_regs.push(next_a(self));
                b_regs.push(take_b(&mut rb));
            }
            2 => {
                a_regs.push(next_a(self));
                b_regs.push(take_b(&mut rb));
                lod = Some("0.0f".into());
            }
            3 | 4 | 10 | 12 => {
                a_regs.push(next_a(self));
                a_regs.push(next_a(self));
                let x = take_b(&mut rb);
                match t {
                    3 => {
                        let l = self.fr(x);
                        lod = Some(l);
                    }
                    4 => depth_ref = Some(x),
                    _ => b_regs.push(x),
                }
            }
            6 | 11 => {
                a_regs.push(next_a(self));
                a_regs.push(next_a(self));
                let x = take_b(&mut rb);
                if t == 6 { depth_ref = Some(x) } else { b_regs.push(x) }
                lod = Some("0.0f".into());
            }
            5 | 13 => {
                a_regs.push(next_a(self));
                a_regs.push(next_a(self));
                let x = take_b(&mut rb);
                let y = take_b(&mut rb);
                if t == 5 {
                    depth_ref = Some(x);
                    lod = Some(self.fr(y));
                } else {
                    b_regs.push(x);
                    lod = Some(self.fr(y));
                }
            }
            _ => bail!("texs target {t}"),
        }
        let coords: Vec<i32> = a_regs.into_iter().chain(b_regs).collect();
        let coords = if t == 0 { vec![coords[0], RZ_] } else { coords };
        let b = self.tex(o.tid_b(), kind);
        let v = self.sample(b, kind, &coords, array, lod, None, depth_ref);
        let mask = TEXS_MASK_LUT[if dest2 == RZ_ { 0 } else { 1 }][o.w_mask() as usize & 7];
        let dests = [dest, if dest == RZ_ { RZ_ } else { dest + 1 }, dest2, if dest2 == RZ_ { RZ_ } else { dest2 + 1 }];
        self.line(format!("{{ float4 v = {v};"));
        self.write_components("v", mask, &dests);
        self.line("}");
        Ok(())
    }

    fn tex_full(&mut self, o: InstTex) -> Result<()> {
        let d = o.dest();
        if d == RZ_ {
            return Ok(());
        }
        let (mut ra, mut rb) = (o.src_a(), o.src_b());
        let mut na = || {
            let r = ra;
            if ra != RZ_ {
                ra += 1;
            }
            r
        };
        let mut nb = || {
            let r = rb;
            if rb != RZ_ {
                rb += 1;
            }
            r
        };
        let dim = o.dim().0;
        let dc = o.dc();
        let kind = match (dim, dc) {
            (3 | 1, true) => TexKind::Depth2DArray,
            (6 | 7, true) => TexKind::DepthCube,
            (_, true) => TexKind::Depth2D,
            (3 | 1, false) => TexKind::T2DArray,
            (4 | 5, _) => TexKind::T3D,
            (6, _) => TexKind::Cube,
            (7, _) => TexKind::CubeArray,
            _ => TexKind::T2D,
        };
        let array = if matches!(dim, 1 | 3 | 5 | 7) { Some(na()) } else { None };
        let n = match dim {
            0 | 1 => 1,
            4 | 5 | 6 | 7 => 3,
            _ => 2,
        };
        let mut coords: Vec<i32> = (0..n).map(|_| na()).collect();
        if n == 1 {
            coords.push(RZ_);
        }
        let lodm = o.lod().0;
        let lod_reg = if lodm >= 2 { Some(nb()) } else { None };
        let offset=if o.aoffi() {Some(self.texture_offset(nb(),o.dim().0)?)} else {None};
        let depth_ref = if dc { Some(nb()) } else { None };
        let (lod, bias) = match lodm {
            1 => (Some("0.0f".to_string()), None),
            3 | 7 => (Some(self.fr(lod_reg.unwrap())), None),
            2 | 6 => (None, Some(self.fr(lod_reg.unwrap()))),
            _ => (None, None),
        };
        let b = self.tex(o.tid_b(), kind);
        let v = self.sample_offset(b, kind, &coords, array, lod, bias, depth_ref,offset);
        self.line(format!("{{ float4 v = {v};"));
        let dests: Vec<i32> = (0..4).map(|k| if d + k >= RZ_ { RZ_ } else { d + k }).collect();
        self.write_components("v", o.w_mask() as u32, &dests);
        self.line("}");
        Ok(())
    }

    fn tex_bindless(&mut self,o:InstTexB) -> Result<()> {
        let d=o.dest();if d==RZ_ {return Ok(());}
        if o.dest_pred()!=7 {bail!("sparse TEX.B residency predicate is unsupported");}
        let dim=o.dim().0;let dc=o.dc();
        let kind=match (dim,dc) {
            (1|3,true)=>TexKind::Depth2DArray,(6|7,true)=>TexKind::DepthCube,(_,true)=>TexKind::Depth2D,
            (1|3,false)=>TexKind::T2DArray,(4,false)=>TexKind::T3D,(6,false)=>TexKind::Cube,(7,false)=>TexKind::CubeArray,
            (5,false)=>bail!("array of 3D textures is unsupported"),_=>TexKind::T2D,
        };
        let (texture,sampler,handle)=self.bindless_refs(o.src_b(),kind)?;
        let mut ra=o.src_a();let mut rb=if o.src_b()==RZ_ {RZ_} else {o.src_b()+1};
        let take=|reg:&mut i32| {let r=*reg;if *reg<RZ_ {*reg+=1;}r.min(RZ_)};
        let array=if matches!(dim,1|3|7) {Some(take(&mut ra))} else {None};
        let mut coords:Vec<_>=(0..if dim<2 {1} else if dim>=4 {3} else {2}).map(|_|take(&mut ra)).collect();
        if dim<2 {coords.push(RZ_);}
        let lodm=o.lodb().0;
        let lodreg=if lodm>1 {Some(take(&mut rb))} else {None};
        let offset=if o.aoffib() {Some(self.texture_offset(take(&mut rb),dim)?)} else {None};
        let depth=if dc {Some(take(&mut rb))} else {None};
        let (lod,bias)=match lodm {
            1=>(Some("0.0f".into()),None),3|7=>(Some(self.fr(lodreg.unwrap())),None),
            2|6=>(None,Some(self.fr(lodreg.unwrap()))),0=>(None,None),_=>bail!("unsupported TEX.B LOD mode {lodm}"),
        };
        let mut sample=self.sample_offset(0,kind,&coords,array,lod,bias,depth,offset);
        sample=sample.replace("t0.",&format!("{texture}.")).replace("(s0,",&format!("({sampler},"));
        self.bindless_open(handle.as_deref(),kind);self.line(format!("sampled={sample};"));if handle.is_some() {self.line("}");}
        let dests:Vec<_>=(0..4).map(|i|(d+i).min(RZ_)).collect();
        self.write_components("sampled",o.w_mask() as u32,&dests);self.line("}");Ok(())
    }

    fn surface_atomic(&mut self,o:InstSuatomB)->Result<()> {
        if self.sph.stage!=Stage::Compute || !matches!(o.dim().0,0|1|3) || o.size().0!=0 || o.op().0!=0 || o.clamp().0!=0 {
            bail!("unsupported surface atomic: compute U32 ADD, 1D/2D, ignore bounds required");
        }
        let id=*self.handle_regs.get(&o.src_c()).ok_or_else(||anyhow::anyhow!("unresolved atomic image handle"))?;
        if !(0x200..0x208).contains(&id) {bail!("atomic operation requires an image handle");}
        let b=id-0x200;self.iface.buffer_metadata=true;
        if self.iface.images.get(&b).is_some_and(|image|image.kind!=TexKind::T2D||image.raw_bytes>4) {bail!("atomic image requires compatible 32-bit 1D/2D accesses");}
        self.iface.images.insert(b,ImageBinding{kind:TexKind::T2D,raw_bytes:4});self.iface.atomic_images|=1<<b;self.iface.image_writes|=1<<b;
        let mut x=self.r(o.src_a());if o.ba() {x=format!("({x}>>2u)");}
        let y=if o.dim().0==3 {self.r((o.src_a()+1).min(RZ_))} else {"0u".into()};
        let value=self.r(o.src_b());
        // Snapshot aliased coordinates/value before replacing the destination.
        // The host preserves the texture through ordered GPU blits around this
        // dispatch; buffer atomics work on Apple7 without texture atomics.
        self.line(format!("{{ uint ax={x},ay={y},addend={value},previous=0u;uint4 layout=image_atomic_layout[{b}],linear=buffer_meta[{}];if(linear.w!=0u) {{if(ax<linear.z && ay==0u) {{ulong address=(ulong(linear.x)|(ulong(linear.y)<<32ul))+ulong(ax)*4ul;ulong pointer=gmem_gpu(res,address);if(pointer==0ul || (address&3ul)!=0ul) shader_fault(res,512u,pc);else previous=atomic_fetch_add_explicit(reinterpret_cast<device atomic_uint*>(pointer),addend,memory_order_relaxed);}}}} else if(ax<layout.x && ay<layout.y) previous=atomic_fetch_add_explicit(image_atomic{b}+ay*layout.z+ax,addend,memory_order_relaxed);",256+b));
        self.set(o.dest(),"previous".into());self.line("}");Ok(())
    }

    fn surface(&mut self,i:&Inst) -> Result<()> {
        let (dim,coord,handle,dest,mask,size,byte_address,pred)=if i.op==Op::SuldDB {
            let o=InstSuldDB(i.raw);(o.dim().0,o.src_a(),o.src_c(),o.dest(),0,o.size().0,o.ba(),o.dest_pred2())
        } else {let o=InstSustB(i.raw);(o.dim().0,o.src_a(),o.src_c(),o.dest(),o.rgba().0,0,false,7)};
        let id=*self.handle_regs.get(&handle).ok_or_else(||anyhow::anyhow!("unresolved image handle in R{handle}"))?;
        if !(0x200..0x208).contains(&id) {bail!("surface instruction uses a non-image handle in R{handle}");}
        let binding=id-0x200;
        if i.op==Op::SustB {self.iface.image_writes|=1<<binding;}
        if self.iface.atomic_images&(1<<binding)!=0 {return self.surface_atomic_plain(i,binding);}
        let kind=match dim {0|1|3=>TexKind::T2D,2|4=>TexKind::T2DArray,5=>TexKind::T3D,_=>bail!("unsupported image dimension {dim}")};
        if dim<=1 {self.iface.buffer_metadata=true;}
        let bytes=if i.op==Op::SuldDB {match size {0|1=>1,2|3=>2,4=>4,5=>8,6|7=>16,_=>unreachable!()}} else {0};
        let entry=self.iface.images.entry(binding).or_insert(ImageBinding{kind,raw_bytes:bytes});
        if entry.kind!=kind {bail!("image binding {binding} has incompatible dimensions");}
        entry.raw_bytes=entry.raw_bytes.max(bytes);
        let mut next=coord;let mut take=|g:&mut Self| {let r=next.min(RZ_);if next<RZ_ {next+=1;}g.r(r)};
        let mut x=take(self);
        if byte_address {x=format!("uint(int({x}) >> {}u)",bytes.trailing_zeros());}
        let y=if dim>=3 {take(self)} else {"0u".into()};
        let z=if dim==5 {take(self)} else {"0u".into()};
        let layer=if matches!(dim,2|4) {take(self)} else {"0u".into()};
        let (coords,extra,condition)=match kind {
            TexKind::T3D=>(format!("uint3({x},{y},{z})"),String::new(),format!("({x})<im{binding}.get_width() && ({y})<im{binding}.get_height() && ({z})<im{binding}.get_depth()")),
            TexKind::T2DArray=>(format!("uint2({x},{y})"),format!(",{layer}"),format!("({x})<im{binding}.get_width() && ({y})<im{binding}.get_height() && ({layer})<im{binding}.get_array_size()")),
            _=>(format!("uint2({x},{y})"),String::new(),format!("({x})<im{binding}.get_width() && ({y})<im{binding}.get_height()")),
        };
        let buffer=format!("buffer_meta[{}]",256+binding);
        if i.op==Op::SuldDB {
            if dim<=1 {
                self.line(format!("{{ bool linear={buffer}.w!=0u;bool in_bounds=linear?({x})<{buffer}.z:({condition});uint4 raw_value=uint4(0u);if(in_bounds) {{if(linear) raw_value=buffer_read_raw(res,{buffer},{x},{}u);else raw_value=image_raw_words(im{binding}.read({coords}{extra}),image_meta[{binding}]);}}",i.addr));
            } else {self.line(format!("{{ bool in_bounds={condition};uint4 raw_value=uint4(0u);if(in_bounds) raw_value=image_raw_words(im{binding}.read({coords}{extra}),image_meta[{binding}]);"));}
            for c in 0..bytes.div_ceil(4) {
                let value=if bytes==1 {format!("(raw_value.x & 255u)")} else if bytes==2 {format!("(raw_value.x & 65535u)")} else {format!("raw_value[{c}]")};
                let value=if size==1 {format!("uint(int({value}<<24)>>24)")} else if size==3 {format!("uint(int({value}<<16)>>16)")} else {value};
                self.set((dest+c as i32).min(RZ_),value);
            }
            self.setp(pred,"in_bounds".into());self.line("}");
        } else {
            if mask!=15 {bail!("partial-component surface store mask {mask:#x} is unsupported");}
            let values:Vec<_>=(0..4).map(|c|self.r((dest+c).min(RZ_))).collect();
            if dim<=1 {
                self.line(format!("if({buffer}.w!=0u) {{if(({x})<{buffer}.z) buffer_write_raw(res,{buffer},{x},image_raw_words(image_encode(uint4({}),image_meta[{binding}]),image_meta[{binding}]),{}u);}} else {{if({condition}) im{binding}.write(image_encode(uint4({}),image_meta[{binding}]),{coords}{extra});}}",values.join(","),i.addr,values.join(",")));
            } else {self.line(format!("if({condition}) im{binding}.write(image_encode(uint4({}),image_meta[{binding}]),{coords}{extra});",values.join(",")));}
        }
        Ok(())
    }

    fn surface_atomic_plain(&mut self,i:&Inst,b:u32)->Result<()>{
        let load=i.op==Op::SuldDB;
        let (dim,coord,dest,bytes,byte_address,pred)=if load{
            let o=InstSuldDB(i.raw);if o.size().0!=4{bail!("mixed atomic image loads require raw32 width");}
            (o.dim().0,o.src_a(),o.dest(),4,o.ba(),o.dest_pred2())
        }else{
            let o=InstSustB(i.raw);if o.rgba().0!=15{bail!("mixed atomic image stores require complete components");}
            (o.dim().0,o.src_a(),o.dest(),4,false,7)
        };
        if self.sph.stage!=Stage::Compute||!matches!(dim,0|1|3){bail!("mixed atomic image requires compute 1D/2D");}
        self.iface.images.insert(b,ImageBinding{kind:TexKind::T2D,raw_bytes:bytes});self.iface.buffer_metadata=true;
        if !load{self.iface.image_writes|=1<<b;}
        let mut x=self.r(coord);if byte_address{x=format!("uint(int({x})>>2u)");}
        let y=if dim==3{self.r((coord+1).min(RZ_))}else{"0u".into()};
        let value=if load{"0u".into()}else{self.r(dest)};
        // R32 integer format and dimensions are checked by the existing host
        // atomic-image staging path. All accesses in this dispatch share that
        // one backing allocation, including stores lexically before SUATOM.
        self.line(format!("{{uint ax={x},ay={y},plain_value={value};uint4 layout=image_atomic_layout[{b}],linear=buffer_meta[{}];bool in_bounds=false;device atomic_uint* pointer=nullptr;uint previous=0u;if(linear.w!=0u){{in_bounds=ax<linear.z&&ay==0u;if(in_bounds){{ulong address=(ulong(linear.x)|(ulong(linear.y)<<32ul))+ulong(ax)*4ul;ulong gpu=gmem_gpu(res,address);if(gpu==0ul||(address&3ul)!=0ul)shader_fault(res,512u,{}u);else pointer=reinterpret_cast<device atomic_uint*>(gpu);}}}}else{{in_bounds=ax<layout.x&&ay<layout.y;if(in_bounds)pointer=image_atomic{b}+ay*layout.z+ax;}}",256+b,i.addr));
        if load{
            self.line("if(pointer!=nullptr)previous=atomic_load_explicit(pointer,memory_order_relaxed);");self.set(dest,"previous".into());self.setp(pred,"in_bounds".into());
        }else{self.line("if(pointer!=nullptr)atomic_store_explicit(pointer,plain_value,memory_order_relaxed);");}
        self.line("}");Ok(())
    }

    fn tld(&mut self,o:InstTld)->Result<()> {
        if o.dest()==RZ_ {return Ok(());}
        if o.ms() || o.cl() || o.dest_pred()!=7 {bail!("TLD multisample/clamp/sparse residency is unsupported");}
        let dim=o.dim().0;
        let kind=match dim {2=>TexKind::T2D,3=>TexKind::T2DArray,4=>TexKind::T3D,_=>bail!("TLD dimension {dim} is unsupported")};
        let mut ra=o.src_a();let take=|r:&mut i32| {let v=(*r).min(RZ_);if *r<RZ_ {*r+=1;}v};
        let layer=if dim==3 {Some(self.r(take(&mut ra)))} else {None};
        let n=if dim==4 {3}else{2};
        let coords=(0..n).map(|_|self.r(take(&mut ra))).collect::<Vec<_>>();
        let mut rb=o.src_b();let lod=if o.lod() {self.r(take(&mut rb))} else {"0u".into()};
        let offset=if o.toff() {self.texture_offset(take(&mut rb),dim)?} else {format!("int{n}(0)")};
        let binding=self.tex(o.tid_b(),kind);
        // Snapshot all operands before writing packed result registers, which
        // may alias the coordinate pair. Out-of-range integer fetches are zero.
        self.line(format!("{{ uint load_lod={lod};int{n} load_coord=int{n}({})+{offset};float4 loaded=float4(0.0f);",coords.join(",")));
        let mut guard=format!("load_lod<t{binding}.get_num_mip_levels() && all(load_coord>=int{n}(0)) && uint(load_coord.x)<t{binding}.get_width(load_lod) && uint(load_coord.y)<t{binding}.get_height(load_lod)");
        let extra=if let Some(layer)=layer {self.line(format!("uint load_layer={layer};"));guard+=&format!(" && load_layer<t{binding}.get_array_size()");",load_layer"} else {""};
        if dim==4 {guard+=&format!(" && uint(load_coord.z)<t{binding}.get_depth(load_lod)");}
        self.line(format!("if({guard}) loaded=t{binding}.read(uint{n}(load_coord){extra},load_lod);"));
        let dests=(0..4).map(|c|(o.dest()+c).min(RZ_)).collect::<Vec<_>>();self.write_components("loaded",o.w_mask() as u32,&dests);self.line("}");Ok(())
    }

    fn tlds(&mut self, o: InstTlds) -> Result<()> {
        let (dest, dest2) = (o.dest(), o.dest2());
        if dest == RZ_ && dest2 == RZ_ {
            return Ok(());
        }
        let (mut ra, mut rb) = (o.src_a(), o.src_b());
        let mut na = || {
            let r = ra;
            if ra != RZ_ {
                ra += 1;
            }
            r
        };
        let mut nb = || {
            let r = rb;
            if rb != RZ_ {
                rb += 1;
            }
            r
        };
        let t = o.target().0;
        let (x, y, lod, kind, arr) = match t {
            0 => (na(), RZ_, None, TexKind::T2D, None),
            1 => (na(), RZ_, Some(nb()), TexKind::T2D, None),
            2 => (na(), nb(), None, TexKind::T2D, None),
            4 => (na(), na(), None, TexKind::T2D, None),
            5 | 0xc => (na(), na(), Some(nb()), TexKind::T2D, None),
            6 => (na(), na(), None, TexKind::T2D, None),
            7 => (na(), na(), None, TexKind::T3D, None),
            8 => {
                let x = nb();
                let y = nb();
                (x, y, None, TexKind::T2DArray, Some(na()))
            }
            _ => bail!("tlds target {t}"),
        };
        // TLDS.3D.LZ splits XYZ as srcA.{x,y},srcB.z. In particular
        // srcB may alias the third packed destination register.
        let z=if t==7 {Some(nb())} else {None};
        let b = self.tex(o.tid_b(), kind);
        if t<=1 {self.iface.buffer_metadata=true;}
        let (xs, ys) = (self.r(x), self.r(y));
        let l = lod.map(|r| self.r(r)).unwrap_or("0u".into());
        let v = match (kind, arr) {
            (TexKind::T2DArray, Some(a)) => {
                let av = self.r(a);
                format!("t{b}.read(uint2({xs}, {ys}), {av}, {l})")
            }
            (TexKind::T3D, _) => {
                let zs=self.r(z.expect("TLDS 3D depth coordinate"));
                format!("((uint({xs})<t{b}.get_width({l}) && uint({ys})<t{b}.get_height({l}) && uint({zs})<t{b}.get_depth({l}))?t{b}.read(uint3({xs},{ys},{zs}),{l}):float4(0.0f))")
            },
            _ => format!("t{b}.read(uint2({xs}, {ys}), {l})"),
        };
        let mask = TEXS_MASK_LUT[if dest2 == RZ_ { 0 } else { 1 }][o.w_mask() as usize & 7];
        let dests = [dest, if dest == RZ_ { RZ_ } else { dest + 1 }, dest2, if dest2 == RZ_ { RZ_ } else { dest2 + 1 }];
        let v=if t<=1 {format!("(buffer_tex{b}.w!=0u?as_type<float4>(buffer_fetch(res,buffer_tex{b},{xs},{l},pc)):{v})")} else {v};
        self.line(format!("{{ float4 v = {v};"));
        self.write_components("v", mask, &dests);
        self.line("}");
        Ok(())
    }

    fn tld4(&mut self, i: &Inst) -> Result<()> {
        let w = i.raw;
        let (d, d2, a, bsrc, comp, dc, tid, mask, two_dest) = if i.op == Op::Tld4s {
            let o = InstTld4s(w);
            (o.dest(), o.dest2(), o.src_a(), o.src_b(), o.tex_comp().0, o.dc(), o.tid_b(), 0xF, true)
        } else {
            let o = InstTld4(w);
            (o.dest(), RZ_, o.src_a(), o.src_b(), o.tex_comp().0, o.dc(), o.tid_b(), o.w_mask() as u32, false)
        };
        let kind = if dc { TexKind::Depth2D } else { TexKind::T2D };
        let b = self.tex(tid, kind);
        let coords = if two_dest && dc { vec![a, if a == RZ_ { RZ_ } else { a + 1 }] } else if two_dest { vec![a, bsrc] } else { vec![a, if a == RZ_ { RZ_ } else { a + 1 }] };
        let c = self.coord(&coords, 2);
        let comp_name = ["x", "y", "z", "w"][comp as usize & 3];
        let v = if dc {
            let rf = if two_dest { bsrc } else { bsrc };
            let rv = self.fr(rf);
            format!("t{b}.gather_compare(s{b}, {c}, {rv})")
        } else {
            format!("t{b}.gather(s{b}, {c}, int2(0), component::{comp_name})")
        };
        self.line(format!("{{ float4 v = {v};"));
        let dests: Vec<i32> = if two_dest {
            vec![d, if d == RZ_ { RZ_ } else { d + 1 }, d2, if d2 == RZ_ { RZ_ } else { d2 + 1 }]
        } else {
            (0..4).map(|k| if d == RZ_ { RZ_ } else { d + k }).collect()
        };
        self.write_components("v", mask, &dests);
        self.line("}");
        Ok(())
    }
}

const TEXS_MASK_LUT: [[u32; 8]; 2] = [
    [0b0001, 0b0010, 0b0100, 0b1000, 0b0011, 0b1001, 0b1010, 0b1100],
    [0b0111, 0b1011, 0b1101, 0b1110, 0b1111, 0b0000, 0b0000, 0b0000],
];

const PRELUDE: &str = r#"#include <metal_stdlib>
using namespace metal;
#define NX_APPROX_MUFU 0
static inline float nx_mufu_rcp(float x) {
#if NX_APPROX_MUFU
    uint magnitude=as_type<uint>(x)&0x7fffffffu;
    if (magnitude>=0x00800000u && magnitude<=0x7e800000u) {
        float estimate=fast::divide(1.0f,x);
        return fma(estimate,fma(-x,estimate,1.0f),estimate);
    }
#endif
    return 1.0f/x;
}
static inline float nx_mufu_rsqrt(float x) {
#if NX_APPROX_MUFU
    uint bits=as_type<uint>(x);
    if (bits>=0x21800000u && bits<=0x5d800000u) {
        float estimate=fast::rsqrt(x);
        return fma(estimate,fma(-0.5f*x,estimate*estimate,0.5f),estimate);
    }
#endif
    return rsqrt(x);
}
#pragma METAL fp contract(off)

struct Res {
    const device uint* cb[18];
    const device uchar* vb[16];
    uint vb_stride[16];
    uint vb_divisor[16];
    uint attr_format[16];
    uint attr_stream[16];
    uint attr_offset[16];
    int base_vertex;
    uint base_instance;
    uint flags;
    float alpha_ref;
    uint alpha_func;
    ulong heap_base;
    ulong heap_gpu[16];
    ulong heap_size;
    ulong heap_chunk_size;
    ulong zero_gpu;
    device const ulong* pool_spans;
    ulong pool_span_count;
    ulong fault_gpu;
    ulong shader_hash;
    ulong pool_root_base;
    ulong pool_root_count;
    device const uint* pool_roots;
    device const ulong* pool_pages;
};

struct GmemCache { ulong base; ulong size; ulong gpu; };

#ifndef NX_PAGED_GMEM
#define NX_PAGED_GMEM 0
#endif
static inline ulong gmem_page(constant Res& res, ulong ga) {
    ulong root = (ga >> 24ul) - res.pool_root_base;
    if (root >= res.pool_root_count) return 0ul;
    uint leaf = res.pool_roots[root];
    if (leaf == 0u) return 0ul;
    ulong page = res.pool_pages[ulong(leaf - 1u) * 4096ul + ((ga >> 12ul) & 4095ul)];
    return page == 0ul ? 0ul : page + (ga & 4095ul);
}

// Pool mappings are immutable for an encoded command. Repeated loads from
// one storage range can reuse its translation without changing guest pointers.
static inline ulong gmem_gpu_cached(thread GmemCache& cache, constant Res& res, ulong ga) {
    ulong off = ga - res.heap_base;
    if (off < res.heap_size && res.heap_chunk_size != 0ul) {
        ulong chunk = off / res.heap_chunk_size;
        return res.heap_gpu[chunk] + off % res.heap_chunk_size;
    }
#if NX_PAGED_GMEM
    ulong page = gmem_page(res, ga);
    if (page != 0ul) return page;
#endif
    if (ga - cache.base < cache.size) return cache.gpu + ga - cache.base;
    ulong lo = 0ul, hi = res.pool_span_count;
    while (lo < hi) {
        ulong mid = (lo + hi) / 2ul;
        ulong base = res.pool_spans[mid*3ul];
        if (ga < base) hi = mid;
        else if (ga - base >= res.pool_spans[mid*3ul+1ul]) lo = mid + 1ul;
        else {
            cache.base = base;
            cache.size = res.pool_spans[mid*3ul+1ul];
            cache.gpu = res.pool_spans[mid*3ul+2ul];
            return cache.gpu + ga - base;
        }
    }
    return 0ul;
}

static inline ulong gmem_gpu(constant Res& res, ulong ga) {
    ulong off = ga - res.heap_base;
    if (off < res.heap_size && res.heap_chunk_size != 0ul) {
        ulong chunk = off / res.heap_chunk_size;
        return res.heap_gpu[chunk] + off % res.heap_chunk_size;
    }
#if NX_PAGED_GMEM
    ulong page = gmem_page(res, ga);
    if (page != 0ul) return page;
#endif
    ulong lo = 0ul, hi = res.pool_span_count;
    while (lo < hi) {
        ulong mid = (lo + hi) / 2ul;
        ulong base = res.pool_spans[mid*3ul];
        if (ga < base) hi = mid;
        else if (ga - base >= res.pool_spans[mid*3ul+1ul]) lo = mid + 1ul;
        else return res.pool_spans[mid*3ul+2ul] + ga - base;
    }
    return 0ul;
}

static inline device uint* gmem_cached(thread GmemCache& cache, constant Res& res, ulong ga) {
    ulong gpu = gmem_gpu_cached(cache, res, ga);
    return reinterpret_cast<device uint*>(gpu != 0ul ? gpu : res.zero_gpu);
}

static inline device uint* gmem(constant Res& res, ulong ga) {
    ulong gpu = gmem_gpu(res, ga);
    return reinterpret_cast<device uint*>(gpu != 0ul ? gpu : res.zero_gpu);
}

// One record per command buffer. The host reads it only after that buffer
// completes, so a later in-flight shader cannot publish a partial record.
static inline bool shader_fault(constant Res& res,uint code,uint pc) {
    device atomic_uint* flag=reinterpret_cast<device atomic_uint*>(res.fault_gpu);
    uint expected=0u;bool first=false;
    while(expected==0u && !(first=atomic_compare_exchange_weak_explicit(flag,&expected,code,memory_order_relaxed,memory_order_relaxed))) {}
    if(first) {
        reinterpret_cast<device uint*>(flag)[1]=pc;
        reinterpret_cast<device uint*>(flag)[2]=uint(res.shader_hash);
        reinterpret_cast<device uint*>(flag)[3]=uint(res.shader_hash>>32ul);
    }
    return first;
}

struct DynamicTextures {
    array<texture2d<float>,4096> color [[id(0)]];
    array<depth2d<float>,4096> depth [[id(4096)]];
    array<sampler,256> samplers [[id(8192)]];
    device const uint2* color_lookup [[id(8448)]];
    device const uint2* depth_lookup [[id(8449)]];
    device const uint2* sampler_lookup [[id(8450)]];
    uint4 counts [[id(8451)]];
};
inline uint bindless_find(device const uint2* table,uint count,uint id) {
    uint lo=0u,hi=count;
    while(lo<hi) {uint mid=lo+(hi-lo)/2u;if(table[mid].x<id)lo=mid+1u;else hi=mid;}
    return lo<count && table[lo].x==id?table[lo].y:0xffffffffu;
}

static inline bool bop(uint op, bool a, bool b) {
    return op == 0u ? (a && b) : op == 1u ? (a || b) : (a != b);
}
static inline bool fcmp(uint op, float a, float b) {
    bool un = isnan(a) || isnan(b);
    switch (op) {
    case 0u: return false;
    case 1u: return a < b;
    case 2u: return a == b;
    case 3u: return a <= b;
    case 4u: return a > b;
    case 5u: return !un && a != b;
    case 6u: return a >= b;
    case 7u: return !un;
    case 8u: return un;
    case 9u: return un || a < b;
    case 10u: return un || a == b;
    case 11u: return un || a <= b;
    case 12u: return un || a > b;
    case 13u: return a != b;
    case 14u: return un || a >= b;
    default: return true;
    }
}
static inline bool icmp(uint op, uint a, uint b, bool s) {
    int sa = int(a), sb = int(b);
    switch (op) {
    case 1u: return s ? sa < sb : a < b;
    case 2u: return a == b;
    case 3u: return s ? sa <= sb : a <= b;
    case 4u: return s ? sa > sb : a > b;
    case 5u: return a != b;
    case 6u: return s ? sa >= sb : a >= b;
    case 7u: return true;
    default: return false;
    }
}
static inline bool ccc(uint c, bool zf, bool nf, bool cf, bool vf) {
    switch (c) {
    case 0u: return false;
    case 1u: return nf != vf;
    case 2u: return zf;
    case 10u: return zf;
    case 3u: return zf || (nf != vf);
    case 4u: return !zf && (nf == vf);
    case 5u: return !zf;
    case 13u: return !zf;
    case 6u: return nf == vf;
    case 15u: return true;
    case 16u: return vf;
    case 17u: return cf;
    case 18u: return nf;
    case 19u: return !cf || zf;
    case 20u: return cf && !zf;
    case 21u: return !nf;
    case 22u: return !cf;
    case 23u: return !vf;
    default: return true;
    }
}
static inline uint lop3(uint a, uint b, uint c, uint lut) {
    uint r = 0u;
    for (uint i = 0u; i < 8u; i++) {
        if ((lut >> i) & 1u) {
            r |= ((i & 4u) ? a : ~a) & ((i & 2u) ? b : ~b) & ((i & 1u) ? c : ~c);
        }
    }
    return r;
}
static inline uint bfe(uint a, uint b, bool s) {
    uint pos = b & 0xffu, len = (b >> 8) & 0xffu;
    if (len == 0u) return 0u;
    if (pos >= 32u) return s && int(a) < 0 ? 0xffffffffu : 0u;
    len = min(len, 32u - pos);
    uint v = a >> pos;
    if (len < 32u) {
        uint m = (1u << len) - 1u;
        v &= m;
        if (s && ((v >> (len - 1u)) & 1u)) v |= ~m;
    }
    return v;
}
static inline uint ext16(uint x, bool high, bool s) {
    if (s) return high ? uint(int(x) >> 16) : uint(int(x << 16) >> 16);
    return high ? (x >> 16) : (x & 0xffffu);
}
static inline uint xmad(uint a, uint b, uint c, uint mode, bool sa, bool sb, bool ha, bool hb, bool psl, bool mrg, uint carry) {
    uint ea = ext16(a, ha, sa), eb = ext16(b, hb, sb);
    uint p = ea * eb;
    if (psl) p <<= 16;
    switch (mode) {
    case 1u: c = c & 0xffffu; break;
    case 2u: c = c >> 16; break;
    case 3u: c = c - (((ea >> 31) << 16) + ((eb >> 31) << 16)); break;
    case 4u: c = c + (b << 16); break;
    default: break;
    }
    uint r = carry != 0xffffffffu ? p + carry : p + c;
    if (mrg) r = (r & 0xffffu) | (b << 16);
    return r;
}
static inline float i2f(uint x, uint fmt, uint sel) {
    switch (fmt) {
    case 0u: return float((x >> (sel * 8u)) & 0xffu);
    case 1u: return float((x >> ((sel >> 1) * 16u)) & 0xffffu);
    case 4u: return float(int(x << (24u - sel * 8u)) >> 24);
    case 5u: return float(int(x << (16u - (sel >> 1) * 16u)) >> 16);
    case 6u: case 7u: return float(int(x));
    default: return float(x);
    }
}
static inline float tof32(uint x, uint fmt, bool high) {
    if (fmt == 1u) return float(as_type<half2>(x)[high ? 1 : 0]);
    return as_type<float>(x);
}
static inline float fround(float x, uint mode) {
    switch (mode) {
    case 4u: return rint(x);
    case 5u: return floor(x);
    case 6u: return ceil(x);
    case 7u: return trunc(x);
    default: return x;
    }
}
static inline uint f2i(float x, uint rnd, uint fmt) {
    if (isnan(x)) return 0u;
    x = rnd == 1u ? floor(x) : rnd == 2u ? ceil(x) : rnd == 3u ? trunc(x) : rint(x);
    switch (fmt) {
    case 1u: return uint(clamp(x, 0.0f, 65535.0f));
    case 5u: return uint(int(clamp(x, -32768.0f, 32767.0f)));
    case 6u: case 7u: return x >= 2147483648.0f ? 0x7fffffffu : x <= -2147483648.0f ? 0x80000000u : uint(int(x));
    default: return x >= 4294967296.0f ? 0xffffffffu : x <= 0.0f ? 0u : uint(x);
    }
}
static inline uint i2i(uint x, uint sel, uint src, uint dst, bool sat, bool ab, bool ng) {
    int v;
    switch (src) {
    case 0u: v = int((x >> (sel * 8u)) & 0xffu); break;
    case 1u: v = int((x >> ((sel >> 1) * 16u)) & 0xffffu); break;
    case 4u: v = int(x << (24u - sel * 8u)) >> 24; break;
    case 5u: v = int(x << (16u - (sel >> 1) * 16u)) >> 16; break;
    default: v = int(x); break;
    }
    if (ab) v = abs(v);
    if (ng) v = -v;
    if (sat) {
        switch (dst) {
        case 0u: v = clamp(v, 0, 255); break;
        case 1u: v = clamp(v, 0, 65535); break;
        case 4u: v = clamp(v, -128, 127); break;
        case 5u: v = clamp(v, -32768, 32767); break;
        default: break;
        }
    }
    return uint(v);
}

static inline uint4 fetch_attr(constant Res& res, uint i, uint vid, uint iid) {
    uint fmt = res.attr_format[i];
    if (fmt == 0u) return uint4(0u, 0u, 0u, 0x3f800000u);
    uint s = res.attr_stream[i];
    uint div = res.vb_divisor[s];
    uint index = div == 0u ? vid : res.base_instance + iid / div;
    const device uchar* p = res.vb[s] + index * res.vb_stride[s] + res.attr_offset[i];
    float4 f = float4(0.0f, 0.0f, 0.0f, 1.0f);
    uint4 u = uint4(0u, 0u, 0u, 1u);
    bool integer = false;
    switch (fmt) {
    case 10u: f.x = *(const device float*)p; break;
    case 22u: f.xy = float2(*(const device packed_float2*)p); break;
    case 34u: f.xyz = float3(*(const device packed_float3*)p); break;
    case 46u: f = float4(*(const device packed_float4*)p); break;
    case 11u: case 12u: u.x = *(const device uint*)p; integer = true; break;
    case 23u: case 24u: u.xy = uint2(*(const device packed_uint2*)p); integer = true; break;
    case 35u: case 36u: u.xyz = uint3(*(const device packed_uint3*)p); integer = true; break;
    case 47u: case 48u: u = uint4(*(const device packed_uint4*)p); integer = true; break;
    case 5u: f.x = float(*(const device half*)p); break;
    case 17u: f.xy = float2(*(const device packed_half2*)p); break;
    case 29u: f.xyz = float3(*(const device packed_half3*)p); break;
    case 41u: f = float4(*(const device packed_half4*)p); break;
    case 6u: f.x = float(*(const device ushort*)p) / 65535.0f; break;
    case 18u: f.xy = float2(*(const device packed_ushort2*)p) / 65535.0f; break;
    case 30u: f.xyz = float3(*(const device packed_ushort3*)p) / 65535.0f; break;
    case 42u: f = float4(*(const device packed_ushort4*)p) / 65535.0f; break;
    case 7u: f.x = max(float(*(const device short*)p) / 32767.0f, -1.0f); break;
    case 19u: f.xy = max(float2(*(const device packed_short2*)p) / 32767.0f, -1.0f); break;
    case 31u: f.xyz = max(float3(*(const device packed_short3*)p) / 32767.0f, -1.0f); break;
    case 43u: f = max(float4(*(const device packed_short4*)p) / 32767.0f, -1.0f); break;
    case 8u: u.x = *(const device ushort*)p; integer = true; break;
    case 20u: u.xy = uint2(*(const device packed_ushort2*)p); integer = true; break;
    case 32u: u.xyz = uint3(*(const device packed_ushort3*)p); integer = true; break;
    case 44u: u = uint4(*(const device packed_ushort4*)p); integer = true; break;
    case 9u: u.x = uint(int(*(const device short*)p)); integer = true; break;
    case 21u: u.xy = uint2(int2(*(const device packed_short2*)p)); integer = true; break;
    case 33u: u.xyz = uint3(int3(*(const device packed_short3*)p)); integer = true; break;
    case 45u: u = uint4(int4(*(const device packed_short4*)p)); integer = true; break;
    case 1u: f.x = float(p[0]) / 255.0f; break;
    case 13u: f.xy = float2(p[0], p[1]) / 255.0f; break;
    case 25u: f.xyz = float3(p[0], p[1], p[2]) / 255.0f; break;
    case 37u: case 56u: f = float4(p[0], p[1], p[2], p[3]) / 255.0f; break;
    case 2u: f.x = max(float(char(p[0])) / 127.0f, -1.0f); break;
    case 14u: f.xy = max(float2(char(p[0]), char(p[1])) / 127.0f, -1.0f); break;
    case 26u: f.xyz = max(float3(char(p[0]), char(p[1]), char(p[2])) / 127.0f, -1.0f); break;
    case 38u: f = max(float4(char(p[0]), char(p[1]), char(p[2]), char(p[3])) / 127.0f, -1.0f); break;
    case 3u: u.x = p[0]; integer = true; break;
    case 15u: u.xy = uint2(p[0], p[1]); integer = true; break;
    case 27u: u.xyz = uint3(p[0], p[1], p[2]); integer = true; break;
    case 39u: u = uint4(p[0], p[1], p[2], p[3]); integer = true; break;
    case 4u: u.x = uint(int(char(p[0]))); integer = true; break;
    case 16u: u.xy = uint2(int2(char(p[0]), char(p[1]))); integer = true; break;
    case 28u: u.xyz = uint3(int3(char(p[0]), char(p[1]), char(p[2]))); integer = true; break;
    case 40u: u = uint4(int4(char(p[0]), char(p[1]), char(p[2]), char(p[3]))); integer = true; break;
    case 81u: f.x = float(p[0]); break;
    case 82u: f.x = float(char(p[0])); break;
    case 83u: f.x = float(*(const device ushort*)p); break;
    case 84u: f.x = float(*(const device short*)p); break;
    case 85u: f.x = float(*(const device uint*)p); break;
    case 86u: f.x = float(*(const device int*)p); break;
    case 87u: f.xy = float2(p[0], p[1]); break;
    case 88u: f.xy = float2(char(p[0]), char(p[1])); break;
    case 89u: f.xy = float2(*(const device packed_ushort2*)p); break;
    case 90u: f.xy = float2(*(const device packed_short2*)p); break;
    case 91u: f.xy = float2(*(const device packed_uint2*)p); break;
    case 92u: f.xy = float2(*(const device packed_int2*)p); break;
    case 93u: f.xyz = float3(p[0], p[1], p[2]); break;
    case 94u: f.xyz = float3(char(p[0]), char(p[1]), char(p[2])); break;
    case 95u: f.xyz = float3(*(const device packed_ushort3*)p); break;
    case 96u: f.xyz = float3(*(const device packed_short3*)p); break;
    case 97u: f.xyz = float3(*(const device packed_uint3*)p); break;
    case 98u: f.xyz = float3(*(const device packed_int3*)p); break;
    case 99u: f = float4(p[0], p[1], p[2], p[3]); break;
    case 100u: f = float4(char(p[0]), char(p[1]), char(p[2]), char(p[3])); break;
    case 101u: f = float4(*(const device packed_ushort4*)p); break;
    case 102u: f = float4(*(const device packed_short4*)p); break;
    case 103u: f = float4(*(const device packed_uint4*)p); break;
    case 104u: f = float4(*(const device packed_int4*)p); break;
    case 105u: case 106u: case 108u: {
        int v = *(const device int*)p;
        int4 c = int4((v << 22) >> 22, (v << 12) >> 22, (v << 2) >> 22, v >> 30);
        if (fmt == 105u) f = max(float4(c) / float4(511.0f, 511.0f, 511.0f, 1.0f), -1.0f);
        else if (fmt == 108u) f = float4(c);
        else { u = as_type<uint4>(c); integer = true; }
        break;
    }
    case 107u: { uint v = *(const device uint*)p; f = float4(v & 0x3ffu, (v >> 10) & 0x3ffu, (v >> 20) & 0x3ffu, v >> 30); break; }
    case 61u: { uint v = *(const device uint*)p; f = float4(v & 0x3ffu, (v >> 10) & 0x3ffu, (v >> 20) & 0x3ffu, v >> 30) / float4(1023.0f, 1023.0f, 1023.0f, 3.0f); break; }
    case 62u: { uint v = *(const device uint*)p; u = uint4(v & 0x3ffu, (v >> 10) & 0x3ffu, (v >> 20) & 0x3ffu, v >> 30); integer = true; break; }
    default: break;
    }
    return integer ? u : as_type<uint4>(f);
}
// Unsigned five-bit exponent (bias 15), with round-to-nearest-even directly
// from float32. Avoid an intermediate half conversion and double rounding.
inline uint image_ufloat(uint value,uint mantissa) {
    uint e=(value>>23u)&255u, f=value&0x7fffffu;
    if(e==255u && f!=0u) return (31u<<mantissa)|(1u<<(mantissa-1u));
    if((value&0x80000000u)!=0u) return 0u;
    if(e==255u || value>=0x47800000u) return 31u<<mantissa;
    if(e<113u) return uint(rint(ldexp(as_type<float>(value),int(14u+mantissa))));
    uint shift=23u-mantissa;
    uint rounded=value+((1u<<(shift-1u))-1u)+((value>>shift)&1u);
    return min((rounded-(112u<<23u))>>shift,31u<<mantissa);
}
inline uint4 image_encode(uint4 value,uint4 meta) {
    const uint bits=meta.x, kind=meta.z;
    if(kind==6u) return uint4(image_ufloat(value.x,6u)|(image_ufloat(value.y,6u)<<11u)|(image_ufloat(value.z,5u)<<22u),0u,0u,0u);
    if(kind==7u || kind==8u) {
        uint4 v=kind==7u?uint4(rint(clamp(as_type<float4>(value),float4(0.0f),float4(1.0f))*float4(1023.0f,1023.0f,1023.0f,3.0f))):value;
        return uint4((v.x&1023u)|((v.y&1023u)<<10u)|((v.z&1023u)<<20u)|((v.w&3u)<<30u),0u,0u,0u);
    }
    uint4 encoded=value;
    if(kind==3u) encoded=uint4(as_type<ushort4>(half4(as_type<float4>(value))));
    else if(kind==4u) encoded=uint4(rint(clamp(as_type<float4>(value),float4(0.0f),float4(1.0f))*float((1u<<bits)-1u)));
    else if(kind==5u) encoded=uint4(int4(rint(clamp(as_type<float4>(value),float4(-1.0f),float4(1.0f))*float((1u<<(bits-1u))-1u))));
    if(bits<32u) encoded&=uint4((1u<<bits)-1u);
    return encoded;
}
inline uint4 image_raw_words(uint4 value,uint4 meta) {
    if(meta.x==8u) return uint4((value.x&255u)|((value.y&255u)<<8)|((value.z&255u)<<16)|((value.w&255u)<<24),0u,0u,0u);
    if(meta.x==16u) return uint4((value.x&65535u)|((value.y&65535u)<<16),(value.z&65535u)|((value.w&65535u)<<16),0u,0u);
    return value;
}

// NVN buffer textures address the same pool bytes as storage/vertex buffers.
// Keep their logical length independent of Metal's 2D texture width limit.
inline uint4 buffer_format(uint packed) {
    uint f=packed&255u,components=1u,bits=0u,kind=0u,index=0u;
    if(f>=1u && f<=48u) {
        components=f>=37u?4u:f>=25u?3u:f>=13u?2u:1u;
        uint n=f-(components-1u)*12u;
        if(n<=4u) {bits=8u;index=n-1u;kind=index==0u?4u:index==1u?5u:index-2u;}
        else if(n<=9u) {bits=16u;index=n-5u;kind=index==0u?3u:index==1u?4u:index==2u?5u:index-3u;}
        else {bits=32u;index=n-10u;kind=index==0u?2u:index-1u;}
    }
    return uint4(bits,components,kind,0u);
}
inline uint4 buffer_read_raw(constant Res& res,uint4 meta,uint x,uint pc) {
    uint4 result=uint4(0u);uint4 format=buffer_format(meta.w);
    uint bytes=format.x*format.y/8u;
    if(x>=meta.z || bytes==0u) return result;
    ulong address=ulong(meta.x)|(ulong(meta.y)<<32ul);address+=ulong(x)*bytes;
    if((address&3ul)==0ul && (bytes&3u)==0u) {
        for(uint i=0u;i<bytes/4u;i++) {
            ulong gpu=gmem_gpu(res,address+ulong(i)*4ul);
            if(gpu==0ul) {shader_fault(res,512u,pc);return uint4(0u);}
            result[i]=*reinterpret_cast<const device uint*>(gpu);
        }
    } else {
        for(uint i=0u;i<bytes;i++) {
            ulong gpu=gmem_gpu(res,address+ulong(i));
            if(gpu==0ul) {shader_fault(res,512u,pc);return uint4(0u);}
            result[i/4u]|=uint(*reinterpret_cast<const device uchar*>(gpu))<<((i%4u)*8u);
        }
    }
    return result;
}
inline void buffer_write_raw(constant Res& res,uint4 meta,uint x,uint4 value,uint pc) {
    uint4 format=buffer_format(meta.w);uint bytes=format.x*format.y/8u;
    if(x>=meta.z || bytes==0u) return;
    ulong address=ulong(meta.x)|(ulong(meta.y)<<32ul);address+=ulong(x)*bytes;
    if((address&3ul)==0ul && (bytes&3u)==0u) {
        for(uint i=0u;i<bytes/4u;i++) {
            ulong gpu=gmem_gpu(res,address+ulong(i)*4ul);
            if(gpu==0ul) {shader_fault(res,512u,pc);return;}
            *reinterpret_cast<device uint*>(gpu)=value[i];
        }
    } else {
        // Subword pixels may share a word between concurrent shader lanes.
        // Atomic replacement preserves the neighboring pixels and padding.
        for(uint i=0u;i<bytes;i++) {
            ulong byte_address=address+ulong(i),aligned=byte_address&~3ul;
            ulong gpu=gmem_gpu(res,aligned);
            if(gpu==0ul || gmem_gpu(res,aligned+3ul)!=gpu+3ul) {shader_fault(res,512u,pc);return;}
            uint shift=uint(byte_address&3ul)*8u,mask=255u<<shift;
            uint byte_value=(value[i/4u]>>((i%4u)*8u))&255u;
            device atomic_uint* pointer=reinterpret_cast<device atomic_uint*>(gpu);
            uint old=atomic_load_explicit(pointer,memory_order_relaxed);
            while(!atomic_compare_exchange_weak_explicit(pointer,&old,(old&~mask)|(byte_value<<shift),memory_order_relaxed,memory_order_relaxed)) {}
        }
    }
}
inline uint4 buffer_fetch(constant Res& res,uint4 meta,uint x,uint lod,uint pc) {
    if(x>=meta.z || lod!=0u) return uint4(0u);
    uint4 format=buffer_format(meta.w),raw=buffer_read_raw(res,meta,x,pc);
    uint bits=format.x,components=format.y,kind=format.z;
    uint4 value=uint4(0u,0u,0u,kind<=1u?1u:0x3f800000u);
    for(uint c=0u;c<components;c++) {
        uint v=raw[(c*bits)/32u]>>((c*bits)%32u);
        if(bits<32u) v&=(1u<<bits)-1u;
        if(kind==1u && bits<32u) v=uint(int(v<<(32u-bits))>>(32u-bits));
        if(kind==3u) v=as_type<uint>(float(as_type<half>(ushort(v))));
        else if(kind==4u) v=as_type<uint>(float(v)/float((1u<<bits)-1u));
        else if(kind==5u) v=as_type<uint>(max(-1.0f,float(int(v<<(32u-bits))>>(32u-bits))/float((1u<<(bits-1u))-1u)));
        value[c]=v;
    }
    uint4 result;
    for(uint c=0u;c<4u;c++) {
        uint swizzle=(meta.w>>(8u+c*3u))&7u;
        result[c]=swizzle==0u?0u:swizzle==1u?(kind<=1u?1u:0x3f800000u):value[min(swizzle-2u,3u)];
    }
    return result;
}

"#;

pub fn translate_configured(blob:&[u8],extra_outputs:u32,uint_targets:u32,sint_targets:u32,constants:Option<&[u8]>,config:TranslationConfig)->Result<Translated> {
    if let Some((local_memory,shared_memory))=config.compute_memory {
        if shared_memory>32768 || local_memory>32768 {bail!("compute memory requirement exceeds current Metal translator limits");}
        translate_code(&Sph::compute(local_memory),blob,extra_outputs,uint_targets,sint_targets,shared_memory,constants,GraphicsDispatch::Switch,config.vector_global_loads)
    } else {
        let sph=blob.get(0x30..).and_then(Sph::parse).ok_or_else(||anyhow::anyhow!("bad SPH"))?;
        let code=blob.get(0x80..).ok_or_else(||anyhow::anyhow!("truncated graphics program"))?;
        translate_code(&sph,code,extra_outputs,uint_targets,sint_targets,0,constants,config.graphics_dispatch,config.vector_global_loads)
    }
}

pub fn translate(blob: &[u8]) -> Result<Translated> {
    translate_with(blob, 0, 0, 0)
}

/// Fragment-stage interpretation of `extra_outputs`: export the second Maxwell
/// color as the secondary blend source of target zero. Vertex stages continue
/// to interpret the argument as their generic output mask.
pub const FRAGMENT_DUAL_SOURCE: u32 = 1;
/// Emit the NVN fixed-function alpha test only for draws that enable it.
pub const FRAGMENT_ALPHA_TEST:u32=2;
pub const FRAGMENT_EARLY_TESTS:u32=4;

pub fn can_force_early_fragment_tests(blob:&[u8])->bool {
    let Some(sph)=blob.get(0x30..).and_then(Sph::parse) else {return false;};
    if sph.stage!=Stage::Fragment || sph.kills_pixels || sph.omap_depth || sph.omap_sample_mask {return false;}
    let Some(code)=blob.get(0x80..) else {return false;};
    let Ok(program)=Program::decode(code) else {return false;};
    // Fault reporting added by the translator is diagnostic. Original shader
    // stores, kills, atomics and depth/sample-mask exports must retain their
    // execution order relative to depth/stencil tests.
    !program.insts.iter().any(|i|matches!(i.op,Op::Invalid|Op::Kil|Op::St|Op::Stg|Op::SustB|Op::SuatomB|Op::Atom|Op::Atoms|Op::AtomCas|Op::Red))
}

pub fn translate_with(blob: &[u8], extra_outputs: u32, uint_targets: u32, sint_targets: u32) -> Result<Translated> {
    let sph = Sph::parse(&blob[0x30..]).ok_or_else(|| anyhow::anyhow!("bad SPH"))?;
    translate_code(&sph, &blob[0x80..], extra_outputs, uint_targets, sint_targets,0,None,GraphicsDispatch::Switch,false)
}

pub fn translate_with_constants(blob:&[u8],extra_outputs:u32,uint_targets:u32,sint_targets:u32,constants:&[u8])->Result<Translated> {
    let sph=Sph::parse(&blob[0x30..]).ok_or_else(||anyhow::anyhow!("bad SPH"))?;
    translate_code(&sph,&blob[0x80..],extra_outputs,uint_targets,sint_targets,0,Some(constants),GraphicsDispatch::Switch,false)
}

/// Experimental graphics dispatch: execute forward edges in the same sweep.
/// Backedges still enter the outer loop; dynamic targets retain every decoded
/// label. The compute token scheduler is deliberately unaffected.
pub fn translate_graphics_forward(blob:&[u8],extra_outputs:u32,uint_targets:u32,sint_targets:u32,constants:Option<&[u8]>)->Result<Translated> {
    let sph=blob.get(0x30..).and_then(Sph::parse).ok_or_else(||anyhow::anyhow!("bad SPH"))?;
    translate_code(&sph,&blob[0x80..],extra_outputs,uint_targets,sint_targets,0,constants,GraphicsDispatch::Forward,false)
}

pub fn translate_graphics_regions(blob:&[u8],extra_outputs:u32,uint_targets:u32,sint_targets:u32,constants:Option<&[u8]>)->Result<Translated> {
    let sph=blob.get(0x30..).and_then(Sph::parse).ok_or_else(||anyhow::anyhow!("bad SPH"))?;
    translate_code(&sph,&blob[0x80..],extra_outputs,uint_targets,sint_targets,0,constants,GraphicsDispatch::Regions,false)
}

pub fn translate_graphics_structured(blob:&[u8],extra_outputs:u32,uint_targets:u32,sint_targets:u32,constants:Option<&[u8]>)->Result<Translated> {
    let sph=blob.get(0x30..).and_then(Sph::parse).ok_or_else(||anyhow::anyhow!("bad SPH"))?;
    translate_code(&sph,&blob[0x80..],extra_outputs,uint_targets,sint_targets,0,constants,GraphicsDispatch::Structured,false)
}

fn independent_compute(prog:&Program,shared_memory:u32)->bool {
    shared_memory==0 && prog.insts.iter().all(|i| {
        !matches!(i.op,Op::Bra|Op::Brx|Op::Ssy|Op::Pbk|Op::Sync|Op::Brk|
            Op::Bar|Op::Membar|Op::Lds|Op::Sts|Op::Atom|Op::Atoms|Op::Red|
            Op::Shfl|Op::Vote|Op::Ld|Op::Ldg|Op::St|Op::Stg|Op::SuldDB|Op::SuatomB) &&
        (i.op!=Op::S2r || matches!(InstS2r(i.raw).s_reg().0,0|0x20..=0x23|0x25..=0x27))
    })
}

pub fn translate_compute(code: &[u8], local_memory: u32) -> Result<Translated> {
    translate_compute_with_memory(code,local_memory,0)
}
pub fn translate_compute_with_memory(code:&[u8],local_memory:u32,shared_memory:u32)->Result<Translated> {
    if shared_memory>32768 || local_memory>32768 {bail!("compute memory requirement exceeds current Metal translator limits");}
    translate_code(&Sph::compute(local_memory),code,0,0,0,shared_memory,None,GraphicsDispatch::Switch,false)
}

pub fn translate_compute_with_constants(code:&[u8],local_memory:u32,shared_memory:u32,constants:&[u8])->Result<Translated> {
    if shared_memory>32768 || local_memory>32768 {bail!("compute memory requirement exceeds current Metal translator limits");}
    translate_code(&Sph::compute(local_memory),code,0,0,0,shared_memory,Some(constants),GraphicsDispatch::Switch,false)
}

fn forward_dispatch(body:&str,prog:&Program)->String {
    let blocks=prog.insts.iter().filter(|i|prog.targets.contains(&i.addr)).map(|i| {
        let marker=format!("    case {}u:\n",i.addr);
        let start=body.find(&marker).expect("generated graphics block label");
        (i.addr,start,start+marker.len())
    }).collect::<Vec<_>>();
    let mut out=format!("    uint pc = {}u;\n    for (;;) {{\n    bool dispatch_progress = false;\n",prog.insts[0].addr);
    for (n,&(addr,_,start)) in blocks.iter().enumerate() {
        let end=blocks.get(n+1).map_or(body.len(),|b|b.1);
        writeln!(out,"    if (pc == {addr}u) {{\n        dispatch_progress = true;\n        do {{").unwrap();
        out.push_str(&body[start..end]);
        if let Some(next)=blocks.get(n+1) {writeln!(out,"        pc = {}u;",next.0).unwrap();}
        // A branch's existing continue exits this one-iteration do loop.
        // Fallthrough alone reaches the assignment to the next block.
        out.push_str("        } while (false);\n    }\n");
    }
    out.push_str("    if (!dispatch_progress) return epilogue();\n    }\n}\n");out
}

fn translate_code(sph: &Sph, code: &[u8], extra_outputs: u32, uint_targets: u32, sint_targets: u32,shared_memory:u32,constants:Option<&[u8]>,dispatch:GraphicsDispatch,vector_global_loads:bool) -> Result<Translated> {
    if sph.stage==Stage::Fragment && extra_outputs&FRAGMENT_EARLY_TESTS!=0 && (extra_outputs&FRAGMENT_ALPHA_TEST!=0 || sph.kills_pixels || sph.omap_depth || sph.omap_sample_mask) {bail!("early fragment tests conflict with original late fragment effects");}
    let dual_source=sph.stage==Stage::Fragment && extra_outputs&FRAGMENT_DUAL_SOURCE!=0;
    if dual_source && (sph.omap_targets&0xf==0 || sph.omap_targets&0xf0==0 || sph.omap_targets&!0xff!=0) {
        bail!("dual-source blending requires original primary and secondary color exports only");
    }
    let mut prog = Program::decode_with_constants(code,constants)?;
    let regions=if matches!(dispatch,GraphicsDispatch::Regions|GraphicsDispatch::Structured) && sph.stage!=Stage::Compute {graphics_regions::prepare(&mut prog,constants)} else {None};
    if sph.stage==Stage::Fragment && extra_outputs&FRAGMENT_EARLY_TESTS!=0 && prog.insts.iter().any(|i|matches!(i.op,Op::Kil|Op::St|Op::Stg|Op::SustB|Op::SuatomB|Op::Atom|Op::Atoms|Op::AtomCas|Op::Red)) {bail!("early fragment tests would suppress original shader side effects");}
    let make_gen = || Gen {
        embedded_constants:constants,
        vector_global_loads,
        // A straight-line kernel without shared/warp operations has no
        // inter-lane execution dependency. Native Metal lane execution is
        // sufficient; keep the token scheduler for every other compute CFG.
        simt:sph.stage==Stage::Compute && !independent_compute(&prog,shared_memory),
        shared_memory,
        sph,
        prog: &prog,
        body: String::new(),
        instruction_writes:Vec::new(),
        regs: BTreeSet::new(),
        preds: BTreeSet::new(),
        uses_cc: false,
        uses_quad_lane: false,
        uses_lmem: false,
        uses_point_coord: false,
        iface: Interface::default(),
        warnings: Vec::new(),
        handle_regs: BTreeMap::new(),
        sampler_regs: BTreeMap::new(),
        query_only_textures:BTreeSet::new(),
    };
    let mut g=make_gen();
    if sph.stage==Stage::Compute&&prog.insts.iter().any(|i|i.op==Op::SuatomB){
        // Handle provenance is already interpreted by Gen. Discover the full
        // image-use union without running guest code, then emit afresh using
        // a single coherent backing representation per atomic binding.
        let mut discovery=make_gen();
        for i in &prog.insts{discovery.inst(i)?;}
        g.iface.atomic_images=discovery.iface.atomic_images;
    }
    for i in &prog.insts {
        g.instruction_writes.push(Vec::new());
        if prog.targets.contains(&i.addr) {
            g.body.push_str(&format!("    case {}u:\n", i.addr));
        }

        let (pred, inv) = ((i.raw >> 16) & 7, (i.raw >> 19) & 1 != 0);
        let no_pred = i.props.has(crate::table::InstProps::NOPRED);
        let guarded = !no_pred && (pred != 7 || inv) && i.name != InstName::Ssy;
        if guarded {
            let p = g.p(pred as i32, inv);
            g.line(format!("if ({p}) {{"));
        }
        g.inst(i)?;
        if guarded {
            g.line("}");
        }
    }
    g.line(if g.simt {"live=false;continue;"} else {"return epilogue();"});

    let stage = sph.stage;
    if stage == Stage::Vertex {
        g.iface.outputs |= extra_outputs;
    }
    let entry = match stage {
        Stage::Vertex => "vmain",
        Stage::Fragment => "fmain",
        Stage::Compute => "cmain",
        s => bail!("unsupported stage {s:?}"),
    };
    let mut out = String::from(PRELUDE);
    if g.iface.dynamic_textures!=0 {out.push_str(&dynamic_texture_layout());}
    if g.body.contains("nx_gmem_ld") {out.push_str(memory_loads::MULTIWORD);}
    if g.body.contains("embedded_c1[") {
        let words=constants.unwrap().chunks_exact(4).map(|b|format!("{:#x}u",u32::from_le_bytes(b.try_into().unwrap()))).collect::<Vec<_>>();
        writeln!(out,"constant uint embedded_c1[{}] = {{{}}};",words.len(),words.join(",")).unwrap();
    }
    let c = ["x", "y", "z", "w"];
    let _ = c;

    if stage == Stage::Vertex {
        out.push_str("struct VOut {\n    float4 pos [[position, invariant]];\n");
        for i in 0..32 {
            if g.iface.outputs >> i & 1 != 0 {
                writeln!(out, "    float4 v{i} [[user(locn{i})]];").unwrap();
            }
        }
        out.push_str("    float psize [[point_size]];\n};\n");
    } else if stage == Stage::Fragment {
        out.push_str("struct FIn {\n    float4 pos [[position]];\n");
        for i in 0..32 {
            if g.iface.attributes >> i & 1 != 0 {
                let q = match sph.imap[i] {
                    Imap::Constant => " [[flat]]",
                    Imap::ScreenLinear => " [[center_no_perspective]]",
                    _ => "",
                };
                writeln!(out, "    float4 v{i} [[user(locn{i})]]{q};").unwrap();
            }
        }
        out.push_str("};\nstruct FOut {\n");
        let mut base = 0;
        for rt in 0..8 {
            if sph.omap_targets >> (rt * 4) & 0xF != 0 {
                let target=if dual_source {0} else {rt};
                let ty = if uint_targets >> target & 1 != 0 { "uint4" } else if sint_targets >> target & 1 != 0 { "int4" } else { "float4" };
                let attribute=if dual_source && rt==1 {"color(0), index(1)".to_string()} else {format!("color({rt})")};
                writeln!(out, "    {ty} c{rt} [[{attribute}]];").unwrap();
                g.iface.color_targets |= 1 << target;
                base += 4;
            }
        }
        let _ = base;
        if sph.omap_depth {
            out.push_str("    float depth [[depth(any)]];\n");
            g.iface.writes_depth = true;
        }
        if g.iface.color_targets == 0 && !sph.omap_depth {
            out.push_str("    float4 c0 [[color(0)]];\n");
        }
        out.push_str("};\n");
    }

    let mut params = vec!["constant Res& res [[buffer(0)]]".to_string()];
    if g.iface.dynamic_textures!=0 {params.push("constant DynamicTexturesWide& dyntex [[buffer(2)]]".into());}
    if stage == Stage::Vertex {
        params.insert(0, "uint vid [[vertex_id]], uint iid [[instance_id]]".into());
    } else if stage == Stage::Compute {
        params.insert(0, "uint3 tid [[thread_position_in_threadgroup]], uint3 ctaid [[threadgroup_position_in_grid]], uint simd_lane [[thread_index_in_simdgroup]]".into());
    } else {
        params.insert(0, "FIn in [[stage_in]], bool front [[front_facing]]".into());
        if g.uses_quad_lane {params.push("uint quad_lane [[thread_index_in_quadgroup]]".into());}
        if g.uses_point_coord {
            params.push("float2 pcoord [[point_coord]]".into());
        }
    }
    let bindings: Vec<(u32, TexKind)> = g.iface.textures.iter().map(|(b, k)| (*b, *k)).collect();
    g.iface.argument_textures=bindings.len()>16;
    if bindings.len()>256 {bail!("static sampler bank exceeds argument-buffer capacity");}
    if g.iface.argument_textures {
        out.push_str("struct StaticTextures {\n");
        for (slot,(b,k)) in bindings.iter().enumerate() {
            writeln!(out,"    {} t{b} [[id({})]]; sampler s{b} [[id({})]];",k.decl(),slot*2,slot*2+1).unwrap();
            g.body=rename(&g.body,&format!("t{b}"),&format!("statictex.t{b}"));
            g.body=rename(&g.body,&format!("s{b}"),&format!("statictex.s{b}"));
        }
        out.push_str("};\n");
        params.push("constant StaticTextures& statictex [[buffer(12)]]".into());
    }
    for (slot, (b, k)) in bindings.iter().enumerate() {
        g.iface.samplers.insert(*b, slot as u32);
        g.iface.texture_slots.insert(*b,slot as u32);
        g.body=rename(&g.body,&format!("buffer_tex{b}"),&format!("buffer_meta[{slot}]"));
        if !g.iface.argument_textures {params.push(format!("{} t{b} [[texture({slot})]], sampler s{b} [[sampler({slot})]]", k.decl()));}
    }
    if !g.iface.images.is_empty() {params.push("constant uint4* image_meta [[buffer(1)]]".into());}
    if g.iface.buffer_metadata {params.push("constant uint4* buffer_meta [[buffer(13)]]".into());}
    if g.iface.atomic_images!=0 {params.push("constant uint4* image_atomic_layout [[buffer(11)]]".into());}
    for (&b,image) in &g.iface.images {
        if g.iface.atomic_images&(1<<b)!=0 {params.push(format!("device atomic_uint* image_atomic{b} [[buffer({})]]",3+b));continue;}
        let ty=match image.kind {TexKind::T2D=>"texture2d",TexKind::T2DArray=>"texture2d_array",TexKind::T3D=>"texture3d",_=>bail!("unsupported image texture kind")};
        params.push(format!("{ty}<uint,access::read_write> im{b} [[texture({})]]",32+b));
    }
    let (kind, ret) = match stage {
        Stage::Vertex => ("vertex", "VOut"),
        Stage::Compute => ("kernel", "void"),
        _ => ("fragment", "FOut"),
    };
    let early=if stage==Stage::Fragment && extra_outputs&FRAGMENT_EARLY_TESTS!=0 {"[[early_fragment_tests]] "} else {""};
    writeln!(out, "{early}{kind} {ret} {entry}({}) {{", params.join(", ")).unwrap();
    if stage!=Stage::Compute && (g.body.contains("gmem(res,") || g.body.contains("gmem_gpu(res,")) {
        out.push_str("    GmemCache global_memory_cache = {0ul, 0ul, 0ul};\n");
        g.body = g.body.replace("gmem_gpu(res,", "gmem_gpu_cached(global_memory_cache,res,")
            .replace("gmem(res,", "gmem_cached(global_memory_cache,res,");
    }
    if shared_memory!=0 {
        let align=if prog.insts.iter().any(|i|i.op==Op::Atoms) {"alignas(4) "} else {""};
        writeln!(out,"    {align}threadgroup uchar shared_mem[{}];",shared_memory.next_multiple_of(4)).unwrap();
    }

    for r in &g.regs {
        writeln!(out, "    uint r{r} = 0u;").unwrap();
    }
    for p in &g.preds {
        writeln!(out, "    bool p{p} = false;").unwrap();
    }
    if g.uses_cc {
        out.push_str("    bool zf = false, nf = false, cf = false, vf = false;\n");
    }
    if g.uses_lmem {
        writeln!(out,"    thread uchar lmem[{}];",sph.local_memory.max(1)).unwrap();
    }
    if stage == Stage::Vertex {
        out.push_str("    VOut o;\n    o.pos = float4(0.0f, 0.0f, 0.0f, 1.0f);\n    o.psize = 1.0f;\n");
        for i in 0..32 {
            if g.iface.outputs >> i & 1 != 0 {
                writeln!(out, "    o.v{i} = float4(0.0f, 0.0f, 0.0f, 1.0f);").unwrap();
            }
        }
        for i in 0..16 {
            if g.iface.attributes >> i & 1 != 0 {
                writeln!(out, "    uint4 a{i} = fetch_attr(res, {i}u, vid, iid);").unwrap();
            }
        }

        out.push_str(
            "    auto epilogue = [&]() -> VOut {\n        VOut r = o;\n        if (res.flags & 1u) r.pos.y = -r.pos.y;\n        if (res.flags & 2u) r.pos.z = (r.pos.z + r.pos.w) * 0.5f;\n        return r;\n    };\n",
        );
    } else if stage == Stage::Compute {
        out.push_str("    auto epilogue = [&]() {};\n");
    } else {

        let mut s = String::from("    auto epilogue = [&]() -> FOut {\n        FOut r;\n");
        let mut base = 0;
        for rt in 0..8 {
            let m = sph.omap_targets >> (rt * 4) & 0xF;
            if m != 0 {
                let target=if dual_source {0} else {rt};
                let (ty, conv, zero) = if uint_targets >> target & 1 != 0 {
                    ("uint4", "", "0u")
                } else if sint_targets >> target & 1 != 0 {
                    ("int4", "int", "0")
                } else {
                    ("float4", "as_type<float>", "0.0f")
                };
                let comps: Vec<String> = (0..4)
                    .map(|k| {
                        if m >> k & 1 != 0 {
                            g.regs.insert(base + k);
                            format!("{conv}(r{})", base + k)
                        } else {
                            zero.into()
                        }
                    })
                    .collect();
                writeln!(s, "        r.c{rt} = {ty}({});", comps.join(", ")).unwrap();
                base += 4;
            }
        }
        if sph.omap_depth {
            let dr = sph.omap_targets.count_ones() as i32 + 1;
            g.regs.insert(dr);
            writeln!(s, "        r.depth = as_type<float>(r{dr});").unwrap();
        }
        if g.iface.color_targets == 0 && !sph.omap_depth {
            s.push_str("        r.c0 = float4(0.0f);\n");
        }

        if extra_outputs&FRAGMENT_ALPHA_TEST!=0 && sph.omap_targets & 0xF != 0 && (uint_targets | sint_targets) & 1 == 0 {
            s.push_str("        if ((res.flags & 4u) != 0u && !fcmp(res.alpha_func, r.c0.w, res.alpha_ref)) discard_fragment();\n");
        }
        s.push_str("        return r;\n    };\n");

        let declared: String = out.clone();
        for r in &g.regs {
            if !declared.contains(&format!("uint r{r} = 0u;")) {
                writeln!(out, "    uint r{r} = 0u;").unwrap();
            }
        }
        out.push_str(&s);
    }
    if g.simt {
        writeln!(out,"    uint pc={}u;",prog.insts[0].addr).unwrap();out.push_str(compute_control::INIT);
        let mut context=[16,22,8,4].map(|r|if g.regs.contains(&r) {format!("r{r}")} else {"0xffffffffu".into()});
        context[0]=if g.uses_cc {"uint(zf)|(uint(nf)<<1u)|(uint(cf)<<2u)|(uint(vf)<<3u)".into()} else {"0u".into()};
        if prog.insts.iter().any(|i|i.op==Op::Brx) {
            // Fuel failure in an indirect program needs the pending token and
            // jump register, not unrelated ALU values. Keep the fault ABI.
            context=["token_sp==0u?0u:token_id[token_sp-1u]".into(),"token_sp==0u?0u:token_target[token_sp-1u]".into(),"token_sp==0u?0u:token_mask[token_sp-1u]".into(),if g.regs.contains(&17) {"r17".into()} else {"0xffffffffu".into()}];
        }
        let capture=format!("device uint* context=reinterpret_cast<device uint*>(res.fault_gpu);context[4]={};context[5]={};context[6]={};context[7]={};",context[0],context[1],context[2],context[3]);
        out.push_str(&compute_control::HEADER.replace("@FAULT_CONTEXT@",&capture));out.push_str(&g.body);out.push_str(compute_control::TAIL);
    } else if let Some(plan)=regions {
        if let Some(body)=(dispatch==GraphicsDispatch::Structured).then(||graphics_structured::emit(&g.body,&prog,&plan)).flatten() {out.push_str(&body);}
        else {out.push_str(&graphics_regions::emit(&g.body,&prog,&plan));}
    } else if dispatch==GraphicsDispatch::Forward && stage!=Stage::Compute {
        out.push_str(&forward_dispatch(&g.body,&prog));
    } else {
        writeln!(out, "    uint pc = {}u;\n    for (;;) {{\n    switch (pc) {{", prog.insts[0].addr).unwrap();out.push_str(&g.body);out.push_str("    default:\n        return epilogue();\n    }\n    }\n}\n");
    }
    let mut interface = g.iface;
    if stage == Stage::Fragment && interface.color_targets == 0 && !interface.writes_depth {
        interface.color_targets = 1;
    }
    if !g.warnings.is_empty() {
        let mut w: Vec<_> = g.warnings.clone();
        w.sort();
        w.dedup();
        out.push_str(&format!("// warnings: {}\n", w.join("; ")));
    }
    let grid_write_footprint=if stage==Stage::Compute && g.warnings.is_empty() {write_footprints::analyze(&prog,&g.instruction_writes,&g.body)} else {None};
    let descriptor_write_footprint=if stage==Stage::Compute && g.warnings.is_empty() {write_footprints::descriptor_stores(&prog,&g.instruction_writes)} else {None};
    let first_high=match stage {Stage::Vertex=>Some(0x45),Stage::Fragment=>Some(0x145),Stage::Compute=>Some(0xc5),_=>None};
    let write_window_footprint=if g.warnings.is_empty() {first_high.and_then(|first|write_footprints::descriptor_high_windows(&prog,&g.instruction_writes,first))} else {None};
    Ok(Translated { stage, msl: out, entry: entry.into(), interface,grid_write_footprint,descriptor_write_footprint,write_window_footprint })
}
