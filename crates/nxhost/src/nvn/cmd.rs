// Mush8 - cmd.rs
// Constantin Clerc (c22dev) - 0.1
use super::draw;
use super::fx::FastMap;
use super::objects::STATE;
use crate::stubs::Args;
use std::collections::{BTreeMap, VecDeque};
use std::sync::{LazyLock, Mutex};

pub type TexBuilder = [u8; 0x68];

#[derive(Debug, Clone, Copy)]
pub struct Draw {
    pub mode: u32,
    pub index: Option<(u32, u64)>,
    pub first: u32,
    pub count: u32,
    pub base_vertex: i32,
    pub base_instance: u32,
    pub instances: u32,
}

impl Draw {
    pub(super) fn validate_indices(&self) -> Result<(), &'static str> {
        let Some((ty, addr)) = self.index else { return Ok(()) };
        if self.count == 0 || self.instances == 0 {
            return Ok(());
        }
        if addr == 0 {
            return Err("null index buffer");
        }
        let size = match ty { 0 => 1, 1 => 2, _ => 4 };
        addr.checked_add(size * self.count as u64).ok_or("index buffer range overflows")?;
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub enum Cmd {
    SetRenderTargets { colors: Vec<Option<Box<TexBuilder>>>, depth: Option<Box<TexBuilder>> },
    ClearColor { index: u32, rgba: [f32; 4] },
    ClearDepthStencil { depth: f32, stencil: u32 },
    Call(Vec<u64>),
    Program {stages:Vec<(u64,u64)>,mask:u32},
    VertexAttribs(Vec<u32>),
    VertexStreams(Vec<(u32, u32)>),
    VertexBuffer { index: u32, addr: u64, size: u64 },
    Uniform { stage: u32, index: u32, addr: u64, size: u64 },
    Texture { stage: u32, index: u32, handle: u64 },
    Image { stage:u32,index:u32,handle:u64,origin:Option<Vec<String>> },
    SeparateTexture {stage:u32,index:u32,handle:u64},
    SeparateSampler {stage:u32,index:u32,handle:u64},
    Storage { stage: u32, index: u32, addr: u64, size: u64 },
    UpdateUniform { addr: u64, size: u64, offset: u64, data: Vec<u8> },
    ColorState(u32),
    ChannelMask(u32),
    Blend(u64),
    DepthStencil(u64),
    Polygon(u32),
    Viewport([f32; 4]),
    Scissor([i32; 4]),
    BlendColor([f32; 4]),
    AlphaRef(f32),
    Multisample(u32),
    SampleMask(u32),
    DebugAnnotation {action:u32,domain:u32,label:String},
    ScratchMemory {pool:u64,offset:u64,size:u64},
    StencilRef(u32, u32),
    StencilValueMask(u32, u32),
    StencilMask(u32, u32),
    TexturePool(u64),
    SamplerPool(u64),
    DepthClamp(bool),
    CopyBufferToTexture { src: u64, dst: Box<TexBuilder>, region: [i32; 6], strides: (u32, u32), flags: u32 },
    CopyTextureToTexture { src: Box<TexBuilder>, src_region: [i32; 6], dst: Box<TexBuilder>, dst_region: [i32; 6], flags: u32 },
    CopyTextureToBuffer { src: Box<TexBuilder>, region: [i32; 6], dst: u64, strides: (u32, u32), flags: u32 },
    ClearTexture { dst: Box<TexBuilder>, region: [i32; 6], color: [f32; 4], mask: u32 },
    RenderEnable(bool),
    Dispatch([u32; 3]),
    DispatchIndirect(u64),
    RasterizerDiscard(bool),
    ReportCounter { counter: u32, addr: u64 },
    CopyBuffer { src: u64, dst: u64, size: u64 },
    ClearBuffer { dst: u64, size: u64, value: u32 },
    Barrier(u32),
    FenceSync(u64),
    WaitSync(u64),
    SignalEvent { address:u64,mode:u32,value:u32 },
    WaitEvent { address:u64,mode:u32,value:u32 },
    DepthRange(f32, f32),
    PolygonOffset([f32; 3]),
    Draw(Draw),
    DrawIndirect {mode:u32,index:Option<(u32,u64)>,arguments:u64},
}

#[derive(Default)]
struct Recorder {
    open: FastMap<u64, Vec<Cmd>>,
    done: FastMap<u64, std::sync::Arc<Vec<Cmd>>>,
    regions: BTreeMap<u64, (u64, u64)>,
    untracked: VecDeque<u64>,
    next: u64,
    memory: FastMap<u64, CommandMemory>,
    open_regions: FastMap<u64, Vec<(u64, u64)>>,
    copy_strides: FastMap<u64, (u32, u32)>,
}

#[derive(Clone, Copy, Default)]
struct MemoryRegion { base: u64, size: u64, used: u64 }

#[derive(Clone)]
struct CommandMemory {
    regions: [MemoryRegion; 2],
    callback: u64,
    data: u64,
    enabled: [bool; 2],
}

impl Default for CommandMemory {
    fn default() -> Self { Self { regions: [MemoryRegion::default(); 2], callback: 0, data: 0, enabled: [true; 2] } }
}

static REC: LazyLock<Mutex<Recorder>> = LazyLock::new(|| Mutex::new(Recorder { next: 1, ..Default::default() }));
const MAX_UNTRACKED: usize = 8192;
const CMD_BYTES: u64 = 16;

pub fn begin(cb: u64) {
    let mut r = REC.lock().unwrap();
    r.open.insert(cb, Vec::new());
    r.open_regions.insert(cb, Vec::new());
}

pub fn initialize(cb: u64) {
    let mut r = REC.lock().unwrap();
    r.memory.insert(cb, CommandMemory::default());
    r.open.remove(&cb);
    r.open_regions.remove(&cb);
    r.copy_strides.insert(cb, (0, 0));
}

pub fn finalize(cb: u64) {
    let mut r = REC.lock().unwrap();
    r.memory.remove(&cb); r.open.remove(&cb); r.open_regions.remove(&cb);
    r.copy_strides.remove(&cb);
}

fn add_region(cb: u64, event: usize, base: u64, size: u64) {
    let mut r = REC.lock().unwrap();
    r.memory.entry(cb).or_default().regions[event] = MemoryRegion { base, size, used: 0 };
}

pub fn add_control_memory(cb: u64, addr: u64, size: u64) {
    add_region(cb, 1, addr, size);
}

fn memory_query(cb: u64, event: usize, field: usize) -> u64 {
    let r = REC.lock().unwrap();
    let m = r.memory.get(&cb).map(|m| m.regions[event]).unwrap_or_default();
    match field { 0 => m.size, 1 => m.used, _ => m.size.saturating_sub(m.used) }
}

/// Reserve our native command representation within the game's supplied
/// budgets. Callbacks run outside REC: the guest callback adds the next region.
fn reserve(cb: u64, event: usize, bytes: u64) -> Option<(u64, u64)> {
    for _ in 0..8 {
        let (callback, data) = {
            let mut r = REC.lock().unwrap();
            let Some(m) = r.memory.get_mut(&cb) else { return None; };
            let region = &mut m.regions[event];
            if bytes <= region.size.saturating_sub(region.used) {
                let base = region.base + region.used;
                region.used += bytes;
                return Some((base, base + bytes));
            }
            if m.callback == 0 || !m.enabled[event] {
                drop(r);
                crate::diag::report_and_exit(&format!("NVN command buffer {cb:#x} exhausted memory kind {event}, needs {bytes:#x} bytes without enabled callback"));
            }
            (m.callback, m.data)
        };
        let f: extern "C" fn(u64, u32, u64, u64) = unsafe { std::mem::transmute(callback as usize) };
        f(cb, event as u32, bytes.max(if event == 0 { 0x1000 } else { 0x100 }), data);
    }
    crate::diag::report_and_exit("NVN command memory callback did not supply sufficient storage")
}

pub fn end(cb: u64) -> u64 {
    let mut guard = REC.lock().unwrap();
    let r = &mut *guard;
    let cmds = r.open.remove(&cb).unwrap_or_default();
    let mut dropped = Vec::new();
    let h = r.next;
    r.next += 1;
    r.done.insert(h, std::sync::Arc::new(cmds));
    let regions = r.open_regions.remove(&cb).unwrap_or_default();
    if !regions.is_empty() {
        for (start, end) in regions {
            let dead: Vec<u64> = r.regions.range(..end).rev().take_while(|(_, (e, _))| *e > start).map(|(s, _)| *s).collect();
            for s in dead {
                if let Some((_, old)) = r.regions.remove(&s) {
                    if old != h { dropped.extend(r.done.remove(&old)); }
                }
            }
            r.regions.insert(start, (end, h));
        }
    } else {
        r.untracked.push_back(h);
        if r.untracked.len() > MAX_UNTRACKED {
            let old = r.untracked.pop_front().unwrap();
            dropped.extend(r.done.remove(&old));
        }
    }
    drop(guard);
    reap(dropped);
    h
}

fn reap(old: Vec<std::sync::Arc<Vec<Cmd>>>) {
    static GARBAGE: Mutex<Vec<std::sync::Arc<Vec<Cmd>>>> = Mutex::new(Vec::new());
    static REAPER: std::sync::Once = std::sync::Once::new();
    if old.is_empty() {
        return;
    }
    REAPER.call_once(|| {
        std::thread::Builder::new()
            .name("nvn-reaper".into())
            .spawn(|| loop {
                std::thread::sleep(std::time::Duration::from_millis(100));
                let batch = std::mem::take(&mut *GARBAGE.lock().unwrap());
                drop(batch);
            })
            .unwrap();
    });
    GARBAGE.lock().unwrap().extend(old);
}

pub fn recording(handle: u64) -> Option<std::sync::Arc<Vec<Cmd>>> {
    let r=REC.lock().unwrap();
    let recording=r.done.get(&handle).cloned();
    static TRACE:LazyLock<bool>=LazyLock::new(||std::env::var("NX_FC27_VIDEO_TRACE").is_ok_and(|v|v=="1") && std::env::var_os("NX_TEST_CONTROL").is_some());
    if *TRACE && recording.is_none() {
        use std::sync::atomic::{AtomicU64,Ordering::Relaxed};
        static MISSING:AtomicU64=AtomicU64::new(0);
        let n=MISSING.fetch_add(1,Relaxed)+1;
        if n<=64 || n.is_power_of_two() {eprintln!("[cmd-lifetime] missing={n} handle={handle:#x} next={:#x} retained={}",r.next,r.done.len());}
    }
    recording
}

fn push(cb: u64, c: Cmd) {
    if !REC.lock().unwrap().open.contains_key(&cb) { crate::diag::report_and_exit("NVN command emitted outside Begin/EndRecording"); }
    let payload = match &c {
        Cmd::UpdateUniform { data, .. } => data.len(),
        Cmd::Call(v) => v.len() * 8,
        Cmd::Program {stages,..} => stages.len() * 16,
        Cmd::VertexAttribs(v) => v.len() * 4,
        Cmd::VertexStreams(v) => v.len() * 8,
        _ => 0,
    };
    reserve(cb, 0, (CMD_BYTES + payload as u64).next_multiple_of(4));
    let region = reserve(cb, 1, CMD_BYTES);
    let mut r = REC.lock().unwrap();
    if let Some(region) = region {
        let ranges = r.open_regions.entry(cb).or_default();
        if let Some(last) = ranges.last_mut().filter(|last| last.1 == region.0) { last.1 = region.1; }
        else { ranges.push(region); }
    }
    if let Some(v) = r.open.get_mut(&cb) {
        v.push(c);
    }
}

fn rd<T: Copy>(addr: u64) -> T {
    unsafe { (addr as *const T).read_unaligned() }
}

fn rd_vec<T: Copy>(addr: u64, n: u64) -> Vec<T> {
    (0..n).map(|i| rd::<T>(addr + i * std::mem::size_of::<T>() as u64)).collect()
}

pub fn set_render_targets(a: &Args) -> u64 {
    let n = a.0[1] as usize;
    let objs: Vec<u64> = (0..n).map(|i| if a.0[2] == 0 { 0 } else { rd::<u64>(a.0[2] + 8 * i as u64) }).collect();
    let builder = |t: u64, view: u64| {
        super::objects::texture(t).map(|t| {
            let mut b = Box::new(t.builder);
            super::objects::apply_view(&mut b, view, true);
            b
        })
    };
    let views: Vec<u64> = (0..n).map(|i| if a.0[3] == 0 { 0 } else { rd::<u64>(a.0[3] + 8 * i as u64) }).collect();
    let colors: Vec<Option<Box<TexBuilder>>> = objs.iter().zip(&views).map(|(&t, &v)| builder(t, v)).collect();
    let depth = builder(a.0[4], a.0[5]);
    push(a.0[0], Cmd::SetRenderTargets { colors, depth });
    0
}

pub fn clear_color(a: &Args) -> u64 {
    let p = a.0[2];
    let rgba = [rd::<f32>(p), rd::<f32>(p + 4), rd::<f32>(p + 8), rd::<f32>(p + 12)];
    push(a.0[0], Cmd::ClearColor { index: a.0[1] as u32, rgba });
    0
}

pub fn clear_depth_stencil(a: &Args) -> u64 {
    push(a.0[0], Cmd::ClearDepthStencil { depth: a.f32(0), stencil: a.0[2] as u32 });
    0
}

pub fn bind_uniform_buffer(a: &Args) -> u64 {
    push(a.0[0], Cmd::Uniform { stage: a.0[1] as u32, index: a.0[2] as u32, addr: a.0[3], size: a.0[4] });
    0
}

pub fn bind_texture(a: &Args) -> u64 {
    push(a.0[0], Cmd::Texture { stage: a.0[1] as u32, index: a.0[2] as u32, handle: a.0[3] });
    0
}

pub fn call_commands(a: &Args) -> u64 {
    let handles = rd_vec::<u64>(a.0[2], a.0[1]);
    push(a.0[0], Cmd::Call(handles));
    0
}

pub fn bind_program(a: &Args) -> u64 {
    let mask=a.0[2] as u32;
    if mask&!0x3f!=0 {crate::diag::report_and_exit("NVN program binding contains invalid shader-stage bits");}
    let stages=if a.0[1]==0 {Vec::new()} else {
        let registered={STATE.lock().unwrap().programs.get(&a.0[1]).cloned()};
        registered.unwrap_or_else(||crate::diag::report_and_exit("NVN binds an uninitialized program"))
    };
    push(a.0[0], Cmd::Program {stages,mask});
    0
}

pub fn bind_vertex_attribs(a: &Args) -> u64 {
    push(a.0[0], Cmd::VertexAttribs(rd_vec::<u32>(a.0[2], a.0[1].min(32))));
    0
}

pub fn bind_vertex_streams(a: &Args) -> u64 {
    push(a.0[0], Cmd::VertexStreams(rd_vec::<(u32, u32)>(a.0[2], a.0[1].min(32))));
    0
}

pub fn bind_vertex_buffer(a: &Args) -> u64 {
    push(a.0[0], Cmd::VertexBuffer { index: a.0[1] as u32, addr: a.0[2], size: a.0[3] });
    0
}

fn bind_ranges(a:&Args,kind:u32) -> u64 {
    let (stage,first,count,ptr,limit) = if kind==0 {(0,a.0[1],a.0[2],a.0[3],16)} else {(a.0[1] as u32,a.0[2],a.0[3],a.0[4],if kind==1 {15} else {16})};
    if first.checked_add(count).is_none_or(|end|end>limit) || (kind!=0 && ![0,1,5].contains(&stage)) {crate::diag::report_and_exit("NVN bulk buffer bindings exceed supported stage or binding range");}
    for (i,(addr,size)) in rd_vec::<(u64,u64)>(ptr,count).into_iter().enumerate() {
        let index=first as u32+i as u32;
        push(a.0[0],match kind {0=>Cmd::VertexBuffer{index,addr,size},1=>Cmd::Uniform{stage,index,addr,size},_=>Cmd::Storage{stage,index,addr,size}});
    }
    0
}

fn bind_textures(a:&Args) -> u64 {
    if a.0[2].checked_add(a.0[3]).is_none_or(|end|end>32) || ![0,1,5].contains(&(a.0[1] as u32)) {crate::diag::report_and_exit("NVN bulk texture bindings exceed supported stage or binding range");}
    for (i,handle) in rd_vec::<u64>(a.0[4],a.0[3]).into_iter().enumerate() {
        push(a.0[0],Cmd::Texture{stage:a.0[1] as u32,index:a.0[2] as u32+i as u32,handle});
    }
    0
}

fn bind_images(a:&Args) -> u64 {
    if a.0[2].checked_add(a.0[3]).is_none_or(|end|end>8) || ![0,1,5].contains(&(a.0[1] as u32)) {crate::diag::report_and_exit("NVN bulk image bindings exceed supported stage or binding range");}
    for (i,handle) in rd_vec::<u64>(a.0[4],a.0[3]).into_iter().enumerate() {
        let origin=if handle==0 && a.0[1]==5 && a.0[3]==1 && std::env::var_os("NX_TRACE_BINDINGS").is_some() {
            let mut lines=vec![format!("null image args{:x?} pointed{:x?}",&a.0[..5],crate::diag::read_words::<8>(a.0[4]))];
            if let Some(frame)=crate::diag::read_words::<2>((a as *const _ as u64)-16) {
                lines.push(format!("caller {} fp{:x}",crate::diag::symbolize(frame[1]),frame[0]));lines.extend(crate::diag::backtrace_lines(frame[0]).into_iter().take(12));
            }
            // Read-only layout probe for the original renderer's binding cache.
            // This is diagnostic data, not an address-dependent runtime patch.
            if let Some(root)=a.0[4].checked_sub(0x2ed0) {
                if let Some(tables)=crate::diag::read_words::<48>(root+0x990) {
                    lines.push(format!("renderer resource tables {:x?}",tables));
                    for (index,table) in tables.chunks_exact(3).enumerate() {
                        if table[0]!=0 {lines.push(format!("renderer group{index} resources{:x?} samplers{:x?} count{}",crate::diag::read_words::<16>(table[0]),crate::diag::read_words::<8>(table[1]),table[2]));}
                    }
                }
                if let Some([shader])=crate::diag::read_words::<1>(root+0x2f30) {
                    lines.push(format!("renderer reflection header{:x?} groups{:x?}",crate::diag::read_words::<4>(shader+0x128),crate::diag::read_words::<8>(shader+0x1a0)));
                    if let Some([entries])=crate::diag::read_words::<1>(shader+0x130) {lines.push(format!("renderer reflection entries{:x?}",crate::diag::read_words::<32>(entries)));}
                }
            }
            Some(lines)
        } else {None};
        push(a.0[0],Cmd::Image{stage:a.0[1] as u32,index:a.0[2] as u32+i as u32,handle,origin});
    }
    0
}

fn bind_separate(a:&Args,sampler:bool) -> u64 {
    if a.0[2].checked_add(a.0[3]).is_none_or(|end|end>128) || ![0,1,5].contains(&(a.0[1] as u32)) {crate::diag::report_and_exit("NVN separate bindings exceed supported stage or range");}
    for (i,handle) in rd_vec::<u64>(a.0[4],a.0[3]).into_iter().enumerate() {
        let (stage,index)=(a.0[1] as u32,a.0[2] as u32+i as u32);
        push(a.0[0],if sampler {Cmd::SeparateSampler{stage,index,handle}} else {Cmd::SeparateTexture{stage,index,handle}});
    }
    0
}

fn debug_label(address:u64) -> String {
    if address==0 {return String::new();}
    let (base,size,block)=crate::kernel::aspace().map.lock().unwrap().query(address);
    if block.perm&crate::mem::perm::R==0 {crate::diag::report_and_exit("NVN debug annotation is not readable");}
    let len=(base+size-address).min(4096) as usize;
    let bytes=unsafe {std::slice::from_raw_parts(address as *const u8,len)};
    String::from_utf8_lossy(&bytes[..bytes.iter().position(|b|*b==0).unwrap_or(len)]).into_owned()
}

fn annotate(a:&Args,action:u32,has_domain:bool) -> u64 {
    let domain=if has_domain {a.0[1] as u32} else {0};
    let label=if action==1 {String::new()} else {debug_label(a.0[if has_domain {2} else {1}])};
    push(a.0[0],Cmd::DebugAnnotation{action,domain,label});0
}

pub fn update_uniform_buffer(a: &Args) -> u64 {
    let len = a.0[4].min(0x10000) as usize;
    let data = unsafe { std::slice::from_raw_parts(a.0[5] as *const u8, len) }.to_vec();
    push(a.0[0], Cmd::UpdateUniform { addr: a.0[1], size: a.0[2], offset: a.0[3], data });
    0
}

pub fn host_impl(name: &str) -> Option<fn(&Args) -> u64> {
    Some(match name {
        "nvnCommandBufferPushDebugGroup"=>|a|annotate(a,0,false),
        "nvnCommandBufferPushDebugGroupStatic"|"nvnCommandBufferPushDebugGroupDynamic"=>|a|annotate(a,0,true),
        "nvnCommandBufferPopDebugGroup"=>|a|annotate(a,1,false),
        "nvnCommandBufferPopDebugGroupId"=>|a|annotate(a,1,true),
        "nvnCommandBufferInsertDebugMarker"=>|a|annotate(a,2,false),
        "nvnCommandBufferInsertDebugMarkerStatic"|"nvnCommandBufferInsertDebugMarkerDynamic"=>|a|annotate(a,2,true),
        "nvnCommandBufferSetMemoryCallback" => |a| { REC.lock().unwrap().memory.entry(a.0[0]).or_default().callback = a.0[1]; 0 },
        "nvnCommandBufferSetMemoryCallbackData" => |a| { REC.lock().unwrap().memory.entry(a.0[0]).or_default().data = a.0[1]; 0 },
        "nvnCommandBufferGetMemoryCallback" => |a| REC.lock().unwrap().memory.get(&a.0[0]).map_or(0, |m| m.callback),
        "nvnCommandBufferGetMemoryCallbackData" => |a| REC.lock().unwrap().memory.get(&a.0[0]).map_or(0, |m| m.data),
        "nvnCommandBufferSetCommandMemoryCallbackEnabled" => |a| { REC.lock().unwrap().memory.entry(a.0[0]).or_default().enabled[0] = a.0[1] as u8 != 0; 0 },
        "nvnCommandBufferSetControlMemoryCallbackEnabled" => |a| { REC.lock().unwrap().memory.entry(a.0[0]).or_default().enabled[1] = a.0[1] as u8 != 0; 0 },
        "nvnCommandBufferAddCommandMemory" => |a| {
            let s = STATE.lock().unwrap();
            let p = s.pools.get(&a.0[1]);
            let base = s.pool_offset_cpu(a.0[1], a.0[2]);
            if a.0[3] == 0 || p.is_none_or(|p| a.0[2].checked_add(a.0[3]).is_none_or(|end| end > p.size)) || base.is_none() {
                drop(s);
                crate::diag::report_and_exit("NVN AddCommandMemory range is outside its physical pool");
            }
            drop(s);
            add_region(a.0[0], 0, base.unwrap(), a.0[3]);
            0
        },
        "nvnCommandBufferGetCommandMemorySize" => |a| memory_query(a.0[0], 0, 0),
        "nvnCommandBufferGetCommandMemoryUsed" => |a| memory_query(a.0[0], 0, 1),
        "nvnCommandBufferGetCommandMemoryFree" => |a| memory_query(a.0[0], 0, 2),
        "nvnCommandBufferGetControlMemorySize" => |a| memory_query(a.0[0], 1, 0),
        "nvnCommandBufferGetControlMemoryUsed" => |a| memory_query(a.0[0], 1, 1),
        "nvnCommandBufferGetControlMemoryFree" => |a| memory_query(a.0[0], 1, 2),
        "nvnCommandBufferIsRecording" => |a| REC.lock().unwrap().open.contains_key(&a.0[0]) as u64,
        "nvnCommandBufferSetCopyRowStride" => |a| { REC.lock().unwrap().copy_strides.entry(a.0[0]).or_default().0 = a.0[1] as u32; 0 },
        "nvnCommandBufferSetCopyImageStride" => |a| { REC.lock().unwrap().copy_strides.entry(a.0[0]).or_default().1 = a.0[1] as u32; 0 },
        "nvnCommandBufferGetCopyRowStride" => |a| REC.lock().unwrap().copy_strides.get(&a.0[0]).map_or(0, |s| s.0 as u64),
        "nvnCommandBufferGetCopyImageStride" => |a| REC.lock().unwrap().copy_strides.get(&a.0[0]).map_or(0, |s| s.1 as u64),
        "nvnCommandBufferCopyBufferToBuffer" => |a| {
            if a.0[3]!=0 && (a.0[1]==0 || a.0[2]==0) && std::env::var_os("NX_TRACE_NULL_GPU").is_some() {
                static COUNT:std::sync::atomic::AtomicU32=std::sync::atomic::AtomicU32::new(0);
                if COUNT.fetch_add(1,std::sync::atomic::Ordering::Relaxed)<4 {crate::diag::trace_stub_context("null CopyBufferToBuffer",a);}
            }
            push(a.0[0], Cmd::CopyBuffer { src: a.0[1], dst: a.0[2], size: a.0[3] }); 0
        },
        "nvnCommandBufferClearBuffer" => |a| { push(a.0[0], Cmd::ClearBuffer { dst: a.0[1], size: a.0[2], value: a.0[3] as u32 }); 0 },
        "nvnCommandBufferBarrier" => |a| { push(a.0[0], Cmd::Barrier(a.0[1] as u32)); 0 },
        "nvnCommandBufferFenceSync" => |a| { push(a.0[0], Cmd::FenceSync(a.0[1])); 0 },
        "nvnCommandBufferWaitSync" => |a| { push(a.0[0], Cmd::WaitSync(a.0[1])); 0 },
        "nvnCommandBufferSignalEvent" => |a| {
            if a.0[2]>1 || a.0[3]>2 || a.0[4]&!1!=0 {crate::diag::report_and_exit("unsupported NVN event signal mode, location or flags");}
            push(a.0[0],Cmd::SignalEvent{address:super::events::storage(a.0[1]),mode:a.0[2] as u32,value:a.0[5] as u32});0
        },
        "nvnCommandBufferWaitEvent" => |a| {
            if a.0[2]>1 {crate::diag::report_and_exit("unsupported NVN event wait mode");}
            push(a.0[0],Cmd::WaitEvent{address:super::events::storage(a.0[1]),mode:a.0[2] as u32,value:a.0[3] as u32});0
        },
        "nvnCommandBufferBindProgram" => bind_program,
        "nvnCommandBufferBindVertexAttribState" => bind_vertex_attribs,
        "nvnCommandBufferBindVertexStreamState" => bind_vertex_streams,
        "nvnCommandBufferBindVertexBuffer" => bind_vertex_buffer,
        "nvnCommandBufferBindVertexBuffers" => |a|bind_ranges(a,0),
        "nvnCommandBufferBindUniformBuffers" => |a|bind_ranges(a,1),
        "nvnCommandBufferBindStorageBuffers" => |a|bind_ranges(a,2),
        "nvnCommandBufferBindTextures" => bind_textures,
        "nvnCommandBufferBindImages" => bind_images,
        "nvnCommandBufferBindImage" => |a| {push(a.0[0],Cmd::Image{stage:a.0[1] as u32,index:a.0[2] as u32,handle:a.0[3],origin:None});0},
        "nvnCommandBufferBindSeparateTextures" => |a|bind_separate(a,false),
        "nvnCommandBufferBindSeparateSamplers" => |a|bind_separate(a,true),
        "nvnCommandBufferBindSeparateTexture" => |a| {push(a.0[0],Cmd::SeparateTexture{stage:a.0[1] as u32,index:a.0[2] as u32,handle:a.0[3]});0},
        "nvnCommandBufferBindSeparateSampler" => |a| {push(a.0[0],Cmd::SeparateSampler{stage:a.0[1] as u32,index:a.0[2] as u32,handle:a.0[3]});0},
        "nvnCommandBufferBindStorageBuffer" => |a| {
            push(a.0[0], Cmd::Storage { stage: a.0[1] as u32, index: a.0[2] as u32, addr: a.0[3], size: a.0[4] });
            0
        },
        "nvnCommandBufferUpdateUniformBuffer" => update_uniform_buffer,
        "nvnCommandBufferBindColorState" => |a| state_cmd(a, |p| Cmd::ColorState(rd(p))),
        "nvnCommandBufferBindChannelMaskState" => |a| state_cmd(a, |p| Cmd::ChannelMask(rd(p))),
        "nvnCommandBufferBindBlendState" => |a| state_cmd(a, |p| Cmd::Blend(rd(p))),
        "nvnCommandBufferBindDepthStencilState" => |a| state_cmd(a, |p| Cmd::DepthStencil(rd(p))),
        "nvnCommandBufferBindPolygonState" => |a| state_cmd(a, |p| Cmd::Polygon(rd(p))),
        "nvnCommandBufferBindMultisampleState" => |a| state_cmd(a,|p|Cmd::Multisample(rd(p))),
        "nvnCommandBufferSetSampleMask" => |a| {push(a.0[0],Cmd::SampleMask(a.0[1] as u32));0},
        "nvnCommandBufferSetShaderScratchMemory"=>|a| {
            let (pool,offset,size)=(a.0[1],a.0[2],a.0[3]);
            if pool!=0 || size!=0 {
                let valid=STATE.lock().unwrap().pools.get(&pool).is_some_and(|p|offset.checked_add(size).is_some_and(|end|end<=p.size));
                if !valid {crate::diag::report_and_exit("NVN shader scratch allocation is outside its pool");}
            }
            push(a.0[0],Cmd::ScratchMemory{pool,offset,size});0
        },
        "nvnCommandBufferSetBlendColor" => |a| state_cmd(a, |p| Cmd::BlendColor(rd(p))),
        "nvnCommandBufferSetAlphaRef" => |a| {push(a.0[0],Cmd::AlphaRef(a.f32(0)));0},
        "nvnCommandBufferSetTexturePool" => |a| state_cmd(a, Cmd::TexturePool),
        "nvnCommandBufferSetSamplerPool" => |a| state_cmd(a, Cmd::SamplerPool),
        "nvnCommandBufferSetViewport" => |a| {
            let v = [a.0[1] as i32 as f32, a.0[2] as i32 as f32, a.0[3] as i32 as f32, a.0[4] as i32 as f32];
            push(a.0[0], Cmd::Viewport(v));
            0
        },
        "nvnCommandBufferSetViewports" => |a| {
            if !single_viewport_range(a) {return 0;}
            push(a.0[0],Cmd::Viewport(rd(a.0[3])));0
        },
        "nvnCommandBufferSetScissor" => |a| {
            push(a.0[0], Cmd::Scissor([a.0[1] as i32, a.0[2] as i32, a.0[3] as i32, a.0[4] as i32]));
            0
        },
        "nvnCommandBufferSetScissors" => |a| {
            if !single_viewport_range(a) {return 0;}
            push(a.0[0],Cmd::Scissor(rd(a.0[3])));0
        },
        "nvnCommandBufferSetStencilRef" => |a| {
            push(a.0[0], Cmd::StencilRef(a.0[1] as u32, a.0[2] as u32));
            0
        },
        "nvnCommandBufferSetStencilValueMask" => |a| {
            push(a.0[0], Cmd::StencilValueMask(a.0[1] as u32, a.0[2] as u32));
            0
        },
        "nvnCommandBufferSetStencilMask" => |a| {
            push(a.0[0], Cmd::StencilMask(a.0[1] as u32, a.0[2] as u32));
            0
        },
        "nvnCommandBufferCopyBufferToTexture" => |a| {
            if let Some(b) = view_builder(a.0[2], a.0[3]) {
                let strides = REC.lock().unwrap().copy_strides.get(&a.0[0]).copied().unwrap_or_default();
                push(a.0[0], Cmd::CopyBufferToTexture { src: a.0[1], dst: b, region: rd(a.0[4]), strides, flags: a.0[5] as u32 });
            }
            0
        },
        "nvnCommandBufferCopyTextureToTexture" => |a| {
            if let (Some(src), Some(dst)) = (view_builder(a.0[1], a.0[2]), view_builder(a.0[4], a.0[5])) {
                push(a.0[0], Cmd::CopyTextureToTexture { src, src_region: rd(a.0[3]), dst, dst_region: rd(a.0[6]), flags: a.0[7] as u32 });
            }
            0
        },
        "nvnCommandBufferCopyTextureToBuffer" => |a| {
            if let Some(src) = view_builder(a.0[1], a.0[2]) {
                let strides = REC.lock().unwrap().copy_strides.get(&a.0[0]).copied().unwrap_or_default();
                push(a.0[0], Cmd::CopyTextureToBuffer { src, region: rd(a.0[3]), dst: a.0[4], strides, flags: a.0[5] as u32 });
            }
            0
        },
        "nvnCommandBufferClearTexture" => |a| {
            if let Some(dst) = view_builder(a.0[1], a.0[2]) {
                push(a.0[0], Cmd::ClearTexture { dst, region: rd(a.0[3]), color: rd(a.0[4]), mask: a.0[5] as u32 });
            }
            0
        },
        "nvnCommandBufferDispatchCompute" => |a| {
            push(a.0[0], Cmd::Dispatch([a.0[1] as u32, a.0[2] as u32, a.0[3] as u32]));
            0
        },
        "nvnCommandBufferDispatchComputeIndirect" => |a| {
            push(a.0[0],Cmd::DispatchIndirect(a.0[1]));0
        },
        "nvnCommandBufferSetRenderEnable" => |a| {
            push(a.0[0], Cmd::RenderEnable(a.0[1] & 0xFF != 0));
            0
        },
        "nvnCommandBufferSetRenderEnableConditional" => |a| {
            push(a.0[0], Cmd::RenderEnable(true));
            0
        },
        "nvnCommandBufferReportCounter" => |a| {
            push(a.0[0], Cmd::ReportCounter { counter: a.0[1] as u32, addr: a.0[2] });
            0
        },
        "nvnCommandBufferSetRasterizerDiscard" => |a| {
            push(a.0[0], Cmd::RasterizerDiscard(a.0[1] & 0xFF != 0));
            0
        },
        "nvnCommandBufferSetDepthRange" => |a| {
            push(a.0[0], Cmd::DepthRange(a.f32(0), a.f32(1)));
            0
        },
        "nvnCommandBufferSetDepthRanges" => |a| {
            if !single_viewport_range(a) {return 0;}
            let values:[f32;2]=rd(a.0[3]);push(a.0[0],Cmd::DepthRange(values[0],values[1]));0
        },
        "nvnCommandBufferSetPolygonOffsetClamp" => |a| {
            push(a.0[0], Cmd::PolygonOffset([a.f32(0), a.f32(1), a.f32(2)]));
            0
        },
        "nvnCommandBufferSetDepthClamp" => |a| {
            push(a.0[0], Cmd::DepthClamp(a.0[1] & 0xFF != 0));
            0
        },
        "nvnCommandBufferSetDepthBounds" => |a| {
            // Apple7 has no depth-bounds test. Its disabled state is exact;
            // enabling it must not silently change the rendered result.
            if a.0[1]&0xff != 0 {crate::diag::report_and_exit(&format!("NVN depth bounds enabled ({}, {}) on an unsupported Metal GPU",a.f32(0),a.f32(1)));}
            0
        },
        "nvnCommandBufferSetConservativeRasterEnable" => |a| {
            if a.0[1]&0xff!=0 {crate::diag::report_and_exit("conservative rasterization is unsupported on this Metal backend");}0
        },
        "nvnCommandBufferSetConservativeRasterDilate" => |a| {
            if a.f32(0)!=0.0 {crate::diag::report_and_exit("nonzero conservative raster dilation is unsupported");}0
        },
        "nvnCommandBufferSetSubpixelPrecisionBias" => |a| {
            if a.0[1]!=0 || a.0[2]!=0 {crate::diag::report_and_exit("nonzero subpixel precision bias is unsupported");}0
        },
        "nvnCommandBufferDrawArrays" => |a| draw(a, Draw { mode: a.0[1] as u32, index: None, first: a.0[2] as u32, count: a.0[3] as u32, base_vertex: 0, base_instance: 0, instances: 1 }),
        "nvnCommandBufferDrawArraysIndirect" => |a| {push(a.0[0],Cmd::DrawIndirect{mode:a.0[1] as u32,index:None,arguments:a.0[2]});0},
        "nvnCommandBufferDrawElementsIndirect" => |a| {push(a.0[0],Cmd::DrawIndirect{mode:a.0[1] as u32,index:Some((a.0[2] as u32,a.0[3])),arguments:a.0[4]});0},
        "nvnCommandBufferDrawArraysInstanced" => |a| {
            draw(a, Draw { mode: a.0[1] as u32, index: None, first: a.0[2] as u32, count: a.0[3] as u32, base_vertex: 0, base_instance: a.0[4] as u32, instances: a.0[5] as u32 })
        },
        "nvnCommandBufferDrawElements" => |a| {
            draw(a, Draw { mode: a.0[1] as u32, index: Some((a.0[2] as u32, a.0[4])), first: 0, count: a.0[3] as u32, base_vertex: 0, base_instance: 0, instances: 1 })
        },
        "nvnCommandBufferDrawElementsBaseVertex" => |a| {
            draw(a, Draw { mode: a.0[1] as u32, index: Some((a.0[2] as u32, a.0[4])), first: 0, count: a.0[3] as u32, base_vertex: a.0[5] as i32, base_instance: 0, instances: 1 })
        },
        "nvnCommandBufferDrawElementsInstanced" => |a| {
            draw(
                a,
                Draw {
                    mode: a.0[1] as u32,
                    index: Some((a.0[2] as u32, a.0[4])),
                    first: 0,
                    count: a.0[3] as u32,
                    base_vertex: a.0[5] as i32,
                    base_instance: a.0[6] as u32,
                    instances: a.0[7] as u32,
                },
            )
        },
        _ => return None,
    })
}

fn single_viewport_range(a:&Args)->bool {
    if a.0[2]==0 {return false;}
    if a.0[1]!=0 || a.0[2]!=1 || a.0[3]==0 {crate::diag::report_and_exit("NVN viewport-array state requires unsupported multiple viewport indices");}
    true
}

fn view_builder(texture: u64, view: u64) -> Option<Box<TexBuilder>> {
    let mut b = Box::new(super::objects::texture(texture)?.builder);
    super::objects::apply_view(&mut b, view, true);
    Some(b)
}

fn state_cmd(a: &Args, f: impl Fn(u64) -> Cmd) -> u64 {
    if a.0[1] != 0 {
        push(a.0[0], f(a.0[1]));
    }
    0
}

fn draw(a: &Args, d: Draw) -> u64 {
    if let Err(reason) = d.validate_indices() {
        static LOGGED: std::sync::Once = std::sync::Once::new();
        LOGGED.call_once(|| eprintln!("[nvn] skipping draw in command buffer {:#x}: {reason}; {d:?}", a.0[0]));
        return 0;
    }
    push(a.0[0], Cmd::Draw(d));
    0
}

pub fn submit(a: &Args) -> u64 {
    super::objects::require_queue(a.0[0]);
    let handles = rd_vec::<u64>(a.0[2], a.0[1]);
    draw::submit(&handles);
    0
}

pub fn fence_sync(a: &Args) -> u64 {
    super::objects::require_queue(a.0[0]);
    draw::fence_sync(a.0[1]);
    0
}

pub fn sync_wait(a: &Args) -> u64 {
    draw::sync_wait(a.0[0],a.0[1])
}

pub fn present(a: &Args) -> u64 {
    super::objects::require_queue(a.0[0]);
    let (tex,layer)={let s=STATE.lock().unwrap();(s.windows.get(&a.0[1]).and_then(|(ts,_)|ts.get(a.0[2] as usize).copied()),s.window_layers.get(&a.0[1]).copied())};
    if let Some(t) = tex {
        draw::present(t,layer.expect("NVN window layer binding missing"));
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

    fn record_draw(name: &str, mut args: [u64; 8]) -> std::sync::Arc<Vec<Cmd>> {
        static NEXT_CB: AtomicU64 = AtomicU64::new(0x1000);
        let cb = NEXT_CB.fetch_add(1, Relaxed);
        args[0] = cb;
        begin(cb);
        host_impl(name).unwrap()(&Args(args, [0; 8]));
        recording(end(cb)).unwrap()
    }

    #[test]
    fn bulk_bindings_preserve_nonzero_first_binding_and_snapshot_array_values() {
        let ranges=[(0x12340000u64,64u64),(0x56780000,128)];
        for (name,stage) in [("nvnCommandBufferBindUniformBuffers",5),("nvnCommandBufferBindStorageBuffers",1)] {
            let commands=record_draw(name,[0,stage,3,2,ranges.as_ptr() as u64,0,0,0]);
            assert_eq!(commands.len(),2);
            for (i,c) in commands.iter().enumerate() {
                let (s,b,a,n)=match c {Cmd::Uniform{stage,index,addr,size}|Cmd::Storage{stage,index,addr,size}=>(*stage,*index,*addr,*size),_=>panic!("wrong buffer command")};
                assert_eq!((s,b,a,n),(stage as u32,3+i as u32,ranges[i].0,ranges[i].1));
            }
        }
        let commands=record_draw("nvnCommandBufferBindVertexBuffers",[0,2,2,ranges.as_ptr() as u64,0,0,0,0]);
        assert!(matches!(commands[1],Cmd::VertexBuffer{index:3,addr:0x56780000,size:128}));
        let textures=[0x100020003u64,0x100040005];
        let commands=record_draw("nvnCommandBufferBindTextures",[0,1,4,2,textures.as_ptr() as u64,0,0,0]);
        assert!(matches!(commands[1],Cmd::Texture{stage:1,index:5,handle:0x100040005}));
        let commands=record_draw("nvnCommandBufferBindImages",[0,5,1,2,textures.as_ptr() as u64,0,0,0]);
        assert!(matches!(commands[1],Cmd::Image{stage:5,index:2,handle:0x100040005,..}));
    }

    #[test]
    fn indexed_draws_with_null_buffers_are_not_recorded() {
        for name in ["nvnCommandBufferDrawElements", "nvnCommandBufferDrawElementsBaseVertex", "nvnCommandBufferDrawElementsInstanced"] {
            let commands = record_draw(name, [0, 4, 1, 36, 0, 0, 0, 1]);
            assert!(commands.is_empty(), "{name} recorded a draw that would dereference null");
        }
    }

    #[test]
    fn overflowing_index_ranges_are_not_recorded() {
        let commands = record_draw("nvnCommandBufferDrawElements", [0, 4, 1, 2, u64::MAX - 1, 0, 0, 1]);
        assert!(commands.is_empty(), "index range overflow reached the renderer");
    }

    #[test]
    fn valid_indexed_draw_preserves_arguments() {
        let indices = [0u16, 1, 2];
        let addr = indices.as_ptr() as u64;
        let commands = record_draw("nvnCommandBufferDrawElementsInstanced", [0, 4, 1, 3, addr, 7, 2, 4]);
        let [Cmd::Draw(d)] = commands.as_slice() else { panic!("expected one indexed draw") };
        assert_eq!(d.index, Some((1, addr)));
        assert_eq!((d.mode, d.count, d.base_vertex, d.base_instance, d.instances), (4, 3, 7, 2, 4));
    }

    #[test]
    fn indirect_draws_record_gpu_addresses_without_reading_arguments() {
        let commands=record_draw("nvnCommandBufferDrawArraysIndirect",[0,4,0x12340000,0,0,0,0,0]);
        assert!(matches!(commands.as_slice(),[Cmd::DrawIndirect{mode:4,index:None,arguments:0x12340000}]));
        let commands=record_draw("nvnCommandBufferDrawElementsIndirect",[0,5,1,0x23450000,0x34560000,0,0,0]);
        assert!(matches!(commands.as_slice(),[Cmd::DrawIndirect{mode:5,index:Some((1,0x23450000)),arguments:0x34560000}]));
    }

    #[test]
    fn non_indexed_draw_does_not_require_an_index_buffer() {
        let commands = record_draw("nvnCommandBufferDrawArrays", [0, 4, 2, 36, 0, 0, 0, 0]);
        let [Cmd::Draw(d)] = commands.as_slice() else { panic!("expected one non-indexed draw") };
        assert_eq!(d.index, None);
        assert_eq!((d.first, d.count), (2, 36));
    }

    struct Refill {
        command: [u8; 0x1000],
        control: [u8; 0x1000],
        calls: [u32; 2],
    }

    extern "C" fn refill(cb: u64, event: u32, minimum: u64, data: u64) {
        let f = unsafe { &mut *(data as *mut Refill) };
        assert!(event <= 1 && minimum <= 0x1000);
        f.calls[event as usize] += 1;
        let base = if event == 0 { f.command.as_ptr() } else { f.control.as_ptr() } as u64;
        add_region(cb, event as usize, base, 0x1000);
    }

    #[test]
    fn recording_refills_both_budgets_and_releases_recorder_lock_for_callback() {
        let cb = 0x8000_0010;
        let mut context = Box::new(Refill { command: [0; 0x1000], control: [0; 0x1000], calls: [0; 2] });
        initialize(cb);
        let data = &mut *context as *mut Refill as u64;
        host_impl("nvnCommandBufferSetMemoryCallback").unwrap()(&Args([cb, refill as *const () as u64, 0, 0, 0, 0, 0, 0], [0; 8]));
        host_impl("nvnCommandBufferSetMemoryCallbackData").unwrap()(&Args([cb, data, 0, 0, 0, 0, 0, 0], [0; 8]));
        add_region(cb, 0, 0x8000_1000, 16);
        add_region(cb, 1, 0x8000_2000, 16);
        begin(cb);
        push(cb, Cmd::DepthClamp(true));
        push(cb, Cmd::DepthClamp(false));
        assert_eq!(context.calls, [1, 1]);
        assert_eq!((memory_query(cb, 0, 0), memory_query(cb, 0, 1), memory_query(cb, 0, 2)), (0x1000, 16, 0xff0));
        assert_eq!((memory_query(cb, 1, 0), memory_query(cb, 1, 1), memory_query(cb, 1, 2)), (0x1000, 16, 0xff0));
        assert!(REC.lock().unwrap().open.contains_key(&cb));
        assert_eq!(recording(end(cb)).unwrap().len(), 2);
        assert!(!REC.lock().unwrap().open.contains_key(&cb));
        finalize(cb);
    }

    #[test]
    fn reused_control_storage_invalidates_only_overwritten_recordings() {
        let cb = 0x8000_0020;
        initialize(cb);
        add_region(cb, 0, 0x8010_0000, 4096);
        add_control_memory(cb, 0x8020_0000, 4096);
        begin(cb); push(cb, Cmd::DepthClamp(true)); let old = end(cb);
        let retained = recording(old).unwrap();
        begin(cb); push(cb, Cmd::DepthClamp(false)); let distinct = end(cb);
        assert!(recording(old).is_some());
        add_control_memory(cb, 0x8020_0000, 16);
        begin(cb); push(cb, Cmd::DepthClamp(false)); let replacement = end(cb);
        assert!(recording(old).is_none());
        assert!(recording(distinct).is_some());
        assert!(recording(replacement).is_some());
        assert_eq!(retained.len(), 1);
        finalize(cb);
    }

    #[test]
    fn texture_copies_capture_each_recorders_current_strides() {
        let cb = 0x8000_0030;
        let mut texture = [0u8; 0x80];
        let builder = [0u8; 0x68];
        let tex = texture.as_mut_ptr() as u64;
        super::super::objects::host_impl("nvnTextureInitialize").unwrap()(&Args([tex, builder.as_ptr() as u64, 0, 0, 0, 0, 0, 0], [0; 8]));
        let region = [0i32, 0, 0, 1, 2, 2];
        begin(cb);
        for (row, image) in [(16u64, 64u64), (32, 128)] {
            host_impl("nvnCommandBufferSetCopyRowStride").unwrap()(&Args([cb, row, 0, 0, 0, 0, 0, 0], [0; 8]));
            host_impl("nvnCommandBufferSetCopyImageStride").unwrap()(&Args([cb, image, 0, 0, 0, 0, 0, 0], [0; 8]));
            host_impl("nvnCommandBufferCopyBufferToTexture").unwrap()(&Args([cb, 0x1000, tex, 0, region.as_ptr() as u64, 0, 0, 0], [0; 8]));
        }
        let cmds = recording(end(cb)).unwrap();
        let got: Vec<_> = cmds.iter().map(|c| match c { Cmd::CopyBufferToTexture { strides, .. } => *strides, _ => panic!("copy") }).collect();
        assert_eq!(got, [(16, 64), (32, 128)]);
        finalize(cb);
    }
}
