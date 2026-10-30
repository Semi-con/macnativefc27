// Mush8 - textures.rs
// Constantin Clerc (c22dev) - 0.1
use super::super::cmd::TexBuilder;
use super::super::objects::STATE;
use crate::gfx::{self, Shared};
use nxshader::TexKind;
use objc2_foundation::NSRange;
use objc2_metal::*;
use super::super::fx::{FastMap,FastSet};
type SourceRanges = FastMap<(u32,u32),Vec<(u64,u64)>>;

pub type Texture = gfx::Texture;
pub type Sampler = objc2::runtime::ProtocolObject<dyn MTLSamplerState>;

fn u32_at(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes(b[o..o + 4].try_into().unwrap())
}
fn f32_at(b: &[u8], o: usize) -> f32 {
    f32::from_bits(u32_at(b, o))
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct TexKey {
    pub addr: u64,
    pub format: u32,
    pub target: u32,
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub levels: u32,
}

impl TexKey {
    pub fn of(b: &TexBuilder) -> Option<TexKey> {
        Self::of_in(b, &STATE.lock().unwrap())
    }

    pub fn of_in(b: &TexBuilder, s: &super::super::objects::State) -> Option<TexKey> {
        let pool = u64::from_le_bytes(b[8..16].try_into().unwrap());
        let offset = u64::from_le_bytes(b[16..24].try_into().unwrap());
        s.pools.get(&pool)?;
        let addr = s.pool_offset_cpu(pool, offset).unwrap_or((pool | 1 << 63) + offset);
        Some(TexKey {
            addr,
            format: u32_at(b, 0x30),
            target: u32_at(b, 0x1c),
            width: u32_at(b, 0x20).max(1),
            height: u32_at(b, 0x24).max(1),
            depth: u32_at(b, 0x28).max(1),
            levels: u32_at(b, 0x2c).max(1),
        })
    }
}

pub fn is_compressed(nvn: u32) -> bool {
    (65..=80).contains(&nvn) || (121..=148).contains(&nvn)
}

// Writable views expose integer components so surface instructions can apply
// the original NVN pixel conversion, including raw loads, in shader code.
fn image_storage_format(nvn:u32) -> Option<(MTLPixelFormat,u32,u32,u32)> {
    use MTLPixelFormat as P;
    // Packed formats share one 32-bit word. Shader conversion preserves the
    // exact memory representation while ordinary sampling uses the float view.
    if matches!(nvn,61..=63) {return Some((P::R32Uint,32,1,match nvn {61=>7,62=>8,_=>6}));}
    let (bits,components,kind)=match nvn {
        1=>(8,1,4),2=>(8,1,5),3=>(8,1,0),4=>(8,1,1),
        5=>(16,1,3),6=>(16,1,4),7=>(16,1,5),8=>(16,1,0),9=>(16,1,1),
        10=>(32,1,2),11=>(32,1,0),12=>(32,1,1),
        13=>(8,2,4),14=>(8,2,5),15=>(8,2,0),16=>(8,2,1),
        17=>(16,2,3),18=>(16,2,4),19=>(16,2,5),20=>(16,2,0),21=>(16,2,1),
        22=>(32,2,2),23=>(32,2,0),24=>(32,2,1),
        37=>(8,4,4),38=>(8,4,5),39=>(8,4,0),40=>(8,4,1),
        41=>(16,4,3),42=>(16,4,4),43=>(16,4,5),44=>(16,4,0),45=>(16,4,1),
        46=>(32,4,2),47=>(32,4,0),48=>(32,4,1),_=>return None,
    };
    let format=match (bits,components) {
        (8,1)=>P::R8Uint,(8,2)=>P::RG8Uint,(8,4)=>P::RGBA8Uint,
        (16,1)=>P::R16Uint,(16,2)=>P::RG16Uint,(16,4)=>P::RGBA16Uint,
        (32,1)=>P::R32Uint,(32,2)=>P::RG32Uint,(32,4)=>P::RGBA32Uint,_=>unreachable!(),
    };
    Some((format,bits,components,kind))
}

pub(super) fn buffer_descriptor(b:&TexBuilder)->Result<([u32;4],[u32;4]),&'static str> {
    let format=u32_at(b,0x30);
    if !(1..=48).contains(&format) {return Err("unsupported buffer texture pixel format");}
    let components=if format>=37 {4} else if format>=25 {3} else if format>=13 {2} else {1};
    let n=format-(components-1)*12;
    let (bits,kind)=if n<=4 {(8,match n {1=>4,2=>5,3=>0,_=>1})} else if n<=9 {(16,match n {5=>3,6=>4,7=>5,8=>0,_=>1})} else {(32,match n {10=>2,11=>0,_=>1})};
    let pool=u64::from_le_bytes(b[8..16].try_into().unwrap());let offset=u64::from_le_bytes(b[16..24].try_into().unwrap());let width=u32_at(b,0x20);
    let size=width as u64*(bits*components/8) as u64;
    let state=STATE.lock().unwrap();
    if width==0 || state.pools.get(&pool).is_none_or(|p|offset.checked_add(size).is_none_or(|end|end>p.size)) {return Err("buffer texture range exceeds its memory pool");}
    // Physical buffer textures can always use their retained canonical alias.
    // Do not add or select a newly allocated logical alias after this submit's
    // immutable mapping snapshot has already been taken.
    let address=state.pools.get(&pool).filter(|p|p.cpu!=0).and_then(|p|p.cpu.checked_add(offset)).or_else(||state.existing_pool_gpu_address(pool,offset)).ok_or("buffer texture has no GPU address")?;
    let mut swizzle=[0x38,0x3c,0x40,0x44].map(|o|u32_at(b,o));
    if swizzle==[0;4] || swizzle.iter().any(|c|*c>5) {swizzle=[2,3,4,5];}
    let packed=swizzle.into_iter().enumerate().fold(format,|value,(index,channel)|value|(channel<<(8+index*3)));
    Ok(([bits,components,kind,1],[address as u32,(address>>32) as u32,width,packed]))
}

#[derive(Default)]
pub struct Textures {
    map: FastMap<TexKey, Shared<Texture>>,
    block_views: Vec<(TexKey, Shared<Texture>)>,
    aliases: FastMap<TexKey, Vec<Shared<Texture>>>,
    storage: FastMap<TexKey, (u32, Shared<Texture>)>,
    views: FastMap<(TexKey, TexKind, [u32; 4], [u32;4]), Shared<Texture>>,
    sources: FastMap<TexKey,SourceRanges>,
    source_views:FastMap<(TexKey,[u32;4]),u64>,
    dummies: FastMap<TexKind, Shared<Texture>>,
    buffer_image_placeholder:Option<Shared<Texture>>,
    samplers: FastMap<[u8; 0x40], Shared<Sampler>>,
    default_sampler: Option<Shared<Sampler>>,
    misses: u64,
    // None means writable backing could not be captured as one complete CPU
    // snapshot. Refresh that resource conservatively once per submit.
    uploaded: FastMap<TexKey, (Option<[u64; WINDOWS]>, u64, u64)>,
    pub frame: u64,
}

impl Textures {
    pub(super) fn buffer_image_view(&mut self,b:&TexBuilder,image:nxshader::ImageBinding)->Result<(Shared<Texture>,[u32;4]),&'static str> {
        let (conversion,_)=buffer_descriptor(b)?;
        if image.kind!=TexKind::T2D {return Err("buffer image requires a one-dimensional surface instruction");}
        if image.raw_bytes!=0 && image.raw_bytes!=conversion[0]*conversion[1]/8 {return Err("buffer image raw load pixel width mismatch");}
        let placeholder=self.buffer_image_placeholder.get_or_insert_with(|| {
            let d=unsafe {MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(MTLPixelFormat::R32Uint,1,1,false)};
            d.setStorageMode(MTLStorageMode::Private);d.setUsage(MTLTextureUsage::ShaderRead|MTLTextureUsage::ShaderWrite);
            Shared(gfx::gpu().device.0.newTextureWithDescriptor(&d).expect("buffer image signature texture"))
        }).clone();
        Ok((placeholder,conversion))
    }
    pub fn flush_cpu_memory(&mut self, addr: u64, size: u64) {
        for (key, (hashes, checked, len)) in &mut self.uploaded {
            if key.addr < addr.saturating_add(size) && addr < key.addr.saturating_add(*len) {
                // An explicit flush is evidence of a write, even if this
                // submit's rolling checksum window lies elsewhere.
                *hashes=None;*checked=u64::MAX;
            }
        }
    }
    pub fn evict(&mut self, gone: &dyn Fn(&TexKey) -> bool) {
        self.map.retain(|k, _| !gone(k));
        self.block_views.retain(|(k, _)| !gone(k));
        self.aliases.retain(|k, _| !gone(k));
        self.storage.retain(|k, _| !gone(k));
        self.views.retain(|(k, _, _, _), _| !gone(k));
        self.sources.retain(|k,_|!gone(k));
        self.source_views.retain(|(k,_),_|!gone(k));
        self.uploaded.retain(|k, _| !gone(k));
    }

    pub fn cached(&self, key: &TexKey) -> Option<(TexKey, Shared<Texture>)> {
        self.map.get(key).map(|t| (*key, t.clone()))
    }

    pub fn aliases_of(&self, key: &TexKey) -> &[Shared<Texture>] {
        self.aliases.get(key).map(Vec::as_slice).unwrap_or(&[])
    }

    // A bindless descriptor pool also contains stale/unselected entries.
    // Omit unavailable backing from the sparse GPU lookup, where selecting
    // that handle still raises the existing translated-shader fault (bit32).
    pub(super) fn bindless_available(&self,key:TexKey,b:&TexBuilder)->bool {
        if self.map.contains_key(&key) && !self.sources.contains_key(&key) {return true;}
        let epoch=super::super::objects::VIRTUAL_MAPPING_EPOCH.load(std::sync::atomic::Ordering::Relaxed);
        if self.source_views.get(&(key,super::super::objects::view_range(b)))==Some(&epoch) {return true;}
        let max_levels=32-key.width.max(key.height).max(if key.target==2 {key.depth}else{1}).leading_zeros();
        try_source_ranges(b,&key,key.levels.min(max_levels)).is_some()
    }

    pub fn get(&mut self, b: &TexBuilder, render_target: bool) -> Option<(TexKey, Shared<Texture>)> {
        self.get_keyed(TexKey::of(b), b, render_target)
    }

    pub(super) fn cpu_reads_for_get(&self,key:Option<TexKey>,render_target:bool)->bool {
        if render_target {return false;}
        let Some(key)=key else {return false;};
        if self.map.contains_key(&key) {self.uploaded.contains_key(&key) || self.sources.contains_key(&key)}
        else {!gfx::is_depth_format(key.format)}
    }

    pub fn get_keyed(&mut self, key: Option<TexKey>, b: &TexBuilder, render_target: bool) -> Option<(TexKey, Shared<Texture>)> {
        if super::submit_profile::active() {
            if let Some(k)=key.filter(|k|k.format==80 && k.width==1024 && k.height==512) {
                let pool=u64::from_le_bytes(b[8..16].try_into().unwrap());
                let (flags,label)={let state=STATE.lock().unwrap();(state.pools.get(&pool).map(|p|p.flags),state.debug_labels.get(&pool).cloned())};
                let len=storage_len(b,&k);
                let hashes=(len!=0 && len<=2*1024*1024 && crate::kernel::aspace().readable(k.addr,len)).then(||memory_hashes(k.addr,len));
                eprintln!("[background-source] frame={} key={k:?} render_target={render_target} pool={pool:#x} flags={flags:x?} label={label:?} cached={} uploaded={:?} len={len} cpu_hashes={hashes:x?}",self.frame,self.map.contains_key(&k),self.uploaded.get(&k));
            }
        }
        static FULL:std::sync::LazyLock<bool>=std::sync::LazyLock::new(||std::env::var("NX_CPU_TEXTURE_FULL_CHECKS").is_ok_and(|v|v=="1"));
        let full=*FULL || crate::input::test_control::cpu_texture_checks_active();
        self.get_keyed_with_checks(key,b,render_target,full)
    }

    fn get_keyed_with_checks(&mut self, key: Option<TexKey>, b: &TexBuilder, render_target: bool, full:bool) -> Option<(TexKey, Shared<Texture>)> {
        let Some(key) = key else {
            self.miss("memory pool not found", b);
            return None;
        };
        if let Some(t) = self.map.get(&key) {
            let t = t.clone();
            let frame = self.frame;
            if !render_target {
                if let Some((mut hashes, checked, len)) = self.uploaded.get(&key).copied() {
                    if checked != frame {
                        let w = window(len, frame);
                        let changed = hashes.is_none_or(|h| if full {memory_hashes(key.addr,len)!=h} else {memory_hash(key.addr, len, w) != h[w]});
                        // The bounded submit profiler can prove that a partial
                        // stripe missed a CPU update. No GPU readback or extra
                        // work occurs on the normal path with profiling off.
                        if !full && !changed && super::submit_profile::active() && hashes.is_some_and(|h|memory_hashes(key.addr,len)!=h) {
                            use std::sync::atomic::{AtomicU32,Ordering::Relaxed};
                            static REPORTED:AtomicU32=AtomicU32::new(0);
                            if REPORTED.fetch_add(1,Relaxed)<32 {eprintln!("[texture-coherence] rolling miss frame={frame} window={w} len={len} key={key:?}");}
                        }
                        if changed {
                            let levels = t.0.mipmapLevelCount() as u32;
                            let (ranges, snapshot_hashes)=upload(b, &key, &t.0, levels,None);
                            self.sources.insert(key,ranges);
                            self.source_views.retain(|(k,_),_|*k!=key);
                            hashes = snapshot_hashes;
                        }
                        self.uploaded.insert(key, (hashes, frame, len));
                    }
                }
            }
            let epoch=super::super::objects::VIRTUAL_MAPPING_EPOCH.load(std::sync::atomic::Ordering::Relaxed);
            let view_key=(key,super::super::objects::view_range(b));
            if !render_target && self.source_views.get(&view_key)!=Some(&epoch) {
                if let Some(previous)=self.sources.get(&key) {
                    let current=source_ranges(b,&key,t.0.mipmapLevelCount() as u32);
                    let changed:FastSet<_>=current.iter().filter(|(sub,ranges)|previous.get(sub)!=Some(ranges)).map(|(sub,_)|*sub).collect();
                    if !changed.is_empty() {
                        let (loaded,_)=upload(b,&key,&t.0,t.0.mipmapLevelCount() as u32,Some(&changed));
                        self.sources.get_mut(&key).unwrap().extend(loaded);
                    }
                    self.source_views.insert(view_key,epoch);
                }
            }
            return Some((key, t));
        }
        let Some(format) = gfx::pixel_format(key.format) else {
            self.miss("no Metal format", b);
            return None;
        };
        let base_key = TexKey { format: 0, ..key };
        if is_compressed(key.format) && gfx::bc::soft(key.format).is_none() {
            if let Some((bk, base)) = self.block_views.iter().find(|(bk, _)| bk.addr == key.addr).map(|(k, t)| (*k, t.clone())) {
                let (bw, bh, bpb) = super::super::format_info(key.format);
                let (_, _, base_bpb) = super::super::format_info(bk.format);
                if bpb == base_bpb && bk.width == key.width.div_ceil(bw) && bk.height == key.height.div_ceil(bh) {
                    let alias = self.create(b, key, format, true)?;
                    let cb = gfx::gpu().queue.0.commandBuffer()?;
                    copy_blocks(&cb, &base.0, &alias.0, None);
                    cb.commit();
                    self.aliases.entry(bk).or_default().push(alias.clone());
                    self.map.insert(key, alias.clone());
                    return Some((key, alias));
                }
            }
        }
        if let Some((base_fmt, base)) = self.storage.get(&base_key).cloned() {
            let (_, _, bpb) = super::super::format_info(key.format);
            let (_, _, base_bpb) = super::super::format_info(base_fmt);
            let plain = |f: u32| !is_compressed(f) && !gfx::is_depth_format(f);
            if bpb == base_bpb && plain(key.format) && plain(base_fmt) {
                let v = unsafe {
                    base.0.newTextureViewWithPixelFormat_textureType_levels_slices(
                        format,
                        base.0.textureType(),
                        NSRange { location: 0, length: base.0.mipmapLevelCount() },
                        NSRange { location: 0, length: base.0.arrayLength() * if base.0.textureType() == MTLTextureType::TypeCube { 6 } else { 1 } },
                    )
                };
                if let Some(v) = v {
                    let v = Shared(v);
                    self.map.insert(key, v.clone());
                    return Some((key, v));
                }
            }
        }
        if !render_target && key.addr >> 63 != 0 && self.misses < 40 {
            self.misses += 1;
            let near: Vec<_> = self.map.keys().filter(|k| k.addr == key.addr).collect();
            eprintln!("[gfx] sampled virtual-pool texture not rendered yet: {key:?} same-address keys {near:?}");
        }
        let tex = self.create(b, key, format, render_target)?;
        self.map.insert(key, tex.clone());
        self.storage.entry(TexKey { format: 0, ..key }).or_insert((key.format, tex.clone()));
        Some((key, tex))
    }

    fn miss(&mut self, why: &str, b: &TexBuilder) {
        self.misses += 1;
        if self.misses <= 10 {
            eprintln!(
                "[gfx] texture unavailable ({why}): pool {:#x} offset {:#x} format {} target {} {}x{}",
                u64::from_le_bytes(b[8..16].try_into().unwrap()),
                u64::from_le_bytes(b[16..24].try_into().unwrap()),
                u32_at(b, 0x30),
                u32_at(b, 0x1c),
                u32_at(b, 0x20),
                u32_at(b, 0x24)
            );
        }
    }

    fn create(&mut self, b: &TexBuilder, key: TexKey, format: MTLPixelFormat, render_target: bool) -> Option<Shared<Texture>> {
        let (ty, slices) = match key.target {
            2 => (MTLTextureType::Type3D, 1),
            3 => (MTLTextureType::Type2DArray, key.height),
            4 | 6 => (MTLTextureType::Type2DArray, key.depth),
            8 => (MTLTextureType::TypeCube, 1),
            9 => (MTLTextureType::TypeCubeArray, (key.depth / 6).max(1)),
            // Buffer texels are represented by a one-row 2D texture; the
            // advertised buffer-texture limit follows this Metal representation.
            10 => (MTLTextureType::Type2D, 1),
            _ => (MTLTextureType::Type2D, 1),
        };
        let height = if matches!(key.target, 3 | 10) { 1 } else { key.height };
        let depth_format = gfx::is_depth_format(key.format);
        let max_levels = 32 - key.width.max(height).leading_zeros();
        let levels = key.levels.min(max_levels).max(1);
        let d = MTLTextureDescriptor::new();
        unsafe {
            d.setTextureType(ty);
            d.setPixelFormat(format);
            d.setWidth(key.width as usize);
            d.setHeight(height as usize);
            d.setDepth(if key.target == 2 { key.depth as usize } else { 1 });
            d.setMipmapLevelCount(levels as usize);
            d.setArrayLength(slices.max(1) as usize);
        }
        let mut usage = MTLTextureUsage::ShaderRead | MTLTextureUsage::PixelFormatView;
        if !is_compressed(key.format) {
            usage |= MTLTextureUsage::RenderTarget | MTLTextureUsage::PixelFormatView;
            if !depth_format {usage |= MTLTextureUsage::ShaderWrite;}
        }
        d.setUsage(usage);
        d.setStorageMode(MTLStorageMode::Private);
        let tex = Shared(gfx::gpu().device.0.newTextureWithDescriptor(&d)?);
        let mut snapshot_hashes=None;
        if !render_target && !depth_format {
            let (ranges,hashes)=upload(b, &key, &tex.0, levels,None);
            snapshot_hashes=hashes;
            self.sources.insert(key,ranges);
            self.source_views.insert((key,super::super::objects::view_range(b)),super::super::objects::VIRTUAL_MAPPING_EPOCH.load(std::sync::atomic::Ordering::Relaxed));
        }
        let len = storage_len(b, &key);
        if !render_target && !depth_format && cpu_writable(b) && len > 0 {
            self.uploaded.insert(key, (snapshot_hashes, self.frame, len));
        }
        if render_target && super::super::gfx_integer_block_format(key.format) {
            self.block_views.push((key, tex.clone()));
        }
        Some(tex)
    }

    pub fn view(&mut self, key: TexKey, b: &TexBuilder, tex: &Shared<Texture>, kind: TexKind) -> Option<Shared<Texture>> {
        let mut swz = [u32_at(b, 0x38), u32_at(b, 0x3c), u32_at(b, 0x40), u32_at(b, 0x44)];
        if swz == [0; 4] || swz.iter().any(|&c| c > 5) {
            swz = [2, 3, 4, 5];
        }
        if key.format == 60 {
            swz = swz.map(|c| match c {
                2 => 4,
                4 => 2,
                c => c,
            });
        }
        if matches!(key.format,153|155) {swz=swz.map(|c|if c==5 {1} else {c});}
        let want = kind_type(kind);
        let have = tex.0.textureType();
        let range=super::super::objects::view_range(b);
        let [layer,count,level,mips]=range;
        let all_mips=tex.0.mipmapLevelCount() as u32;
        let mips=if mips==0 {all_mips.checked_sub(level)?} else {mips};
        if mips==0 || level.checked_add(mips)?>all_mips {return None;}
        if have == want && swz == [2, 3, 4, 5] && range==[0;4] {
            return Some(tex.clone());
        }
        if let Some(v) = self.views.get(&(key, kind, swz,range)) {
            return Some(v.clone());
        }
        let available = tex.0.arrayLength() * if matches!(have, MTLTextureType::TypeCube | MTLTextureType::TypeCubeArray) { 6 } else { 1 };
        let remaining=available.checked_sub(layer as usize)?;
        let requested=if count==0 {remaining} else {count as usize};
        let slices = match (have, want) {
            (h, w) if h == w => requested,
            (MTLTextureType::Type3D, _) | (_, MTLTextureType::Type3D) => return None,
            (_, MTLTextureType::Type2D) => 1,
            (_, MTLTextureType::Type2DArray) => requested,
            (_, MTLTextureType::TypeCube) => 6,
            (_, MTLTextureType::TypeCubeArray) => requested / 6 * 6,
            _ => return None,
        };
        if slices == 0 || slices > remaining || slices>requested {
            return None;
        }
        let s = |c: u32| MTLTextureSwizzle(c as _);
        let v = unsafe {
            tex.0.newTextureViewWithPixelFormat_textureType_levels_slices_swizzle(
                tex.0.pixelFormat(),
                want,
                NSRange { location: level as usize, length: mips as usize },
                NSRange { location: layer as usize, length: slices },
                MTLTextureSwizzleChannels { red: s(swz[0]), green: s(swz[1]), blue: s(swz[2]), alpha: s(swz[3]) },
            )
        }?;
        let v = Shared(v);
        self.views.insert((key, kind, swz,range), v.clone());
        Some(v)
    }

    pub fn image_view(&mut self,b:&TexBuilder,image:nxshader::ImageBinding) -> Result<(Shared<Texture>,[u32;4]),&'static str> {
        let format=u32_at(b,0x30);
        let (uint_format,bits,components,kind)=image_storage_format(format).ok_or("unsupported storage-image format")?;
        if image.raw_bytes!=0 && bits*components/8!=image.raw_bytes {return Err("raw surface load needs a different pixel-width view");}
        let (_,base)=self.get(b,false).ok_or("storage image unavailable")?;
        let (layer,level)=super::super::objects::view_subresource(b);
        let (target,slices)=match image.kind {
            TexKind::T2D=>(MTLTextureType::Type2D,1),
            TexKind::T2DArray=>(MTLTextureType::Type2DArray,base.0.arrayLength().saturating_sub(layer as usize)),
            TexKind::T3D=>(MTLTextureType::Type3D,1),_=>return Err("unsupported storage-image dimensionality"),
        };
        if slices==0 || level as usize>=base.0.mipmapLevelCount() {return Err("storage-image subresource out of range");}
        let view=unsafe {base.0.newTextureViewWithPixelFormat_textureType_levels_slices(uint_format,target,NSRange{location:level as usize,length:1},NSRange{location:if target==MTLTextureType::Type3D {0} else {layer as usize},length:slices})}.ok_or("Metal cannot create storage-image integer view")?;
        Ok((Shared(view),[bits,components,kind,0]))
    }

    pub fn dummy(&mut self, kind: TexKind) -> Shared<Texture> {
        self.dummies
            .entry(kind)
            .or_insert_with(|| {
                let d = MTLTextureDescriptor::new();
                let ty = kind_type(kind);
                let slices = if ty == MTLTextureType::TypeCube || ty == MTLTextureType::TypeCubeArray { 6 } else { 1 };
                unsafe {
                    d.setTextureType(ty);
                    d.setWidth(1);
                    d.setHeight(1);
                }
                if kind.is_depth() {
                    d.setPixelFormat(MTLPixelFormat::Depth32Float);
                    d.setStorageMode(MTLStorageMode::Private);
                } else {
                    d.setPixelFormat(MTLPixelFormat::RGBA8Unorm);
                    d.setStorageMode(MTLStorageMode::Shared);
                }
                let t = gfx::gpu().device.0.newTextureWithDescriptor(&d).expect("dummy texture");
                if !kind.is_depth() {
                    let white = [0xFFu8; 4];
                    for s in 0..slices {
                        unsafe {
                            t.replaceRegion_mipmapLevel_slice_withBytes_bytesPerRow_bytesPerImage(
                                region(1, 1, 1),
                                0,
                                s,
                                std::ptr::NonNull::from(&white).cast(),
                                4,
                                4,
                            )
                        };
                    }
                }
                Shared(t)
            })
            .clone()
    }

    pub fn sampler(&mut self, b: Option<&[u8; 0x60]>) -> Shared<Sampler> {
        let Some(b) = b else {
            return self.default_sampler.get_or_insert_with(|| Shared(make_sampler(None))).clone();
        };
        let mut key = [0u8; 0x40];
        key.copy_from_slice(&b[8..0x48]);
        self.samplers.entry(key).or_insert_with(|| Shared(make_sampler(Some(b)))).clone()
    }
}

pub fn kind_type(kind: TexKind) -> MTLTextureType {
    match kind {
        TexKind::T2D | TexKind::Depth2D => MTLTextureType::Type2D,
        TexKind::T2DArray | TexKind::Depth2DArray => MTLTextureType::Type2DArray,
        TexKind::T3D => MTLTextureType::Type3D,
        TexKind::Cube | TexKind::DepthCube => MTLTextureType::TypeCube,
        TexKind::CubeArray => MTLTextureType::TypeCubeArray,
    }
}

fn region(w: u32, h: u32, d: u32) -> MTLRegion {
    MTLRegion { origin: MTLOrigin { x: 0, y: 0, z: 0 }, size: MTLSize { width: w as usize, height: h as usize, depth: d as usize } }
}

struct Piece {
    offset: usize,
    row: usize,
    image: usize,
    size: (u32, u32, u32),
    sub: (u32, u32),
}

fn source_ranges(b:&TexBuilder,key:&TexKey,levels:u32)->SourceRanges {
    try_source_ranges(b,key,levels).unwrap_or_else(|| {
        let pool=u64::from_le_bytes(b[8..16].try_into().unwrap());let offset=u64::from_le_bytes(b[16..24].try_into().unwrap());
        let state=STATE.lock().unwrap();let maps=state.virtual_maps.get(&pool).map(|maps|maps.iter().filter(|m|m.0<offset.saturating_add(storage_len(b,key)) && m.0+m.1>offset).take(16).copied().collect::<Vec<_>>());
        eprintln!("[texture-upload] incomplete backing: {key:?} pool{pool:#x} offset{offset:#x} view{:?} maps{maps:x?}",super::super::objects::view_range(b));
        let _=std::fs::create_dir_all("work/texture-upload-fail");let _=std::fs::write("work/texture-upload-fail/builder.bin",b);
        crate::diag::report_and_exit("sampled texture subresource has incomplete readable pool backing");
    })
}

fn try_source_ranges(b:&TexBuilder,key:&TexKey,levels:u32)->Option<SourceRanges> {
    let pool=u64::from_le_bytes(b[8..16].try_into().unwrap());let base=u64::from_le_bytes(b[16..24].try_into().unwrap());
    let [first_layer,layer_count,first_level,level_count]=super::super::objects::view_range(b);
    let mut requested=Vec::new();
    if let Some((shape,layout))=super::super::texture_layout(b.as_ptr() as u64) {
        let last_layer=if layer_count==0 {shape.layers} else {first_layer.checked_add(layer_count).unwrap_or(u32::MAX)};
        let last_level=if level_count==0 {levels} else {first_level.checked_add(level_count).unwrap_or(u32::MAX)};
        if first_layer>=last_layer || last_layer>shape.layers || first_level>=last_level || last_level>levels {crate::diag::report_and_exit("NVN sampled texture view exceeds its mip/layer allocation");}
        for layer in first_layer..last_layer {for level in first_level..last_level {
            let row=(shape.width>>level).max(1).div_ceil(shape.block_w)*shape.bytes_per_block;
            let rows=(shape.height>>level).max(1).div_ceil(shape.block_h);
            let depth=(shape.depth>>level).max(1);let (gy,gz)=layout.level_gobs[level as usize];
            let len=row.div_ceil(64) as u64*rows.div_ceil(8).div_ceil(gy) as u64*depth.div_ceil(gz) as u64*gy as u64*gz as u64*512;
            requested.push(((level,layer),base+layout.layer_size*layer as u64+layout.level_offsets[level as usize],len));
        }}
    } else {requested.push(((0,0),base,storage_len(b,key)));}
    let state=STATE.lock().unwrap();let mut sources=SourceRanges::default();
    for (sub,offset,len) in requested {
        let ranges=state.pool_cpu_ranges(pool,offset,len).filter(|ranges|ranges.iter().all(|&(addr,len)|crate::kernel::aspace().readable(addr,len)));
        let ranges=ranges?;
        sources.insert(sub,ranges);
    }
    Some(sources)
}

struct UploadMemory {
    sources:SourceRanges,
    // Only a complete, contiguous and readable mapping can be compared with
    // the existing rolling CPU checksum. The very same bytes feed untile.
    bytes:Option<Vec<u8>>,
    hashes:Option<[u64;WINDOWS]>,
}

fn contiguous_ranges(base:u64,len:u64,ranges:&[(u64,u64)])->bool {
    let Some(end)=base.checked_add(len) else {return false;};
    let mut at=base;
    for &(address,size) in ranges {
        if address!=at || size==0 {return false;}
        let Some(next)=at.checked_add(size) else {return false;};at=next;
        if at>end {return false;}
    }
    len!=0 && at==end
}

impl UploadMemory {
    fn capture(b:&TexBuilder,key:&TexKey,levels:u32)->Self {
        let sources=source_ranges(b,key,levels);
        let len=storage_len(b,key);
        let pool=u64::from_le_bytes(b[8..16].try_into().unwrap());
        let offset=u64::from_le_bytes(b[16..24].try_into().unwrap());
        let contiguous=cpu_writable(b) && len>0 && STATE.lock().unwrap().pool_cpu_ranges(pool,offset,len)
            .is_some_and(|ranges|contiguous_ranges(key.addr,len,&ranges));
        let bytes=if contiguous && crate::kernel::aspace().readable(key.addr,len) {
            Some(unsafe {std::slice::from_raw_parts(key.addr as *const u8,len as usize)}.to_vec())
        } else {None};
        let hashes=bytes.as_ref().map(|bytes|memory_hashes(bytes.as_ptr() as u64,bytes.len() as u64));
        Self{sources,bytes,hashes}
    }
}

fn upload(b: &TexBuilder, key: &TexKey, tex: &Texture, levels: u32, only:Option<&FastSet<(u32,u32)>>) -> (SourceRanges,Option<[u64;WINDOWS]>) {
    upload_captured(b,key,tex,levels,only,UploadMemory::capture(b,key,levels))
}

fn upload_captured(b:&TexBuilder,key:&TexKey,tex:&Texture,levels:u32,only:Option<&FastSet<(u32,u32)>>,memory:UploadMemory)->(SourceRanges,Option<[u64;WINDOWS]>) {
    let (bw, bh, bpb) = super::super::format_info(key.format);
    let soft = gfx::bc::soft(key.format);
    let mut pieces = Vec::new();
    let mut total = 0usize;
    let mut piece = |size: (u32, u32, u32), sub: (u32, u32), row: usize, rows: usize| {
        pieces.push(Piece { offset: total, row, image: row * rows, size, sub });
        total = (total + row * rows * size.2 as usize).next_multiple_of(16);
    };
    let UploadMemory{sources,mut bytes,hashes}=memory;
    match super::super::texture_layout(b.as_ptr() as u64) {
        Some((shape, layout)) => {
            if shape.width!=key.width {crate::diag::report_and_exit("multisampled CPU texture upload is unsupported");}
            let levels = levels.min(layout.level_offsets.len() as u32);
            for layer in 0..shape.layers {
                for level in 0..levels {
                    if !sources.contains_key(&(level,layer)) || only.is_some_and(|set|!set.contains(&(level,layer))) {continue;}
                    let (w, h) = ((shape.width >> level).max(1), (shape.height >> level).max(1));
                    let (cols, rows) = (w.div_ceil(bw), h.div_ceil(bh));
                    piece((w, h, (shape.depth >> level).max(1)), (level, layer), (cols * bpb) as usize, rows as usize);
                }
            }
            let gathered=if let Some(snapshot)=bytes.take() {snapshot} else {
                let mut gathered=vec![0u8;layout.total_size as usize];
                for p in &pieces {
                    let mut at=(p.sub.1 as u64*layout.layer_size+layout.level_offsets[p.sub.0 as usize]) as usize;
                    for &(addr,len) in &sources[&p.sub] {
                        let bytes=unsafe{std::slice::from_raw_parts(addr as *const u8,len as usize)};
                        gathered[at..at+bytes.len()].copy_from_slice(bytes);at+=bytes.len();
                    }
                }
                gathered
            };
            let fill = |dst: &mut [u8]| untile_pieces(&gathered, &layout, &pieces, dst);
            match soft {
                None => stage_with(tex, total, &pieces, fill),
                Some(k) => {
                    let mut data = vec![0u8; total];
                    fill(&mut data);
                    decode_and_stage(tex, k, data, pieces);
                }
            }
        }
        None => {
            let stride = if key.target == 10 { key.width.div_ceil(bw) * bpb } else { u32_at(b, 0x5c) };
            let rows = if key.target == 10 { 1 } else { key.height.div_ceil(bh) };
            let row_bytes = key.width.div_ceil(bw) * bpb;
            if stride < row_bytes {
                crate::diag::report_and_exit("linear texture upload outside readable guest storage or invalid stride");
            }
            let gathered=if let Some(snapshot)=bytes.take() {snapshot} else {
                let mut gathered=Vec::with_capacity((stride*rows) as usize);
                for &(addr,len) in &sources[&(0,0)] {gathered.extend_from_slice(unsafe{std::slice::from_raw_parts(addr as *const u8,len as usize)});}
                gathered
            };
            let src=gathered.as_slice();
            piece((key.width, if key.target == 10 { 1 } else { key.height }, 1), (0, 0), stride as usize, rows as usize);
            match soft {
                None => stage_with(tex, total, &pieces, |dst| dst[..src.len()].copy_from_slice(src)),
                Some(k) => decode_and_stage(tex, k, src.to_vec(), pieces),
            }
        }
    }
    (sources,hashes)
}

fn untile_pieces(src: &[u8], layout: &super::super::tex::Layout, pieces: &[Piece], dst: &mut [u8]) {
    use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
    const BAND: u32 = 64;
    let mut jobs = Vec::new();
    for (i, p) in pieces.iter().enumerate() {
        let rows = (p.image / p.row.max(1)) as u32;
        for z in 0..p.size.2 {
            let mut y = 0;
            while y < rows {
                jobs.push((i, z, y..(y + BAND).min(rows)));
                y += BAND;
            }
        }
    }
    let layer_of = |p: &Piece| p.sub.1 as u64 * layout.layer_size + layout.level_offsets[p.sub.0 as usize];
    let base = dst.as_mut_ptr() as usize;
    let run = |(i, z, ys): &(usize, u32, std::ops::Range<u32>)| {
        let p = &pieces[*i];
        let rows = (p.image / p.row.max(1)) as u32;
        let start = layer_of(p) as usize;
        let at = p.offset + *z as usize * p.image + ys.start as usize * p.row;
        let out = unsafe { std::slice::from_raw_parts_mut((base + at) as *mut u8, ys.len() * p.row) };
        super::super::tex::untile_into(src.get(start..).unwrap_or(&[]), p.row as u32, rows, *z, layout.level_gobs[p.sub.0 as usize], ys.clone(), out);
    };
    if dst.len() < 1 << 20 || jobs.len() < 4 {
        jobs.iter().for_each(run);
        return;
    }
    let next = AtomicUsize::new(0);
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get()).clamp(2, 6);
    std::thread::scope(|s| {
        for _ in 1..threads {
            s.spawn(|| {
                while let Some(j) = jobs.get(next.fetch_add(1, Relaxed)) {
                    run(j);
                }
            });
        }
        while let Some(j) = jobs.get(next.fetch_add(1, Relaxed)) {
            run(j);
        }
    });
}

fn decode_and_stage(tex: &Texture, k: gfx::bc::Kind, data: Vec<u8>, mut pieces: Vec<Piece>) {
    let mut decoded = Vec::with_capacity(data.len() * 4);
    for p in &mut pieces {
        let (w, h, d) = p.size;
        let rows = (h as usize).div_ceil(4);
        let offset = decoded.len();
        let mut bpr = 0;
        for i in 0..d as usize {
            let Some(img) = data.get(p.offset + i * p.image..) else { break };
            let (px, r) = gfx::bc::decode(k, &img[..img.len().min(p.row * rows)], p.row, w as usize, h as usize, 1);
            decoded.extend_from_slice(&px);
            bpr = r;
        }
        (p.offset, p.row, p.image) = (offset, bpr, bpr * h as usize);
        decoded.resize(decoded.len().next_multiple_of(16), 0);
    }
    stage_with(tex, decoded.len(), &pieces, |dst| dst.copy_from_slice(&decoded));
}

fn stage_with(tex: &Texture, total: usize, pieces: &[Piece], fill: impl FnOnce(&mut [u8])) {
    if pieces.is_empty() || total == 0 {
        return;
    }
    let g = gfx::gpu();
    let Some(buf) = g.device.0.newBufferWithLength_options(total, MTLResourceOptions::StorageModeShared | MTLResourceOptions::CPUCacheModeWriteCombined) else { return };
    let dst = unsafe { std::slice::from_raw_parts_mut(buf.contents().as_ptr() as *mut u8, total) };
    fill(dst);
    let Some(cb) = g.queue.0.commandBuffer() else { return };
    let Some(enc) = cb.blitCommandEncoder() else { return };
    for p in pieces {
        let (w, h, d) = p.size;
        unsafe {
            enc.copyFromBuffer_sourceOffset_sourceBytesPerRow_sourceBytesPerImage_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin(
                &buf,
                p.offset,
                p.row,
                p.image,
                MTLSize { width: w as usize, height: h as usize, depth: d as usize },
                tex,
                p.sub.1 as usize,
                p.sub.0 as usize,
                MTLOrigin { x: 0, y: 0, z: 0 },
            )
        };
    }
    enc.endEncoding();
    cb.commit();
}

pub fn compare_fn(nvn: u32) -> MTLCompareFunction {
    use MTLCompareFunction as C;
    match nvn {
        1 => C::Never,
        2 => C::Less,
        3 => C::Equal,
        4 => C::LessEqual,
        5 => C::Greater,
        6 => C::NotEqual,
        7 => C::GreaterEqual,
        _ => C::Always,
    }
}

fn make_sampler(b: Option<&[u8; 0x60]>) -> objc2::rc::Retained<Sampler> {
    let d = MTLSamplerDescriptor::new();
    d.setSupportArgumentBuffers(true);
    if let Some(b) = b {
        let min = u32_at(b, 8);
        d.setMinFilter(if matches!(min, 1 | 3 | 5) { MTLSamplerMinMagFilter::Linear } else { MTLSamplerMinMagFilter::Nearest });
        d.setMipFilter(match min {
            2 | 3 => MTLSamplerMipFilter::Nearest,
            4 | 5 => MTLSamplerMipFilter::Linear,
            _ => MTLSamplerMipFilter::NotMipmapped,
        });
        d.setMagFilter(if u32_at(b, 0xc) == 1 { MTLSamplerMinMagFilter::Linear } else { MTLSamplerMinMagFilter::Nearest });
        let wrap = |w: u32| match w {
            1 => MTLSamplerAddressMode::Repeat,
            2..=4 => MTLSamplerAddressMode::MirrorClampToEdge,
            5 => MTLSamplerAddressMode::ClampToBorderColor,
            6 => MTLSamplerAddressMode::MirrorRepeat,
            _ => MTLSamplerAddressMode::ClampToEdge,
        };
        d.setSAddressMode(wrap(u32_at(b, 0x10)));
        d.setTAddressMode(wrap(u32_at(b, 0x14)));
        d.setRAddressMode(wrap(u32_at(b, 0x18)));
        let (lo, hi) = (f32_at(b, 0x1c), f32_at(b, 0x20));
        if lo.is_finite() && hi.is_finite() && lo <= hi {
            d.setLodMinClamp(lo.max(0.0));
            d.setLodMaxClamp(hi);
        }
        d.setCompareFunction(if u32_at(b, 0x28) == 1 { compare_fn(u32_at(b, 0x2c)) } else { MTLCompareFunction::LessEqual });
        let border = [f32_at(b, 0x30), f32_at(b, 0x34), f32_at(b, 0x38), f32_at(b, 0x3c)];
        d.setBorderColor(if border == [1.0; 4] {
            MTLSamplerBorderColor::OpaqueWhite
        } else if border[3] == 1.0 {
            MTLSamplerBorderColor::OpaqueBlack
        } else {
            MTLSamplerBorderColor::TransparentBlack
        });
        let bias = f32_at(b, 0x24);
        if bias.is_finite() && bias != 0.0 && objc2::runtime::NSObjectProtocol::respondsToSelector(&*d, objc2::sel!(setLodBias:)) {
            d.setLodBias(bias.clamp(-16.0, 15.99));
        }
        let aniso = f32_at(b, 0x40);
        if aniso.is_finite() {
            d.setMaxAnisotropy((aniso as usize).clamp(1, 16));
        }
    } else {
        d.setMinFilter(MTLSamplerMinMagFilter::Linear);
        d.setMagFilter(MTLSamplerMinMagFilter::Linear);
        d.setSAddressMode(MTLSamplerAddressMode::Repeat);
        d.setTAddressMode(MTLSamplerAddressMode::Repeat);
    }
    gfx::gpu().device.0.newSamplerStateWithDescriptor(&d).expect("sampler")
}

fn storage_len(b: &TexBuilder, key: &TexKey) -> u64 {
    match super::super::texture_layout(b.as_ptr() as u64) {
        Some((_, l)) => l.total_size,
        None => {
            let (bw, bh, bpb) = super::super::format_info(key.format);
            if key.target == 10 { return key.width.div_ceil(bw) as u64 * bpb as u64; }
            u32_at(b, 0x5c) as u64 * key.height.div_ceil(bh) as u64
        }
    }
}

const FULL_HASH: u64 = 64 << 10;
const WINDOWS: usize = 16;

fn window(len: u64, frame: u64) -> usize {
    if len <= FULL_HASH { 0 } else { (frame % WINDOWS as u64) as usize }
}

fn memory_hash(addr: u64, len: u64, w: usize) -> u64 {
    let mix = |h: u64, w: u64| (h ^ w).wrapping_mul(0x9E37_79B9_7F4A_7C15).rotate_left(29);
    if len <= FULL_HASH {
        return mix(len, checksum_at(addr, len));
    }
    let mut h = len ^ w as u64;
    let mut o = 256 * w as u64;
    while o < len {
        h = mix(h, checksum_at(addr + o, (len-o).min(256)));
        o += 4096;
    }
    h
}

fn checksum_at(addr: u64, len: u64) -> u64 {
    if len == 0 {
        return 0;
    }
    let data = unsafe { std::slice::from_raw_parts(addr as *const u8, len as usize) };
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("crc") {
        return unsafe { crc32_slice(data) };
    }
    let mut h = 0u64;
    for chunk in data.chunks_exact(8) {
        let w = u64::from_le_bytes(chunk.try_into().unwrap());
        h = (h ^ w).wrapping_mul(0x9E37_79B9_7F4A_7C15).rotate_left(29);
    }
    for &byte in data.chunks_exact(8).remainder() { h = (h ^ byte as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15).rotate_left(29); }
    h
}

#[cfg(test)]
mod hash_tests {
    #[test]
    fn complete_texture_checks_cover_stripes_and_the_final_partial_stripe() {
        let mut bytes=vec![0u8;(super::FULL_HASH+1)as usize];
        let original=super::memory_hashes(bytes.as_ptr()as u64,bytes.len()as u64);
        for offset in [0,255,256,4095,4096,bytes.len()-1] {
            bytes[offset]=1;
            assert_ne!(original,super::memory_hashes(bytes.as_ptr()as u64,bytes.len()as u64),"byte{offset} must affect complete checks");
            bytes[offset]=0;
        }
        bytes[256]=1;
        // A texture accessed every16 submissions repeatedly visits stripe0.
        // This negative control is the actual aliasing condition being tested.
        for frame in [16,32,48,64] {let w=super::window(bytes.len()as u64,frame);assert_eq!(w,0);assert_eq!(original[w],super::memory_hash(bytes.as_ptr()as u64,bytes.len()as u64,w));}
        assert_ne!(original,super::memory_hashes(bytes.as_ptr()as u64,bytes.len()as u64));
    }
    #[test]
    fn upload_snapshot_requires_complete_ordered_contiguous_backing() {
        assert!(super::contiguous_ranges(100,20,&[(100,7),(107,13)]));
        for ranges in [vec![],vec![(100,19)],vec![(100,21)],vec![(101,20)],vec![(100,7),(108,13)],vec![(100,7),(106,13)],vec![(100,0),(100,20)]] {
            assert!(!super::contiguous_ranges(100,20,&ranges));
        }
        assert!(!super::contiguous_ranges(u64::MAX-3,8,&[(u64::MAX-3,8)]));
    }

    #[test]
    fn explicit_cpu_flush_invalidates_hashes_outside_current_rolling_window() {
        let mut textures=super::Textures::default();
        let key=super::TexKey{addr:0x10000,format:37,target:1,width:256,height:256,depth:1,levels:1};
        textures.uploaded.insert(key,(Some([55;super::WINDOWS]),1,0x40000));
        textures.flush_cpu_memory(key.addr+8192,4);
        assert_eq!(textures.uploaded[&key],(None,u64::MAX,0x40000));
    }

    #[test]
    fn tiny_color_tables_and_tail_bytes_affect_upload_detection() {
        let a = [0u8; 13];
        for length in [1, 4, 7, 8, 13] {
            let mut b = a; b[length - 1] = 1;
            assert_ne!(super::memory_hash(a.as_ptr() as u64, length as u64, 0), super::memory_hash(b.as_ptr() as u64, length as u64, 0));
        }
    }
}

#[cfg(all(test,target_os="macos"))]
mod snapshot_gpu_tests {
    use super::*;

    #[test]
    #[ignore="requires a real Metal GPU"]
    fn gpu_virtual_texture_refreshes_cpu_writes_through_physical_backing() {
        objc2::rc::autoreleasepool(|_|unsafe {
            crate::kernel::KERNEL.aspace.get_or_init(||crate::mem::AddressSpace::reserve().unwrap());
            let allocation=0x40000u64;let cpu=crate::kernel::aspace().alias()+0x7a00000;let output=cpu+allocation;
            crate::kernel::aspace().set(cpu,allocation*2,crate::mem::Block{state:crate::mem::state::NORMAL,perm:crate::mem::perm::RW,attr:0}).unwrap();
            let physical=0xcc00;let virtual_pool=0xcc01;let out_pool=0xcc02;
            let pools=gfx::pool_memory(&[(physical,cpu,allocation),(out_pool,output,allocation)]).unwrap();
            {
                use super::super::super::objects::MemoryPool;
                let mut state=STATE.lock().unwrap();
                state.pools.insert(physical,MemoryPool{cpu,size:allocation,flags:0x22});
                state.pools.insert(virtual_pool,MemoryPool{cpu:0,size:allocation,flags:0x221});
                state.virtual_maps.insert(virtual_pool,vec![(0,allocation,physical,0)]);
                assert_ne!(state.pools[&virtual_pool].flags&1,0,"the old virtual-pool-only policy skipped tracking this resource");
            }
            let mut builder=[0u8;0x68];builder[8..16].copy_from_slice(&virtual_pool.to_le_bytes());
            for (at,value)in [(0x1c,1u32),(0x20,64),(0x24,512),(0x28,1),(0x2c,1),(0x30,37)] {builder[at..at+4].copy_from_slice(&value.to_le_bytes());}
            let key=TexKey::of(&builder).unwrap();assert_eq!(key.addr,cpu);
            let len=storage_len(&builder,&key);assert!(len>FULL_HASH && len<=allocation);
            let red=0xff0000ffu32;let green=0xff00ff00u32;
            for n in 0..len/4 {(cpu as *mut u32).add(n as usize).write(red);}
            let readback=|texture:&Texture| {
                let cb=gfx::gpu().queue.0.commandBuffer().unwrap();
                super::super::copy::copy_to_buffer(&cb,texture,(0,0),[0,0,0,64,512,1],37,output,&pools,(0,0)).unwrap();
                cb.commit();cb.waitUntilCompleted();assert_eq!(cb.status(),MTLCommandBufferStatus::Completed);
                std::slice::from_raw_parts(output as *const u32,64*512).to_vec()
            };
            let mut legacy=Textures::default();let mut tracked=Textures::default();
            for textures in [&mut legacy,&mut tracked] {
                let (_,texture)=textures.get_keyed_with_checks(Some(key),&builder,false,true).unwrap();
                assert!(readback(&texture.0).iter().all(|v|*v==red));
            }
            assert!(tracked.uploaded.contains_key(&key),"physical CPU writes must be tracked through a virtual pool");
            // Model the original virtual-pool-only policy on an independent
            // private Metal texture, then compare actual GPU pixels.
            legacy.uploaded.remove(&key);
            for (index,offset)in [256usize,512].into_iter().enumerate() {
                (cpu as *mut u32).add(offset/4).write(green);
                legacy.frame=32+index as u64*16;tracked.frame=legacy.frame;
                let (_,old)=legacy.get_keyed_with_checks(Some(key),&builder,false,true).unwrap();
                assert!(readback(&old.0).iter().all(|v|*v==red),"untracked negative control must retain stale GPU pixels");
                let (_,new)=tracked.get_keyed_with_checks(Some(key),&builder,false,true).unwrap();
                let pixels=readback(&new.0);
                assert_eq!(pixels.iter().filter(|v|**v==green).count(),index+1,"mapped CPU writes must reach the actual Metal texture");
                assert_eq!(pixels.iter().filter(|v|**v==red).count(),pixels.len()-index-1);
            }
            {
                let mut state=STATE.lock().unwrap();state.virtual_maps.remove(&virtual_pool);
                state.pools.remove(&virtual_pool);state.pools.remove(&physical);
            }
            gfx::forget_pool(physical);gfx::forget_pool(out_pool);
        });
    }

    #[test]
    #[ignore="requires a real Metal GPU"]
    fn gpu_complete_cpu_texture_checks_refresh_video_changes_outside_rolling_stripe() {
        objc2::rc::autoreleasepool(|_|unsafe {
            crate::kernel::KERNEL.aspace.get_or_init(||crate::mem::AddressSpace::reserve().unwrap());
            let allocation=0x40000u64;let cpu=crate::kernel::aspace().alias()+0x7900000;let output=cpu+allocation;
            crate::kernel::aspace().set(cpu,allocation*2,crate::mem::Block{state:crate::mem::state::NORMAL,perm:crate::mem::perm::RW,attr:0}).unwrap();
            let pool=0xcb00;let out_pool=0xcb01;
            let pools=gfx::pool_memory(&[(pool,cpu,allocation),(out_pool,output,allocation)]).unwrap();
            STATE.lock().unwrap().pools.insert(pool,super::super::super::objects::MemoryPool{cpu,size:allocation,flags:0x22});
            let mut builder=[0u8;0x68];builder[8..16].copy_from_slice(&pool.to_le_bytes());
            for (at,value)in [(0x1c,1u32),(0x20,64),(0x24,512),(0x28,1),(0x2c,1),(0x30,37)] {builder[at..at+4].copy_from_slice(&value.to_le_bytes());}
            let key=TexKey::of(&builder).unwrap();let len=storage_len(&builder,&key);assert!(len>FULL_HASH && len<=allocation);
            let red=0xff0000ffu32;let green=0xff00ff00u32;
            for n in 0..len/4 {(cpu as *mut u32).add(n as usize).write(red);}
            let readback=|texture:&Texture| {
                let cb=gfx::gpu().queue.0.commandBuffer().unwrap();
                super::super::copy::copy_to_buffer(&cb,texture,(0,0),[0,0,0,64,512,1],37,output,&pools,(0,0)).unwrap();
                cb.commit();cb.waitUntilCompleted();assert_eq!(cb.status(),MTLCommandBufferStatus::Completed);
                std::slice::from_raw_parts(output as *const u32,64*512).to_vec()
            };
            let mut textures=Textures::default();
            let (_,texture)=textures.get_keyed_with_checks(Some(key),&builder,false,false).unwrap();
            assert!(readback(&texture.0).iter().all(|v|*v==red));
            for (index,offset)in [256usize,512].into_iter().enumerate() {
                (cpu as *mut u32).add(offset/4).write(green);
                textures.frame=16+(index as u64)*32;
                let (_,texture)=textures.get_keyed_with_checks(Some(key),&builder,false,false).unwrap();
                assert_eq!(readback(&texture.0).iter().filter(|v|**v==green).count(),index,"negative control: repeated stripe0 still shows the previous image");
                textures.frame+=16;
                let (_,texture)=textures.get_keyed_with_checks(Some(key),&builder,false,true).unwrap();
                let pixels=readback(&texture.0);
                assert_eq!(pixels.iter().filter(|v|**v==green).count(),index+1,"complete checks must upload the actual new CPU pixels");
                assert!(pixels.iter().all(|v|*v==red || *v==green));
            }
            STATE.lock().unwrap().pools.remove(&pool);gfx::forget_pool(pool);gfx::forget_pool(out_pool);
        });
    }

    #[test]
    #[ignore="requires a real Metal GPU"]
    fn gpu_cpu_texture_stamp_matches_uploaded_snapshot_then_refreshes_later_writes() {
        objc2::rc::autoreleasepool(|_|unsafe {
            crate::kernel::KERNEL.aspace.get_or_init(||crate::mem::AddressSpace::reserve().unwrap());
            let page=crate::mem::PAGE;let cpu=crate::kernel::aspace().alias()+0x7800000;let output=cpu+page;
            crate::kernel::aspace().set(cpu,page*2,crate::mem::Block{state:crate::mem::state::NORMAL,perm:crate::mem::perm::RW,attr:0}).unwrap();
            std::ptr::write_bytes(cpu as *mut u8,0,page as usize);
            let pool=0xca00;let out_pool=0xca01;
            let pools=gfx::pool_memory(&[(pool,cpu,page),(out_pool,output,page)]).unwrap();
            STATE.lock().unwrap().pools.insert(pool,super::super::super::objects::MemoryPool{cpu,size:page,flags:0x22});
            let mut builder=[0u8;0x68];builder[8..16].copy_from_slice(&pool.to_le_bytes());
            for (at,value) in [(0x1c,1u32),(0x20,1),(0x24,1),(0x28,1),(0x2c,1),(0x30,37)] {builder[at..at+4].copy_from_slice(&value.to_le_bytes());}
            let key=TexKey::of(&builder).unwrap();let len=storage_len(&builder,&key);
            let red=0xff0000ffu32;let green=0xff00ff00u32;
            (cpu as *mut u32).write(red);
            let memory=UploadMemory::capture(&builder,&key,1);let expected_stamp=memory.hashes.expect("complete upload snapshot");
            // This write happens after capture but before CPU untile/GPU upload.
            // Stamping live memory after staging would falsely mark green as
            // uploaded while the actual texture still contains red.
            (cpu as *mut u32).write(green);
            assert_ne!(expected_stamp,memory_hashes(cpu,len));
            let descriptor=MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(MTLPixelFormat::RGBA8Unorm,1,1,false);
            descriptor.setStorageMode(MTLStorageMode::Private);descriptor.setUsage(MTLTextureUsage::ShaderRead|MTLTextureUsage::RenderTarget);
            let texture=Shared(gfx::gpu().device.0.newTextureWithDescriptor(&descriptor).unwrap());
            let (sources,stamp)=upload_captured(&builder,&key,&texture.0,1,None,memory);
            assert_eq!(stamp,Some(expected_stamp));
            let readback=|texture:&Texture| {
                let cb=gfx::gpu().queue.0.commandBuffer().unwrap();
                super::super::copy::copy_to_buffer(&cb,texture,(0,0),[0,0,0,1,1,1],37,output,&pools,(0,0)).unwrap();
                cb.commit();cb.waitUntilCompleted();assert_eq!(cb.status(),MTLCommandBufferStatus::Completed);
                (output as *const u32).read()
            };
            assert_eq!(readback(&texture.0),red,"GPU must receive the frozen bytes");
            let mut textures=Textures::default();textures.map.insert(key,texture);textures.sources.insert(key,sources);
            textures.uploaded.insert(key,(stamp,0,len));textures.frame=1;
            let (_,texture)=textures.get_keyed(Some(key),&builder,false).unwrap();
            assert_eq!(readback(&texture.0),green,"next access must see the later CPU write");
            assert_eq!(textures.uploaded[&key].0,Some(memory_hashes(cpu,len)));
            // A known flush must also force the source refresh in the same
            // submit rather than skipping because checked==frame.
            (cpu as *mut u32).write(red);textures.flush_cpu_memory(cpu,4);
            let (_,texture)=textures.get_keyed(Some(key),&builder,false).unwrap();
            assert_eq!(readback(&texture.0),red);
            STATE.lock().unwrap().pools.remove(&pool);gfx::forget_pool(pool);gfx::forget_pool(out_pool);
        });
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "crc")]
unsafe fn crc32_slice(data: &[u8]) -> u64 {
    use std::arch::aarch64::{__crc32cb, __crc32cd};
    unsafe {
        let (mut a, mut b, mut c, mut d) = (0u32, 1u32, 2u32, 3u32);
        let mut p = data.as_ptr();
        let mut n = data.len();
        while n >= 32 {
            let q = p.cast::<u64>();
            a = __crc32cd(a, q.read_unaligned());
            b = __crc32cd(b, q.add(1).read_unaligned());
            c = __crc32cd(c, q.add(2).read_unaligned());
            d = __crc32cd(d, q.add(3).read_unaligned());
            p = p.add(32);
            n -= 32;
        }
        while n >= 8 {
            a = __crc32cd(a, p.cast::<u64>().read_unaligned());
            p = p.add(8);
            n -= 8;
        }
        while n > 0 {
            b = __crc32cb(b, *p);
            p = p.add(1);
            n -= 1;
        }
        ((a as u64) << 32 | b as u64) ^ ((c as u64) << 32 | d as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
    }
}

fn cpu_writable(b: &TexBuilder) -> bool {
    let pool = u64::from_le_bytes(b[8..16].try_into().unwrap());
    let offset=u64::from_le_bytes(b[16..24].try_into().unwrap());
    let len=super::super::texture_builder_storage(b.as_ptr() as u64).0;
    cpu_writable_backing(&STATE.lock().unwrap(),pool,offset,len)
}

fn cpu_writable_backing(state:&super::super::objects::State,pool:u64,offset:u64,len:u64)->bool {
    let Some(p)=state.pools.get(&pool) else{return false;};
    if p.flags&1==0 {return true;}
    if p.cpu!=0 || len==0 {return false;}
    let Some(end)=offset.checked_add(len).filter(|end|*end<=p.size) else{return false;};
    // A virtual pool's CPU_NO_ACCESS bit does not describe the CPU mapping
    // permissions of its physical pages. Include writes through those pages.
    state.virtual_maps.get(&pool).is_some_and(|maps|maps.iter().any(|&(start,size,physical,_)|
        offset<start.saturating_add(size) && start<end &&
        state.pools.get(&physical).is_some_and(|backing|backing.cpu!=0 && backing.flags&1==0)))
}

#[cfg(test)]
mod virtual_writable_tests {
    use super::*;
    #[test]
    fn virtual_texture_tracking_uses_only_overlapping_cpu_writable_physical_pages() {
        use super::super::super::objects::{State,MemoryPool};
        let mut state=State::default();
        state.pools.insert(1,MemoryPool{cpu:0,size:1024,flags:0x221});
        state.pools.insert(2,MemoryPool{cpu:0x100000,size:512,flags:0x21});
        state.pools.insert(3,MemoryPool{cpu:0x200000,size:512,flags:0x22});
        state.virtual_maps.insert(1,vec![(0,256,2,0),(256,256,3,0)]);
        assert!(!cpu_writable_backing(&state,1,0,256));
        assert!(cpu_writable_backing(&state,1,256,256));
        assert!(cpu_writable_backing(&state,1,192,128));
        assert!(!cpu_writable_backing(&state,1,512,128));
        assert!(!cpu_writable_backing(&state,1,0,0));
        assert!(!cpu_writable_backing(&state,1,900,256));
        assert!(!cpu_writable_backing(&state,1,u64::MAX-16,64));
        assert!(!cpu_writable_backing(&state,99,0,64));
        assert!(!cpu_writable_backing(&state,2,0,64));
        assert!(cpu_writable_backing(&state,3,0,64));
        state.pools.remove(&3);assert!(!cpu_writable_backing(&state,1,256,256));
    }
}

fn memory_hashes(addr: u64, len: u64) -> [u64; WINDOWS] {
    let mut hashes = [0; WINDOWS];
    let n = if len <= FULL_HASH { 1 } else { WINDOWS };
    for (w, h) in hashes.iter_mut().enumerate().take(n) {
        *h = memory_hash(addr, len, w);
    }
    hashes
}

pub fn copy_blocks(cb: &gfx::CommandBuffer, src: &Texture, dst: &Texture, only: Option<(u32, u32)>) {
    let Some(enc) = cb.blitCommandEncoder() else { return };
    let cube = |t: &Texture| if matches!(t.textureType(), MTLTextureType::TypeCube | MTLTextureType::TypeCubeArray) { 6 } else { 1 };
    let slices = (src.arrayLength() * cube(src)).min(dst.arrayLength() * cube(dst));
    let bpb = match src.pixelFormat() {
        MTLPixelFormat::RGBA32Uint | MTLPixelFormat::RGBA32Sint | MTLPixelFormat::RGBA32Float => 16,
        _ => 8,
    };
    let levels = src.mipmapLevelCount().min(dst.mipmapLevelCount());
    let level_size = |level: usize| (src.width() >> level).max(1) * (src.height() >> level).max(1) * bpb * slices;
    let total: usize = (0..levels).map(level_size).sum();
    static SCRATCH: std::sync::Mutex<Option<Shared<gfx::Buffer>>> = std::sync::Mutex::new(None);
    let mut scratch = SCRATCH.lock().unwrap();
    if scratch.as_ref().is_none_or(|b| b.0.length() < total) {
        let Some(b) = gfx::gpu().device.0.newBufferWithLength_options(total.next_power_of_two(), MTLResourceOptions::StorageModePrivate) else {
            enc.endEncoding();
            return;
        };
        *scratch = Some(Shared(b));
    }
    let buf = scratch.as_ref().unwrap().0.clone();
    drop(scratch);
    let mut base = 0;
    for level in 0..levels {
        let (w, h) = ((src.width() >> level).max(1), (src.height() >> level).max(1));
        let (dw, dh) = ((dst.width() >> level).max(1), (dst.height() >> level).max(1));
        let level_base = base;
        base += level_size(level);
        if only.is_some_and(|(_, l)| l as usize != level) {
            continue;
        }
        for slice in 0..slices {
            if only.is_some_and(|(s, _)| s as usize != slice) {
                continue;
            }
            let off = level_base + slice * w * h * bpb;
            unsafe {
                enc.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toBuffer_destinationOffset_destinationBytesPerRow_destinationBytesPerImage(
                    src, slice, level, MTLOrigin { x: 0, y: 0, z: 0 }, MTLSize { width: w, height: h, depth: 1 }, &buf, off, w * bpb, w * h * bpb,
                );
                enc.copyFromBuffer_sourceOffset_sourceBytesPerRow_sourceBytesPerImage_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin(
                    &buf, off, w * bpb, w * h * bpb, MTLSize { width: dw, height: dh, depth: 1 }, dst, slice, level, MTLOrigin { x: 0, y: 0, z: 0 },
                );
            }
        }
    }
    enc.endEncoding();
}
