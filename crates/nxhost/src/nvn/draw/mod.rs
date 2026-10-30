// Mush8 - mod.rs
// Constantin Clerc (c22dev) - 0.1
mod copy;
mod bindless;
mod exec;
mod shaders;
mod dynamic_vertex;
mod vertex_formats;
mod state;
mod textures;
mod cpu_events;
mod shader_writes;
mod write_windows;
mod fault_records;
mod gpu_counters;
#[cfg(test)]
mod gpu_counter_tests;
#[cfg(test)]
mod fault_record_tests;
// Capture-derived write regression fixtures remain private; see PROVENANCE.json.
#[cfg(test)]
mod transfer_wait_tests;
mod address_table;
mod buffer_direct;
mod gpu_wait;
mod submit_profile;
#[cfg(all(test,target_os="macos"))]
mod structured_tests;
#[cfg(all(test,target_os="macos"))]
mod mufu_tests;
#[cfg(all(test,target_os="macos"))]
mod memory_load_tests;
#[cfg(all(test,target_os="macos"))]
mod buffer_read_tests;
#[cfg(all(test,target_os="macos"))]
mod pool_lifetime_tests;
#[cfg(all(test,target_os="macos"))]
mod page_cache_tests;

use super::cmd::{self, Cmd, Draw, TexBuilder};
use super::objects::STATE;
use crate::gfx::{self, Attachments, Load, RenderEncoder, Shared};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::*;
use state::{Arena, GState, States, Targets};
use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicBool, Ordering::Relaxed};
use std::sync::{LazyLock, Mutex};

pub(crate) static TRACE_DRAWS: AtomicBool = AtomicBool::new(false);
static DUMP_NEXT: AtomicBool = AtomicBool::new(false);
static UPPER_LEFT: AtomicBool = AtomicBool::new(false);
static ZERO_TO_ONE: AtomicBool = AtomicBool::new(false);

pub fn set_origin_upper_left(v: bool) {
    UPPER_LEFT.store(v, Relaxed);
}
pub fn set_depth_zero_to_one(v: bool) {
    ZERO_TO_ONE.store(v, Relaxed);
}

#[derive(Default)]
struct Engine {
    st: GState,
    textures: textures::Textures,
    shaders: shaders::Shaders,
    states: States,
    arena: Arena,
    syncs: HashMap<u64, SyncPoint>,
    presented: HashMap<u64, Shared<gfx::CommandBuffer>>,
    last_submit: Option<Shared<gfx::CommandBuffer>>,
    colors: Vec<Option<Box<TexBuilder>>>,
    depth: Option<Box<TexBuilder>>,
    overlays: BTreeMap<u64, (u64, u64, usize)>,
    target_first_frame: HashMap<textures::TexKey, u64>,
    copier: copy::Copier,
    kernels: HashMap<u64, Shared<ProtocolObject<dyn MTLComputePipelineState>>>,
    bank_tex: [[u64; 32]; 3],
    bank_gpu: [u64; 3],
    bank_ok: [bool; 3],
    sampled: HashMap<textures::TexKey, Shared<gfx::Texture>>,
    touched: Vec<(textures::TexKey, Shared<gfx::Texture>)>,
    clears: Vec<PendingClear>,
    draws: u32,
    skipped: u32,
    held: Held,
    fill_pipeline: Option<Shared<ProtocolObject<dyn MTLComputePipelineState>>>,
    event_pipeline: Option<Shared<ProtocolObject<dyn MTLComputePipelineState>>>,
    debug_groups: Vec<(u32,String)>,
    resource_history: std::collections::VecDeque<String>,
    // Pending GPU writes into pool buffers may be read by CPU texture uploads.
    // Keep this across submits until a queue completion makes them visible.
    gpu_buffer_dirty:bool,
    window_batch_writes:bool,
    cpu_events:cpu_events::CpuEvents,
    address_table:address_table::Cache,
}

#[derive(Default)]
struct Held {
    last: bool,
}

#[derive(Clone)]
enum SyncPoint { Initialized, Ready, Fenced(Shared<gfx::CommandBuffer>) }

fn wait_sync_point(point:Option<SyncPoint>,timeout_ns:u64) -> u64 {
    let cb=match point {Some(SyncPoint::Ready)=>return 0,Some(SyncPoint::Fenced(cb))=>cb,_=>return 3};
    match cb.0.status() {MTLCommandBufferStatus::Completed=>return 0,MTLCommandBufferStatus::Error=>return 3,_=>{}}
    if timeout_ns==0 {return 2;}
    crate::kernel::blocking(|| {
        if timeout_ns==u64::MAX {cb.0.waitUntilCompleted();}
        else {
            let start=std::time::Instant::now();let limit=std::time::Duration::from_nanos(timeout_ns);
            loop {
                match cb.0.status() {MTLCommandBufferStatus::Completed=>return 1,MTLCommandBufferStatus::Error=>return 3,_=>{}}
                let elapsed=start.elapsed();if elapsed>=limit {return 2;}
                std::thread::sleep((limit-elapsed).min(std::time::Duration::from_micros(50)));
            }
        }
        if cb.0.status()==MTLCommandBufferStatus::Completed {1} else {3}
    })
}

impl Held {
    fn present(&mut self, skipped: u32, draws: u32) -> bool {
        self.last = !self.last && skipped * 50 >= draws.max(1);
        !self.last
    }
}

struct PendingClear {
    key: textures::TexKey,
    sub: (u32, u32),
    tex: Shared<gfx::Texture>,
    load: Load,
    stencil: bool,
}

impl PendingClear {
    fn is_depth(&self) -> bool {
        matches!(self.load, Load::ClearDepthStencil(..))
    }
}

impl Engine {
    fn forget_dead(&mut self) {
        for (addr, size) in std::mem::take(&mut *FLUSHED.lock().unwrap()) { self.textures.flush_cpu_memory(addr, size); }
        let dead = std::mem::take(&mut *DEAD.lock().unwrap());
        if !dead.is_empty() {
            let gone = |k: &textures::TexKey| dead.iter().any(|&(lo, hi)| k.addr >= lo && k.addr < hi);
            self.textures.evict(&gone);
            self.target_first_frame.retain(|k, _| !gone(k));
        }
    }
}

static ENGINE: LazyLock<Mutex<Engine>> = LazyLock::new(Default::default);

fn engine() -> std::sync::MutexGuard<'static, Engine> {
    // A host mutex may block while another replay waits for its GPU work.
    // Waiting callers must release their logical guest scheduling slot.
    if gpu_wait::enabled() {
        // Keep the cooperative priority/affinity checkpoint before locking;
        // an uncontended host mutex does not itself require blocking the guest.
        crate::kernel::sched_point();
        match ENGINE.try_lock() {
            Ok(guard)=>return guard,
            Err(std::sync::TryLockError::WouldBlock)=>{},
            Err(std::sync::TryLockError::Poisoned(_))=>panic!("NVN engine mutex poisoned"),
        }
    }
    crate::kernel::blocking(|| ENGINE.lock().unwrap())
}

struct Pass {
    enc: Retained<RenderEncoder>,
    counter:Option<gfx::Shared<ProtocolObject<dyn MTLCounterSampleBuffer>>>,
    targets: Targets,
    size: (u32, u32),
    heap: u64,
    chunks: usize,
    keys: Vec<(textures::TexKey, (u32, u32))>,
    fresh: bool,
    fixed: Option<EncState>,
    profile_context_reported:bool,
}

#[derive(Clone, Copy)]
struct EncState {
    pipeline: usize,
    depth: usize,
    cull: u32,
    winding: u32,
    fill: u32,
    bias: [u32; 3],
    depth_clamp: bool,
    origin: [u32; 4],
    zrange: [u32; 2],
    scissor: [u32; 4],
    stencil: [u32; 2],
    blend: [u32; 4],
}

struct Replay<'a> {
    e: &'a mut Engine,
    cb: Retained<gfx::CommandBuffer>,
    pass: Option<Pass>,
    heap: Option<&'static gfx::HeapBuffer>,
    pools: Vec<std::sync::Arc<gfx::PoolMemory>>,
    passes: u32,
    copies: HashMap<(u64, u64), u64>,
    fault:Option<state::Alloc>,
    pool_table:Option<[u64;6]>,
    resource_buffers:std::cell::OnceCell<Vec<std::ptr::NonNull<ProtocolObject<dyn MTLResource>>>>,
}

impl Replay<'_> {
    fn pool_resources(&self)->&[std::ptr::NonNull<ProtocolObject<dyn MTLResource>>] {
        self.resource_buffers.get_or_init(|| {
            let mut seen=std::collections::HashSet::new();
            let out=self.pools.iter().flat_map(|p|&p.chunks).filter(|c|seen.insert(Retained::as_ptr(&c.buffer.0) as usize)).map(|c|std::ptr::NonNull::from(ProtocolObject::from_ref(&*c.buffer.0))).collect::<Vec<_>>();
            if TRACE_DRAWS.load(Relaxed) {eprintln!("[frame] indirect buffers: {} address spans, {} unique Metal buffers",self.pools.iter().map(|p|p.chunks.len()).sum::<usize>(),out.len());}
            out
        })
    }
    fn run(&mut self, cmds: &[Cmd], depth: usize) {
        for c in cmds {
            static TRACE:LazyLock<bool>=LazyLock::new(||std::env::var_os("NX_TRACE_BINDINGS").is_some());
            if *TRACE {
                let entry=match c {
                    Cmd::Program{stages,mask}=>Some(format!("program mask{mask:x} {stages:x?}")),
                    Cmd::Image{stage,index,handle,origin}=>Some(format!("image stage{stage} slot{index} handle{handle:x} origin{origin:?}")),
                    Cmd::TexturePool(pool)=>Some(format!("texture pool{pool:x}")),
                    Cmd::Dispatch(groups)=>Some(format!("dispatch {groups:?}")),
                    Cmd::DispatchIndirect(address)=>Some(format!("dispatch indirect{address:x}")),
                    Cmd::Call(handles)=>Some(format!("call {handles:x?}")),
                    Cmd::RenderEnable(enable)=>Some(format!("render enable{enable}")),
                    _=>None,
                };
                if let Some(entry)=entry {
                    if self.e.resource_history.len()==128 {self.e.resource_history.pop_front();}
                    self.e.resource_history.push_back(format!("depth{depth} {entry}"));
                }
            }
            let st = &mut self.e.st;
            match c {
                Cmd::SetRenderTargets { colors, depth } => {
                    if self.pass.is_some() && self.e.colors == *colors && self.e.depth == *depth {
                        continue;
                    }
                    self.end_pass();
                    self.e.colors = colors.clone();
                    self.e.depth = depth.clone();
                }
                Cmd::ClearColor { index, rgba } => {
                    if !st.render_enable {
                        continue;
                    }
                    self.end_pass();
                    let Some(Some(b)) = self.e.colors.get(*index as usize).cloned() else { continue };
                    if textures::is_compressed(u32::from_le_bytes(b[0x30..0x34].try_into().unwrap())) {
                        continue;
                    }
                    let Some((key, t)) = self.texture(&b, true) else { continue };
                    if TRACE_DRAWS.load(Relaxed) {
                        eprintln!("[frame] clear color rt{index} {key:?} = {rgba:?}");
                    }
                    let c = rgba.map(|x| x as f64);
                    let sub = super::objects::view_subresource(&b);
                    self.defer_clear(PendingClear { key, sub, tex: t, load: Load::ClearColor(c), stencil: false });
                }
                Cmd::ClearDepthStencil { depth: z, stencil } => {
                    if !st.render_enable {
                        continue;
                    }
                    self.end_pass();
                    let Some(b) = self.e.depth.clone() else { continue };
                    let fmt = u32::from_le_bytes(b[0x30..0x34].try_into().unwrap());
                    let Some((key, t)) = self.texture(&b, true) else { continue };
                    let sub = super::objects::view_subresource(&b);
                    self.defer_clear(PendingClear { key, sub, tex: t, load: Load::ClearDepthStencil(*z as f64, *stencil), stencil: gfx::has_stencil(fmt) });
                }
                Cmd::Call(handles) => {
                    if depth < 8 {
                        for h in handles {
                            if let Some(sub) = cmd::recording(*h) {
                                self.run(&sub, depth + 1);
                            }
                        }
                    }
                }
                Cmd::Program {stages,mask} => st.bind_program(stages,*mask).unwrap_or_else(|e|crate::diag::report_and_exit(e)),
                Cmd::VertexAttribs(v) => {
                    for (i, w) in v.iter().take(16).enumerate() {
                        st.attribs[i] = *w;
                    }
                }
                Cmd::VertexStreams(v) => {
                    for (i, s) in v.iter().take(16).enumerate() {
                        st.streams[i] = *s;
                    }
                }
                Cmd::VertexBuffer { index, addr, size } => {
                    if let Some(v) = st.vbufs.get_mut(*index as usize) {
                        *v = (*addr, *size);
                    }
                }
                Cmd::Uniform { stage, index, addr, size } => {
                    if let Some(u) = stage_index(*stage).and_then(|s| st.ubos[s].get_mut(*index as usize)) {
                        *u = (*addr, *size);
                    }
                }
                Cmd::Texture { stage, index, handle } => {
                    if let Some(t) = stage_index(*stage).and_then(|s| st.textures[s].get_mut(*index as usize)) {
                        *t = *handle;
                    }
                }
                Cmd::Image{stage,index,handle,..} => {
                    if let Some(t)=stage_index(*stage).and_then(|s|st.images[s].get_mut(*index as usize)) {*t=*handle;self.e.bank_ok[stage_index(*stage).unwrap()]=false;}
                    else {crate::diag::report_and_exit("NVN image binding is outside supported stage/range");}
                }
                Cmd::SeparateTexture{stage,index,handle}|Cmd::SeparateSampler{stage,index,handle} => {
                    let s=stage_index(*stage).unwrap_or_else(||crate::diag::report_and_exit("unsupported NVN separate binding stage"));
                    let array=if matches!(c,Cmd::SeparateTexture{..}) {&mut st.separate_textures[s]} else {&mut st.separate_samplers[s]};
                    let slot=array.get_mut(*index as usize).unwrap_or_else(||crate::diag::report_and_exit("unsupported NVN separate binding index"));
                    *slot=*handle;self.e.bank_ok[s]=false;
                }
                Cmd::Storage { stage, index, addr, size } => {
                    if let Some(u) = stage_index(*stage).and_then(|s| st.storage[s].get_mut(*index as usize)) {
                        *u = (*addr, *size);
                    }
                }
                Cmd::UpdateUniform { addr, size, offset, data } => self.update_uniform(*addr, *size, *offset, data),
                Cmd::ColorState(v) => st.color_state = *v,
                Cmd::ChannelMask(v) => st.channel_mask = *v,
                Cmd::Blend(v) => st.blend[(*v & 7) as usize] = *v,
                Cmd::DepthStencil(v) => st.depth_stencil = *v,
                Cmd::Polygon(v) => st.polygon = *v,
                Cmd::Viewport(v) => st.viewport = Some(*v),
                Cmd::Scissor(v) => st.scissor = Some(*v),
                Cmd::BlendColor(v) => st.blend_color = *v,
                Cmd::AlphaRef(v) => st.alpha_ref = *v,
                Cmd::Multisample(v) => st.multisample = *v,
                Cmd::SampleMask(v) => st.sample_mask = *v,
                // Metal owns thread-private shader backing. Preserve and
                // validate the NVN binding without treating Tegra scratch as
                // a user-visible storage buffer in translated shaders.
                Cmd::ScratchMemory{pool,offset,size}=>st.scratch_memory=(*pool,*offset,*size),
                Cmd::DebugAnnotation{action,domain,label} => {
                    match action {
                        0=>{if self.e.debug_groups.len()<256 {self.e.debug_groups.push((*domain,label.clone()));}},
                        1=>{if let Some(i)=self.e.debug_groups.iter().rposition(|g|g.0==*domain) {self.e.debug_groups.remove(i);}},
                        _=>{},
                    }
                    let mut parts=self.e.debug_groups.iter().map(|g|g.1.as_str()).collect::<Vec<_>>();
                    if *action==2 {parts.push(label);}
                    self.cb.setLabel(Some(&objc2_foundation::NSString::from_str(&parts.join(" / "))));
                }
                Cmd::StencilRef(f, v) => set_faces(&mut st.stencil_ref, *f, *v),
                Cmd::StencilValueMask(f, v) => set_faces(&mut st.stencil_read, *f, *v),
                Cmd::StencilMask(f, v) => set_faces(&mut st.stencil_write, *f, *v),
                Cmd::TexturePool(p) => st.texture_pool = *p,
                Cmd::SamplerPool(p) => st.sampler_pool = *p,
                Cmd::DepthClamp(v) => st.depth_clamp = *v,
                Cmd::RasterizerDiscard(v) => st.rasterizer_discard = *v,
                Cmd::ReportCounter { counter, addr } => {
                    if !matches!(*counter, 0 | 16) { crate::diag::report_and_exit(&format!("NVN counter type {counter} has no Metal implementation")); }
                    let handled=self.gpu_report_counter(*counter,*addr).unwrap_or_else(|e|crate::diag::report_and_exit(e));
                    if !handled {
                        if submit_profile::active() {eprintln!("[cpu-wait-reason] counter type={counter} destination={addr:#x}");}
                        self.commit_boundary(true);
                        let ns = super::objects::timestamp_ns();
                        if *addr != 0 {let data=[ns.to_le_bytes(),ns.to_le_bytes()].concat();self.write_cpu_bytes(*addr,&data).unwrap_or_else(|e|crate::diag::report_and_exit(e));}
                    }
                }
                Cmd::CopyBuffer { src, dst, size } => {
                    self.end_pass(); self.flush_clears();
                    self.copy_buffer(*src, *dst, *size).unwrap_or_else(|e| crate::diag::report_and_exit(&e));
                }
                Cmd::ClearBuffer { dst, size, value } => {
                    self.end_pass(); self.flush_clears();
                    self.clear_buffer(*dst, *size, *value).unwrap_or_else(|e| crate::diag::report_and_exit(e));
                }
                Cmd::Barrier(bits) => {
                    if TRACE_DRAWS.load(Relaxed) {eprintln!("[frame] barrier {bits:#x} shared_dirty={}",self.e.gpu_buffer_dirty);}
                    if submit_profile::active() && bits&0x10!=0 && self.e.gpu_buffer_dirty {eprintln!("[cpu-write] barrier {bits:#x} waits for unknown/shared writer");}
                    self.end_pass(); self.flush_clears();
                    // Encoder boundaries and tracked Metal textures order GPU
                    // image writes/sampling. A CPU wait is only required when
                    // shared pool buffers may feed the CPU upload path.
                    if bits & 0x10 != 0 && self.e.gpu_buffer_dirty { self.commit_boundary(true); }
                    else if bits & 0x10 != 0 && self.e.window_batch_writes {self.commit_boundary(false);}
                }
                Cmd::FenceSync(sync) => {
                    let cb = self.commit_boundary(false);
                    if !self.e.syncs.contains_key(sync) {crate::diag::report_and_exit("NVN FenceSync uses an uninitialized sync");}
                    self.e.syncs.insert(*sync, SyncPoint::Fenced(Shared(cb)));
                }
                Cmd::WaitSync(sync) => {
                    if wait_sync_point(self.e.syncs.get(sync).cloned(),u64::MAX)==3 {crate::diag::report_and_exit(&format!("NVN WaitSync has invalid or unfenced sync {sync:#x}"));}
                }
                Cmd::SignalEvent{address,mode,value} => {
                    if TRACE_DRAWS.load(Relaxed) {eprintln!("[frame] signal event {address:#x} mode{mode} value{value}");}
                    self.end_pass(); self.flush_clears();
                    self.signal_event(*address,*mode,*value).unwrap_or_else(|e|crate::diag::report_and_exit(e));
                }
                Cmd::WaitEvent{address,mode,value} => {
                    if submit_profile::active() {eprintln!("[cpu-wait-reason] event address={address:#x} mode={mode} value={value}");}
                    self.commit_boundary(true);
                    super::events::wait_memory(*address,*mode,*value).unwrap_or_else(|e|crate::diag::report_and_exit(e));
                }
                Cmd::DepthRange(n, f) => st.depth_range = (*n, *f),
                Cmd::PolygonOffset(v) => st.polygon_offset = *v,
                Cmd::CopyBufferToTexture { src, dst, region, strides, flags } => {
                    if submit_profile::active() {eprintln!("[texture-flow] buffer={src:#x} destination={:?} region={region:?}",textures::TexKey::of(dst));}
                    if *flags != 0 { crate::diag::report_and_exit(&format!("NVN buffer-to-texture copy flags {flags:#x} are unsupported")); }
                    self.end_pass();
                    self.flush_clears();
                    self.copy_buffer_to_texture(*src, dst, *region, *strides).unwrap_or_else(|e| crate::diag::report_and_exit(e));
                }
                Cmd::CopyTextureToTexture { src, src_region, dst, dst_region, flags } => {
                    if submit_profile::active() {eprintln!("[texture-flow] source={:?} destination={:?} source_region={src_region:?} destination_region={dst_region:?}",textures::TexKey::of(src),textures::TexKey::of(dst));}
                    self.end_pass();
                    self.flush_clears();
                    let r = match (self.texture(src, false), self.texture(dst, true)) {
                        (Some((_, s)), Some((_, d))) => {
                            let (ss, ds) = (super::objects::view_subresource(src), super::objects::view_subresource(dst));
                            self.e.copier.copy(&self.cb, &s, ss, *src_region, &d, ds, *dst_region, *flags)
                        }
                        _ => Err("copy of an unavailable texture"),
                    };
                    r.unwrap_or_else(|e| crate::diag::report_and_exit(&format!("{e}: src {:?} sub {:?} region {src_region:?}; dst {:?} sub {:?} region {dst_region:?}; flags {flags:#x}",textures::TexKey::of(src),super::objects::view_subresource(src),textures::TexKey::of(dst),super::objects::view_subresource(dst))));
                }
                Cmd::CopyTextureToBuffer { src, region, dst, strides, flags } => {
                    if *flags != 0 { crate::diag::report_and_exit(&format!("NVN texture-to-buffer copy flags {flags:#x} are unsupported")); }
                    self.end_pass();
                    self.flush_clears();
                    let format = u32::from_le_bytes(src[0x30..0x34].try_into().unwrap());
                    let r = match self.texture(src, false) {
                        Some((_, s)) => copy::copy_to_buffer(&self.cb, &s.0, super::objects::view_subresource(src), *region, format, *dst, &self.pools, *strides),
                        None => Err("copy of an unavailable texture"),
                    };
                    let writes=r.unwrap_or_else(|e| crate::diag::report_and_exit(e));
                    for (gpu,cpu,len) in writes {self.record_transfer_write(gpu,cpu,len);}
                    if submit_profile::active() {eprintln!("[cpu-write] texture-copy ->{dst:#x} region{region:?}");}
                }
                Cmd::ClearTexture { dst, region, color, mask } => {
                    if !st.render_enable {
                        continue;
                    }
                    self.end_pass();
                    self.flush_clears();
                    let r = match self.texture(dst, true) {
                        Some((_, d)) => self.e.copier.clear(&self.cb, &d, super::objects::view_subresource(dst), *region, *color, *mask),
                        None => Err("clear of an unavailable texture"),
                    };
                    r.unwrap_or_else(|e| crate::diag::report_and_exit(e));
                }
                Cmd::RenderEnable(v) => st.render_enable = *v,
                Cmd::Dispatch(groups) => {
                    self.end_pass();
                    self.flush_clears();
                    let r = self.dispatch(*groups);
                    r.unwrap_or_else(|e| crate::diag::report_and_exit(e));
                }
                Cmd::DispatchIndirect(address)=>{
                    self.end_pass();self.flush_clears();
                    self.dispatch_indirect(*address).unwrap_or_else(|e|crate::diag::report_and_exit(e));
                }
                Cmd::Draw(d) => {
                    self.e.draws += 1;
                    if let Err(error) = self.draw(d) {
                        if submit_profile::active() {eprintln!("[draw-reason] error={error} count={} instances={}",d.count,d.instances);}
                        if error.ends_with("compiling") { self.e.skipped += 1; }
                        else { crate::diag::report_and_exit(error); }
                    }
                }
                Cmd::DrawIndirect{mode,index,arguments}=>{
                    self.e.draws+=1;
                    let d=Draw{mode:*mode,index:*index,first:0,count:1,base_vertex:0,base_instance:0,instances:1};
                    if let Err(error)=self.draw_indirect(&d,*arguments) {
                        if submit_profile::active() {eprintln!("[draw-reason] indirect error={error} arguments={arguments:#x}");}
                        if error.ends_with("compiling") {self.e.skipped+=1;} else {crate::diag::report_and_exit(error);}
                    }
                }
            }
        }
    }

    fn commit_boundary(&mut self, wait: bool) -> Retained<gfx::CommandBuffer> {
        self.end_pass(); self.flush_clears();
        let next = gfx::gpu().queue.0.commandBuffer().expect("Metal command buffer");
        let completed = std::mem::replace(&mut self.cb, next);
        self.fault=None;
        self.pool_table=None;
        commit(&completed, &self.pools);
        self.e.window_batch_writes=false;
        if wait {
            let timed=submit_profile::active() || (TRACE_DRAWS.load(Relaxed) && std::path::Path::new("work/profile-cpu-waits").exists());
            let start=timed.then(std::time::Instant::now);
            let handoff=gpu_wait::complete(&completed);self.e.gpu_buffer_dirty=false;
            self.e.cpu_events.clear();
            if let Some(start)=start {eprintln!("[cpu-wait] submit {:.3}ms handoff{handoff}",start.elapsed().as_secs_f64()*1000.0);}
        }
        completed
    }

    fn fault_record(&mut self)->u64 {
        if let Some(a)=&self.fault {return a.gpu;}
        let isolated=fault_records::enabled();
        let a=if isolated {fault_records::allocate()} else {self.e.arena.alloc(32)};
        unsafe {std::ptr::write_bytes(a.ptr,0,32);}
        self.use_chunk(&a.buffer);
        let keep=a.buffer.clone();let offset=a.offset;
        let handler=block2::RcBlock::new(move |_:std::ptr::NonNull<gfx::CommandBuffer>| {
            let record=unsafe {std::slice::from_raw_parts((keep.0.contents().as_ptr() as *const u8).add(offset).cast::<u32>(),8)};
            if record[0]!=0 {
                let hash=record[2] as u64|((record[3] as u64)<<32);
                crate::diag::report_and_exit(&format!("translated shader {hash:016x} fault {:#x} at instruction {:#x}, context {:x?}",record[0],record[1],&record[4..]));
            }
            if isolated {fault_records::recycle(keep.clone());}
        });
        unsafe {self.cb.addCompletedHandler(block2::RcBlock::as_ptr(&handler));}
        let gpu=a.gpu;self.fault=Some(a);gpu
    }

    fn copy_mapping_error(&self,src:u64,dst:u64,size:u64,offset:u64,source:bool)->String {
        let address=if source {src+offset} else {dst+offset};
        let mut nearest=self.pools.iter().flat_map(|p|&p.chunks).map(|c| {
            let distance=if address<c.base {c.base-address} else {address.saturating_sub(c.base+c.size)};
            (distance,c.base,c.size,c.offset as u64)
        }).collect::<Vec<_>>();nearest.sort_unstable();nearest.truncate(3);
        let (current_pools,aliases,virtual_pools)={
            let s=STATE.lock().unwrap();
            let current=s.pools.iter().filter(|(_,p)|p.cpu!=0 && address>=p.cpu && address-p.cpu<p.size).take(4).map(|(&id,p)|(id,p.cpu,p.size,p.flags)).collect::<Vec<_>>();
            let aliases=s.gpu_aliases().into_iter().filter(|(base,_,len)|address>=*base && address-*base<*len).take(4).collect::<Vec<_>>();
            let virtuals=s.virtual_bases.iter().filter_map(|(&id,&base)|s.pools.get(&id).filter(|p|address>=base && address-base<p.size).map(|p|(id,base,p.size))).take(4).collect::<Vec<_>>();
            (current,aliases,virtuals)
        };
        let heap=self.heap.map(|h|(h.base,h.size));let current_heap=gfx::heap().map(|h|(h.base,h.size));
        format!("NVN copy {} is not mapped to Metal: src{src:#x} dst{dst:#x} size{size:#x} offset{offset:#x}; nearest(distance,base,size,offset){nearest:x?}; snapshotHeap{heap:x?} currentHeap{current_heap:x?}; currentPools{current_pools:x?} aliases{aliases:x?} virtualPools{virtual_pools:x?}",if source {"source"} else {"destination"})
    }

    fn copy_buffer(&mut self, src: u64, dst: u64, size: u64) -> Result<(), String> {
        if size == 0 { return Ok(()); }
        let src_end = src.checked_add(size).ok_or("NVN buffer copy source overflow")?;
        let dst_end = dst.checked_add(size).ok_or("NVN buffer copy destination overflow")?;
        if src < dst_end && dst < src_end { return Err("overlapping NVN buffer copy".into()); }
        // Validate the whole transfer before encoding any portion. An alias
        // hole must not leave an active encoder or partially modified prefix.
        let mut pieces=Vec::new();
        let mut offset = 0;
        while offset < size {
            let (s, so) = self.buffer_range(src + offset, 1).ok_or_else(||self.copy_mapping_error(src,dst,size,offset,true))?;
            let (d, doff) = self.buffer_range(dst + offset, 1).ok_or_else(||self.copy_mapping_error(src,dst,size,offset,false))?;
            let len = (size - offset).min(s.base + s.size - src - offset).min(d.base + d.size - dst - offset);
            pieces.push((s.buffer.clone(),so,d.buffer.clone(),doff,len as usize));
            offset += len;
        }
        let enc = self.cb.blitCommandEncoder().ok_or("Metal buffer blit encoder")?;
        let mut writes=Vec::with_capacity(pieces.len());let mut offset=0u64;
        for (source,so,destination,doff,len) in pieces {
            unsafe {enc.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(&source.0,so,&destination.0,doff,len);}
            writes.push((dst+offset,destination.0.contents().as_ptr() as u64+doff as u64,len as u64));offset+=len as u64;
        }
        enc.endEncoding();
        for (gpu,cpu,len) in writes {self.record_transfer_write(gpu,cpu,len);}
        if submit_profile::active() {eprintln!("[cpu-write] copy {src:#x}->{dst:#x}+{size:#x}");}
        Ok(())
    }

    fn clear_buffer(&mut self, dst: u64, size: u64, value: u32) -> Result<(), &'static str> {
        if size == 0 { return Ok(()); }
        if dst % 4 != 0 || size % 4 != 0 { return Err("NVN ClearBuffer requires 32-bit alignment"); }
        if self.e.fill_pipeline.is_none() {
            let source = "#include <metal_stdlib>\nusing namespace metal; kernel void fill32(device uint* b [[buffer(0)]], constant uint2& p [[buffer(1)]], uint i [[thread_position_in_grid]]) { if (i < p.y) b[i] = p.x; }";
            let lib = gfx::gpu().device.0.newLibraryWithSource_options_error(&objc2_foundation::NSString::from_str(source), None).map_err(|_| "buffer fill shader")?;
            let f = lib.newFunctionWithName(&objc2_foundation::NSString::from_str("fill32")).ok_or("buffer fill function")?;
            self.e.fill_pipeline = Some(Shared(gfx::gpu().device.0.newComputePipelineStateWithFunction_error(&f).map_err(|_| "buffer fill pipeline")?));
        }
        let enc = self.cb.computeCommandEncoder().ok_or("Metal buffer fill encoder")?;
        enc.setComputePipelineState(&self.e.fill_pipeline.as_ref().unwrap().0);
        let mut offset = 0;let mut writes=Vec::new();
        while offset < size {
            let (d, doff) = self.buffer_range(dst + offset, 1).ok_or("NVN fill destination is not mapped to Metal")?;
            let len = (size - offset).min(d.base + d.size - dst - offset);
            let p = [value, (len / 4) as u32];
            unsafe {
                enc.setBuffer_offset_atIndex(Some(&d.buffer.0), doff, 0);
                enc.setBytes_length_atIndex(std::ptr::NonNull::from(&p).cast(), size_of_val(&p), 1);
            }
            enc.dispatchThreads_threadsPerThreadgroup(MTLSize { width: p[1] as usize, height: 1, depth: 1 }, MTLSize { width: 64, height: 1, depth: 1 });
            writes.push((dst+offset,d.buffer.0.contents().as_ptr() as u64+doff as u64,len));
            offset += len;
        }
        enc.endEncoding();
        for (gpu,cpu,len) in writes {self.record_transfer_write(gpu,cpu,len);}
        if submit_profile::active() {eprintln!("[cpu-write] clear {dst:#x}+{size:#x}");}
        Ok(())
    }

    fn signal_event(&mut self,address:u64,mode:u32,value:u32) -> Result<(), &'static str> {
        if mode>1 {return Err("unsupported NVN GPU event signal mode");}
        if self.e.event_pipeline.is_none() {
            let source = "#include <metal_stdlib>\nusing namespace metal; kernel void event32(device atomic_uint* b [[buffer(0)]], constant uint2& p [[buffer(1)]]) { if (p.x==0) atomic_store_explicit(b,p.y,memory_order_relaxed); else atomic_fetch_add_explicit(b,p.y,memory_order_relaxed); }";
            let lib = gfx::gpu().device.0.newLibraryWithSource_options_error(&objc2_foundation::NSString::from_str(source),None).map_err(|_|"event signal shader")?;
            let f = lib.newFunctionWithName(&objc2_foundation::NSString::from_str("event32")).ok_or("event signal function")?;
            self.e.event_pipeline = Some(Shared(gfx::gpu().device.0.newComputePipelineStateWithFunction_error(&f).map_err(|_|"event signal pipeline")?));
        }
        let (chunk,off) = self.buffer_range(address,4).ok_or("NVN GPU event storage is not mapped to Metal")?;
        let enc = self.cb.computeCommandEncoder().ok_or("Metal event signal encoder")?;
        enc.setComputePipelineState(&self.e.event_pipeline.as_ref().unwrap().0);
        let args = [mode,value];
        unsafe {enc.setBuffer_offset_atIndex(Some(&chunk.buffer.0),off,0);enc.setBytes_length_atIndex(std::ptr::NonNull::from(&args).cast(),size_of_val(&args),1);}
        enc.dispatchThreads_threadsPerThreadgroup(MTLSize{width:1,height:1,depth:1},MTLSize{width:1,height:1,depth:1});
        let cpu=chunk.buffer.0.contents().as_ptr() as u64+off as u64;
        enc.endEncoding(); self.record_cpu_event(address,cpu); Ok(())
    }

    fn copy_buffer_to_texture(&mut self, src: u64, b: &TexBuilder, region: [i32; 6], strides: (u32, u32)) -> Result<(), &'static str> {
        let (key, tex) = self.texture(b, true).ok_or("buffer copy destination texture unavailable")?;
        let [x, y, z, w, h, d] = region.map(|v| v.max(0) as usize);
        if w == 0 || h == 0 {
            return Ok(());
        }
        let (layer, level) = super::objects::view_subresource(b);
        let (mut row, mut image, len) = copy::buffer_layout(w, h, d, key.format, strides)?;
        if key.target==2 && key.format==46 && std::env::var_os("NX_PROBE_COMPUTE_HASH").is_some() {
            static COUNT:std::sync::atomic::AtomicU32=std::sync::atomic::AtomicU32::new(0);
            let n=COUNT.fetch_add(1,Relaxed);
            if n<8 && len<=4*1024*1024 {
                self.commit_boundary(true);
                let _=std::fs::create_dir_all("work/volume-upload-probes");
                if let Ok(data)=self.cpu_bytes(src,len) {let _=std::fs::write(format!("work/volume-upload-probes/{n}.source"),data);}
                let _=std::fs::write(format!("work/volume-upload-probes/{n}.builder"),b);
                let _=std::fs::write(format!("work/volume-upload-probes/{n}.txt"),format!("key{key:?} source{src:#x} region{region:?} strides{strides:?} row{row} image{image} len{len}"));
            }
        }
        let layered = matches!(tex.0.textureType(), MTLTextureType::Type2DArray | MTLTextureType::TypeCube | MTLTextureType::TypeCubeArray);
        let (buf, off) = if let Some(kind) = gfx::bc::soft(key.format) {
            // The destination Metal texture is uncompressed on devices without
            // BC support. Complete GPU writes before CPU decoding source blocks.
            self.commit_boundary(true);
            let source = self.cpu_bytes(src,len)?;
            let mut decoded = Vec::new(); let mut decoded_row = 0;
            for i in 0..d.max(1) {
                let (bytes,pitch) = gfx::bc::decode(kind,&source[i*image..],row,w,h,1);
                decoded_row = pitch; decoded.extend_from_slice(&bytes);
            }
            row = decoded_row; image = row*h;
            let a = self.e.arena.alloc_bytes(&decoded); (a.buffer,a.offset)
        } else { match self.buffer_range(src, len as u64) {
            Some((chunk, off)) => (chunk.buffer.clone(), off),
            None => {
                if self.buffer_range(src,1).is_some() {
                    let staging = self.e.arena.alloc_bytes(&vec![0u8;len]);
                    let enc = self.cb.blitCommandEncoder().ok_or("segmented upload staging encoder")?;
                    let mut at=0;
                    while at<len {
                        let (part,part_offset)=self.buffer_range(src+at as u64,1).ok_or("unmapped hole in segmented texture upload")?;
                        let n=(len-at).min((part.base+part.size-src-at as u64) as usize);
                        unsafe {enc.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(&part.buffer.0,part_offset,&staging.buffer.0,staging.offset+at,n);}
                        at+=n;
                    }
                    enc.endEncoding(); (staging.buffer,staging.offset)
                } else {
                if !crate::kernel::aspace().readable(src, len as u64) { return Err("buffer-to-texture source outside guest storage"); }
                let data = unsafe { std::slice::from_raw_parts(src as *const u8, len) };
                let a = self.e.arena.alloc_bytes(data);
                (a.buffer, a.offset)
                }
            }
        }};
        let enc = self.cb.blitCommandEncoder().ok_or("buffer-to-texture blit encoder")?;
        for i in 0..d.max(1) {
            let (slice, oz, depth) = if layered { (layer as usize + z + i, 0, 1) } else if i == 0 { (0, z, d.max(1)) } else { break };
            unsafe {
                enc.copyFromBuffer_sourceOffset_sourceBytesPerRow_sourceBytesPerImage_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin(
                    &buf.0,
                    off + i * image,
                    row,
                    image,
                    MTLSize { width: w, height: h, depth },
                    &tex.0,
                    slice,
                    level as usize,
                    MTLOrigin { x, y, z: oz },
                )
            };
        }
        enc.endEncoding();
        Ok(())
    }

    fn end_pass(&mut self) {
        if let Some(p) = self.pass.take() {
            p.enc.endEncoding();
            if submit_profile::active() {
                if let Some(counter)=p.counter {submit_profile::watch(&self.cb,counter,format!("render={}x{} pass={}",p.size.0,p.size.1,self.passes),4);}
            }
            for (k, sub) in &p.keys {
                if let Some((_, src)) = self.e.textures.cached(k) {
                    for dst in self.e.textures.aliases_of(k) {
                        textures::copy_blocks(&self.cb, &src.0, &dst.0, Some(*sub));
                    }
                }
            }
            self.passes += 1;
            if self.passes % PASSES_PER_COMMIT == 0 {
                if let Some(next) = gfx::gpu().queue.0.commandBuffer() {
                    commit(&std::mem::replace::<Retained<gfx::CommandBuffer>>(&mut self.cb, next), &self.pools);
                    self.fault=None;
                    self.pool_table=None;self.e.window_batch_writes=false;
                }
            }
        }
    }

    fn update_uniform(&mut self, addr: u64, size: u64, offset: u64, data: &[u8]) {
        let size = size.clamp(offset + data.len() as u64, 0x10000);
        self.prepare_cpu_read(addr,size);
        let source;
        let src = match self.e.overlays.get(&addr) {
            Some(&(osize, _, ptr)) if osize >= size => ptr as *const u8,
            _ => {source=self.cpu_bytes(addr,size as usize).unwrap_or_else(|e|crate::diag::report_and_exit(e));source.as_ptr()},
        };
        let a = self.e.arena.alloc(size as usize);
        unsafe {
            std::ptr::copy_nonoverlapping(src, a.ptr, size as usize);
            std::ptr::copy_nonoverlapping(data.as_ptr(), a.ptr.add(offset as usize), data.len());
        }
        self.e.overlays.insert(addr, (size, a.gpu, a.ptr as usize));
        if a.new_chunk {
            self.use_chunk(&a.buffer);
        }
    }

    fn use_chunk(&mut self, b: &Shared<state::Buffer>) {
        if let Some(p) = &mut self.pass {
            p.enc.useResource_usage_stages(ProtocolObject::from_ref(&*b.0), MTLResourceUsage::Read|MTLResourceUsage::Write, MTLRenderStages::Vertex | MTLRenderStages::Fragment);
            p.chunks += 1;
        }
    }

    fn gpu_range(&mut self, addr: u64, size: u64, overlay: bool) -> u64 {
        if overlay {
            if let Some((&base, &(size, gpu, _))) = self.e.overlays.range(..=addr).next_back() {
                if addr < base + size {
                    return gpu + (addr - base);
                }
            }
        }
        if let Some((chunk, off)) = self.buffer_range(addr, size.max(1)) {
            return chunk.gpu + off as u64;
        }
        if let Some(&gpu) = self.copies.get(&(addr, size)) { return gpu; }
        let available = self.pools.iter().find(|p| addr >= p.base && addr - p.base < p.size)
            .map(|p| p.base + p.size - addr)
            .or_else(|| self.heap.filter(|h| addr >= h.base && addr - h.base < h.size).map(|h| h.base + h.size - addr));
        if let Some(len) = available {
            if let Some((chunk, off)) = self.buffer_range(addr, len.min(size).max(1)) { return chunk.gpu + off as u64; }
        }
        let len = size.min(available.unwrap_or(size));
        if addr != 0 && len != 0 && crate::kernel::aspace().readable(addr, len) {
            self.prepare_cpu_read(addr,len);
            let data = unsafe { std::slice::from_raw_parts(addr as *const u8, len as usize) };
            let a = self.e.arena.alloc_bytes(data);
            if a.new_chunk { self.use_chunk(&a.buffer); }
            self.copies.insert((addr, size), a.gpu);
            return a.gpu;
        }
        gfx::gpu().zeros.0.gpuAddress()
    }

    fn buffer_range(&self, addr: u64, size: u64) -> Option<(&gfx::HeapChunk, usize)> {
        gfx::pool_buffer_range(&self.pools, addr, size).or_else(|| self.heap.and_then(|h| h.buffer_range(addr, size)))
    }

    fn cpu_bytes(&mut self,addr:u64,len:usize) -> Result<Vec<u8>,&'static str> {
        self.prepare_cpu_read(addr,len as u64);
        let mut out=Vec::with_capacity(len); let mut at=0;
        while at<len {
            if let Some((part,offset))=self.buffer_range(addr+at as u64,1) {
                let n=(len-at).min((part.base+part.size-addr-at as u64) as usize);
                let bytes=unsafe {std::slice::from_raw_parts((part.buffer.0.contents().as_ptr() as *const u8).add(offset),n)};
                out.extend_from_slice(bytes); at+=n;
            } else {
                if !crate::kernel::aspace().readable(addr+at as u64,(len-at) as u64) {return Err("CPU transfer crosses unmapped guest or GPU storage");}
                out.extend_from_slice(unsafe {std::slice::from_raw_parts((addr+at as u64) as *const u8,len-at)}); break;
            }
        }
        Ok(out)
    }

    // GPU addresses need not be CPU pointers. Resolve every destination
    // before mutating any bytes, including discontiguous virtual pool aliases.
    fn write_cpu_bytes(&self,addr:u64,bytes:&[u8])->Result<(),&'static str> {
        let mut parts=Vec::new();let mut at=0;
        while at<bytes.len() {
            let address=addr.checked_add(at as u64).ok_or("CPU write GPU address overflow")?;
            if let Some((part,offset))=self.buffer_range(address,1) {
                let n=(bytes.len()-at).min((part.size-(address-part.base)) as usize);
                let ptr=unsafe{part.buffer.0.contents().as_ptr().cast::<u8>().add(offset)};
                parts.push((ptr,at,n));at+=n;
            } else {
                let n=bytes.len()-at;if !crate::kernel::aspace().readable(address,n as u64){return Err("CPU write crosses unmapped GPU storage");}
                parts.push((address as *mut u8,at,n));break;
            }
        }
        for (ptr,at,n) in parts {unsafe{std::ptr::copy_nonoverlapping(bytes.as_ptr().add(at),ptr,n);}}
        Ok(())
    }
    fn flush_uniform_overlays(&self)->Result<(),&'static str> {
        for (&addr,&(size,_,ptr)) in &self.e.overlays {
            let bytes=unsafe{std::slice::from_raw_parts(ptr as *const u8,size as usize)};self.write_cpu_bytes(addr,bytes)?;
        }
        Ok(())
    }

    fn defer_clear(&mut self, c: PendingClear) {
        self.e.clears.retain(|p| !(p.key == c.key && p.sub == c.sub && p.is_depth() == c.is_depth()));
        self.e.clears.push(c);
    }

    fn take_clear(&mut self, key: textures::TexKey, sub: (u32, u32), depth: bool) -> Option<PendingClear> {
        let i = self.e.clears.iter().position(|p| p.key == key && p.sub == sub && p.is_depth() == depth)?;
        Some(self.e.clears.remove(i))
    }

    fn flush_clears(&mut self) {
        for c in std::mem::take(&mut self.e.clears) {
            let a = if c.is_depth() {
                Attachments { colors: vec![], depth: Some((c.tex, c.load, c.stencil)), subs: vec![], depth_sub: c.sub }
            } else {
                Attachments { colors: vec![Some((c.tex, c.load))], depth: None, subs: vec![c.sub], depth_sub: (0, 0) }
            };
            gfx::run_pass(&self.cb, &a);
        }
    }

    fn ensure_pass(&mut self) -> Result<(), &'static str> {
        if self.pass.is_none() {
            let mut colors = Vec::new();
            let mut subs = Vec::new();
            let mut fmts = Vec::new();
            let mut size = None;
            let mut bound = Vec::new();
            for b in self.e.colors.clone() {
                let b = b.filter(|b| !textures::is_compressed(u32::from_le_bytes(b[0x30..0x34].try_into().unwrap())));
                let sub = b.as_ref().map_or((0, 0), |b| super::objects::view_subresource(b));
                let t = b.and_then(|b| self.texture(&b, true));
                let t = t.filter(|(k, _)| {
                    let dup = bound.contains(&(*k, sub));
                    bound.push((*k, sub));
                    !dup
                });
                fmts.push(t.as_ref().map(|(_, t)| t.0.pixelFormat()));
                colors.push(t.map(|(k, t)| {
                    if !self.e.touched.iter().any(|(tk, _)| *tk == k) {
                        self.e.touched.push((k, t.clone()));
                    }
                    size.get_or_insert(((k.width >> sub.1).max(1), (k.height >> sub.1).max(1)));
                    let load = self.take_clear(k, sub, false).map_or(Load::Keep, |c| c.load);
                    (t, load)
                }));
                subs.push(sub);
            }
            let mut depth = None;
            let mut dfmt = None;
            let mut stencil = false;
            let mut depth_sub = (0, 0);
            if let Some(b) = self.e.depth.clone() {
                if let Some((k, t)) = self.texture(&b, true) {
                    depth_sub = super::objects::view_subresource(&b);
                    stencil = gfx::has_stencil(k.format);
                    dfmt = Some(t.0.pixelFormat());
                    size.get_or_insert(((k.width >> depth_sub.1).max(1), (k.height >> depth_sub.1).max(1)));
                    let load = self.take_clear(k, depth_sub, true).map_or(Load::Keep, |c| c.load);
                    depth = Some((t, load, stencil));
                }
            }
            let Some(size) = size else { return Err("no render target") };
            let keys: Vec<(textures::TexKey, (u32, u32))> =
                self.e.colors.iter().flatten().filter_map(|b| Some((textures::TexKey::of(b)?, super::objects::view_subresource(b)))).collect();
            let frame = self.e.textures.frame;
            let mut fresh = false;
            for b in self.e.colors.iter().flatten().chain(self.e.depth.iter()) {
                if let Some(k) = textures::TexKey::of(b) {
                    let first = *self.e.target_first_frame.entry(k).or_insert(frame);
                    fresh |= first == frame;
                }
            }
            while fmts.last() == Some(&None) {
                fmts.pop();
            }
            self.flush_clears();
            let counter=if submit_profile::active() || (TRACE_DRAWS.load(Relaxed) && std::path::Path::new("work/profile-gpu-stages").exists()) {gfx::timestamp_samples(4)} else {None};
            let enc = gfx::begin_pass_sampled(&self.cb, &Attachments { colors, depth, subs, depth_sub },counter.as_ref().map(|c|&*c.0)).ok_or("no encoder")?;
            let chunks: Vec<_> = self.e.arena.used.clone();
            for c in &chunks {
                enc.useResource_usage_stages(ProtocolObject::from_ref(&*c.0), MTLResourceUsage::Read, MTLRenderStages::Vertex | MTLRenderStages::Fragment);
            }
            if let Some(table)=&self.e.address_table.buffer {
                enc.useResource_usage_stages(ProtocolObject::from_ref(&*table.0),MTLResourceUsage::Read,MTLRenderStages::Vertex|MTLRenderStages::Fragment);
            }
            if let Some(fault)=&self.fault {enc.useResource_usage_stages(ProtocolObject::from_ref(&*fault.buffer.0),MTLResourceUsage::Read|MTLResourceUsage::Write,MTLRenderStages::Vertex|MTLRenderStages::Fragment);}
            let z = &gfx::gpu().zeros.0;
            enc.useResource_usage_stages(ProtocolObject::from_ref(&**z), MTLResourceUsage::Read, MTLRenderStages::Vertex | MTLRenderStages::Fragment);
            let resources=self.pool_resources();
            if !resources.is_empty() {
                unsafe {enc.useResources_count_usage_stages(std::ptr::NonNull::new(resources.as_ptr().cast_mut()).unwrap(),resources.len(),MTLResourceUsage::Read,MTLRenderStages::Vertex|MTLRenderStages::Fragment);}
            }
            self.pass = Some(Pass { enc, counter,targets: Targets { colors: fmts, depth: dfmt, stencil }, size, heap: 0, chunks: chunks.len(), fresh, keys, fixed: None,profile_context_reported:false });
        }
        Ok(())
    }
}

fn stage_index(nvn_stage: u32) -> Option<usize> {
    match nvn_stage {
        0 => Some(0),
        1 => Some(1),
        5 => Some(2),
        _ => None,
    }
}

fn set_faces(v: &mut [u32; 2], faces: u32, value: u32) {
    if faces & 1 != 0 {
        v[0] = value;
    }
    if faces & 2 != 0 {
        v[1] = value;
    }
}

struct Debug {
    trace_frame: Option<u32>,
    trace_at: Option<f32>,
    trace_count: u32,
    dump_rt_at: Option<f32>,
    dump_frames: Vec<u32>,
    capture_file: bool,
}

static DEBUG: LazyLock<Debug> = LazyLock::new(|| {
    let var = |k: &str| std::env::var(k).ok();
    Debug {
        trace_frame: var("NX_TRACE_FRAME").and_then(|v| v.parse().ok()),
        trace_at: var("NX_TRACE_AT").and_then(|v| v.parse().ok()),
        trace_count: var("NX_TRACE_COUNT").and_then(|v| v.parse().ok()).unwrap_or(1),
        dump_rt_at: var("NX_DUMP_RT_AT").and_then(|v| v.parse().ok()),
        dump_frames: var("NX_DUMP_FRAME").map(|v| v.split(',').filter_map(|f| f.parse().ok()).collect()).unwrap_or_default(),
        capture_file: var("NX_NO_CAPTURE_FILE").is_none(),
    }
});

fn mapped_pool_snapshot(after_snapshot: impl FnOnce()) -> Result<Vec<std::sync::Arc<gfx::PoolMemory>>, String> {
    // Finalize may return to a guest thread that immediately unmaps storage.
    // Keep its registry entry live until NoCopy wrapping has retained the
    // physical buffers; copying only the numeric addresses is insufficient.
    let registry = STATE.lock().unwrap();
    let physical = registry.pools.iter().map(|(&object,p)|(object,p.cpu,p.size)).collect::<Vec<_>>();
    let aliases = registry.gpu_aliases();
    after_snapshot();
    let mut pools = gfx::pool_memory(&physical)?;
    gfx::add_pool_aliases(&mut pools,&aliases)?;
    drop(registry);
    Ok(pools)
}

pub fn submit(handles: &[u64]) {
    gfx::wait_foreground();
    objc2::rc::autoreleasepool(|_| {
        let mut e = engine();
        e.arena.recycle();
        e.forget_dead();
        e.bank_ok = [false; 3];
        e.cpu_events.begin_submit();
        e.window_batch_writes=false;
        e.textures.frame += 1;
        let Some(cb) = gfx::gpu().queue.0.commandBuffer() else { return };
        let pools = mapped_pool_snapshot(||{}).unwrap_or_else(|error| crate::diag::report_and_exit(&error));
        let mut r = Replay { e: &mut e, cb, pass: None, heap: gfx::heap(), pools, passes: 0, copies: HashMap::new(), fault:None, pool_table:None,resource_buffers:Default::default() };
        let mut updated = BTreeMap::new();
        for h in handles {
            if let Some(cmds) = cmd::recording(*h) {
                collect_updates(&cmds, 0, &mut updated);
            }
        }
        if !updated.is_empty() {r.commit_boundary(true);}
        for (&addr, &size) in &updated {
            let size = size.min(0x10000) as usize;
            let src=r.cpu_bytes(addr,size).unwrap_or_else(|e|crate::diag::report_and_exit(e));
            let a = r.e.arena.alloc_bytes(&src);
            r.e.overlays.insert(addr, (size as u64, a.gpu, a.ptr as usize));
        }
        static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let n = N.fetch_add(1, Relaxed);
        super::virtual_lookup::poll_test_mode(n);
        static T0: LazyLock<std::time::Instant> = LazyLock::new(std::time::Instant::now);
        static TRACED_AT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let dbg = &*DEBUG;
        let at = dbg.trace_at.is_some_and(|t| T0.elapsed().as_secs_f32() >= t) && TRACED_AT.fetch_add(1, Relaxed) < dbg.trace_count;
        static CAPTURE_LEFT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        if dbg.capture_file && n % 32 == 0 {
            if let Ok(v) = std::fs::read_to_string("work/capture") {
                let _ = std::fs::remove_file("work/capture");
                CAPTURE_LEFT.store(v.trim().parse::<u32>().unwrap_or(1).max(1), Relaxed);
            }
        }
        let capture = CAPTURE_LEFT.load(Relaxed) > 0;
        if capture {
            CAPTURE_LEFT.fetch_sub(1, Relaxed);
            DUMP_NEXT.store(true, Relaxed);
        }
        TRACE_DRAWS.store(capture || dbg.trace_frame == Some(n) || at, Relaxed);
        if gpu_counters::poll_test_mode(n) {r.commit_boundary(true);}
        let profile_start=submit_profile::begin(n);
        for h in handles {
            if let Some(cmds) = cmd::recording(*h) {
                r.run(&cmds, 0);
            }
        }
        r.end_pass();
        r.flush_clears();
        r.flush_uniform_overlays().unwrap_or_else(|e|crate::diag::report_and_exit(e));
        r.e.window_batch_writes=false;
        let cb = r.cb;
        commit(&cb, &r.pools);
        submit_profile::finish(profile_start);
        static DUMPED_AT: AtomicBool = AtomicBool::new(false);
        let dump_at = dbg.dump_rt_at.is_some_and(|t| T0.elapsed().as_secs_f32() >= t) && !DUMPED_AT.swap(true, Relaxed);
        if DUMP_NEXT.swap(false, Relaxed) || dump_at || dbg.dump_frames.contains(&n) {
            cb.waitUntilCompleted();
            let _ = std::fs::create_dir_all("work/frame");
            for (i, (k, t)) in std::mem::take(&mut e.touched).into_iter().enumerate() {
                let path = format!("work/frame/{n:04}_{i:03}_{:x}_f{}_{}x{}.png", k.addr, k.format, k.width, k.height);
                gfx::dump_png(&t.0, &path);
            }
            let _ = std::fs::create_dir_all("work/frame/tex");
            for (k, t) in std::mem::take(&mut e.sampled) {
                if t.0.textureType()==MTLTextureType::Type3D {
                    dump_volume(&t.0,&k,&format!("work/volume-probes/{n:04}_{:x}_f{}",k.addr,k.format));
                } else {
                    gfx::dump_png(&t.0, &format!("work/frame/tex/{n:04}_{:x}_f{}_{}x{}.png", k.addr, k.format, k.width, k.height));
                }
            }
            eprintln!("[gfx] frame {n} dumped to work/frame");
        }
        e.touched.clear();
        e.sampled.clear();
        let cb = Shared(cb);
        e.arena.retire(cb.clone());
        e.overlays.clear();
        e.last_submit = Some(cb);
    });
}

fn dump_volume(texture:&gfx::Texture,key:&textures::TexKey,path:&str) {
    if key.target==10 || gfx::is_depth_format(key.format) {return;}
    let (bw,bh,bpp)=super::format_info(key.format);
    if bw!=1 || bh!=1 {eprintln!("[volume] compressed volume readback omitted: {key:?}");return;}
    let (w,h,d)=(texture.width(),texture.height(),texture.depth());
    let row=(w*bpp as usize).next_multiple_of(256);let image=row*h;let size=image*d;
    if size==0 || size>16*1024*1024 {eprintln!("[volume] readback size {size} outside capture bound");return;}
    let device=&gfx::gpu().device.0;
    let Some(buffer)=device.newBufferWithLength_options(size,MTLResourceOptions::StorageModeShared) else {return;};
    let Some(cb)=gfx::gpu().queue.0.commandBuffer() else {return;};
    let Some(encoder)=cb.blitCommandEncoder() else {return;};
    unsafe {encoder.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toBuffer_destinationOffset_destinationBytesPerRow_destinationBytesPerImage(texture,0,0,MTLOrigin{x:0,y:0,z:0},MTLSize{width:w,height:h,depth:d},&buffer,0,row,image);}
    encoder.endEncoding();
    cb.commit();cb.waitUntilCompleted();
    if cb.status()!=MTLCommandBufferStatus::Completed {eprintln!("[volume] readback failed: {:?}",cb.status());return;}
    let tight_row=w*bpp as usize;let tight_size=tight_row*h*d;
    let padded=unsafe {std::slice::from_raw_parts(buffer.contents().as_ptr() as *const u8,size)};
    let mut data=Vec::with_capacity(tight_size);
    for z in 0..d {for y in 0..h {let start=z*image+y*row;data.extend_from_slice(&padded[start..start+tight_row]);}}
    let _=std::fs::create_dir_all("work/volume-probes");
    let _=std::fs::write(format!("{path}.gpu"),data);
    let _=std::fs::write(format!("{path}.txt"),format!("width={w}\nheight={h}\ndepth={d}\nrow={tight_row}\nbytes_per_pixel={bpp}\nnvn_format={}\n",key.format));
}

static DEAD: Mutex<Vec<(u64, u64)>> = Mutex::new(Vec::new());
static FLUSHED: Mutex<Vec<(u64, u64)>> = Mutex::new(Vec::new());

pub fn flush_memory(addr: u64, size: u64) { if size > 0 { FLUSHED.lock().unwrap().push((addr, size)); } }

pub fn forget_memory(addr: u64, size: u64) {
    if size > 0 {
        DEAD.lock().unwrap().push((addr, addr + size));
    }
}

pub fn fence_sync(sync: u64) {
    let mut e = engine();
    if !e.syncs.contains_key(&sync) {crate::diag::report_and_exit("NVN queue fence uses an uninitialized sync");}
    let point=e.last_submit.clone().map(SyncPoint::Fenced).unwrap_or(SyncPoint::Ready);
    e.syncs.insert(sync, point);
}

pub fn initialize_sync(sync:u64,device:u64) -> u64 {
    if sync==0 || device==0 {return 0;}
    // SDK 22 initializes device+state, its native-fence payload, and offset30.
    unsafe {std::ptr::write_bytes(sync as *mut u8,0,0x20);(sync as *mut u64).write_unaligned(device);((sync+0x30) as *mut u64).write_unaligned(0);}
    engine().syncs.insert(sync,SyncPoint::Initialized);1
}
pub fn finalize_sync(sync:u64) {engine().syncs.remove(&sync);}
pub fn acquire_sync(sync:u64,texture:u64) {
    let mut e=engine();
    if !e.syncs.contains_key(&sync) {crate::diag::report_and_exit("NVN window acquire uses an uninitialized sync");}
    let point=e.presented.get(&texture).cloned().map(SyncPoint::Fenced).unwrap_or(SyncPoint::Ready);
    e.syncs.insert(sync,point);
}

pub use shaders::precompile;
pub use state::prewarm;

pub fn finish() {
    let cb = engine().last_submit.clone();
    if let Some(cb) = cb {
        gpu_wait::complete(&cb.0);
    }
}

pub fn sync_wait(sync: u64,timeout_ns:u64) -> u64 {
    let point=engine().syncs.get(&sync).cloned();
    wait_sync_point(point,timeout_ns)
}

const PASSES_PER_COMMIT: u32 = 24;

fn commit(cb: &gfx::CommandBuffer, pools: &[std::sync::Arc<gfx::PoolMemory>]) {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = NEXT.fetch_add(1, Relaxed);
    let keep = pools.to_vec();
    let handler = block2::RcBlock::new(move |command: std::ptr::NonNull<ProtocolObject<dyn MTLCommandBuffer>>| {
        std::hint::black_box(&keep);
        let command = unsafe { command.as_ref() };
        if n < 8 { eprintln!("[gfx] GPU completion #{n}, status {:?}", command.status()); }
        if command.status() == MTLCommandBufferStatus::Error {
            eprintln!("[gfx] Metal command buffer failed: {:?}", command.error());
            crate::diag::report_and_exit("Metal command buffer execution failed");
        }
    });
    unsafe { cb.addCompletedHandler(block2::RcBlock::as_ptr(&handler)) };
    if *FPS {
        let h = block2::RcBlock::new(|cb: std::ptr::NonNull<ProtocolObject<dyn MTLCommandBuffer>>| {
            let cb = unsafe { cb.as_ref() };
            static LAST_END: Mutex<f64> = Mutex::new(0.0);
            let (start, end) = (cb.GPUStartTime(), cb.GPUEndTime());
            let mut last = LAST_END.lock().unwrap();
            GPU_NS.fetch_add(((end - start.max(*last)).max(0.0) * 1e9) as u64, Relaxed);
            *last = last.max(end);
        });
        unsafe { cb.addCompletedHandler(block2::RcBlock::as_ptr(&h)) };
    }
    cb.commit();
}

static FPS: LazyLock<bool> = LazyLock::new(|| std::env::var_os("NX_FPS").is_some());
static GPU_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn count_fps() {
    static STATS: Mutex<Option<(std::time::Instant, std::time::Instant, u32, f32)>> = Mutex::new(None);
    if !*FPS {
        return;
    }
    let now = std::time::Instant::now();
    let mut s = STATS.lock().unwrap();
    let (start, last, n, worst) = s.get_or_insert((now, now, 0, 0.0));
    *worst = worst.max(now.duration_since(*last).as_secs_f32() * 1000.0);
    *last = now;
    *n += 1;
    if now.duration_since(*start).as_secs() >= 1 {
        let gpu = GPU_NS.swap(0, Relaxed) as f32 / 1e6 / *n as f32;
        eprintln!("[fps] {:.1} fps, worst frame {worst:.1} ms, GPU {gpu:.1} ms/frame", *n as f32 / now.duration_since(*start).as_secs_f32());
        *s = Some((now, now, 0, 0.0));
    }
}

pub fn present(texture: u64,layer:u64) {
    count_fps();
    objc2::rc::autoreleasepool(|_| {
        let b = super::objects::texture(texture).map(|t| t.builder);
        let Some(b) = b else { return };
        let (t, show) = {
            let mut e = engine();
            e.forget_dead();
            let (skipped, draws) = (std::mem::take(&mut e.skipped), std::mem::take(&mut e.draws));
            (e.textures.get(&b, true), e.held.present(skipped, draws))
        };
        if let Some((_, t)) = t {
            if show {
                dump_at(&t);
            }
            let history=show.then(||(layer,crate::services::vi::queue_frame(layer)));
            if let Some(cb)=crate::kernel::blocking(|| gfx::present(show.then_some(&*t.0), !UPPER_LEFT.load(Relaxed),history)) {
                let mut e=engine();e.presented.insert(texture,cb.clone());e.last_submit=Some(cb);
            }
        }
    });
}

pub fn write_texels(mut b: TexBuilder, view: u64, region: [i32; 6], data: *const u8, strides: Option<(usize, usize)>) {
    super::objects::apply_view(&mut b, view, true);
    objc2::rc::autoreleasepool(|_| {
        let mut e = engine();
        e.wait_cpu_events();
        e.forget_dead();
        let Some((key, tex)) = e.textures.get(&b, false) else { return };
        let (bw, bh, bpb) = super::format_info(key.format);
        let [x, y, z, w, h, d] = region.map(|v| v.max(0) as usize);
        if w == 0 || h == 0 || data.is_null() {
            return;
        }
        let row = w.div_ceil(bw as usize) * bpb as usize;
        let image = row * h.div_ceil(bh as usize);
        let (row, image) = strides.filter(|s| s.0 >= row).unwrap_or((row, image));
        let decoded = gfx::bc::soft(key.format).map(|k| {
            let blocks = row * h.div_ceil(4);
            (0..d.max(1)).flat_map(|i| gfx::bc::decode(k, unsafe { std::slice::from_raw_parts(data.add(i * image), blocks) }, row, w, h, 1).0).collect::<Vec<u8>>()
        });
        let (data, row, image) = match &decoded {
            Some(px) => (px.as_ptr(), px.len() / d.max(1) / h, px.len() / d.max(1)),
            None => (data, row, image),
        };
        let level = if view != 0 && unsafe { *(view as *const u8) } & 1 != 0 { (unsafe { *((view + 1) as *const u8) }) as usize } else { 0 };
        let layered = matches!(tex.0.textureType(), MTLTextureType::Type2DArray | MTLTextureType::TypeCube | MTLTextureType::TypeCubeArray);
        let slices = if layered { z..z + d.max(1) } else { 0..1 };
        if tex.0.storageMode() == MTLStorageMode::Private {
            let count = if layered { d.max(1) } else { 1 };
            let depth = if layered { 1 } else { d.max(1) };
            let len = image * (count * depth - 1) + row * h.div_ceil(bh as usize);
            let g = gfx::gpu();
            let Some(buf) = (unsafe { g.device.0.newBufferWithBytes_length_options(std::ptr::NonNull::new(data as *mut u8).unwrap().cast(), len, MTLResourceOptions::StorageModeShared) }) else { return };
            let Some(cb) = g.queue.0.commandBuffer() else { return };
            let Some(enc) = cb.blitCommandEncoder() else { return };
            for (i, slice) in slices.enumerate() {
                unsafe {
                    enc.copyFromBuffer_sourceOffset_sourceBytesPerRow_sourceBytesPerImage_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin(
                        &buf,
                        i * image,
                        row,
                        image,
                        MTLSize { width: w, height: h, depth },
                        &tex.0,
                        slice,
                        level,
                        MTLOrigin { x, y, z: if layered { 0 } else { z } },
                    )
                };
            }
            enc.endEncoding();
            cb.commit();
            return;
        }
        for (i, slice) in slices.enumerate() {
            let region = MTLRegion {
                origin: MTLOrigin { x, y, z: if layered { 0 } else { z } },
                size: MTLSize { width: w, height: h, depth: if layered { 1 } else { d.max(1) } },
            };
            let p = unsafe { data.add(i * image) } as *mut u8;
            unsafe {
                tex.0.replaceRegion_mipmapLevel_slice_withBytes_bytesPerRow_bytesPerImage(
                    region,
                    level,
                    slice,
                    std::ptr::NonNull::new(p).unwrap().cast(),
                    row,
                    image,
                )
            };
        }
    });
}

pub fn read_texels(mut b: TexBuilder, view: u64, region: [i32; 6], data: *mut u8, strides: Option<(usize, usize)>) {
    super::objects::apply_view(&mut b, view, true);
    let [_, _, _, w, h, d] = region.map(|v| v.max(0) as usize);
    if w == 0 || h == 0 || data.is_null() {
        return;
    }
    finish();
    objc2::rc::autoreleasepool(|_| {
        let mut e = engine();
        e.forget_dead();
        let Some((key, tex)) = e.textures.get(&b, true) else { return };
        drop(e);
        let (bw, bh, bpb) = super::format_info(key.format);
        let row = w.div_ceil(bw as usize) * bpb as usize;
        let image = row * h.div_ceil(bh as usize);
        let len = image * d.max(1);
        let g = gfx::gpu();
        let Some(buf) = g.device.0.newBufferWithLength_options(len, MTLResourceOptions::StorageModeShared) else { return };
        let Some(cb) = g.queue.0.commandBuffer() else { return };
        if copy::copy_into(&cb, &tex.0, super::objects::view_subresource(&b), region, key.format, &buf, 0).is_err() {
            return;
        }
        cb.commit();
        crate::kernel::blocking(|| cb.waitUntilCompleted());
        let src = unsafe { std::slice::from_raw_parts(buf.contents().as_ptr() as *const u8, len) };
        let (dst_row, dst_image) = strides.filter(|s| s.0 >= row).unwrap_or((row, image));
        let rows = h.div_ceil(bh as usize);
        for z in 0..d.max(1) {
            for y in 0..rows {
                let from = &src[z * image + y * row..][..row];
                unsafe { std::ptr::copy_nonoverlapping(from.as_ptr(), data.add(z * dst_image + y * dst_row), row) };
            }
        }
    });
}

fn dump_at(t: &Shared<gfx::Texture>) {
    if let Some(sequence)=crate::input::test_control::frame() {
        let _=std::fs::create_dir_all("work/frame");
        gfx::dump_png(&t.0,&format!("work/frame/manual-{sequence:06}.png"));
        eprintln!("[gfx] manual screen {sequence} dumped");
    }
    static TIMES: LazyLock<Mutex<Vec<f32>>> = LazyLock::new(|| {
        let mut v: Vec<f32> = std::env::var("NX_DUMP_AT").unwrap_or_default().split(',').filter_map(|x| x.trim().parse().ok()).collect();
        v.sort_by(|a, b| b.partial_cmp(a).unwrap());
        Mutex::new(v)
    });
    let now = gfx::clock();
    let mut times = TIMES.lock().unwrap();
    if times.last().is_some_and(|&at| now >= at) {
        let at = times.pop().unwrap();
        let _ = std::fs::create_dir_all("work/frame");
        gfx::dump_png(&t.0, &format!("work/frame/t{at:05.1}.png"));
        eprintln!("[gfx] screen at {at}s dumped");
    }
}

fn collect_updates(cmds: &[Cmd], depth: usize, out: &mut BTreeMap<u64, u64>) {
    for c in cmds {
        match c {
            Cmd::UpdateUniform { addr, size, offset, data } => {
                let s = out.entry(*addr).or_default();
                *s = (*s).max(*size).max(offset + data.len() as u64);
            }
            Cmd::Call(handles) if depth < 8 => {
                for h in handles {
                    if let Some(sub) = cmd::recording(*h) {
                        collect_updates(&sub, depth + 1, out);
                    }
                }
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Held;

    #[cfg(target_os="macos")]
    fn maxwell_program(instructions:&[u64]) -> Vec<u8> {
        let mut instructions=instructions.to_vec();instructions.extend([0xe30000000007000fu64,0xe2400fffff87000f]);
        let mut words=Vec::new();for group in instructions.chunks(3) {words.push(0u64);words.extend_from_slice(group);}
        let mut bytes=words.iter().flat_map(|w|w.to_le_bytes()).collect::<Vec<_>>();bytes.resize(bytes.len().max(0x100),0);bytes
    }
    #[cfg(target_os="macos")]
    fn mov32(reg:u64,value:u32) -> u64 {0x010000000007f000|((value as u64)<<20)|reg}
    #[cfg(target_os="macos")]
    fn mov_handle(reg:u64,offset:u64) -> u64 {0x4c98078000070000|(2u64<<34)|((offset/4)<<20)|reg}

    #[cfg(target_os="macos")]
    #[test]
    fn early_fragment_eligibility_preserves_original_late_effects() {
        let make=|words:&[u64]| {let mut b=vec![0u8;0x80];b[..4].copy_from_slice(&0x12345678u32.to_le_bytes());b[0x30..0x34].copy_from_slice(&(5u32<<10).to_le_bytes());b[0x78..0x7c].copy_from_slice(&15u32.to_le_bytes());b.extend(maxwell_program(words));b};
        let opaque=make(&[mov32(0,1.0f32.to_bits())]);assert!(nxshader::can_force_early_fragment_tests(&opaque));
        assert!(nxshader::translate_with(&opaque,nxshader::FRAGMENT_EARLY_TESTS,0,0).unwrap().msl.contains("[[early_fragment_tests]]"));
        assert!(nxshader::translate_with(&opaque,nxshader::FRAGMENT_EARLY_TESTS|nxshader::FRAGMENT_ALPHA_TEST,0,0).is_err());
        for (offset,bits) in [(0x30,1u32<<15),(0x7c,1),(0x7c,2)] {
            let mut changed=opaque.clone();let old=u32::from_le_bytes(changed[offset..offset+4].try_into().unwrap());changed[offset..offset+4].copy_from_slice(&(old|bits).to_le_bytes());
            assert!(!nxshader::can_force_early_fragment_tests(&changed));assert!(nxshader::translate_with(&changed,nxshader::FRAGMENT_EARLY_TESTS,0,0).is_err());
        }
        for (word,op) in [(0xe33000000007000fu64,nxshader::table::Op::Kil),(0xeed8000000070000,nxshader::table::Op::Stg)] {
            assert_eq!(nxshader::table::decode(word).1,op);let changed=make(&[word]);assert!(!nxshader::can_force_early_fragment_tests(&changed));assert!(nxshader::translate_with(&changed,nxshader::FRAGMENT_EARLY_TESTS,0,0).is_err());
        }
    }

    #[cfg(target_os="macos")]
    #[test]
    #[ignore="requires real Metal depth tests and diagnostic fault-buffer visibility"]
    fn gpu_opaque_early_tests_reject_occluded_pixels_before_fault_instrumentation() {
        use super::*;
        objc2::rc::autoreleasepool(|_| unsafe {
            let dev=&gfx::gpu().device.0;let vs_source="#include <metal_stdlib>\nusing namespace metal;vertex float4 vertex_probe(uint id [[vertex_id]]) {float2 p=float2((id<<1)&2,id&2);return float4(p*2.0f-1.0f,0.5f,1.0f);}";
            let lib=dev.newLibraryWithSource_options_error(&objc2_foundation::NSString::from_str(vs_source),None).unwrap();
            let vs=std::sync::Arc::new(shaders::Compiled{write_window_footprint:None,descriptor_write_footprint:None,grid_write_footprint:None,func:gfx::Shared(lib.newFunctionWithName(&objc2_foundation::NSString::from_str("vertex_probe")).unwrap()),iface:Default::default(),source:None,argument_encoders:Default::default(),vertex_formats:None,dynamic_vertex:None});
            let mut blob=vec![0u8;0x80];blob[..4].copy_from_slice(&0x12345678u32.to_le_bytes());blob[0x30..0x34].copy_from_slice(&(5u32<<10).to_le_bytes());blob[0x78..0x7c].copy_from_slice(&15u32.to_le_bytes());
            let rgba=[0.25f32,0.5,0.75,1.0];blob.extend(maxwell_program(&rgba.iter().enumerate().map(|(i,v)|mov32(i as u64,v.to_bits())).collect::<Vec<_>>()));
            let make=|format| {let d=MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(format,4,4,false);d.setStorageMode(MTLStorageMode::Private);d.setUsage(MTLTextureUsage::RenderTarget|MTLTextureUsage::ShaderRead);dev.newTextureWithDescriptor(&d).unwrap()};
            let color=make(MTLPixelFormat::RGBA32Float);let depth=make(MTLPixelFormat::Depth32Float);let output=dev.newBufferWithLength_options(320,MTLResourceOptions::StorageModeShared).unwrap();let fault=dev.newBufferWithLength_options(32,MTLResourceOptions::StorageModeShared).unwrap();
            let mut res:state::Res=std::mem::zeroed();res.fault_gpu=fault.gpuAddress();res.shader_hash=0xfc27e17;
            let targets=state::Targets{colors:vec![Some(MTLPixelFormat::RGBA32Float)],depth:Some(MTLPixelFormat::Depth32Float),stencil:false};let mut state=state::GState::default();state.depth_stencil=3|(2<<4);let mut states=state::States::default();let ds=states.depth_stencil(&state,true,false);
            for options in [0,nxshader::FRAGMENT_EARLY_TESTS] {
                let t=nxshader::translate_with(&blob,options,0,0).unwrap();
                // This is precisely the extra observable effect that ordinary
                // translated buffer loads can have on their error path.
                let instrumented=t.msl.replacen("        return r;\n    };","        shader_fault(res,16u,0u);\n        return r;\n    };",1);assert_ne!(instrumented,t.msl);
                let library=dev.newLibraryWithSource_options_error(&objc2_foundation::NSString::from_str(&instrumented),None).unwrap();let fs=std::sync::Arc::new(shaders::Compiled{write_window_footprint:None,descriptor_write_footprint:None,grid_write_footprint:None,func:gfx::Shared(library.newFunctionWithName(&objc2_foundation::NSString::from_str("fmain")).unwrap()),iface:t.interface,source:None,argument_encoders:Default::default(),vertex_formats:None,dynamic_vertex:None});
                let pipeline=states.pipeline(&vs,(0xfc27e16,0),Some(&fs),Some((0xfc27e17,options)),&targets,&state,true).unwrap();
                for visible in [false,true] {
                    std::ptr::write_bytes(fault.contents().as_ptr(),0,32);
                    let pass=MTLRenderPassDescriptor::new();let c=pass.colorAttachments().objectAtIndexedSubscript(0);c.setTexture(Some(&color));c.setLoadAction(MTLLoadAction::Clear);c.setStoreAction(MTLStoreAction::Store);c.setClearColor(MTLClearColor{red:0.125,green:0.125,blue:0.125,alpha:0.125});let d=pass.depthAttachment();d.setTexture(Some(&depth));d.setLoadAction(MTLLoadAction::Clear);d.setStoreAction(MTLStoreAction::Store);d.setClearDepth(if visible {1.0}else{0.25});
                    let cb=gfx::gpu().queue.0.commandBuffer().unwrap();let enc=cb.renderCommandEncoderWithDescriptor(&pass).unwrap();enc.setRenderPipelineState(&pipeline.0);enc.setDepthStencilState(Some(&ds.0));enc.useResource_usage_stages(ProtocolObject::from_ref(&*fault),MTLResourceUsage::Read|MTLResourceUsage::Write,MTLRenderStages::Fragment);enc.setFragmentBytes_length_atIndex(std::ptr::NonNull::from(&res).cast(),size_of_val(&res),0);enc.drawPrimitives_vertexStart_vertexCount(MTLPrimitiveType::Triangle,0,3);enc.endEncoding();
                    copy::copy_into(&cb,&color,(0,0),[0,0,0,4,4,1],47,&output,0).unwrap();cb.commit();cb.waitUntilCompleted();assert_eq!(cb.status(),MTLCommandBufferStatus::Completed);
                    let record=std::slice::from_raw_parts(fault.contents().as_ptr().cast::<u32>(),8);if visible {assert_eq!(record[0],16);} else if options!=0 {assert_eq!(record,[0;8],"occluded opaque pixels must not execute diagnostic stores");}
                    let expected=if visible {rgba}else{[0.125;4]};for pixel in std::slice::from_raw_parts(output.contents().as_ptr().cast::<f32>(),64).chunks_exact(4) {assert_eq!(pixel,expected);}
                    eprintln!("[verify] opaque early={options} visible={visible} fault={}",record[0]);
                }
            }
        });
    }

    #[cfg(target_os="macos")]
    #[test]
    #[ignore="requires a real Metal GPU"]
    fn gpu_fragment_y_direction_tracks_window_origin() {
        use super::*;
        objc2::rc::autoreleasepool(|_| unsafe {
            let ipa=|d:u64,attr:u64|(0xe003ff87cff7ff03u64&!(255u64|(0x3ff<<28)))|d|(attr<<28);
            let shuffle=|d:u64,a:u64,delta:u64|(0xef17700cf027020cu64&!(255u64|(255<<8)|(31<<20)))|d|(a<<8)|(delta<<20);
            let code=maxwell_program(&[0xf0c8000001270000,ipa(4,0x70),ipa(5,0x74),shuffle(1,4,1),shuffle(2,5,2),mov32(3,0)]);
            let mut blob=vec![0u8;0x80];blob[0x30..0x34].copy_from_slice(&(5u32<<10).to_le_bytes());blob[0x78..0x7c].copy_from_slice(&15u32.to_le_bytes());blob.extend(code);
            let t=nxshader::translate(&blob).unwrap();
            let source=format!("{}\nvertex float4 direction_triangle(uint id [[vertex_id]]) {{float2 p=float2((id<<1)&2,id&2);return float4(p*2.0f-1.0f,0.0f,1.0f);}}",t.msl);
            let dev=&gfx::gpu().device.0;let lib=dev.newLibraryWithSource_options_error(&objc2_foundation::NSString::from_str(&source),None).unwrap();
            let desc=MTLRenderPipelineDescriptor::new();desc.setVertexFunction(Some(&lib.newFunctionWithName(&objc2_foundation::NSString::from_str("direction_triangle")).unwrap()));desc.setFragmentFunction(Some(&lib.newFunctionWithName(&objc2_foundation::NSString::from_str("fmain")).unwrap()));desc.colorAttachments().objectAtIndexedSubscript(0).setPixelFormat(MTLPixelFormat::RGBA32Float);
            let pipeline=dev.newRenderPipelineStateWithDescriptor_error(&desc).unwrap();let td=MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(MTLPixelFormat::RGBA32Float,4,4,false);td.setStorageMode(MTLStorageMode::Shared);td.setUsage(MTLTextureUsage::RenderTarget);let texture=dev.newTextureWithDescriptor(&td).unwrap();
            for flags in [0u32,1] {
                let mut res:state::Res=std::mem::zeroed();res.flags=flags;
                let pass=MTLRenderPassDescriptor::new();let color=pass.colorAttachments().objectAtIndexedSubscript(0);color.setTexture(Some(&texture));color.setLoadAction(MTLLoadAction::Clear);color.setStoreAction(MTLStoreAction::Store);
                let cb=gfx::gpu().queue.0.commandBuffer().unwrap();let enc=cb.renderCommandEncoderWithDescriptor(&pass).unwrap();enc.setRenderPipelineState(&pipeline);enc.setFragmentBytes_length_atIndex(std::ptr::NonNull::from(&res).cast(),size_of_val(&res),0);enc.drawPrimitives_vertexStart_vertexCount(MTLPrimitiveType::Triangle,0,3);enc.endEncoding();cb.commit();cb.waitUntilCompleted();assert_eq!(cb.status(),MTLCommandBufferStatus::Completed);
                let mut pixels=[0f32;64];texture.getBytes_bytesPerRow_fromRegion_mipmapLevel(std::ptr::NonNull::from(&mut pixels).cast(),64,MTLRegion{origin:MTLOrigin{x:0,y:0,z:0},size:MTLSize{width:4,height:4,depth:1}},0);
                for (i,pixel) in pixels.chunks_exact(4).enumerate() {let x=i%4;let y=i/4;assert_eq!(pixel[0],if flags==0 {1.0} else {-1.0});assert_eq!(pixel[1],(x^1) as f32+0.5);assert_eq!(pixel[2],(y^1) as f32+0.5);}
            }
        });
    }

    #[cfg(target_os="macos")]
    #[test]
    #[ignore="requires real Metal GPU and NX_COMPUTE_CAPTURE pointing to a local run"]
    fn gpu_captured_original_compute_replays_without_fault() {
        use super::*;
        let root=std::path::PathBuf::from(std::env::var("NX_COMPUTE_CAPTURE").expect("set NX_COMPUTE_CAPTURE"));
        let hash=std::env::var("NX_COMPUTE_CAPTURE_HASH").unwrap_or("0d1cd0394b05dc9b".into());
        let code=std::fs::read(root.join(format!("work/msl_rt/{hash}.bin"))).unwrap();
        objc2::rc::autoreleasepool(|_| unsafe {
            let compiled=shaders::compile_config(&code,true,0,0,0,0xfc270d1,(4096,3060)).unwrap();let dev=&gfx::gpu().device.0;let pipeline=dev.newComputePipelineStateWithFunction_error(&compiled.func.0).unwrap();
            let mut res:state::Res=std::mem::zeroed();let zero=&gfx::gpu().zeros.0;res.zero_gpu=zero.gpuAddress();res.cb.fill(zero.gpuAddress());
            let mut buffers=Vec::new();let mut spans=Vec::new();
            let mut bank0=[0u32;0x200];
            for i in 0..3 {
                let bytes=std::fs::read(root.join(format!("work/compute-probes/{hash}.storage{i}.bin"))).unwrap();
                let buffer=dev.newBufferWithBytes_length_options(std::ptr::NonNull::new(bytes.as_ptr() as *mut u8).unwrap().cast(),bytes.len(),MTLResourceOptions::StorageModeShared).unwrap();
                let address=0x100000000u64*(i as u64+1);spans.push([address,bytes.len() as u64,buffer.gpuAddress()]);let at=0xc4+4*i;bank0[at]=address as u32;bank0[at+1]=(address>>32) as u32;bank0[at+2]=bytes.len() as u32;buffers.push(buffer);
            }
            res.pool_span_count=3;let table=dev.newBufferWithBytes_length_options(std::ptr::NonNull::new(spans.as_ptr() as *mut u8).unwrap().cast(),spans.len()*24,MTLResourceOptions::StorageModeShared).unwrap();res.pool_spans=table.gpuAddress();buffers.push(table);
            // A zero-initialized output could conceal an early exit. Require
            // the original kernel to replace every output word as well.
            std::ptr::write_bytes(buffers[1].contents().as_ptr(),0xcd,48);
            let bank=dev.newBufferWithBytes_length_options(std::ptr::NonNull::from(&bank0).cast(),size_of_val(&bank0),MTLResourceOptions::StorageModeShared).unwrap();res.cb[0]=bank.gpuAddress();buffers.push(bank);
            for slot in [1usize,3] {
                let bytes=std::fs::read(root.join(format!("work/compute-probes/{hash}.cb{slot}.bin"))).unwrap();let buffer=dev.newBufferWithBytes_length_options(std::ptr::NonNull::new(bytes.as_ptr() as *mut u8).unwrap().cast(),bytes.len(),MTLResourceOptions::StorageModeShared).unwrap();res.cb[slot]=buffer.gpuAddress();buffers.push(buffer);
            }
            let fault=dev.newBufferWithLength_options(32,MTLResourceOptions::StorageModeShared).unwrap();std::ptr::write_bytes(fault.contents().as_ptr(),0,32);res.fault_gpu=fault.gpuAddress();res.shader_hash=u64::from_str_radix(&hash,16).unwrap();
            let cb=gfx::gpu().queue.0.commandBuffer().unwrap();let enc=cb.computeCommandEncoder().unwrap();enc.setComputePipelineState(&pipeline);for b in buffers.iter().chain([&fault,zero]) {enc.useResource_usage(ProtocolObject::from_ref(&**b),MTLResourceUsage::Read|MTLResourceUsage::Write);}enc.setBytes_length_atIndex(std::ptr::NonNull::from(&res).cast(),size_of_val(&res),0);enc.dispatchThreadgroups_threadsPerThreadgroup(MTLSize{width:1,height:1,depth:1},MTLSize{width:64,height:1,depth:1});enc.endEncoding();cb.commit();cb.waitUntilCompleted();assert_eq!(cb.status(),MTLCommandBufferStatus::Completed);
            let record=std::slice::from_raw_parts(fault.contents().as_ptr().cast::<u32>(),8);let output=std::slice::from_raw_parts(buffers[1].contents().as_ptr().cast::<u32>(),12);eprintln!("[verify] original compute fault {:x?}; output {:x?}",record,output);assert_eq!(record[0],0,"original compute fault");assert!(output.iter().all(|v|*v!=0xcdcdcdcd),"original kernel did not replace all output words");
        });
    }

    #[cfg(target_os="macos")]
    #[test]
    #[ignore="requires a real Metal GPU"]
    fn gpu_sdk22_storage_banks_read_write_first_and_last_descriptor_in_each_stage() {
        use super::*;
        objc2::rc::autoreleasepool(|_| unsafe {
            crate::kernel::KERNEL.aspace.get_or_init(||crate::mem::AddressSpace::reserve().unwrap());let page=crate::mem::PAGE;let cpu=crate::kernel::aspace().alias()+0x2c00000;
            crate::kernel::aspace().set(cpu,page*2,crate::mem::Block{state:crate::mem::state::NORMAL,perm:crate::mem::perm::RW,attr:0}).unwrap();let output=cpu+page;
            let pools=gfx::pool_memory(&[(0xa400,cpu,page),(0xa401,output,page)]).unwrap();
            let values=[9u32,15,25,49];std::ptr::copy_nonoverlapping(values.as_ptr(),cpu as *mut u32,4);
            let prelude=nxshader::prelude().replace("#define NX_PAGED_GMEM 0","#define NX_PAGED_GMEM 1");
            let source=format!("{}\nkernel void ssbo_probe(constant Res& res [[buffer(0)]],constant uint2& bindings [[buffer(1)]],uint n [[thread_position_in_grid]]) {{uint i=bindings.x,o=bindings.y;ulong input=ulong(res.cb[0][i])|(ulong(res.cb[0][i+1])<<32ul);ulong output=ulong(res.cb[0][o])|(ulong(res.cb[0][o+1])<<32ul);if(n<4 && gmem_gpu(res,input+n*4ul)!=0ul && gmem_gpu(res,output+n*4ul)!=0ul) *gmem(res,output+n*4ul)=*gmem(res,input+n*4ul)+res.cb[0][i+2]+res.cb[0][o+2];}}",prelude);
            let dev=&gfx::gpu().device.0;let lib=dev.newLibraryWithSource_options_error(&objc2_foundation::NSString::from_str(&source),None).unwrap();let f=lib.newFunctionWithName(&objc2_foundation::NSString::from_str("ssbo_probe")).unwrap();let pipeline=dev.newComputePipelineStateWithFunction_error(&f).unwrap();
            let mut e=Engine::default();let cb=gfx::gpu().queue.0.commandBuffer().unwrap();let mut r=Replay{e:&mut e,cb,pass:None,heap:None,pools,passes:0,copies:HashMap::new(),fault:None,pool_table:None,resource_buffers:Default::default()};
            for (stage,base) in [(0,0x44u32),(1,0x144),(2,0xc4)] {
                for slot in [0usize,15] {
                    let mut st=GState::default();st.storage[stage][slot]=(cpu,16);st.storage[stage][1]=(output,16);let res=r.res(&st,stage,0,0);let bindings=[base+slot as u32*4,base+4];
                    std::ptr::write_bytes(output as *mut u8,0xcc,16);let enc=r.cb.computeCommandEncoder().unwrap();enc.setComputePipelineState(&pipeline);
                    for c in r.pools.iter().flat_map(|p|&p.chunks) {enc.useResource_usage(ProtocolObject::from_ref(&*c.buffer.0),MTLResourceUsage::Read|MTLResourceUsage::Write);}
                    for b in &r.e.arena.used {enc.useResource_usage(ProtocolObject::from_ref(&*b.0),MTLResourceUsage::Read|MTLResourceUsage::Write);}
                    if let Some(table)=&r.e.address_table.buffer {enc.useResource_usage(ProtocolObject::from_ref(&*table.0),MTLResourceUsage::Read);}
                    enc.useResource_usage(ProtocolObject::from_ref(&*gfx::gpu().zeros.0),MTLResourceUsage::Read);enc.setBytes_length_atIndex(std::ptr::NonNull::from(&res).cast(),size_of_val(&res),0);enc.setBytes_length_atIndex(std::ptr::NonNull::from(&bindings).cast(),size_of_val(&bindings),1);enc.dispatchThreads_threadsPerThreadgroup(MTLSize{width:4,height:1,depth:1},MTLSize{width:4,height:1,depth:1});enc.endEncoding();r.commit_boundary(true);
                    assert_eq!(std::slice::from_raw_parts(output as *const u32,4),values.map(|n|n+32),"stage{stage} slot{slot}");
                }
            }
            gfx::forget_pool(0xa400);gfx::forget_pool(0xa401);
        });
    }

    #[cfg(target_os="macos")]
    #[test]
    #[ignore="requires a real Metal GPU"]
    fn gpu_buffer_textures_exceed_2d_limit_and_share_pool_storage() {
        use super::*;
        objc2::rc::autoreleasepool(|_| unsafe {
            crate::kernel::KERNEL.aspace.get_or_init(||crate::mem::AddressSpace::reserve().unwrap());
            let page=crate::mem::PAGE;let size=page*64;let cpu=crate::kernel::aspace().alias()+0x3000000;
            crate::kernel::aspace().set(cpu,size*2,crate::mem::Block{state:crate::mem::state::NORMAL,perm:crate::mem::perm::RW,attr:0}).unwrap();
            let output=cpu+size;let width=32771u32;
            {let mut state=STATE.lock().unwrap();state.pools.insert(0xd100,super::super::objects::MemoryPool{cpu,size,flags:0x22});state.pools.insert(0xd101,super::super::objects::MemoryPool{cpu:output,size,flags:0x21});}
            let builder=|pool:u64,format:u32| {let mut b=[0u8;0x68];b[8..16].copy_from_slice(&pool.to_le_bytes());for (at,value) in [(0x1c,10u32),(0x20,width),(0x24,1),(0x28,1),(0x2c,1),(0x30,format)] {b[at..at+4].copy_from_slice(&value.to_le_bytes());}b};
            let pools=gfx::pool_memory(&[(0xd100,cpu,size),(0xd101,output,size)]).unwrap();let mut engine=Engine::default();engine.st.texture_pool=0xd180;engine.st.textures[2][0]=0x100;engine.st.images[2][0]=0x101;
            let cb=gfx::gpu().queue.0.commandBuffer().unwrap();let mut r=Replay{e:&mut engine,cb,pass:None,heap:None,pools,passes:0,copies:HashMap::new(),fault:None,pool_table:None,resource_buffers:Default::default()};
            let mut control=vec![0u8;0x880];for at in [0x748,0x74c,0x750] {control[at..at+4].copy_from_slice(&1u32.to_le_bytes());}
            // TLDS.1D reads four components into R4..R7. SUST.1D writes
            // those same logical values into a distinct view of pool bytes.
            let fetch=0xda00000000000000u64|(4<<50)|(8<<36)|(6<<28)|(255<<20)|(7<<16)|4;
            let store=(0xeb20060600f70a04u64&!((7u64<<33)|(255<<8)))|(1<<33);
            for (format,bytes,pixel) in [(46u32,16usize,vec![0x3f000000u32,0x3f400000,0x3e800000,0x3f800000].into_iter().flat_map(u32::to_le_bytes).collect::<Vec<_>>()),(34,12,vec![0x3f000000u32,0x3f400000,0x3e800000].into_iter().flat_map(u32::to_le_bytes).collect()),(1,1,vec![127]),(5,2,vec![0,0x38]),(4,1,vec![0x80]),(11,4,0xf1234567u32.to_le_bytes().to_vec())] {
                let source=builder(0xd100,format);let destination=builder(0xd101,format);
                {let mut state=STATE.lock().unwrap();state.registered_textures.insert((0xd180,0x100),source);state.registered_textures.insert((0xd180,0x101),destination);}
                std::ptr::write_bytes(output as *mut u8,0xcd,size as usize);
                for index in [0u32,16383,16384,width-1,width] {
                    std::ptr::copy_nonoverlapping(pixel.as_ptr(),(cpu+index as u64*bytes as u64) as *mut u8,bytes);
                    let shader=maxwell_program(&[mov32(0,index),fetch,mov_handle(12,0x120),store]);
                    control[0x700..0x704].copy_from_slice(&(shader.len() as u32).to_le_bytes());r.e.st.set_program(&[(shader.as_ptr() as u64,control.as_ptr() as u64)]);r.e.bank_ok[2]=false;
                    r.dispatch([1,1,1]).unwrap();r.run(&[Cmd::Barrier(0x10)],0);
                    if index<width {assert_eq!(r.cpu_bytes(output+index as u64*bytes as u64,bytes).unwrap(),pixel,"format{format} index{index}");}
                }
                r.prepare_cpu_read(output,size);
                assert_eq!(((output+width as u64*bytes as u64) as *const u8).read(),0xcd,"buffer image wrote past the logical view");
            }
            // A GPU buffer writer and the buffer texture read the same bytes,
            // without a CPU upload or a private texture copy in between.
            let source=builder(0xd100,46);let destination=builder(0xd101,46);
            {let mut state=STATE.lock().unwrap();state.registered_textures.insert((0xd180,0x100),source);state.registered_textures.insert((0xd180,0x101),destination);}
            let index=width-1;r.clear_buffer(cpu+index as u64*16,16,0x3f000000).unwrap();
            let shader=maxwell_program(&[mov32(0,index),fetch,mov_handle(12,0x120),store]);control[0x700..0x704].copy_from_slice(&(shader.len() as u32).to_le_bytes());r.e.st.set_program(&[(shader.as_ptr() as u64,control.as_ptr() as u64)]);r.e.bank_ok[2]=false;
            r.dispatch([1,1,1]).unwrap();r.copy_buffer(output+index as u64*16,cpu+0x80000,16).unwrap();r.commit_boundary(true);
            assert_eq!(std::slice::from_raw_parts((cpu+0x80000) as *const u32,4),[0x3f000000;4]);
            // TXQ's C2 image reference follows the thirty-two sampled slots.
            // Query image0 and write its full logical width through image1.
            let queried=builder(0xd100,46);let destination=builder(0xd101,11);
            {let mut state=STATE.lock().unwrap();state.registered_textures.insert((0xd180,0x100),queried);state.registered_textures.insert((0xd180,0x101),destination);}
            r.e.st.images[2][0]=0x100;r.e.st.images[2][1]=0x101;
            let query=0xdf48000000000000u64|(72<<36)|(1<<31)|(1<<22)|(7<<16)|(5<<8)|4;
            let shader=maxwell_program(&[mov32(5,0),query,mov32(0,0),mov_handle(12,0x128),store]);control[0x700..0x704].copy_from_slice(&(shader.len() as u32).to_le_bytes());r.e.st.set_program(&[(shader.as_ptr() as u64,control.as_ptr() as u64)]);r.e.bank_ok[2]=false;r.dispatch([1,1,1]).unwrap();r.commit_boundary(true);
            assert_eq!((output as *const u32).read(),width,"image-slot TXQ lost the logical buffer width");
            r.e.st.images[2][0]=0x101;
            // Neighboring R8 pixels share a 32-bit storage word. Exercise
            // concurrent lane writes and preserve the surrounding sentinel.
            let source=builder(0xd100,1);let destination=builder(0xd101,1);
            {let mut state=STATE.lock().unwrap();state.registered_textures.insert((0xd180,0x100),source);state.registered_textures.insert((0xd180,0x101),destination);}
            std::ptr::write_bytes(output as *mut u8,0xcd,size as usize);
            let values=(0..64).map(|i|(i*3+1) as u8).collect::<Vec<_>>();std::ptr::copy_nonoverlapping(values.as_ptr(),(cpu+16384) as *mut u8,64);
            let shader=maxwell_program(&[0xf0c8000002170000,(0x1c00000000070000u64|(16384<<20)),fetch,mov_handle(12,0x120),store]);control[0x700..0x704].copy_from_slice(&(shader.len() as u32).to_le_bytes());control[0x748..0x74c].copy_from_slice(&64u32.to_le_bytes());r.e.st.set_program(&[(shader.as_ptr() as u64,control.as_ptr() as u64)]);r.e.bank_ok[2]=false;r.dispatch([1,1,1]).unwrap();r.commit_boundary(true);
            assert_eq!(std::slice::from_raw_parts((output+16384) as *const u8,64),values.as_slice());assert_eq!(((output+16383) as *const u8).read(),0xcd);assert_eq!(((output+16448) as *const u8).read(),0xcd);
            // A narrow R8 view touches whole aligned atomic words. Its
            // barrier must submit without a CPU wait, then the real CPU read
            // waits for those exact words and observes preserved padding.
            let mut destination=builder(0xd101,1);destination[16..24].copy_from_slice(&3u64.to_le_bytes());destination[0x20..0x24].copy_from_slice(&5u32.to_le_bytes());STATE.lock().unwrap().registered_textures.insert((0xd180,0x101),destination);
            std::ptr::write_bytes(output as *mut u8,0xcd,16);
            let shader=maxwell_program(&[mov32(0,0),mov32(4,(90.0f32/255.0).to_bits()),mov32(5,0),mov32(6,0),mov32(7,0),mov_handle(12,0x120),store]);control[0x700..0x704].copy_from_slice(&(shader.len() as u32).to_le_bytes());control[0x748..0x74c].copy_from_slice(&1u32.to_le_bytes());r.e.st.set_program(&[(shader.as_ptr() as u64,control.as_ptr() as u64)]);r.e.bank_ok[2]=false;
            let gate=gfx::Shared(gfx::gpu().device.0.newSharedEvent().unwrap());r.cb.encodeWaitForEvent_value(ProtocolObject::from_ref(&*gate.0),1);r.dispatch([1,1,1]).unwrap();let before=Retained::as_ptr(&r.cb);
            let event=gate.clone();let (release,receive)=std::sync::mpsc::channel();let worker=std::thread::spawn(move||{let _=receive.recv_timeout(std::time::Duration::from_secs(2));event.0.setSignaledValue(1);});
            r.run(&[Cmd::Barrier(0x10)],0);if write_windows::enabled(){assert_eq!(gate.0.signaledValue(),0);assert_ne!(before,Retained::as_ptr(&r.cb));assert!(!r.e.gpu_buffer_dirty);}
            let _=release.send(());worker.join().unwrap();let bytes=r.cpu_bytes(output,12).unwrap();assert_eq!(bytes,[0xcd,0xcd,0xcd,0x5a,0xcd,0xcd,0xcd,0xcd,0xcd,0xcd,0xcd,0xcd]);r.commit_boundary(true);
            eprintln!("[verify] buffer-image fenced asynchronous barrier and exact CPU readback preserve unalignedR8 atomic word padding");
            STATE.lock().unwrap().registered_textures.retain(|(pool,_),_|*pool!=0xd180);STATE.lock().unwrap().pools.remove(&0xd100);STATE.lock().unwrap().pools.remove(&0xd101);gfx::forget_pool(0xd100);gfx::forget_pool(0xd101);
        });
    }

    #[cfg(target_os="macos")]
    #[test]
    #[ignore="requires a real Metal GPU"]
    fn gpu_zero_register_comparison_flags_bound_short_loop_after_other_lanes_exit() {
        use super::*;
        objc2::rc::autoreleasepool(|_| unsafe {
            crate::kernel::KERNEL.aspace.get_or_init(||crate::mem::AddressSpace::reserve().unwrap());let page=crate::mem::PAGE;let cpu=crate::kernel::aspace().alias()+0x2400000;
            crate::kernel::aspace().set(cpu,page*2,crate::mem::Block{state:crate::mem::state::NORMAL,perm:crate::mem::perm::RW,attr:0}).unwrap();
            let mut b=[0u8;0x68];b[8..16].copy_from_slice(&0xa200u64.to_le_bytes());for (o,v) in [(0x1c,1u32),(0x20,1),(0x24,1),(0x28,1),(0x2c,1),(0x30,47)] {b[o..o+4].copy_from_slice(&v.to_le_bytes());}
            {let mut s=STATE.lock().unwrap();s.pools.insert(0xa200,super::super::objects::MemoryPool{cpu,size:page,flags:0x22});s.registered_textures.insert((0xa280,0x100),b);}
            let output=cpu+page;let pools=gfx::pool_memory(&[(0xa201,output,page)]).unwrap();let mut e=Engine::default();e.st.texture_pool=0xa280;e.st.images[2][0]=0x100;
            let cb=gfx::gpu().queue.0.commandBuffer().unwrap();let mut r=Replay{e:&mut e,cb,pass:None,heap:None,pools,passes:0,copies:HashMap::new(),fault:None,pool_table:None,resource_buffers:Default::default()};
            let mut control=vec![0u8;0x880];for (o,v) in [(0x748,64u32),(0x74c,1),(0x750,1)] {control[o..o+4].copy_from_slice(&v.to_le_bytes());}
            for with_shared in [false,true] {
             control[0x754..0x758].copy_from_slice(&(if with_shared {256u32} else {0}).to_le_bytes());
             for (floating,float_bool) in [(false,false),(false,true),(true,false),(true,true)] {
                let pred=(0x3662038002071107u64&!((7<<49)|(255<<8)|(0x7ffff<<20)))|(5<<49);
                // ISET.LT.CC RZ,R4,2 followed by BRA.NEU: the result register
                // is discarded, but ZF must still terminate this two-pass loop.
                let compare=if floating {
                    let clear=(15u64<<48)|(0x7ffff<<20)|(1<<56)|(255<<8)|(1<<43)|(1<<44)|(1<<53)|(1<<54)|(1<<52);
                    (0x30c283ff80071affu64&!clear)|(1<<48)|(0x40000<<20)|(4<<8)|((float_bool as u64)<<52)
                } else {0x36538380002704ffu64|((float_bool as u64)<<44)};
                let increment=if floating {0x0883f80000070404u64} else {0x1c00000000170404};
                let mut words=vec![0xf0c8000002170000];
                if with_shared {words.extend([0x3848000000270014,0xef5c000000071400,0xf0a81b8000070000]);}
                words.extend([pred,0xe30000000000000f,mov32(4,0)]);let start=words.len();words.push(increment);
                if with_shared {words.push(0xef4c10000047ff06);}
                words.push(compare);let at=words.len();let address=|n:usize|((n/3*4+1+n%3)*8) as i32;
                let delta=address(start)-address(at)-8;words.push(0xe24000000007000du64|(((delta as u32&0xffffff) as u64)<<20));words.push(0x50d9e38000070005);
                if !with_shared {words.push(mov32(6,0x12345678));}
                words.extend([mov32(7,0),mov_handle(12,0x120),mov32(10,0),mov32(11,0),0xeb20060600f70a04]);
                let shader=maxwell_program(&words);control[0x700..0x704].copy_from_slice(&(shader.len() as u32).to_le_bytes());r.e.st.set_program(&[(shader.as_ptr() as u64,control.as_ptr() as u64)]);r.dispatch([1,1,1]).unwrap();
                let (_,tex)=r.e.textures.get(&b,true).unwrap();copy::copy_to_buffer(&r.cb,&tex.0,(0,0),[0,0,0,1,1,1],47,output,&r.pools,(0,0)).unwrap();r.commit_boundary(true);assert_eq!(std::slice::from_raw_parts(output as *const u32,4),[if floating {2f32.to_bits()} else {2},1,if with_shared {1} else {0x12345678},0]);
             }
            }
            STATE.lock().unwrap().registered_textures.remove(&(0xa280,0x100));STATE.lock().unwrap().pools.remove(&0xa200);gfx::forget_pool(0xa201);
        });
    }

    #[cfg(target_os="macos")]
    #[test]
    #[ignore="requires a real Metal GPU"]
    fn gpu_fault_records_publish_own_pc_and_shader_identity_without_touching_zero_buffer() {
        use super::*;
        objc2::rc::autoreleasepool(|_| unsafe {
            let dev=&gfx::gpu().device.0;
            let source=format!("{}\nkernel void fault_probe(constant Res& res [[buffer(0)]],uint lane [[thread_position_in_grid]]) {{shader_fault(res,2u,uint(res.shader_hash));}}",nxshader::prelude());
            let lib=dev.newLibraryWithSource_options_error(&objc2_foundation::NSString::from_str(&source),None).unwrap();let f=lib.newFunctionWithName(&objc2_foundation::NSString::from_str("fault_probe")).unwrap();let p=dev.newComputePipelineStateWithFunction_error(&f).unwrap();
            let records=(0..2).map(|_|dev.newBufferWithLength_options(16,MTLResourceOptions::StorageModeShared).unwrap()).collect::<Vec<_>>();let mut commands=Vec::new();
            for (i,record) in records.iter().enumerate() {
                std::ptr::write_bytes(record.contents().as_ptr(),0,16);let mut res:state::Res=std::mem::zeroed();res.fault_gpu=record.gpuAddress();res.shader_hash=0x123456780000abcdu64+i as u64;res.zero_gpu=gfx::gpu().zeros.0.gpuAddress();
                let cb=gfx::gpu().queue.0.commandBuffer().unwrap();let enc=cb.computeCommandEncoder().unwrap();enc.setComputePipelineState(&p);enc.useResource_usage(ProtocolObject::from_ref(&**record),MTLResourceUsage::Read|MTLResourceUsage::Write);enc.setBytes_length_atIndex(std::ptr::NonNull::from(&res).cast(),size_of_val(&res),0);enc.dispatchThreads_threadsPerThreadgroup(MTLSize{width:64,height:1,depth:1},MTLSize{width:64,height:1,depth:1});enc.endEncoding();cb.commit();commands.push(cb);
            }
            for cb in commands {cb.waitUntilCompleted();assert_eq!(cb.status(),MTLCommandBufferStatus::Completed);}
            for (i,record) in records.iter().enumerate() {assert_eq!(std::slice::from_raw_parts(record.contents().as_ptr().cast::<u32>(),4),[2,0xabcd+i as u32,0xabcd+i as u32,0x12345678]);}
            let zeros=std::slice::from_raw_parts(gfx::gpu().zeros.0.contents().as_ptr().cast::<u8>(),65536);assert!(zeros.iter().all(|n|*n==0));
        });
    }

    #[cfg(target_os="macos")]
    #[test]
    #[ignore="requires a real Metal GPU"]
    fn gpu_bgra_cpu_upload_sampling_srgb_and_opaque_x_channel() {
        use super::*;
        objc2::rc::autoreleasepool(|_| unsafe {
            crate::kernel::KERNEL.aspace.get_or_init(||crate::mem::AddressSpace::reserve().unwrap());let page=crate::mem::PAGE;let cpu=crate::kernel::aspace().alias()+0x1c00000;
            crate::kernel::aspace().set(cpu,page*4,crate::mem::Block{state:crate::mem::state::NORMAL,perm:crate::mem::perm::RW,attr:0}).unwrap();
            let dev=&gfx::gpu().device.0;
            let lib=dev.newLibraryWithSource_options_error(&objc2_foundation::NSString::from_str("#include <metal_stdlib>\nusing namespace metal;kernel void bgra_probe(texture2d<float> source [[texture(0)]],device float4* out [[buffer(0)]],uint n [[thread_position_in_grid]]) {if(n<4) out[n]=source.read(uint2(n%2,n/2));}"),None).unwrap();
            let f=lib.newFunctionWithName(&objc2_foundation::NSString::from_str("bgra_probe")).unwrap();let pipeline=dev.newComputePipelineStateWithFunction_error(&f).unwrap();let output=dev.newBufferWithLength_options(64,MTLResourceOptions::StorageModeShared).unwrap();
            let pixels=[0x80ff4000u32,0x0000ff80,0xffff0000,0x400080ff];let mut textures=textures::Textures::default();
            for (i,format) in (153u32..=156).enumerate() {
                let address=cpu+i as u64*page;std::ptr::copy_nonoverlapping(pixels.as_ptr(),address as *mut u32,4);
                let pool=0xa100+i as u64;STATE.lock().unwrap().pools.insert(pool,super::super::objects::MemoryPool{cpu:address,size:page,flags:0x22});
                let mut b=[0u8;0x68];b[8..16].copy_from_slice(&pool.to_le_bytes());for (o,v) in [(0x18,0x10u32),(0x1c,1),(0x20,2),(0x24,2),(0x28,1),(0x2c,1),(0x30,format),(0x5c,8)] {b[o..o+4].copy_from_slice(&v.to_le_bytes());}
                assert!(!textures::is_compressed(format));let (key,tex)=textures.get(&b,false).unwrap();let tex=textures.view(key,&b,&tex,nxshader::TexKind::T2D).unwrap();
                let cb=gfx::gpu().queue.0.commandBuffer().unwrap();let enc=cb.computeCommandEncoder().unwrap();enc.setComputePipelineState(&pipeline);enc.setTexture_atIndex(Some(&tex.0),0);enc.setBuffer_offset_atIndex(Some(&output),0,0);enc.dispatchThreads_threadsPerThreadgroup(MTLSize{width:4,height:1,depth:1},MTLSize{width:4,height:1,depth:1});enc.endEncoding();cb.commit();cb.waitUntilCompleted();assert_eq!(cb.status(),MTLCommandBufferStatus::Completed);
                let values=std::slice::from_raw_parts(output.contents().as_ptr().cast::<f32>(),16);
                for n in 0..4 {for c in 0..4 {
                    let shift=[16,8,0,24][c];let mut expected=((pixels[n]>>shift)&255) as f32/255.0;
                    if c==3 && matches!(format,153|155) {expected=1.0;}
                    if c<3 && format>=155 {expected=if expected<=0.04045 {expected/12.92} else {((expected+0.055)/1.055).powf(2.4)};}
                    assert!((values[n*4+c]-expected).abs()<0.001,"format{format} pixel{n} channel{c}: {} vs{expected}",values[n*4+c]);
                }}
                STATE.lock().unwrap().pools.remove(&pool);
            }
        });
    }

    #[cfg(target_os="macos")]
    #[test]
    #[ignore="requires a real Metal GPU"]
    fn gpu_depth_stencil_to_float_depth_copy_preserves_planes_layers_and_crop() {
        use super::*;
        objc2::rc::autoreleasepool(|_| unsafe {
            let dev=&gfx::gpu().device.0;
            let make=|format| {let d=MTLTextureDescriptor::new();d.setTextureType(MTLTextureType::Type2DArray);d.setPixelFormat(format);d.setWidth(8);d.setHeight(8);d.setArrayLength(2);d.setMipmapLevelCount(2);d.setStorageMode(MTLStorageMode::Private);d.setUsage(MTLTextureUsage::RenderTarget|MTLTextureUsage::ShaderRead|MTLTextureUsage::PixelFormatView);Shared(dev.newTextureWithDescriptor(&d).unwrap())};
            let source=make(MTLPixelFormat::Depth32Float_Stencil8);let destination=make(MTLPixelFormat::Depth32Float);
            let cb=gfx::gpu().queue.0.commandBuffer().unwrap();
            for (layer,depth) in [0.25f64,0.75].into_iter().enumerate() {
                for (tex,value,stencil) in [(&source,depth,true),(&destination,0.9,false)] {
                    gfx::run_pass(&cb,&Attachments{colors:vec![],depth:Some((tex.clone(),Load::ClearDepthStencil(value,if layer==0 {7} else {99}),stencil)),subs:vec![],depth_sub:(layer as u32,1)});
                }
            }
            copy::Copier::default().copy(&cb,&source,(0,1),[1,0,0,3,4,2],&destination,(0,1),[0,0,0,3,4,2],0).unwrap();
            let output=dev.newBufferWithLength_options(128,MTLResourceOptions::StorageModeShared).unwrap();
            copy::copy_into(&cb,&destination.0,(0,1),[0,0,0,4,4,2],52,&output,0).unwrap();
            cb.commit();cb.waitUntilCompleted();assert_eq!(cb.status(),MTLCommandBufferStatus::Completed);
            let values=std::slice::from_raw_parts(output.contents().as_ptr().cast::<f32>(),32);
            for layer in 0..2 {for y in 0..4 {for x in 0..4 {let expected=if x==3 {0.9f32} else if layer==0 {0.25} else {0.75};assert_eq!(values[layer*16+y*4+x].to_bits(),expected.to_bits(),"layer{layer} pixel{x},{y}");}}}
        });
    }

    #[cfg(target_os="macos")]
    #[test]
    #[ignore="requires real Metal GPU and NX_STARTUP_SHADER_CORPUS pointing to captured local shaders"]
    fn gpu_original_startup_shader_corpus_compiles() {
        use super::*;
        let root=std::path::PathBuf::from(std::env::var("NX_STARTUP_SHADER_CORPUS").expect("set NX_STARTUP_SHADER_CORPUS"));
        objc2::rc::autoreleasepool(|_| {
            let mut files=std::fs::read_dir(&root).unwrap().filter_map(Result::ok).map(|e|e.path()).filter(|p|p.extension().is_some_and(|e|e=="code")).collect::<Vec<_>>();files.sort();
            let expected=std::env::var("NX_STARTUP_SHADER_COUNT").ok().and_then(|n|n.parse::<usize>().ok()).unwrap_or(13);
            assert_eq!(files.len(),expected,"original stage corpus count mismatch");
            for (index,file) in files.iter().enumerate() {
                let bytes=std::fs::read(file).unwrap();let compute=bytes.starts_with(&0x12345679u32.to_le_bytes());
                let blob=if compute {&bytes[0x100..]} else {&bytes[..]};
                let memory=std::fs::read(file.with_extension("control")).ok().filter(|c|c.len()>=0x758).map(|c| {
                    (u32::from_le_bytes(c[0x710..0x714].try_into().unwrap()),u32::from_le_bytes(c[0x754..0x758].try_into().unwrap()))
                }).unwrap_or((0,0));
                let compiled=shaders::compile_config(blob,compute,0,0,0,0xfc270000+index as u64,memory);
                assert!(compiled.is_some(),"original startup shader {} did not translate and compile on this GPU",file.display());
                let compiled=compiled.unwrap();
                if std::env::var_os("NX_DYNAMIC_VERTEX_CACHE").is_some() && compiled.func.0.functionType()==MTLFunctionType::Vertex {
                    assert!(compiled.dynamic_vertex.is_some(),"original vertex {} fell back during dynamic corpus verification",file.display());
                }
                if let Some(library)=&compiled.vertex_formats {assert!(library.function(vertex_formats::enabled().then_some(&[10,22,34,46,37,13,29,43,62,105,106,108,41,1,0,9]),vertex_formats::buffer_enabled().then_some(&[0;32])).is_some(),"original vertex format specialization failed: {}",file.display());}
                eprintln!("[verify] original startup shader {} compiled dynamic_vertex={}",file.file_name().unwrap().to_string_lossy(),compiled.dynamic_vertex.is_some());
            }
        });
    }

    #[cfg(target_os="macos")]
    #[test]
    #[ignore="requires a real Metal GPU"]
    fn gpu_bindless_derivative_mip_selection_and_offset_color_depth_gathers() {
        use super::*;
        objc2::rc::autoreleasepool(|_| unsafe {
            crate::kernel::KERNEL.aspace.get_or_init(||crate::mem::AddressSpace::reserve().unwrap());let page=crate::mem::PAGE;let cpu=crate::kernel::aspace().alias()+0x1800000;
            crate::kernel::aspace().set(cpu,page*4,crate::mem::Block{state:crate::mem::state::NORMAL,perm:crate::mem::perm::RW,attr:0}).unwrap();
            let builder=|pool:u64,format:u32,width:u32,levels:u32| {let mut b=[0u8;0x68];b[8..16].copy_from_slice(&pool.to_le_bytes());for (o,v) in [(0x1c,1u32),(0x20,width),(0x24,width),(0x28,1),(0x2c,levels),(0x30,format)] {b[o..o+4].copy_from_slice(&v.to_le_bytes());}b};
            let color=builder(0x9f00,37,4,3);let target=builder(0x9f01,47,1,1);let depth=builder(0x9f02,52,4,1);
            for (i,b) in [color,target,depth].into_iter().enumerate() {let mut s=STATE.lock().unwrap();s.pools.insert(0x9f00+i as u64,super::super::objects::MemoryPool{cpu:cpu+i as u64*page,size:page,flags:0x22});s.registered_textures.insert((0x9f80,0x100+i as u32),b);s.registered_textures.insert((0x9f80,0x20100+i as u32),b);}
            {let mut state=STATE.lock().unwrap();
                for id in [0x9ffe,0x9fff] {state.pools.insert(id,super::super::objects::MemoryPool{cpu:0,size:0x10000,flags:0x221});}
                state.virtual_maps.insert(0x9ffe,vec![(0,4096,0x9f00,0)]);
                state.registered_textures.insert((0x9f80,0x203fe),builder(0x9ffe,37,64,7));
                state.registered_textures.insert((0x9f80,0x203ff),builder(0x9fff,37,4,3));
            }
            let io=cpu+3*page;let output=io+512;let pools=gfx::pool_memory(&[(0x9f03,io,page)]).unwrap();
            for n in 0..16 {((io as *mut u32).add(n)).write(0xff000000|n as u32+1);((io+256) as *mut f32).add(n).write((n as f32+1.0)*0.05);}
            for n in 16..20 {(io as *mut u32).add(n).write(0xff00ff00);}((io+80) as *mut u32).write(0xffff0000);
            let mut sampler=[0u8;0x60];sampler[8..12].copy_from_slice(&2u32.to_le_bytes());sampler[0x20..0x24].copy_from_slice(&4f32.to_le_bytes());sampler[0x28..0x2c].copy_from_slice(&1u32.to_le_bytes());sampler[0x2c..0x30].copy_from_slice(&4u32.to_le_bytes());
            STATE.lock().unwrap().registered_samplers.insert((0x9f90,2),sampler);
            let mut e=Engine::default();e.st.texture_pool=0x9f80;e.st.sampler_pool=0x9f90;e.st.images[2][0]=0x101;e.st.textures[2][0]=0x100|(2<<20);
            let cb=gfx::gpu().queue.0.commandBuffer().unwrap();let mut r=Replay{e:&mut e,cb,pass:None,heap:None,pools,passes:0,copies:HashMap::new(),fault:None,pool_table:None,resource_buffers:Default::default()};
            for (level,size,offset) in [(0,4,0),(1,2,64),(2,1,80)] {let mut view=color;view[0x66]=level;view[0x67]=0xb7;r.copy_buffer_to_texture(io+offset,&view,[0,0,0,size,size,1],(0,0)).unwrap();}
            r.copy_buffer_to_texture(io+256,&depth,[0,0,0,4,4,1],(0,0)).unwrap();
            let mut control=vec![0u8;0x880];for o in [0x748,0x74c,0x750] {control[o..o+4].copy_from_slice(&1u32.to_le_bytes());}
            let txd=(0xde780007a2472824&!(255u64|(255<<8)|(255<<20)))|8|(4<<20);
            let derivative=vec![mov_handle(0,0x20),mov32(1,0.5f32.to_bits()),mov32(2,0.5f32.to_bits()),mov32(3,0),mov32(4,0.5f32.to_bits()),mov32(5,0),mov32(6,0),mov32(7,0.5f32.to_bits()),txd];
            let gather=|compare:bool| {
                let op=(0xdefa0017a0270014&!(255u64|(255<<8)|(255<<20)|(1<<50)))|8|(2<<20)|((compare as u64)<<50);
                vec![mov32(0,0.5f32.to_bits()),mov32(1,0.5f32.to_bits()),mov_handle(2,0x20),mov32(3,0xff01),mov32(4,0.375f32.to_bits()),op]
            };
            let mut cases=vec![(derivative,0x100u64,[0f32,1.0,0.0,1.0]),(gather(false),0x100,[7.0/255.0,8.0/255.0,4.0/255.0,3.0/255.0]),(gather(true),0x102,[0.0,1.0,0.0,0.0])];
            for (offset,red,compare_value) in [(0xf1u32,3.0/255.0,0.0),(0x1f,9.0/255.0,1.0)] {
                for compare in [false,true] {
                    let address=io+1032+if offset==0xf1 {0} else {4};(address as *mut u32).write(offset);
                    let op=(0xdeba0017a0670404u64&!(255u64|(255<<8)|(255<<20)|(7<<37)|(1<<50)))|8|(2<<20)|(1<<37)|((compare as u64)<<50);
                    let words=vec![mov32(0,0.375f32.to_bits()),mov32(1,0.375f32.to_bits()),mov_handle(2,0x20),mov32(26,address as u32),mov32(27,(address>>32) as u32),0xeed4200000071a03,mov32(4,0.375f32.to_bits()),op];
                    cases.push((words,if compare {0x102} else {0x100},if compare {[compare_value;4]} else {[red,0.0,0.0,1.0]}));
                }
            }
            for (base,handle,expected) in cases {
              for dynamic in [false,true] {
                let mut words=base.clone();
                if dynamic {
                    let reg=if words[0]==mov_handle(0,0x20) {0u64} else {2};
                    let index=words.iter().position(|w|*w==mov_handle(reg,0x20)).unwrap();
                    words[index]=0xeed4200000071800|reg;
                    let address=io+1024;(address as *mut u32).write((handle as u32+0x20000)|(2<<20));
                    words.splice(..0,[mov32(24,address as u32),mov32(25,(address>>32) as u32)]);
                }
                r.e.st.textures[2][0]=handle|(2<<20);r.e.bank_ok[2]=false;
                words.extend([mov_handle(12,0x120),mov32(14,0),mov32(15,0),0xeb20060600f70e08]);let shader=maxwell_program(&words);control[0x700..0x704].copy_from_slice(&(shader.len() as u32).to_le_bytes());
                r.e.st.set_program(&[(shader.as_ptr() as u64,control.as_ptr() as u64)]);r.dispatch([1,1,1]).unwrap();let (_,tex)=r.e.textures.get(&target,true).unwrap();copy::copy_to_buffer(&r.cb,&tex.0,(0,0),[0,0,0,1,1,1],47,output,&r.pools,(0,0)).unwrap();r.commit_boundary(true);
                let actual=std::slice::from_raw_parts(output as *const f32,4);for (channel,(got,want)) in actual.iter().zip(expected).enumerate() {assert!((*got-want).abs()<1e-6,"channel {channel}: got {actual:?}, expected {expected:?}");}
              }
            }
            // TLD uses integer coordinates, integer LOD and packed destination
            // components. Coordinates deliberately alias its result registers.
            for (x,y,lod,offset,mask,expected) in [
                (1u32,2,0,None,15u64,[10.0/255.0,0.0,0.0,1.0]),
                (1,1,1,None,15,[0.0,1.0,0.0,1.0]),
                (1,1,0,Some(0x1fu32),15,[9.0/255.0,0.0,0.0,1.0]),
                (u32::MAX,0,0,None,15,[0.0;4]),
                (0,0,4,None,15,[0.0;4]),
                (1,2,0,None,5,[10.0/255.0,0.0,0.25,0.5]),
            ] {
                let clear=255u64|(255<<8)|(255<<20)|(15<<31)|(8191<<36)|(1<<55)|(1<<35);
                let load=(0xdc3805a2aff71002u64&!clear)|8|(8<<8)|(2<<20)|(mask<<31)|(8<<36)|(1<<55)|((offset.is_some() as u64)<<35);
                let shader=maxwell_program(&[mov32(8,x),mov32(9,y),mov32(10,0.25f32.to_bits()),mov32(11,0.5f32.to_bits()),mov32(2,lod),mov32(3,offset.unwrap_or(0)),load,mov_handle(12,0x120),mov32(14,0),mov32(15,0),0xeb20060600f70e08]);
                r.e.st.textures[2][0]=0x100|(2<<20);r.e.bank_ok[2]=false;control[0x700..0x704].copy_from_slice(&(shader.len() as u32).to_le_bytes());r.e.st.set_program(&[(shader.as_ptr() as u64,control.as_ptr() as u64)]);r.dispatch([1,1,1]).unwrap();
                let (_,tex)=r.e.textures.get(&target,true).unwrap();copy::copy_to_buffer(&r.cb,&tex.0,(0,0),[0,0,0,1,1,1],47,output,&r.pools,(0,0)).unwrap();r.commit_boundary(true);
                let actual=std::slice::from_raw_parts(output as *const f32,4);for (got,want) in actual.iter().zip(expected) {assert!((*got-want).abs()<1e-6,"TLD {x},{y} lod{lod} mask{mask}: {actual:?} != {expected:?}");}
            }
            for compare in [false,true] {
                let mut array=builder(if compare {0x9f02}else{0x9f00},if compare {52}else{37},4,1);
                array[0x10..0x18].copy_from_slice(&0x800u64.to_le_bytes());array[0x1c..0x20].copy_from_slice(&4u32.to_le_bytes());array[0x28..0x2c].copy_from_slice(&2u32.to_le_bytes());
                STATE.lock().unwrap().registered_textures.insert((0x9f80,0x103),array);
                for n in 0..32 {((io+2048) as *mut u32).add(n).write(if compare {(if n<16 {0.2f32}else{0.8f32}).to_bits()} else {0xff000000|if n<16 {1}else{2}});}
                r.copy_buffer_to_texture(io+2048,&array,[0,0,0,4,4,2],(0,0)).unwrap();
                for layer in 0..2 {
                    let clear=255u64|(255<<8)|(255<<20)|(7<<28)|(1<<50);
                    let gather=(0xdefa0017a0270014u64&!clear)|8|(4<<20)|(3<<28)|((compare as u64)<<50);
                    let shader=maxwell_program(&[mov32(0,layer),mov32(1,0.5f32.to_bits()),mov32(2,0.5f32.to_bits()),mov_handle(4,0x20),mov32(5,0),mov32(6,0.5f32.to_bits()),gather,mov_handle(12,0x120),mov32(14,0),mov32(15,0),0xeb20060600f70e08]);
                    r.e.st.textures[2][0]=0x103|(2<<20);r.e.bank_ok[2]=false;control[0x700..0x704].copy_from_slice(&(shader.len() as u32).to_le_bytes());r.e.st.set_program(&[(shader.as_ptr() as u64,control.as_ptr() as u64)]);r.dispatch([1,1,1]).unwrap();
                    let (_,tex)=r.e.textures.get(&target,true).unwrap();copy::copy_to_buffer(&r.cb,&tex.0,(0,0),[0,0,0,1,1,1],47,output,&r.pools,(0,0)).unwrap();r.commit_boundary(true);
                    let expected=if compare {layer as f32}else{(layer+1) as f32/255.0};let actual=std::slice::from_raw_parts(output as *const f32,4);assert!(actual.iter().all(|got|(*got-expected).abs()<1e-6),"array gather compare{compare} layer{layer}: {actual:?} != {expected}");
                }
            }
            for id in 0x9f00..0x9f04 {STATE.lock().unwrap().pools.remove(&id);gfx::forget_pool(id);}
            let mut s=STATE.lock().unwrap();s.registered_textures.retain(|(p,_),_|*p!=0x9f80);s.registered_samplers.remove(&(0x9f90,2));
            for id in [0x9ffe,0x9fff] {s.pools.remove(&id);s.virtual_maps.remove(&id);s.virtual_bases.remove(&id);}
        });
    }

    #[cfg(target_os="macos")]
    #[test]
    #[ignore="requires a real Metal GPU"]
    fn gpu_scalar_compare_select_bit_insert_and_funnel_boundaries() {
        use super::*;
        fn select(base:u64,d:u64,a:u64,b:u64,c:u64,compare:u64,signed:bool)->u64 {
            let floating=base>>56==0x5b && (base>>52)&15==0xa;
            let compare_mask=if floating {15u64<<48} else {(7<<49)|(1<<48)};
            let clear=255u64|(255<<8)|(255<<20)|(255<<39)|compare_mask;
            (base&!clear)|d|(a<<8)|(b<<20)|(c<<39)|if floating {compare<<48} else {(compare<<49)|((signed as u64)<<48)}
        }
        objc2::rc::autoreleasepool(|_| unsafe {
            crate::kernel::KERNEL.aspace.get_or_init(||crate::mem::AddressSpace::reserve().unwrap());let page=crate::mem::PAGE;let cpu=crate::kernel::aspace().alias()+0x1400000;
            crate::kernel::aspace().set(cpu,page*2,crate::mem::Block{state:crate::mem::state::NORMAL,perm:crate::mem::perm::RW,attr:0}).unwrap();
            STATE.lock().unwrap().pools.insert(0x9b00,super::super::objects::MemoryPool{cpu,size:page,flags:0x22});let mut b=[0u8;0x68];b[8..16].copy_from_slice(&0x9b00u64.to_le_bytes());
            for (o,v) in [(0x1c,1u32),(0x20,1),(0x24,1),(0x28,1),(0x2c,1),(0x30,47)] {b[o..o+4].copy_from_slice(&v.to_le_bytes());}
            STATE.lock().unwrap().registered_textures.insert((0x9b80,0x100),b);
            let output=cpu+page;let pools=gfx::pool_memory(&[(0x9b01,output,page)]).unwrap();let mut e=Engine::default();e.st.texture_pool=0x9b80;e.st.images[2][0]=0x100;
            let cb=gfx::gpu().queue.0.commandBuffer().unwrap();let mut r=Replay{e:&mut e,cb,pass:None,heap:None,pools,passes:0,copies:HashMap::new(),fault:None,pool_table:None,resource_buffers:Default::default()};
            let mut control=vec![0u8;0x880];for o in [0x748,0x74c,0x750] {control[o..o+4].copy_from_slice(&1u32.to_le_bytes());}
            let mut cases=vec![(vec![mov32(0,0x12345678),mov32(1,0xaabbccdd),mov32(2,f32::NAN.to_bits()),mov32(3,u32::MAX),select(0x5ba481000ff70000,4,0,1,2,13,false),select(0x5b4a058000070d04,5,0,1,3,1,true),0x5c30000000070006,0x5c08000000170007],[0x12345678u32,0x12345678,28,0xaabbccddu32.count_ones()])];
            let i2f=|d:u64,source:u64,format:u64,destination:u64,round:u64|0x5cb8000000070000u64|d|(source<<20)|((format&3)<<10)|((format&4)<<11)|(destination<<8)|(round<<39);
            // The exact original immediate I2F.F16.S16 -> F2I.U32.F16 pair
            // controls FC27's second-stage workgroup luminance reduction.
            for (mut words,half_bits,roundtrip) in [
                (vec![0x38b8000000172504],0x3c00,1),
                (vec![mov32(4,(-7i32) as u32),i2f(4,4,5,1,0)],0xc700,0),
                (vec![0x4cb8000c00072504],0x7800,32768),
            ] {
                words.extend([0x5cb0018000470605,mov32(6,16777219),mov32(7,u32::MAX),i2f(6,6,2,2,3),i2f(7,7,2,2,3)]);
                cases.push((words,[half_bits,roundtrip,0x4b800001,0x4f7fffff]));
            }
            for (round,expected_float,expected_half) in [
                (0,[0x4b800002,0x4f800000,0xcb800002,0xcf000000],[0x7800,0xf800,0x7c00,0xfc00]),
                (1,[0x4b800001,0x4f7fffff,0xcb800002,0xcf000000],[0x77ff,0xf800,0x7bff,0xfc00]),
                (2,[0x4b800002,0x4f800000,0xcb800001,0xceffffff],[0x7800,0xf7ff,0x7c00,0xfbff]),
                (3,[0x4b800001,0x4f7fffff,0xcb800001,0xceffffff],[0x77ff,0xf7ff,0x7bff,0xfbff]),
            ] {
                cases.push((vec![mov32(0,16777219),mov32(1,u32::MAX),mov32(2,(-16777219i32) as u32),mov32(3,(-2147483647i32) as u32),i2f(4,0,2,2,round),i2f(5,1,2,2,round),i2f(6,2,6,2,round),i2f(7,3,6,2,round)],expected_float));
                cases.push((vec![mov32(0,32767),mov32(1,(-32767i32) as u32),mov32(2,u32::MAX),mov32(3,i32::MIN as u32),i2f(4,0,2,1,round),i2f(5,1,6,1,round),i2f(6,2,2,1,round),i2f(7,3,6,1,round)],expected_half));
            }
            for shift in [0u32,31,32,63] {
                let bfi=(0x36f07f8010470025&!(255u64|(255<<8)|(0x7ffff<<20)|(1<<56)|(255<<39)))|4|(0x808<<20)|(1<<39);
                let left=(0x5bfc02000017ff04&!(255u64|(255<<8)|(255<<20)|(255<<39)))|5|(1<<8)|(3<<20)|(2<<39);
                let right=(0x38fc7f8001070101&!(255u64|(255<<8)|(63<<20)|(255<<39)))|6|(1<<8)|((shift as u64)<<20)|(2<<39);
                let pair=(0xdeadbeefu64<<32)|0x12345678;
                cases.push((vec![mov32(0,u32::MAX),mov32(1,0x12345678),mov32(2,0xdeadbeef),mov32(3,shift),bfi,left,right,0x5c3002000ff70007],[0x1234ff78,((pair<<(shift&31))>>32) as u32,(pair>>(shift&31)) as u32,u32::MAX]));
            }
            for shift in [0u32,1,31,32,63,u32::MAX] {
                let clear=255u64|(255<<8)|(255<<20)|(255<<39)|(1<<50);
                let left=(0x5bf8000000070000&!clear)|4|(1<<8)|(3<<20)|(2<<39);
                let right=(0x5cf8000000070000&!clear)|5|(1<<8)|(3<<20)|(2<<39);
                let immediate=(0x36f8000000070000&!clear)|6|(1<<8)|((shift.min(63) as u64)<<20)|(2<<39);
                let pair=(0xdeadbeefu64<<32)|0x12345678;
                let l=if shift<32 {((pair<<shift)>>32) as u32} else {0};let r=if shift<32 {(pair>>shift) as u32} else {0};
                cases.push((vec![mov32(1,0x12345678),mov32(2,0xdeadbeef),mov32(3,shift),left,right,immediate,mov32(7,0)],[l,r,l,0]));
            }
            r.e.st.ubos[2][0]=(output+512,512);((output+512) as *mut u32).write(32767);((output+512+66*4) as *mut u32).write(17);
            for shift in [0u32,16,31] {for negate in [false,true] {
                let clear=255u64|(255<<8)|(255<<39)|(31<<51)|(1<<56);
                let op=(0x18877f8c04270520u64&!clear)|4|(1<<39)|((shift as u64)<<51)|((negate as u64)<<56);
                let value=0x89abcdef01234567u64.wrapping_shl(shift);let value=if negate {0u64.wrapping_sub(value)} else {value};
                cases.push((vec![mov32(0,0x01234567),mov32(1,0x89abcdef),op,mov32(5,0),mov32(6,0),mov32(7,0)],[((value>>32) as u32).wrapping_add(17),0,0,0]));
            }}
            for (a,b,c,asel,bsel,maximum,signed_dest,saturate,merge,expected) in [
                (7u32,23,2,6u64,6u64,true,false,false,6u64,23u32),
                (9,2,5,6,6,false,false,false,6,5),
                (u32::MAX-4,3,0,14,14,false,false,true,4,0),
                (u32::MAX,u32::MAX,0,6,14,true,true,true,4,i32::MAX as u32),
                (0x80,0x81,0xdeadbeef,8,8,false,true,false,0,0xff80beef),
                (0x80ff0000,0x007f0000,0,10,12,false,true,false,4,0xffffffff),
            ] {
                let clear=255u64|(255<<8)|(255<<20)|(255<<39)|(7<<36)|(1<<48)|(7<<28)|(1<<49)|(7<<51)|(1<<54)|(1<<55)|(1<<56);
                let op=(0x3b34136061c72526u64&!clear)|4|(1<<20)|(2<<39)|((asel&7)<<36)|((asel&8)<<45)|((bsel&7)<<28)|((bsel&8)<<46)|(merge<<51)|((signed_dest as u64)<<54)|((saturate as u64)<<55)|((maximum as u64)<<56);
                cases.push((vec![mov32(0,a),mov32(1,b),mov32(2,c),op,mov32(5,0),mov32(6,0),mov32(7,0)],[expected,0,0,0]));
            }
            ((output+512+68*4) as *mut u32).write(0x7654);
            let mask=255u64|(255<<8)|(0x7ffff<<20)|(255<<39);
            let pi=|d:u64,a:u64,s:u64|(0x36c000054107130cu64&!mask)|d|(a<<8)|(s<<20)|(2<<39);
            let pr=(0x5bc0000000070000u64&!mask)|6|(1<<8)|(3<<20)|(2<<39);
            let pc=(0x4bc0000000070000u64&!mask)|7|(1<<8)|(68<<20)|(3<<34)|(2<<39);
            cases.push((vec![mov32(1,0x80ff7f01),mov32(2,0x12345678),mov32(3,0x3210),mov32(4,0x80ff7f01),pi(4,4,0x5410),pi(5,1,0x89ab),pr,pc],[0x56787f01,0x0000ffff,0x80ff7f01,0x12345678]));
            for (mut words,expected) in cases {
                words.extend([mov_handle(12,0x120),mov32(10,0),mov32(11,0),0xeb20060600f70a04]);let shader=maxwell_program(&words);
                control[0x700..0x704].copy_from_slice(&(shader.len() as u32).to_le_bytes());r.e.st.set_program(&[(shader.as_ptr() as u64,control.as_ptr() as u64)]);r.dispatch([1,1,1]).unwrap();
                let (_,tex)=r.e.textures.get(&b,true).unwrap();copy::copy_to_buffer(&r.cb,&tex.0,(0,0),[0,0,0,1,1,1],47,output,&r.pools,(0,0)).unwrap();r.commit_boundary(true);
                assert_eq!(std::slice::from_raw_parts(output as *const u32,4),expected);
            }
            STATE.lock().unwrap().registered_textures.remove(&(0x9b80,0x100));STATE.lock().unwrap().pools.remove(&0x9b00);gfx::forget_pool(0x9b01);
        });
    }

    #[cfg(target_os="macos")]
    #[test]
    #[ignore="requires a real Metal GPU"]
    fn gpu_compute_convergence_collectives_shared_memory_and_indirect_branches() {
        use super::*;
        fn address(n:usize)->u32 {((n/3*4+1+n%3)*8) as u32}
        fn branch(words:&mut [u64],at:usize,target:usize) {
            let delta=address(target) as i64-address(at) as i64-8;
            words[at]=(words[at]&!(0xffffff<<20))|(((delta as u64)&0xffffff)<<20);
        }
        fn mov(d:u64,s:u64)->u64 {(0x5c9807800ff70007&!(255u64|(255<<20)))|d|(s<<20)}
        fn shuffle(d:u64,s:u64,delta:u64,clamp:u64,mode:u64,p:u64)->u64 {
            let clear=255u64|(255<<8)|(31<<20)|(8191<<34)|(3<<30)|(7<<48);
            (0xef17007cf0170f00&!clear)|d|(s<<8)|(delta<<20)|(clamp<<34)|(mode<<30)|(p<<48)
        }
        fn store(handle:u64,data:u64)->u64 {(0xeb20008600f70204&!(255u64|(255<<8)|(255<<39)))|data|(handle<<39)}
        for nthreads in [64u32,512,1024] {
        objc2::rc::autoreleasepool(|_| unsafe {
            crate::kernel::KERNEL.aspace.get_or_init(||crate::mem::AddressSpace::reserve().unwrap());
            let page=crate::mem::PAGE;let span=page*8;let cpu=crate::kernel::aspace().alias()+0x1000000;
            crate::kernel::aspace().set(cpu,span*5,crate::mem::Block{state:crate::mem::state::NORMAL,perm:crate::mem::perm::RW,attr:0}).unwrap();
            let builder=|pool:u64| {let mut b=[0u8;0x68];b[8..16].copy_from_slice(&pool.to_le_bytes());for (o,v) in [(0x1c,1u32),(0x20,nthreads),(0x24,1),(0x28,1),(0x2c,1),(0x30,47)] {b[o..o+4].copy_from_slice(&v.to_le_bytes());}b};
            let images=[builder(0x9e00),builder(0x9e01)];
            for i in 0..2 {let mut s=STATE.lock().unwrap();s.pools.insert(0x9e00+i,super::super::objects::MemoryPool{cpu:cpu+i*span,size:span,flags:0x22});s.registered_textures.insert((0x9e80,0x100+i as u32),images[i as usize]);}
            let output=cpu+2*span;let atomic=cpu+4*span;(atomic as *mut u32).write(0);
            let pools=gfx::pool_memory(&[(0x9e02,output,span*2),(0x9e03,atomic,page)]).unwrap();
            let mut words=vec![0xf0c8000002170000,0xf0c8000000070001,0x38f0000000170000,0xe290000000000000,0xe24000000000000f,mov32(15,100),0xf0f800000007000f,mov32(15,200),0xe24000000007000f,0x50b0000000070f00,0xf0f800000007000f,0x50d9e38000070010];
            branch(&mut words,3,11);branch(&mut words,4,7);branch(&mut words,8,10);
            words.extend([0x0400000000170012,0x3848000000571212]);
            let join=words.len();words.push(0xe290000000000000);
            let add=words.len();words.push(0);
            let jump=words.len();words.push(0xe25000000007120f);
            let table=words.len();words.extend([mov32(17,10),0xf0f800000007000f,0x50b0000000070f00,mov32(17,20),0xf0f800000007000f,0x50b0000000070f00]);let done=words.len();
            words[add]=0x1c00000000071212|((address(table) as u64)<<20);
            let base=-(address(jump) as i64+8);words[jump]|=((base as u64)&0xffffff)<<20;
            branch(&mut words,join,done);
            words.extend([
                0x3848000000270003,0xef5c000000070300,
                shuffle(5,0,1,31,3,7),0x3848000000270506,
                0xf0a81b8000070000,0xef4c100000070607,
                0x50d9e00000070008,shuffle(9,0,1,0x1000,1,1),0x50d9e0800007000a,
                mov32(30,atomic as u32),mov32(31,(atomic>>32) as u32),mov32(13,1),0xebf9000000071e0d,
                0xf0a81b8000070000,0xeed4200000071e0b,
                mov_handle(12,0x120),mov_handle(13,0x128),mov32(1,0),mov(4,8),mov(5,9),mov(6,7),mov(7,11),store(12,4),
                mov(4,15),mov(5,16),mov(6,10),mov(7,17),store(13,4),
            ]);
            let shader=maxwell_program(&words);let mut control=vec![0u8;0x880];
            for (offset,value) in [(0x700,shader.len() as u32),(0x748,nthreads),(0x74c,1),(0x750,1),(0x754,nthreads*4)] {control[offset..offset+4].copy_from_slice(&value.to_le_bytes());}
            let mut e=Engine::default();e.st.texture_pool=0x9e80;e.st.images[2][0]=0x100;e.st.images[2][1]=0x101;e.st.set_program(&[(shader.as_ptr() as u64,control.as_ptr() as u64)]);
            let cb=gfx::gpu().queue.0.commandBuffer().unwrap();let mut r=Replay{e:&mut e,cb,pass:None,heap:None,pools,passes:0,copies:HashMap::new(),fault:None,pool_table:None,resource_buffers:Default::default()};
            r.clear_buffer(atomic+16,12,1).unwrap();r.dispatch_indirect(atomic+16).unwrap();
            for (i,b) in images.iter().enumerate() {let (_,tex)=r.e.textures.get(b,true).unwrap();copy::copy_to_buffer(&r.cb,&tex.0,(0,0),[0,0,0,nthreads as i32,1,1],47,output+i as u64*span,&r.pools,(0,0)).unwrap();}
            r.commit_boundary(true);
            for tid in 0..nthreads as usize {
                let first=std::slice::from_raw_parts((output as *const u32).add(tid*4),4);
                assert_eq!(first,[0xaaaaaaaau32,if tid%16==0 {tid as u32} else {tid as u32-1},tid as u32^1,nthreads],"thread {tid} shared/collective result");
                let second=std::slice::from_raw_parts((output as *const u32).add(span as usize/4+tid*4),4);
                assert_eq!(second,[if tid%2==0 {100u32} else {200},u32::MAX,0xfffefffe,if tid%2==0 {10} else {20}],"thread {tid} convergence/indirect branch result");
            }
            assert_eq!((atomic as *const u32).read(),nthreads);
            for id in 0x9e00..0x9e04 {STATE.lock().unwrap().pools.remove(&id);gfx::forget_pool(id);}
            STATE.lock().unwrap().registered_textures.retain(|(p,_),_|*p!=0x9e80);
        });
        }
    }

    #[cfg(target_os="macos")]
    #[test]
    #[ignore="requires a real Metal GPU"]
    fn gpu_shared_cache_invalidation_does_not_wait_for_unrelated_queue_work() {
        use super::*;
        objc2::rc::autoreleasepool(|_| unsafe {
            let buffer=gfx::gpu().device.0.newBufferWithLength_options(16384,MTLResourceOptions::StorageModeShared).unwrap();
            let pool=0xca201;STATE.lock().unwrap().pools.insert(pool,super::super::objects::MemoryPool{cpu:buffer.contents().as_ptr() as u64,size:16384,flags:0x22});
            let mut object=[0u64;4];let builder=[0u64,0,pool,0,16384];
            let invoke=|name:&str,args:[u64;8]|super::super::objects::host_impl(name).unwrap()(&crate::stubs::Args(args,[0;8]));
            assert_eq!(invoke("nvnBufferInitialize",[object.as_mut_ptr() as u64,builder.as_ptr() as u64,0,0,0,0,0,0]),1);
            let event=Shared(gfx::gpu().device.0.newSharedEvent().unwrap());let cb=Shared(gfx::gpu().queue.0.commandBuffer().unwrap());
            cb.0.encodeWaitForEvent_value(ProtocolObject::from_ref(&*event.0),1);
            let blit=cb.0.blitCommandEncoder().unwrap();blit.fillBuffer_range_value(&buffer,objc2_foundation::NSRange::new(0,16384),0x5a);blit.endEncoding();cb.0.commit();
            let previous=engine().last_submit.replace(cb.clone());
            let (release,wait)=std::sync::mpsc::channel();
            // The fallback always releases the GPU, including against the old
            // implementation, so a regression fails without hanging the queue.
            let signal=std::thread::spawn(move || {let timely=wait.recv_timeout(std::time::Duration::from_secs(1)).is_ok();event.0.setSignaledValue(1);timely});
            assert_eq!(invoke("nvnMemoryPoolInvalidateMappedRange",[pool,0,16384,0,0,0,0,0]),0);
            assert_eq!(invoke("nvnBufferInvalidateMappedRange",[object.as_ptr() as u64,0,16384,0,0,0,0,0]),0);
            let _=release.send(());let nonblocking=signal.join().unwrap();cb.0.waitUntilCompleted();
            engine().last_submit=previous;invoke("nvnBufferFinalize",[object.as_mut_ptr() as u64,0,0,0,0,0,0,0]);STATE.lock().unwrap().pools.remove(&pool);
            assert!(nonblocking,"cache maintenance waited for unrelated GPU completion");assert_eq!(cb.0.status(),MTLCommandBufferStatus::Completed);
            assert!(std::slice::from_raw_parts(buffer.contents().as_ptr().cast::<u8>(),16384).iter().all(|&b|b==0x5a),"shared bytes must be visible after the actual fence");
        });
    }

    #[cfg(target_os="macos")]
    #[test]
    #[ignore="requires a real Metal GPU"]
    fn gpu_sync_poll_timeout_acquire_and_repeated_wait_preserve_fence() {
        use super::*;
        objc2::rc::autoreleasepool(|_| {
            let mut object=[0xaabbccddeeff0011u64;9];let sync=object.as_mut_ptr() as u64;
            assert_eq!(initialize_sync(sync,0x1234),1);assert_eq!(object[8],0xaabbccddeeff0011);
            assert_eq!(sync_wait(sync,0),3,"initialized but unfenced sync must fail");
            acquire_sync(sync,0x9e00);assert_eq!(sync_wait(sync,0),0,"never-presented image is ready");
            let event=gfx::Shared(gfx::gpu().device.0.newSharedEvent().unwrap());
            let cb=gfx::Shared(gfx::gpu().queue.0.commandBuffer().unwrap());
            cb.0.encodeWaitForEvent_value(ProtocolObject::from_ref(&*event.0),1);cb.0.commit();
            engine().syncs.insert(sync,SyncPoint::Fenced(cb));
            let signal=std::thread::spawn(move || {std::thread::sleep(std::time::Duration::from_millis(40));event.0.setSignaledValue(1);});
            assert_eq!(sync_wait(sync,0),2);assert_eq!(sync_wait(sync,1_000_000),2);
            assert!(sync_wait(sync,u64::MAX)<=1);signal.join().unwrap();
            assert_eq!(sync_wait(sync,0),0);assert_eq!(sync_wait(sync,0),0);
            let mut e=engine();let cb=gfx::gpu().queue.0.commandBuffer().unwrap();
            let mut r=Replay{e:&mut e,cb,pass:None,heap:None,pools:Vec::new(),passes:0,copies:HashMap::new(),fault:None,pool_table:None,resource_buffers:Default::default()};
            r.run(&[Cmd::WaitSync(sync)],0);r.commit_boundary(true);drop(r);drop(e);
            finalize_sync(sync);assert_eq!(sync_wait(sync,0),3);
        });
    }

    #[cfg(target_os="macos")]
    #[test]
    #[ignore="requires a real Metal GPU"]
    fn gpu_translated_packed_float_surface_3d_rounding_and_special_values() {
        use super::*;
        objc2::rc::autoreleasepool(|_| unsafe {
            crate::kernel::KERNEL.aspace.get_or_init(||crate::mem::AddressSpace::reserve().unwrap());
            let page=crate::mem::PAGE;let cpu=crate::kernel::aspace().alias()+0xc00000;
            crate::kernel::aspace().set(cpu,page*2,crate::mem::Block{state:crate::mem::state::NORMAL,perm:crate::mem::perm::RW,attr:0}).unwrap();
            STATE.lock().unwrap().pools.insert(0x9d00,super::super::objects::MemoryPool{cpu,size:page,flags:0x22});
            let mut b=[0u8;0x68];b[8..16].copy_from_slice(&0x9d00u64.to_le_bytes());
            for (o,v) in [(0x1c,2u32),(0x20,1),(0x24,1),(0x28,1),(0x2c,1),(0x30,63)] {b[o..o+4].copy_from_slice(&v.to_le_bytes());}
            STATE.lock().unwrap().registered_textures.insert((0x9d80,0x100),b);
            let mut control=vec![0u8;0x880];for o in [0x748,0x74c,0x750] {control[o..o+4].copy_from_slice(&1u32.to_le_bytes());}
            let output=cpu+page;let pools=gfx::pool_memory(&[(0x9d01,output,page)]).unwrap();
            let mut e=Engine::default();let cb=gfx::gpu().queue.0.commandBuffer().unwrap();
            let mut r=Replay{e:&mut e,cb,pass:None,heap:None,pools,passes:0,copies:HashMap::new(),fault:None,pool_table:None,resource_buffers:Default::default()};
            r.e.st.texture_pool=0x9d80;r.e.st.images[2][0]=0x100;
            let cases=[
                ([0.25f32,0.5,0.75],832u32|(896<<11)|(464<<22)),
                ([-1.0,f32::INFINITY,f32::NAN],(1984<<11)|(1008<<22)),
                ([2f32.powi(-20),2f32.powi(-21),2f32.powi(-19)],1|(1<<22)),
                ([1.0+1.0/128.0,1.0+3.0/128.0,65504.0],960|(962<<11)|(992<<22)),
            ];
            for (values,expected) in cases {
                let shader=maxwell_program(&[mov_handle(12,0x120),mov32(0,0),mov32(1,0),mov32(2,0),mov32(4,values[0].to_bits()),mov32(5,values[1].to_bits()),mov32(6,values[2].to_bits()),mov32(7,1f32.to_bits()),0xeb20060a00f70004]);
                control[0x700..0x704].copy_from_slice(&(shader.len() as u32).to_le_bytes());r.e.st.set_program(&[(shader.as_ptr() as u64,control.as_ptr() as u64)]);
                r.dispatch([1,1,1]).unwrap();let (_,tex)=r.e.textures.get(&b,true).unwrap();
                assert!(!r.e.gpu_buffer_dirty,"private image stores should not dirty CPU buffer storage");
                r.run(&[Cmd::Barrier(0x10)],0);assert_eq!(r.cb.status(),MTLCommandBufferStatus::NotEnqueued);
                copy::copy_to_buffer(&r.cb,&tex.0,(0,0),[0,0,0,1,1,1],63,output,&r.pools,(0,0)).unwrap();r.commit_boundary(true);
                assert_eq!((output as *const u32).read(),expected,"packed float {values:?}");
            }
            STATE.lock().unwrap().registered_textures.remove(&(0x9d80,0x100));STATE.lock().unwrap().pools.remove(&0x9d00);gfx::forget_pool(0x9d01);
        });
    }

    #[cfg(target_os="macos")]
    #[test]
    #[ignore="requires a real Metal GPU"]
    fn gpu_translated_surface_half_address_and_bindless_texture_operations() {
        use super::*;
        objc2::rc::autoreleasepool(|_| unsafe {
            crate::kernel::KERNEL.aspace.get_or_init(||crate::mem::AddressSpace::reserve().unwrap());
            let page=crate::mem::PAGE;
            let cpu=crate::kernel::aspace().alias()+0x800000;
            crate::kernel::aspace().set(cpu,page*8,crate::mem::Block{state:crate::mem::state::NORMAL,perm:crate::mem::perm::RW,attr:0}).unwrap();
            for i in 0..8 {STATE.lock().unwrap().pools.insert(0x9c00+i,super::super::objects::MemoryPool{cpu:cpu+i*page,size:page,flags:0x22});}
            let builder=|pool:u64,format:u32,stride:u32| {
                let mut b=[0u8;0x68];b[8..16].copy_from_slice(&pool.to_le_bytes());
                for (o,v) in [(0x18,0x10u32),(0x1c,1),(0x20,1),(0x24,1),(0x28,1),(0x2c,1),(0x30,format),(0x5c,stride)] {b[o..o+4].copy_from_slice(&v.to_le_bytes());}b
            };
            let mut control=vec![0u8;0x880];for o in [0x748,0x74c,0x750] {control[o..o+4].copy_from_slice(&1u32.to_le_bytes());}
            let output=cpu+7*page;
            let pools=gfx::pool_memory(&[(0x9c07,output,page)]).unwrap();
            let mut e=Engine::default();let cb=gfx::gpu().queue.0.commandBuffer().unwrap();
            let mut r=Replay{e:&mut e,cb,pass:None,heap:None,pools,passes:0,copies:HashMap::new(),fault:None,pool_table:None,resource_buffers:Default::default()};
            r.e.st.texture_pool=0x9c80;
            let store=maxwell_program(&[mov_handle(1,0x120),mov32(2,0),mov32(3,0),mov32(4,0.25f32.to_bits()),mov32(5,0.5f32.to_bits()),mov32(6,0.75f32.to_bits()),mov32(7,1f32.to_bits()),0xeb20008600f70204]);
            for (i,format,stride) in [(0u64,37,4),(1,41,8),(2,47,16)] {
                let b=builder(0x9c00+i,format,stride);STATE.lock().unwrap().registered_textures.insert((0x9c80,0x100+i as u32),b);
                r.e.st.images[2][0]=0x100+i;r.e.bank_ok[2]=false;
                control[0x700..0x704].copy_from_slice(&(store.len() as u32).to_le_bytes());r.e.st.set_program(&[(store.as_ptr() as u64,control.as_ptr() as u64)]);
                r.dispatch([1,1,1]).unwrap();let (_,texture)=r.e.textures.get(&b,true).unwrap();
                copy::copy_to_buffer(&r.cb,&texture.0,(0,0),[0,0,0,1,1,1],format,output,&r.pools,(0,0)).unwrap();r.commit_boundary(true);
                match format {
                    37=>assert_eq!(std::slice::from_raw_parts(output as *const u8,4),[64,128,191,255]),
                    41=>assert_eq!(std::slice::from_raw_parts(output as *const u16,4),[0x3400,0x3800,0x3a00,0x3c00]),
                    _=>assert_eq!(std::slice::from_raw_parts(output as *const u32,4),[0.25f32.to_bits(),0.5f32.to_bits(),0.75f32.to_bits(),1f32.to_bits()]),
                }
            }
            let b=builder(0x9c03,47,16);STATE.lock().unwrap().registered_textures.insert((0x9c80,0x103),b);
            let mut lea=0x5bdf7f8100270303u64;
            lea&=!(255u64|(255<<8)|(255<<20)|(255<<39)|(31<<28));lea|=6|(8<<8)|(10<<20)|(9<<39)|(16<<28);
            let arithmetic=maxwell_program(&[mov_handle(1,0x120),mov32(2,0),mov32(3,0),mov32(6,0x3c00),mov32(5,0.5f32.to_bits()),0x5d130000aff70604,0x5d148000aff70505,mov32(8,0xffff0000),mov32(9,1),mov32(10,3),lea,0xeb20008600f70204]);
            r.e.st.images[2][0]=0x103;r.e.bank_ok[2]=false;control[0x700..0x704].copy_from_slice(&(arithmetic.len() as u32).to_le_bytes());r.e.st.set_program(&[(arithmetic.as_ptr() as u64,control.as_ptr() as u64)]);
            r.dispatch([1,1,1]).unwrap();let (_,texture)=r.e.textures.get(&b,true).unwrap();copy::copy_to_buffer(&r.cb,&texture.0,(0,0),[0,0,0,1,1,1],47,output,&r.pools,(0,0)).unwrap();r.commit_boundary(true);
            assert_eq!((output as *const u32).read(),1f32.to_bits());assert_eq!(((output+4) as *const u32).read(),0x3f003800);assert_eq!(((output+8) as *const u32).read(),0x20002);
            // Stale descriptors are legal while unselected. In particular,
            // building TEX.B's sparse table must not eagerly upload this hole.
            let unavailable=builder(0x9cff,37,4);
            {let mut state=STATE.lock().unwrap();state.pools.insert(0x9cff,super::super::objects::MemoryPool{cpu:0,size:4096,flags:0x221});state.registered_textures.insert((0x9c80,0x1ff),unavailable);}
            let source=builder(0x9c04,23,8);let destination=builder(0x9c05,23,8);
            ((cpu+4*page) as *mut u32).write(0x12345678);((cpu+4*page+4) as *mut u32).write(0x9abcdef0);
            STATE.lock().unwrap().registered_textures.insert((0x9c80,0x104),source);STATE.lock().unwrap().registered_textures.insert((0x9c80,0x105),destination);
            let raw=maxwell_program(&[mov_handle(11,0x120),mov_handle(1,0x128),mov32(10,0),mov32(0,0),0xeb10058200570a08,0xeb20008000f70008]);
            r.e.st.images[2][0]=0x104;r.e.st.images[2][1]=0x105;r.e.bank_ok[2]=false;control[0x700..0x704].copy_from_slice(&(raw.len() as u32).to_le_bytes());r.e.st.set_program(&[(raw.as_ptr() as u64,control.as_ptr() as u64)]);
            r.dispatch([1,1,1]).unwrap();let (_,texture)=r.e.textures.get(&destination,true).unwrap();copy::copy_to_buffer(&r.cb,&texture.0,(0,0),[0,0,0,1,1,1],23,output,&r.pools,(0,0)).unwrap();r.commit_boundary(true);
            assert_eq!(std::slice::from_raw_parts(output as *const u32,2),[0x12345678,0x9abcdef0]);
            let sampled=builder(0x9c06,37,4);((cpu+6*page) as *mut u32).write(0xff804020);
            STATE.lock().unwrap().registered_textures.insert((0x9c80,0x106),sampled);
            let sample=maxwell_program(&[mov32(0,0.25f32.to_bits()),mov32(1,0.75f32.to_bits()),mov_handle(5,0x168),0x4c47020815c70502,mov_handle(12,0x120),mov32(10,0),mov32(11,0),0xdeb80023a0270000,0xeb20060600f70a00]);
            r.e.st.images[2][0]=0x103;r.e.st.separate_textures[2][0]=0x100000106;r.e.st.separate_samplers[2][1]=0x201<<20;r.e.bank_ok[2]=false;
            control[0x700..0x704].copy_from_slice(&(sample.len() as u32).to_le_bytes());r.e.st.set_program(&[(sample.as_ptr() as u64,control.as_ptr() as u64)]);
            r.dispatch([1,1,1]).unwrap();let (_,texture)=r.e.textures.get(&b,true).unwrap();copy::copy_to_buffer(&r.cb,&texture.0,(0,0),[0,0,0,1,1,1],47,output,&r.pools,(0,0)).unwrap();r.commit_boundary(true);
            // This TEX.B has RGB mask 0b0111; R3 remains zero.
            for (channel,(actual,expected)) in std::slice::from_raw_parts(output as *const f32,4).iter().zip([32.0/255.0,64.0/255.0,128.0/255.0,0.0]).enumerate() {assert!((*actual-expected).abs()<1e-6,"channel {channel}: sampled {actual}, expected {expected}");}
            // More sparse descriptor IDs than physical argument slots are
            // legal when all aliases retain the exact same Metal view.
            let last_id=0x80000+nxshader::msl::DYNAMIC_TEXTURE_CAPACITY as u32+16;
            {let mut s=STATE.lock().unwrap();for id in 0x80000..=last_id{s.registered_textures.insert((0x9c80,id),sampled);}}
            r.e.st.separate_textures[2][0]=0x100000000|u64::from(last_id);r.e.bank_ok[2]=false;
            r.dispatch([1,1,1]).unwrap();let(_,texture)=r.e.textures.get(&b,true).unwrap();copy::copy_to_buffer(&r.cb,&texture.0,(0,0),[0,0,0,1,1,1],47,output,&r.pools,(0,0)).unwrap();r.commit_boundary(true);
            for (actual,expected)in std::slice::from_raw_parts(output as *const f32,4).iter().zip([32.0/255.0,64.0/255.0,128.0/255.0,0.0]){assert!((*actual-expected).abs()<1e-6,"last sparse texture alias must preserve actual sampling");}
            for id in 0x9c00..0x9c08 {STATE.lock().unwrap().pools.remove(&id);gfx::forget_pool(id);}
            STATE.lock().unwrap().registered_textures.retain(|(pool,_),_|*pool!=0x9c80);
            STATE.lock().unwrap().pools.remove(&0x9cff);
        });
    }

    #[cfg(target_os="macos")]
    #[test]
    #[ignore="requires a real Metal GPU"]
    fn gpu_surface_atomic_add_preserves_contention_return_values_and_texture_coherence() {
        use super::*;
        objc2::rc::autoreleasepool(|_| unsafe {
            crate::kernel::KERNEL.aspace.get_or_init(||crate::mem::AddressSpace::reserve().unwrap());
            let page=crate::mem::PAGE;let cpu=crate::kernel::aspace().alias()+0x2400000;
            crate::kernel::aspace().set(cpu,page*4,crate::mem::Block{state:crate::mem::state::NORMAL,perm:crate::mem::perm::RW,attr:0}).unwrap();
            for i in 0..2 {STATE.lock().unwrap().pools.insert(0xa100+i,super::super::objects::MemoryPool{cpu:cpu+i*page,size:page,flags:0x22});}
            let builder=|pool:u64,format:u32,width:u32,height:u32,stride:u32| {
                let mut b=[0u8;0x68];b[8..16].copy_from_slice(&pool.to_le_bytes());
                for (o,v) in [(0x18,0x10u32),(0x1c,1),(0x20,width),(0x24,height),(0x28,1),(0x2c,1),(0x30,format),(0x5c,stride)] {b[o..o+4].copy_from_slice(&v.to_le_bytes());}b
            };
            let counter=builder(0xa100,11,3,2,12);let results=builder(0xa101,47,64,1,1024);
            std::ptr::copy_nonoverlapping([11u32,100,300,7,9,11].as_ptr(),cpu as *mut u32,6);
            for (id,b) in [(0x100,counter),(0x101,results)] {STATE.lock().unwrap().registered_textures.insert((0xa180,id),b);}
            let output=cpu+2*page;let pools=gfx::pool_memory(&[(0xa102,output,page*2)]).unwrap();let mut e=Engine::default();e.st.texture_pool=0xa180;e.st.images[2][0]=0x100;e.st.images[2][1]=0x101;
            let cb=gfx::gpu().queue.0.commandBuffer().unwrap();let mut r=Replay{e:&mut e,cb,pass:None,heap:None,pools,passes:0,copies:HashMap::new(),fault:None,pool_table:None,resource_buffers:Default::default()};
            let mut control=vec![0u8;0x880];for (o,v) in [(0x748,64u32),(0x74c,1),(0x750,1)] {control[o..o+4].copy_from_slice(&v.to_le_bytes());}
            let atomic=|dest:u64,dim:u64,ba:bool| {
                let clear=255u64|(255<<8)|(255<<20)|(255<<39)|(7<<33)|(1<<28);
                (0xea70038200670404u64&!clear)|dest|(4<<8)|(6<<20)|(7<<39)|(dim<<33)|((ba as u64)<<28)
            };
            let mut expected=[11u32,100,300,7,9,11];
            for (coord,y,dim,ba,add,dest,repeats,index) in [(0u32,0u32,1u64,false,1u32,4u64,1u32,Some(0usize)),(0,0,1,false,1,4,2,Some(0)),(u32::MAX,0,1,false,5,4,1,None),(8,0,1,true,2,4,1,Some(2)),(1,1,3,false,3,4,1,Some(4)),(0,0,1,false,1,255,1,Some(0))] {
                let shader=maxwell_program(&[0xf0c8000002170000,mov_handle(7,0x120),mov32(4,coord),mov32(5,y),mov32(6,add),atomic(dest,dim,ba),mov_handle(12,0x128),mov32(1,0),0xeb20060600f70004]);
                control[0x700..0x704].copy_from_slice(&(shader.len() as u32).to_le_bytes());r.e.st.set_program(&[(shader.as_ptr() as u64,control.as_ptr() as u64)]);
                let old=index.map_or(0,|i|expected[i]);
                for _ in 0..repeats {r.dispatch([1,1,1]).unwrap();}
                if let Some(i)=index {expected[i]=expected[i].wrapping_add(64*add*repeats);}
                let (_,texture)=r.e.textures.get(&counter,true).unwrap();copy::copy_to_buffer(&r.cb,&texture.0,(0,0),[0,0,0,3,2,1],11,output,&r.pools,(0,0)).unwrap();
                let (_,texture)=r.e.textures.get(&results,true).unwrap();copy::copy_to_buffer(&r.cb,&texture.0,(0,0),[0,0,0,64,1,1],47,output+page,&r.pools,(0,0)).unwrap();r.commit_boundary(true);
                assert_eq!(std::slice::from_raw_parts(output as *const u32,6),expected);
                if dest!=255 {
                    let mut returned=std::slice::from_raw_parts((output+page) as *const u32,256).chunks_exact(4).map(|v|v[0]).collect::<Vec<_>>();returned.sort_unstable();
                    let want=if index.is_none() {vec![0;64]} else {(0..64).map(|n|old+64*add*(repeats-1)+n*add).collect()};assert_eq!(returned,want);
                }
            }
            for id in 0xa100..0xa102 {STATE.lock().unwrap().pools.remove(&id);}gfx::forget_pool(0xa102);STATE.lock().unwrap().registered_textures.retain(|(p,_),_|*p!=0xa180);
        });
    }

    #[cfg(target_os="macos")]
    #[test]
    #[ignore="requires a real Metal GPU; run serially"]
    fn gpu_mixed_surface_load_store_and_atomics_share_texture_and_linear_backing(){
        use super::*;
        objc2::rc::autoreleasepool(|_|unsafe{
            crate::kernel::KERNEL.aspace.get_or_init(||crate::mem::AddressSpace::reserve().unwrap());
            let page=crate::mem::PAGE;
            for (variant,target)in [1u32,10].into_iter().enumerate(){
                let cpu=crate::kernel::aspace().alias()+0xa000000+variant as u64*page*4;crate::kernel::aspace().set(cpu,page*4,crate::mem::Block{state:crate::mem::state::NORMAL,perm:crate::mem::perm::RW,attr:0}).unwrap();std::ptr::write_bytes(cpu as *mut u8,0,(page*4)as usize);
                let pool=0xe100+variant as u64*16;let descriptor_pool=pool+8;
                for i in 0..4 {STATE.lock().unwrap().pools.insert(pool+i,super::super::objects::MemoryPool{cpu:cpu+i*page,size:page,flags:0x22});}
                let builder=|p:u64,target:u32,width:u32|{let mut b=[0u8;0x68];b[8..16].copy_from_slice(&p.to_le_bytes());for(at,v)in[(0x18,0x10u32),(0x1c,target),(0x20,width),(0x24,1),(0x28,1),(0x2c,1),(0x30,11),(0x5c,width*4)]{b[at..at+4].copy_from_slice(&v.to_le_bytes());}b};
                let counter=builder(pool,target,64);let results=builder(pool+1,1,512);
                for i in 0..64{((cpu+i*4)as*mut u32).write(100+i as u32);}
                {let mut s=STATE.lock().unwrap();s.registered_textures.insert((descriptor_pool,0x100),counter);s.registered_textures.insert((descriptor_pool,0x101),results);}
                let load=|d:u64|{let clear=255u64|(255<<8)|(255<<39)|(7<<33)|(7<<20)|(1<<23);(0xeb10058200570a08&!clear)|d|(7<<39)|(1<<33)|(4<<20)};
                let store=|d:u64,coordinate:u64,handle:u64|{let clear=255u64|(255<<8)|(255<<39)|(7<<33);(0xeb20008600f70204&!clear)|d|(coordinate<<8)|(handle<<39)|(1<<33)};
                let atomic=|d:u64,value:u64|{let clear=255u64|(255<<8)|(255<<20)|(255<<39)|(7<<33)|(1<<28);(0xea70038200670404&!clear)|d|(value<<20)|(7<<39)|(1<<33)};
                // Global index = local thread.x + block.x*64. The second
                // block deliberately exercises the64 out-of-range indices.
                let mut words=vec![0xf0c8000002170000,0xf0c8000002570001,0x3848000000670101,0x5c10000000170000,mov_handle(7,0x120),load(8),store(8,0,7),atomic(9,8),load(10),store(9,0,7),mov32(6,1),atomic(11,6),mov_handle(12,0x128),0x3848000000270001];
                for channel in 0..4 {words.extend([store(8+channel,1,12),0x1c00000000170101]);}
                let shader=maxwell_program(&words);let translated=nxshader::translate_compute_with_memory(&shader,0,0).unwrap();assert_eq!(translated.interface.atomic_images,1);assert!(translated.msl.contains("atomic_load_explicit(pointer"));assert!(translated.msl.contains("atomic_store_explicit(pointer"));
                let mut control=vec![0u8;0x880];for(at,v)in[(0x700,shader.len()as u32),(0x748,64),(0x74c,1),(0x750,1)]{control[at..at+4].copy_from_slice(&v.to_le_bytes());}
                let pools=gfx::pool_memory(&[(pool,cpu,page),(pool+1,cpu+page,page),(pool+2,cpu+page*2,page),(pool+3,cpu+page*3,page)]).unwrap();let mut engine=Engine::default();engine.st.texture_pool=descriptor_pool;engine.st.images[2][0]=0x100;engine.st.images[2][1]=0x101;engine.st.set_program(&[(shader.as_ptr()as u64,control.as_ptr()as u64)]);
                let cb=gfx::gpu().queue.0.commandBuffer().unwrap();let mut r=Replay{e:&mut engine,cb,pass:None,heap:None,pools,passes:0,copies:HashMap::new(),fault:None,pool_table:None,resource_buffers:Default::default()};r.dispatch([2,1,1]).unwrap();let(_,out)=r.e.textures.get(&results,true).unwrap();copy::copy_to_buffer(&r.cb,&out.0,(0,0),[0,0,0,512,1,1],11,cpu+page*2,&r.pools,(0,0)).unwrap();
                if target==1{let(_,source)=r.e.textures.get(&counter,true).unwrap();copy::copy_to_buffer(&r.cb,&source.0,(0,0),[0,0,0,64,1,1],11,cpu+page*3,&r.pools,(0,0)).unwrap();}
                r.commit_boundary(true);
                let actual=std::slice::from_raw_parts((cpu+page*2)as*const u32,512);
                for (i,pixel)in actual.chunks_exact(4).enumerate(){let v=if i<64{100+i as u32}else{0};assert_eq!(pixel,&[v,v,v*2,v],"target{target} lane{i}: before/return/after/finalreturn");}
                let data=if target==1{std::slice::from_raw_parts((cpu+page*3)as*const u32,64).to_vec()}else{r.cpu_bytes(cpu,256).unwrap().chunks_exact(4).map(|b|u32::from_le_bytes(b.try_into().unwrap())).collect()};assert_eq!(data,(0..64).map(|i|101+i).collect::<Vec<u32>>(),"target{target} final source words");
                for id in pool..pool+4{STATE.lock().unwrap().pools.remove(&id);gfx::forget_pool(id);}STATE.lock().unwrap().registered_textures.retain(|(p,_),_|*p!=descriptor_pool);
                eprintln!("[verify] mixed SULD/SUST before+after SUATOM target{target}:64 owned words,64 out-of-bounds lanes,all returned/intermediate/final values exact");
            }
        });
    }

    #[cfg(target_os="macos")]
    #[test]
    #[ignore="requires a real Metal GPU"]
    fn gpu_indirect_draws_consume_gpu_produced_parameters_and_index_offsets() {
        use super::*;
        objc2::rc::autoreleasepool(|_| unsafe {
            crate::kernel::KERNEL.aspace.get_or_init(||crate::mem::AddressSpace::reserve().unwrap());let page=crate::mem::PAGE;let cpu=crate::kernel::aspace().alias()+0x3400000;
            crate::kernel::aspace().set(cpu,page*4,crate::mem::Block{state:crate::mem::state::NORMAL,perm:crate::mem::perm::RW,attr:0}).unwrap();
            let positions=[[10f32,10.0,0.5,1.0],[10.0,10.0,0.5,1.0],[-1.0,-1.0,0.5,1.0],[3.0,-1.0,0.5,1.0],[-1.0,3.0,0.5,1.0]];
            std::ptr::copy_nonoverlapping(positions.as_ptr().cast::<u8>(),cpu as *mut u8,size_of_val(&positions));
            let make_shader=|stage:u32,words:&[u64]| {let mut b=vec![0u8;0x80];b[..4].copy_from_slice(&0x12345678u32.to_le_bytes());b[0x30..0x34].copy_from_slice(&(stage<<10).to_le_bytes());if stage==5 {b[0x78..0x7c].copy_from_slice(&15u32.to_le_bytes());}b.extend(maxwell_program(words));b};
            let vs=make_shader(1,&[0xefd800000007ff00|(3<<47)|(0x80<<20),0xeff000000007ff00|(3<<47)|(0x70<<20)]);
            let fs=make_shader(5,&[mov32(0,0.75f32.to_bits()),mov32(1,0.5f32.to_bits()),mov32(2,0.25f32.to_bits()),mov32(3,1f32.to_bits())]);
            for (id,address,size) in [(0xb000,cpu,page),(0xb001,cpu+page,page),(0xb010,vs.as_ptr() as u64,vs.len() as u64),(0xb011,fs.as_ptr() as u64,fs.len() as u64)] {STATE.lock().unwrap().pools.insert(id,super::super::objects::MemoryPool{cpu:address,size,flags:0x22});}
            let mut target=[0u8;0x68];target[8..16].copy_from_slice(&0xb001u64.to_le_bytes());for (o,v) in [(0x1c,1u32),(0x20,4),(0x24,4),(0x28,1),(0x2c,1),(0x30,46)] {target[o..o+4].copy_from_slice(&v.to_le_bytes());}
            let mut vc=vec![0u8;0x880];let mut fc=vc.clone();vc[0x700..0x704].copy_from_slice(&(vs.len() as u32).to_le_bytes());fc[0x700..0x704].copy_from_slice(&(fs.len() as u32).to_le_bytes());
            let output=cpu+2*page;let pools=gfx::pool_memory(&[(0xb000,cpu,page),(0xb002,output,page)]).unwrap();let mut e=Engine::default();e.colors=vec![Some(Box::new(target))];e.st.attribs[0]=0x40|(46<<21);e.st.streams[0]=(16,0);e.st.vbufs[0]=(cpu,size_of_val(&positions) as u64);e.st.set_program(&[(vs.as_ptr() as u64,vc.as_ptr() as u64),(fs.as_ptr() as u64,fc.as_ptr() as u64)]);
            let cb=gfx::gpu().queue.0.commandBuffer().unwrap();let mut r=Replay{e:&mut e,cb,pass:None,heap:None,pools,passes:0,copies:HashMap::new(),fault:None,pool_table:None,resource_buffers:Default::default()};
            let source=cpu+512;let arguments=cpu+768;let indices=cpu+1024;
            for (index,params,visible) in [(None,[3u32,1,2,7,0],true),(None,[3,0,2,7,0],false),(Some((1u32,indices)),[3,1,1,2,7],true),(Some((2,indices)),[3,1,1,2,7],true)] {
                if index.is_some_and(|(ty,_)|ty==1) {std::ptr::copy_nonoverlapping([999u16,0,1,2].as_ptr(),indices as *mut u16,4);} else {std::ptr::copy_nonoverlapping([999u32,0,1,2].as_ptr(),indices as *mut u32,4);}
                std::ptr::copy_nonoverlapping(params.as_ptr(),source as *mut u32,5);std::ptr::write_bytes(arguments as *mut u8,0,20);
                // The CPU argument buffer is still zero when commands are
                // encoded. Only the preceding GPU copy makes this draw live.
                r.run(&[Cmd::ClearColor{index:0,rgba:[0.125;4]},Cmd::CopyBuffer{src:source,dst:arguments,size:20},Cmd::DrawIndirect{mode:4,index,arguments}],0);
                r.end_pass();let (_,texture)=r.e.textures.get(&target,true).unwrap();copy::copy_to_buffer(&r.cb,&texture.0,(0,0),[0,0,0,4,4,1],46,output,&r.pools,(0,0)).unwrap();r.commit_boundary(true);
                let expected=if visible {[0.75f32,0.5,0.25,1.0]} else {[0.125;4]};
                for pixel in std::slice::from_raw_parts(output as *const f32,64).chunks_exact(4) {assert_eq!(pixel,expected,"indirect index {index:?}, args {params:?}");}
            }
            let d=Draw{mode:4,index:None,first:0,count:1,base_vertex:0,base_instance:0,instances:1};assert!(r.draw_indirect(&d,arguments+1).is_err());assert!(r.draw_indirect(&d,0x12340000).is_err());
            r.end_pass();r.commit_boundary(true);
            for id in [0xb000,0xb001,0xb010,0xb011] {STATE.lock().unwrap().pools.remove(&id);}gfx::forget_pool(0xb000);gfx::forget_pool(0xb002);
        });
    }

    #[cfg(target_os="macos")]
    #[test]
    #[ignore="requires a real Metal GPU"]
    fn gpu_shared_and_global_atomic_add_return_unique_values_with_aliased_destinations() {
        use super::*;
        objc2::rc::autoreleasepool(|_| unsafe {
            crate::kernel::KERNEL.aspace.get_or_init(||crate::mem::AddressSpace::reserve().unwrap());let page=crate::mem::PAGE;let cpu=crate::kernel::aspace().alias()+0x3c00000;
            crate::kernel::aspace().set(cpu,page*3,crate::mem::Block{state:crate::mem::state::NORMAL,perm:crate::mem::perm::RW,attr:0}).unwrap();let global=cpu+128;(global as *mut u32).write(11);
            let mut target=[0u8;0x68];target[8..16].copy_from_slice(&0xc001u64.to_le_bytes());for (o,v) in [(0x1c,1u32),(0x20,64),(0x24,1),(0x28,1),(0x2c,1),(0x30,47)] {target[o..o+4].copy_from_slice(&v.to_le_bytes());}
            {let mut s=STATE.lock().unwrap();s.pools.insert(0xc001,super::super::objects::MemoryPool{cpu:cpu+page,size:page,flags:0x22});s.registered_textures.insert((0xc080,0x100),target);}
            let shared=(0xec0000008020ff02u64&!(255u64|(255<<20)|(7<<16)))|4|(4<<20)|(7<<16);
            let atomic=(0xed01000000500606u64&!(255u64|(255<<8)|(255<<20)|(7<<16)))|8|(8<<8)|(6<<20)|(7<<16);
            let shader=maxwell_program(&[0xf0c8000002170000,0x3848000000270014,0xef5c0000000714ff|(8<<20),0xf0a81b8000070000,mov32(4,1),shared,0xf0a81b8000070000,0xef4c10000007ff05|(8<<20),mov32(8,global as u32),mov32(9,(global>>32) as u32),mov32(6,1),atomic,0x5c98078000870006,mov32(7,0xfeedbeef),mov_handle(12,0x120),mov32(1,0),0xeb20060600f70004]);
            let mut control=vec![0u8;0x880];for (o,v) in [(0x700,shader.len() as u32),(0x748,64),(0x74c,1),(0x750,1),(0x754,264)] {control[o..o+4].copy_from_slice(&v.to_le_bytes());}
            let output=cpu+2*page;let pools=gfx::pool_memory(&[(0xc000,cpu,page),(0xc002,output,page)]).unwrap();let mut e=Engine::default();e.st.texture_pool=0xc080;e.st.images[2][0]=0x100;e.st.set_program(&[(shader.as_ptr() as u64,control.as_ptr() as u64)]);
            let cb=gfx::gpu().queue.0.commandBuffer().unwrap();let mut r=Replay{e:&mut e,cb,pass:None,heap:None,pools,passes:0,copies:HashMap::new(),fault:None,pool_table:None,resource_buffers:Default::default()};r.dispatch([1,1,1]).unwrap();let (_,texture)=r.e.textures.get(&target,true).unwrap();copy::copy_to_buffer(&r.cb,&texture.0,(0,0),[0,0,0,64,1,1],47,output,&r.pools,(0,0)).unwrap();r.commit_boundary(true);
            assert_eq!((global as *const u32).read(),75);
            let pixels=std::slice::from_raw_parts(output as *const u32,256);let mut shared=pixels.chunks_exact(4).map(|v|v[0]).collect::<Vec<_>>();let mut global=pixels.chunks_exact(4).map(|v|v[2]).collect::<Vec<_>>();shared.sort_unstable();global.sort_unstable();assert_eq!(shared,(0..64).collect::<Vec<_>>());assert_eq!(global,(11..75).collect::<Vec<_>>());for p in pixels.chunks_exact(4) {assert_eq!(p[1],64);assert_eq!(p[3],0xfeedbeef);}
            STATE.lock().unwrap().pools.remove(&0xc001);STATE.lock().unwrap().registered_textures.remove(&(0xc080,0x100));gfx::forget_pool(0xc000);gfx::forget_pool(0xc002);
        });
    }

    #[cfg(target_os="macos")]
    #[test]
    #[ignore="requires a real Metal GPU; run serially"]
    fn gpu_shared_integer_atomics_preserve_signed_extremes_contention_and_old_values(){
        use super::*;
        objc2::rc::autoreleasepool(|_|unsafe{
            crate::kernel::KERNEL.aspace.get_or_init(||crate::mem::AddressSpace::reserve().unwrap());let page=crate::mem::PAGE;let cpu=crate::kernel::aspace().alias()+0x7800000;
            crate::kernel::aspace().set(cpu,page*2,crate::mem::Block{state:crate::mem::state::NORMAL,perm:crate::mem::perm::RW,attr:0}).unwrap();
            let mut target=[0u8;0x68];target[8..16].copy_from_slice(&0xc701u64.to_le_bytes());for(o,v)in[(0x1c,1u32),(0x20,64),(0x24,1),(0x28,1),(0x2c,1),(0x30,47)]{target[o..o+4].copy_from_slice(&v.to_le_bytes());}
            {let mut s=STATE.lock().unwrap();s.pools.insert(0xc701,super::super::objects::MemoryPool{cpu,size:page,flags:0x22});s.registered_textures.insert((0xc780,0x100),target);}
            let output=cpu+page;let pools=gfx::pool_memory(&[(0xc702,output,page)]).unwrap();let mut e=Engine::default();e.st.texture_pool=0xc780;e.st.images[2][0]=0x100;
            let cb=gfx::gpu().queue.0.commandBuffer().unwrap();let mut r=Replay{e:&mut e,cb,pass:None,heap:None,pools,passes:0,copies:HashMap::new(),fault:None,pool_table:None,resource_buffers:Default::default()};
            for(op,signed,initial,operand,expected)in[
                (1u64,0u64,u32::MAX,0x80000001,0x80000001),(1,1,0x7fffffff,0x80000001,0x80000001),
                (2,0,0,0xffffffff,0xffffffff),(2,1,0x80000001,0x7fffffff,0x7fffffff),
                (5,0,0xaaffffff,0xffaa0000,0xaaaa0000),(6,0,0x55000000,0x0088aaaa,0x5588aaaa),
                (7,0,0xf0f0f0f0,0x5555aaaa,0xf0f0f0f0),(8,0,123,456,456),
            ]{
                let atomic=0xec00000000070000u64|(op<<52)|(signed<<28)|(2<<30)|(4<<20)|(255<<8)|4;
                let shader=maxwell_program(&[0xf0c8000002170000,0x3848000000270014,mov32(4,initial),0xef5c000000071404,0xf0a81b8000070000,mov32(4,operand),atomic,0xf0a81b8000070000,0xef4c10000007ff05|(8<<20),mov32(6,operand),0x5c98078000070007,mov_handle(12,0x120),mov32(1,0),0xeb20060600f70004]);
                let mut control=vec![0u8;0x880];for(o,v)in[(0x700,shader.len()as u32),(0x748,64),(0x74c,1),(0x750,1),(0x754,256)]{control[o..o+4].copy_from_slice(&v.to_le_bytes());}
                r.e.st.set_program(&[(shader.as_ptr()as u64,control.as_ptr()as u64)]);r.dispatch([1,1,1]).unwrap();
                let(_,texture)=r.e.textures.get(&target,true).unwrap();copy::copy_to_buffer(&r.cb,&texture.0,(0,0),[0,0,0,64,1,1],47,output,&r.pools,(0,0)).unwrap();r.commit_boundary(true);
                let mut old=Vec::new();for(n,p)in std::slice::from_raw_parts(output as*const u32,256).chunks_exact(4).enumerate(){old.push(p[0]);assert_eq!(&p[1..],&[expected,operand,n as u32],"op{op} signed{signed} thread{n}");}
                old.sort_unstable();let mut wanted=if op==7{vec![initial;32].into_iter().chain(vec![initial^operand;32]).collect::<Vec<_>>()}else{vec![initial].into_iter().chain(vec![expected;63]).collect::<Vec<_>>()};wanted.sort_unstable();assert_eq!(old,wanted,"op{op} signed{signed} previous values");
            }
            STATE.lock().unwrap().pools.remove(&0xc701);STATE.lock().unwrap().registered_textures.remove(&(0xc780,0x100));gfx::forget_pool(0xc702);
            eprintln!("[verify] shared MIN/MAX signed+unsigned extremes, AND/OR/XOR/EXCH:64 contending lanes, aliased operand/destination, all previous values and final words exact");
        });
    }

    #[cfg(target_os="macos")]
    #[test]
    #[ignore="requires a real Metal GPU"]
    fn gpu_private_local_memory_over_16k_preserves_offsets_narrow_signs_and_thread_isolation() {
        use super::*;
        objc2::rc::autoreleasepool(|_| unsafe {
            crate::kernel::KERNEL.aspace.get_or_init(||crate::mem::AddressSpace::reserve().unwrap());let page=crate::mem::PAGE;let cpu=crate::kernel::aspace().alias()+0x4400000;
            crate::kernel::aspace().set(cpu,page*2,crate::mem::Block{state:crate::mem::state::NORMAL,perm:crate::mem::perm::RW,attr:0}).unwrap();
            let mut target=[0u8;0x68];target[8..16].copy_from_slice(&0xc101u64.to_le_bytes());for (o,v) in [(0x1c,1u32),(0x20,64),(0x24,1),(0x28,1),(0x2c,1),(0x30,47)] {target[o..o+4].copy_from_slice(&v.to_le_bytes());}
            {let mut s=STATE.lock().unwrap();s.pools.insert(0xc101,super::super::objects::MemoryPool{cpu,size:page,flags:0x22});s.registered_textures.insert((0xc180,0x100),target);}
            let local=|store:bool,reg:u64,offset:u64,size:u64| {let base=if store {0xef5400000007ff00u64} else {0xef4400000007ff00};(base&!(255u64|(0xffffff<<20)|(7<<48)))|reg|(offset<<20)|(size<<48)};
            let add=0x1c00000000070005u64|(0x20000000u64<<20);
            let shader=maxwell_program(&[0xf0c8000002170000,mov32(4,0x12345678),local(true,4,0,4),add,local(true,5,0x4000,4),mov32(6,0x80),local(true,6,0x45ff,0),local(false,6,0x45ff,1),mov32(7,0x8001),local(true,7,0x45fd,2),local(false,7,0x45fd,3),local(false,4,0,4),local(false,5,0x4000,4),mov_handle(12,0x120),mov32(1,0),0xeb20060600f70004]);
            let mut control=vec![0u8;0x880];for (o,v) in [(0x700,shader.len() as u32),(0x710,0x4600),(0x748,64),(0x74c,1),(0x750,1)] {control[o..o+4].copy_from_slice(&v.to_le_bytes());}
            let output=cpu+page;let pools=gfx::pool_memory(&[(0xc102,output,page)]).unwrap();let mut e=Engine::default();e.st.texture_pool=0xc180;e.st.images[2][0]=0x100;e.st.set_program(&[(shader.as_ptr() as u64,control.as_ptr() as u64)]);
            let cb=gfx::gpu().queue.0.commandBuffer().unwrap();let mut r=Replay{e:&mut e,cb,pass:None,heap:None,pools,passes:0,copies:HashMap::new(),fault:None,pool_table:None,resource_buffers:Default::default()};r.dispatch([1,1,1]).unwrap();let (_,texture)=r.e.textures.get(&target,true).unwrap();copy::copy_to_buffer(&r.cb,&texture.0,(0,0),[0,0,0,64,1,1],47,output,&r.pools,(0,0)).unwrap();r.commit_boundary(true);
            for (n,pixel) in std::slice::from_raw_parts(output as *const u32,256).chunks_exact(4).enumerate() {assert_eq!(pixel,[0x12345678,0x20000000+n as u32,0xffffff80,0xffff8001]);}
            STATE.lock().unwrap().pools.remove(&0xc101);STATE.lock().unwrap().registered_textures.remove(&(0xc180,0x100));gfx::forget_pool(0xc102);
        });
    }

    #[test]
    fn only_frames_missing_many_draws_are_held_back() {
        let mut h = Held::default();
        assert!(h.present(0, 3200));
        assert!(!h.present(700, 3200), "the race behind the loading screen, missing a fifth of its draws, is held back");
        assert!(h.present(30, 2500), "a frame missing about 1% of its draws is shown");
        assert!(h.present(13, 1200), "a race frame missing about 1% of its draws is shown");
        assert!(!h.present(4, 143), "a menu fade missing 4 of 143 draws turns grey and is held back");
    }

    #[test]
    fn never_two_frames_in_a_row_are_held_back() {
        let mut h = Held::default();
        assert!(!h.present(700, 3200));
        assert!(h.present(700, 3200), "the second frame in a row is shown, so a hold costs at most one repeated frame");
        assert!(!h.present(700, 3200));
    }

    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires a real Metal GPU"]
    fn gpu_virtual_aliases_cross_physical_pages_without_copying_storage() {
        use super::*;
        objc2::rc::autoreleasepool(|_| unsafe {
            let page=crate::mem::PAGE as usize;
            let memory=libc::mmap(std::ptr::null_mut(),page*6,libc::PROT_READ|libc::PROT_WRITE,libc::MAP_PRIVATE|libc::MAP_ANON,-1,0);
            assert_ne!(memory,libc::MAP_FAILED);
            let a=memory as u64+4096; let b=memory as u64+page as u64*2+4096; let dst=memory as u64+page as u64*4;
            let mut pools=gfx::pool_memory(&[(0x9a00,a,4096),(0x9a01,b,4096),(0x9a02,dst,4096)]).unwrap();
            let virtual_base=0x20_0000_0000;
            gfx::add_pool_aliases(&mut pools,&[(virtual_base,a,4096),(virtual_base+4096,b,4096)]).unwrap();
            assert!(gfx::pool_buffer_range(&pools,virtual_base+8192,1).is_none());
            let mut e=Engine::default();let cb=gfx::gpu().queue.0.commandBuffer().unwrap();
            let mut r=Replay{e:&mut e,cb,pass:None,heap:None,pools,passes:0,copies:HashMap::new(),fault:None,pool_table:None,resource_buffers:Default::default()};
            assert_eq!(r.pools.iter().map(|p|p.chunks.len()).sum::<usize>(),5);assert_eq!(r.pool_resources().len(),3,"physical buffers must be registered once despite two virtual aliases");
            r.clear_buffer(a,4096,0x11223344).unwrap();r.clear_buffer(b,4096,0x55667788).unwrap();
            r.copy_buffer(virtual_base+4092,dst,8).unwrap();
            r.clear_buffer(virtual_base+4092,8,0xaabbccdd).unwrap();r.commit_boundary(true);
            assert_eq!((dst as *const u32).read(),0x11223344);assert_eq!(((dst+4) as *const u32).read(),0x55667788);
            assert_eq!(((a+4092) as *const u32).read(),0xaabbccdd);assert_eq!((b as *const u32).read(),0xaabbccdd);
            assert_eq!(r.cpu_bytes(virtual_base+4092,8).unwrap(),[0xdd,0xcc,0xbb,0xaa,0xdd,0xcc,0xbb,0xaa]);
            for id in [0x9a00,0x9a01,0x9a02] {gfx::forget_pool(id);}
        });
    }

    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires a real Metal GPU"]
    fn gpu_event_signals_add_write_and_preserve_command_order() {
        use super::*;
        objc2::rc::autoreleasepool(|_| unsafe {
            let page = crate::mem::PAGE as usize;
            let memory = libc::mmap(std::ptr::null_mut(),page*3,libc::PROT_READ|libc::PROT_WRITE,libc::MAP_PRIVATE|libc::MAP_ANON,-1,0);
            assert_ne!(memory,libc::MAP_FAILED);
            let a = memory as u64+4096;
            let pools = gfx::pool_memory(&[(0x9800,a,8192)]).unwrap();
            let mut e = Engine::default(); let cb = gfx::gpu().queue.0.commandBuffer().unwrap();
            let mut r = Replay{e:&mut e,cb,pass:None,heap:None,pools,passes:0,copies:HashMap::new(),fault:None,pool_table:None,resource_buffers:Default::default()};
            r.clear_buffer(a,4,0xfffffffc).unwrap();
            r.signal_event(a,1,8).unwrap(); r.copy_buffer(a,a+256,4).unwrap();
            r.signal_event(a,0,17).unwrap(); r.signal_event(a,1,3).unwrap();
            r.commit_boundary(true);
            assert_eq!((a as *const u32).read(),20); assert_eq!(((a+256) as *const u32).read(),4);
            super::super::events::wait_memory(a,0,20).unwrap();
            gfx::forget_pool(0x9800);
        });
    }

    #[cfg(target_os="macos")]
    #[test]
    #[ignore="requires a real Metal GPU"]
    fn gpu_event_host_reads_preserve_virtual_aliases_uniforms_and_texture_uploads() {
        use super::*;
        objc2::rc::autoreleasepool(|_| unsafe {
            crate::kernel::KERNEL.aspace.get_or_init(||crate::mem::AddressSpace::reserve().unwrap());
            let page=crate::mem::PAGE;let cpu=crate::kernel::aspace().alias()+0x6000000;
            crate::kernel::aspace().set(cpu,page*4,crate::mem::Block{state:crate::mem::state::NORMAL,perm:crate::mem::perm::RW,attr:0}).unwrap();
            // Deliberately start4KiB into a16KiB Metal allocation. Both virtual
            // aliases must include the underlying HeapChunk.offset.
            let a=cpu+4096;let output=cpu+page*2;let va=0x25_0000_0000;let vb=va+4096;
            let mut pools=gfx::pool_memory(&[(0x9700,a,8192),(0x9701,output,page)]).unwrap();
            gfx::add_pool_aliases(&mut pools,&[(va,a+128,4096),(vb,a+128,4096)]).unwrap();
            let mut e=Engine::default();let cb=gfx::gpu().queue.0.commandBuffer().unwrap();
            let mut r=Replay{e:&mut e,cb,pass:None,heap:None,pools,passes:0,copies:HashMap::new(),fault:None,pool_table:None,resource_buffers:Default::default()};
            ((a+512) as *mut u32).write(0);
            r.signal_event(va+384,0,0x12345678).unwrap();
            let before=Retained::as_ptr(&r.cb);r.run(&[Cmd::Barrier(0x10)],0);
            if cpu_events::enabled() {
                assert_eq!(Retained::as_ptr(&r.cb),before,"event-only shader barrier must remain uncommitted");
                assert_eq!(((a+512) as *const u32).read(),0,"the event shader has not run yet");
                r.prepare_cpu_read(vb+1024,4);
                assert_eq!(Retained::as_ptr(&r.cb),before,"an unrelated CPU read must not flush the queue");
                let mut empty=GState::default();empty.texture_pool=0x9780;r.prepare_stage_cpu_reads(&empty,true);
                assert_eq!(Retained::as_ptr(&r.cb),before,"an empty descriptor pool must not flush the queue");
            }
            assert_eq!(r.cpu_bytes(vb+384,4).unwrap(),0x12345678u32.to_le_bytes(),"read through a different virtual alias");

            r.signal_event(a+512,0,55).unwrap();r.run(&[Cmd::Barrier(0x10)],0);
            r.update_uniform(a+512,16,4,&42u32.to_le_bytes());
            let ptr=r.e.overlays.get(&(a+512)).unwrap().2 as *const u32;
            assert_eq!(std::slice::from_raw_parts(ptr,2),[55,42],"uniform snapshot must include the preceding GPU event");
            r.e.overlays.clear();

            let mut b=[0u8;0x68];b[8..16].copy_from_slice(&0x9700u64.to_le_bytes());b[16..24].copy_from_slice(&512u64.to_le_bytes());
            for (at,value) in [(0x1c,1u32),(0x20,1),(0x24,1),(0x28,1),(0x2c,1),(0x30,11)] {b[at..at+4].copy_from_slice(&value.to_le_bytes());}
            {let mut state=STATE.lock().unwrap();state.pools.insert(0x9700,super::super::objects::MemoryPool{cpu:a,size:8192,flags:0x22});state.registered_textures.insert((0x9780,0x100),b);super::super::objects::TEXTURE_DESCRIPTOR_EPOCH.fetch_add(1,Relaxed);}
            r.signal_event(a+512,0,0xabcdef01).unwrap();r.run(&[Cmd::Barrier(0x10)],0);
            let mut st=GState::default();st.texture_pool=0x9780;r.prepare_stage_cpu_reads(&st,true);
            let (_,texture)=r.texture(&b,false).unwrap();
            copy::copy_to_buffer(&r.cb,&texture.0,(0,0),[0,0,0,1,1,1],11,output,&r.pools,(0,0)).unwrap();r.commit_boundary(true);
            assert_eq!((output as *const u32).read(),0xabcdef01,"CPU texture hash/upload must see the actual GPU event write");
            {let mut state=STATE.lock().unwrap();state.pools.remove(&0x9700);state.registered_textures.remove(&(0x9780,0x100));super::super::objects::TEXTURE_DESCRIPTOR_EPOCH.fetch_add(1,Relaxed);}
            gfx::forget_pool(0x9700);gfx::forget_pool(0x9701);
        });
    }

    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires a real Metal GPU; run with NX_SOFT_BC=1"]
    fn gpu_software_bc_upload_waits_for_gpu_source_and_respects_image_stride() {
        use super::*;
        assert!(gfx::bc::enabled(),"this test requires NX_SOFT_BC=1");
        objc2::rc::autoreleasepool(|_| unsafe {
            let page = crate::mem::PAGE as usize;
            let memory = libc::mmap(std::ptr::null_mut(),page*4,libc::PROT_READ|libc::PROT_WRITE,libc::MAP_PRIVATE|libc::MAP_ANON,-1,0);
            assert_ne!(memory,libc::MAP_FAILED);
            let src = memory as u64+4096; let staging = src+512; let dst = src+1024;
            let pools = gfx::pool_memory(&[(0x9900,src,8192)]).unwrap();
            for (image,color) in [0xf800u16,0x07e0].into_iter().enumerate() {
                for row in 0..2 {for col in 0..2 { ((src+image as u64*64+row*24+col*8) as *mut u16).write(color); }}
            }
            let storage = memory as u64+page as u64*3;
            STATE.lock().unwrap().pools.insert(0x9901,super::super::objects::MemoryPool{cpu:storage,size:page as u64,flags:0x22});
            let mut b = [0u8;0x68]; b[8..16].copy_from_slice(&0x9901u64.to_le_bytes());
            for (o,v) in [(0x1c,4u32),(0x20,6),(0x24,5),(0x28,2),(0x2c,1),(0x30,65)] {b[o..o+4].copy_from_slice(&v.to_le_bytes());}
            let mut e = Engine::default(); let cb = gfx::gpu().queue.0.commandBuffer().unwrap();
            let mut r = Replay{e:&mut e,cb,pass:None,heap:None,pools,passes:0,copies:HashMap::new(),fault:None,pool_table:None,resource_buffers:Default::default()};
            r.copy_buffer(src,staging,128).unwrap();
            let region = [0,0,0,6,5,2];
            r.copy_buffer_to_texture(staging,&b,region,(24,64)).unwrap();
            let (_,tex) = r.e.textures.get(&b,true).unwrap();
            copy::copy_to_buffer(&r.cb,&tex.0,(0,0),region,37,dst,&r.pools,(0,0)).unwrap();
            r.commit_boundary(true);
            for (i,color) in [0xff0000ffu32,0xff00ff00].into_iter().enumerate() {
                for pixel in 0..30 {assert_eq!(((dst+i as u64*120) as *const u32).add(pixel).read(),color);}
            }
            assert!(copy::copy_to_buffer(&r.cb,&tex.0,(0,0),region,65,dst,&r.pools,(0,0)).is_err());
            STATE.lock().unwrap().pools.remove(&0x9901); gfx::forget_pool(0x9900);
        });
    }

    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires a real Metal GPU"]
    fn gpu_native_pool_copies_and_pattern_clears_execute_in_order() {
        use super::*;
        objc2::rc::autoreleasepool(|_| unsafe {
            let page = crate::mem::PAGE as usize;
            let memory = libc::mmap(std::ptr::null_mut(), page * 6, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_PRIVATE | libc::MAP_ANON, -1, 0);
            assert_ne!(memory, libc::MAP_FAILED);
            let base = memory as u64;
            let a = base + 4096; let b = base + page as u64 * 3 + 4096;
            let mut pools = gfx::pool_memory(&[(0x9500, a, 8192), (0x9501, b, 8192)]).unwrap();
            // The original renderer computes GPU + CPU-region - BufferMap.
            // A CPU_NO_ACCESS destination's region and Map must both be zero;
            // leaking its physical backing pointer cancels the GPU address.
            STATE.lock().unwrap().pools.insert(0x9500,super::super::objects::MemoryPool{cpu:a,size:8192,flags:0x12});
            STATE.lock().unwrap().pools.insert(0x9501,super::super::objects::MemoryPool{cpu:b,size:8192,flags:0x21});
            let mut objects=[[0u64;16];2];
            for (i,pool) in [0x9500u64,0x9501].into_iter().enumerate() {
                let builder=[0,0,pool,0,128];let args=crate::stubs::Args([objects[i].as_mut_ptr() as u64,builder.as_ptr() as u64,0,0,0,0,0,0],[0;8]);
                assert_eq!(super::super::objects::host_impl("nvnBufferInitialize").unwrap()(&args),1);
            }
            let query=|name:&str,object:u64|super::super::objects::host_impl(name).unwrap()(&crate::stubs::Args([object,0,0,0,0,0,0,0],[0;8]));
            let src_object=objects[0].as_ptr() as u64;let dst_object=objects[1].as_ptr() as u64;
            assert_eq!(query("nvnBufferMap",dst_object),0);assert_eq!(query("nvnMemoryPoolMap",0x9501),0);
            let src=query("nvnBufferGetAddress",src_object)+a-query("nvnBufferMap",src_object);
            let dst=query("nvnBufferGetAddress",dst_object)-query("nvnBufferMap",dst_object);
            assert_eq!(src,query("nvnBufferGetAddress",src_object));assert_eq!(dst,query("nvnBufferGetAddress",dst_object));
            if super::super::gpu_domains::enabled() {assert_ne!(src,a);assert_ne!(dst,b);}else{assert_eq!((src,dst),(a,b));}
            let aliases=STATE.lock().unwrap().gpu_aliases();gfx::add_pool_aliases(&mut pools,&aliases).unwrap();
            let mut e = Engine::default();
            let cb = gfx::gpu().queue.0.commandBuffer().unwrap();
            let mut r = Replay { e: &mut e, cb, pass: None, heap: None, pools, passes: 0, copies: HashMap::new(), fault:None, pool_table:None,resource_buffers:Default::default() };
            r.clear_buffer(a, 128, 0x11223344).unwrap();
            r.copy_buffer(src, dst, 128).unwrap();
            r.clear_buffer(a, 128, 0x88776655).unwrap();
            r.copy_buffer(b, a + 256, 128).unwrap();
            assert!(r.copy_buffer(a, a + 4, 16).is_err());
            ((b+8192-4) as *mut u32).write(0xdecafbad);
            let error=r.copy_buffer(a,b+8192-4,8).unwrap_err();
            assert!(error.contains(&format!("dst{:#x}",b+8192-4)));assert!(error.contains("offset0x4"));assert!(error.contains("currentPools"));
            assert!(r.copy_buffer(a+8192-4,b+512,8).is_err());
            assert!(r.clear_buffer(a + 1, 16, 0).is_err());
            // A pending write survives a nonblocking submit. Texture-cache
            // invalidation must expose that GPU output before CPU uploads.
            let writer=r.commit_boundary(false);assert_eq!(r.e.gpu_buffer_dirty,!cpu_events::transfers_enabled());
            r.run(&[Cmd::Barrier(0x10)],0);
            // The gated path can leave this known transfer on the GPU until
            // the host consumes its destination. Unknown writers still wait
            // at the original barrier.
            r.prepare_cpu_read(b,128);
            assert_eq!(writer.status(),MTLCommandBufferStatus::Completed);assert!(!r.e.gpu_buffer_dirty);
            assert_eq!((b as *const u32).read(),0x11223344,"barrier returned before shared-buffer data became visible");
            // Repeated texture-only barriers preserve the unsubmitted batch.
            r.run(&[Cmd::Barrier(0x10),Cmd::Barrier(0x10)],0);
            assert_eq!(r.cb.status(),MTLCommandBufferStatus::NotEnqueued);
            let completed = r.commit_boundary(true);
            assert_eq!(completed.status(), MTLCommandBufferStatus::Completed);
            assert_eq!(((b+8192-4) as *const u32).read(),0xdecafbad,"failed transfer modified its mapped prefix");
            for i in 0..32 {
                assert_eq!((a as *const u32).add(i).read(), 0x88776655);
                assert_eq!((b as *const u32).add(i).read(), 0x11223344);
                assert_eq!(((a + 256) as *const u32).add(i).read(), 0x11223344);
            }
            query("nvnMemoryPoolFinalize",0x9500);query("nvnMemoryPoolFinalize",0x9501);
            let mut state=STATE.lock().unwrap();state.buffers.remove(&src_object);state.buffers.remove(&dst_object);
        });
    }

    #[cfg(target_os="macos")]
    #[test]
    #[ignore="requires a real Metal GPU"]
    fn gpu_sdk_default_non_power_of_two_volume_and_2d_upload_pixels() {
        use super::*;
        objc2::rc::autoreleasepool(|_|unsafe {
            crate::kernel::KERNEL.aspace.get_or_init(||crate::mem::AddressSpace::reserve().unwrap());let page=crate::mem::PAGE;let cpu=crate::kernel::aspace().alias()+0x6000000;
            crate::kernel::aspace().set(cpu,page*224,crate::mem::Block{state:crate::mem::state::NORMAL,perm:crate::mem::perm::RW,attr:0}).unwrap();
            let output=cpu+2*1024*1024;let pools=gfx::pool_memory(&[(0xc701,output,1024*1024)]).unwrap();
            for (target,height,depth,gy,gz,expected_size) in [(2u32,33u32,33u32,1u32,16u32,1105920u64),(1,17,1,2,1,18432)] {
                let width=33u32;let mut b=[0u8;0x68];b[8..16].copy_from_slice(&0xc700u64.to_le_bytes());for (o,v) in [(0x1c,target),(0x20,width),(0x24,height),(0x28,depth),(0x2c,1),(0x30,46)] {b[o..o+4].copy_from_slice(&v.to_le_bytes());}
                let (_,layout)=super::super::texture_layout(b.as_ptr() as u64).unwrap();assert_eq!(layout.total_size,expected_size);assert_eq!(layout.level_gobs[0],(gy,gz));
                STATE.lock().unwrap().pools.insert(0xc700,super::super::objects::MemoryPool{cpu,size:expected_size,flags:0x22});std::ptr::write_bytes(cpu as *mut u8,0,expected_size as usize);
                // Independent Tegra address construction with original SDK GOB
                // constants, not the upload layout/untile helper under test.
                let width_gobs=(width*16).div_ceil(64);let blocks_y=height.div_ceil(8*gy);
                let expected=|x:u32,y:u32,z:u32|[x as f32/32.0,y as f32/32.0,z as f32/32.0,1.0];
                for z in 0..depth {for y in 0..height {for x in 0..width {
                    let xb=x*16;let block=((z/gz)*blocks_y+y/(8*gy))*width_gobs+xb/64;
                    let gob=(z%gz)*gy+(y/8)%gy;
                    let inside=((xb%64)/32)*256+((y%8)/2)*64+((xb%32)/16)*32+(y%2)*16;
                    let address=cpu+(block*gy*gz*512+gob*512+inside) as u64;
                    std::ptr::copy_nonoverlapping(expected(x,y,z).as_ptr(),address as *mut f32,4);
                }}}
                let mut textures=textures::Textures::default();let (_,texture)=textures.get(&b,false).unwrap();
                let row=(width*16).next_multiple_of(256);let image=row*height;
                let cb=gfx::gpu().queue.0.commandBuffer().unwrap();copy::copy_to_buffer(&cb,&texture.0,(0,0),[0,0,0,width as i32,height as i32,depth as i32],46,output,&pools,(row,image)).unwrap();cb.commit();cb.waitUntilCompleted();assert_eq!(cb.status(),MTLCommandBufferStatus::Completed);
                for z in 0..depth {for y in 0..height {for x in 0..width {
                    let at=output+(z*image+y*row+x*16) as u64;assert_eq!(std::slice::from_raw_parts(at as *const f32,4),expected(x,y,z),"target{target} pixel{x},{y},{z}");
                }}}
                STATE.lock().unwrap().pools.remove(&0xc700);
            }
            gfx::forget_pool(0xc701);
        });
    }

    #[cfg(target_os="macos")]
    #[test]
    #[ignore="requires a real Metal GPU"]
    fn gpu_tlds_3d_preserves_depth_and_overlapping_packed_destinations() {
        use super::*;
        objc2::rc::autoreleasepool(|_|unsafe {
            crate::kernel::KERNEL.aspace.get_or_init(||crate::mem::AddressSpace::reserve().unwrap());let page=crate::mem::PAGE;let cpu=crate::kernel::aspace().alias()+0x5800000;
            crate::kernel::aspace().set(cpu,page*4,crate::mem::Block{state:crate::mem::state::NORMAL,perm:crate::mem::perm::RW,attr:0}).unwrap();
            let builder=|pool:u64,target:u32,w:u32,h:u32,d:u32| {let mut b=[0u8;0x68];b[8..16].copy_from_slice(&pool.to_le_bytes());for (o,v) in [(0x1c,target),(0x20,w),(0x24,h),(0x28,d),(0x2c,1),(0x30,46)] {b[o..o+4].copy_from_slice(&v.to_le_bytes());}b};
            let source=builder(0xc600,2,4,3,5);let target=builder(0xc601,1,1,1,1);
            {let mut state=STATE.lock().unwrap();for (i,b) in [source,target].into_iter().enumerate() {state.pools.insert(0xc600+i as u64,super::super::objects::MemoryPool{cpu:cpu+i as u64*page,size:page,flags:0x22});state.registered_textures.insert((0xc680,0x100+i as u32),b);}}
            let io=cpu+2*page;for z in 0..5 {for y in 0..3 {for x in 0..4 {let values=[x as f32+1.0,y as f32+11.0,z as f32+101.0,1.0];std::ptr::copy_nonoverlapping(values.as_ptr(),(io+((z*3+y)*4+x)*16) as *mut f32,4);}}}
            let output=cpu+3*page;let pools=gfx::pool_memory(&[(0xc602,io,page),(0xc603,output,page)]).unwrap();let mut e=Engine::default();e.st.texture_pool=0xc680;e.st.textures[2][0]=0x100;e.st.images[2][0]=0x101;
            let cb=gfx::gpu().queue.0.commandBuffer().unwrap();let mut r=Replay{e:&mut e,cb,pass:None,heap:None,pools,passes:0,copies:HashMap::new(),fault:None,pool_table:None,resource_buffers:Default::default()};r.copy_buffer_to_texture(io,&source,[0,0,0,4,3,5],(0,0)).unwrap();
            let mut control=vec![0u8;0x880];for o in [0x748,0x74c,0x750] {control[o..o+4].copy_from_slice(&1u32.to_le_bytes());}
            // The original color-LUT instruction aliases srcA with dst0/1 and
            // srcB.z with dst2; output RGB must snapshot all three coordinates.
            let clear=255u64|(255<<8)|(255<<20)|(255<<28)|(8191<<36);
            let load=(0xdae205a171770808u64&!clear)|8|(8<<8)|(23<<20)|(23<<28)|(8<<36);
            for (x,y,z,expected) in [(0u32,0u32,0u32,[1.0,11.0,101.0,0.0]),(3,2,1,[4.0,13.0,102.0,0.0]),(2,1,4,[3.0,12.0,105.0,0.0]),(0,0,5,[0.0;4]),(u32::MAX,0,1,[0.0;4])] {
                let shader=maxwell_program(&[mov32(8,x),mov32(9,y),mov32(23,z),load,0x5c9807800177000a,mov32(11,0),mov_handle(12,0x120),mov32(14,0),mov32(15,0),0xeb20060600f70e08]);
                control[0x700..0x704].copy_from_slice(&(shader.len() as u32).to_le_bytes());r.e.st.set_program(&[(shader.as_ptr() as u64,control.as_ptr() as u64)]);r.e.bank_ok[2]=false;r.dispatch([1,1,1]).unwrap();let (_,tex)=r.e.textures.get(&target,true).unwrap();copy::copy_to_buffer(&r.cb,&tex.0,(0,0),[0,0,0,1,1,1],46,output,&r.pools,(0,0)).unwrap();r.commit_boundary(true);
                assert_eq!(std::slice::from_raw_parts(output as *const f32,4),expected,"3D TLDS coordinate{x},{y},{z}");
            }
            for id in 0xc600..=0xc603 {STATE.lock().unwrap().pools.remove(&id);gfx::forget_pool(id);}STATE.lock().unwrap().registered_textures.retain(|(pool,_),_|*pool!=0xc680);
        });
    }

    #[cfg(target_os="macos")]
    #[test]
    #[ignore="requires a real Metal GPU"]
    fn gpu_sampled_views_resolve_fragmented_backing_and_only_the_requested_mips() {
        use super::*;
        objc2::rc::autoreleasepool(|_|unsafe {
            crate::kernel::KERNEL.aspace.get_or_init(||crate::mem::AddressSpace::reserve().unwrap());
            let page=crate::mem::PAGE;let cpu=crate::kernel::aspace().alias()+0x5400000;
            crate::kernel::aspace().set(cpu,page*12,crate::mem::Block{state:crate::mem::state::NORMAL,perm:crate::mem::perm::RW,attr:0}).unwrap();
            let mut b=[0u8;0x68];b[8..16].copy_from_slice(&0xc500u64.to_le_bytes());
            for (o,v) in [(0x1c,4u32),(0x20,64),(0x24,64),(0x28,2),(0x2c,7),(0x30,37)] {b[o..o+4].copy_from_slice(&v.to_le_bytes());}
            let (shape,layout)=super::super::texture_layout(b.as_ptr() as u64).unwrap();
            let size=layout.total_size.next_multiple_of(4096);assert!(size<page*6);
            let colors=[0xff332211u32,0xff665544,0xff998877,0xffccbbaa,0xff193b5d,0xff426486,0xff315273];
            // Deliberately scatter each virtual 4 KiB page, including the base
            // mip itself, and leave the final allocation page unmapped.
            {let mut state=STATE.lock().unwrap();state.pools.insert(0xc500,super::super::objects::MemoryPool{cpu:0,size,flags:0x221});
                let mut maps=Vec::new();for n in 0..size/4096 {let phys=0xc510+n;let address=cpu+n*8192;
                    state.pools.insert(phys,super::super::objects::MemoryPool{cpu:address,size:4096,flags:0x10a});
                    maps.push((n*4096,4096,phys,0));
                }state.virtual_maps.insert(0xc500,maps);
            }
            // Every byte in a mip stores the same RGBA word, so the independent
            // expected readback does not depend on the untile implementation.
            {let state=STATE.lock().unwrap();for layer in 0..shape.layers {for level in 0..shape.levels {
                let start=layout.layer_size*layer as u64+layout.level_offsets[level as usize];
                let end=if level+1<shape.levels {layout.layer_size*layer as u64+layout.level_offsets[level as usize+1]} else {layout.layer_size*(layer+1) as u64};
                for (addr,len) in state.pool_cpu_ranges(0xc500,start,end-start).unwrap() {for off in (0..len).step_by(4) {((addr+off) as *mut u32).write(colors[level as usize]+layer*0x00010101);}}
            }}}
            let last_map=STATE.lock().unwrap().virtual_maps.get_mut(&0xc500).unwrap().pop().unwrap();
            let output=cpu+10*page;let pools=gfx::pool_memory(&[(0xc5f0,output,page)]).unwrap();
            let mut textures=textures::Textures::default();
            for (layer,first,count) in [(0,0,2),(0,1,1),(1,0,1)] {
                let mut view=b;super::super::objects::set_view_range(&mut view,Some((layer,1)),Some((first,count)));
                let (key,base)=textures.get(&view,false).unwrap();let selected=textures.view(key,&view,&base,nxshader::TexKind::T2D).unwrap();
                assert_eq!(selected.0.width(),(64>>first) as usize);assert_eq!(selected.0.mipmapLevelCount(),count as usize);
                let cb=gfx::gpu().queue.0.commandBuffer().unwrap();
                copy::copy_to_buffer(&cb,&selected.0,(0,0),[0,0,0,1,1,1],37,output,&pools,(0,0)).unwrap();cb.commit();cb.waitUntilCompleted();
                assert_eq!((output as *const u32).read(),colors[first as usize]+layer*0x00010101);
            }
            // Restore the tail and request a previously unuploaded mip, then
            // remap that mip without changing the physical first page/key.
            STATE.lock().unwrap().virtual_maps.get_mut(&0xc500).unwrap().push(last_map);
            let mut view=b;super::super::objects::set_view_range(&mut view,Some((1,1)),Some((6,1)));
            let (key,base)=textures.get(&view,false).unwrap();let selected=textures.view(key,&view,&base,nxshader::TexKind::T2D).unwrap();
            let read=|tex:&gfx::Texture| {let cb=gfx::gpu().queue.0.commandBuffer().unwrap();copy::copy_to_buffer(&cb,tex,(0,0),[0,0,0,1,1,1],37,output,&pools,(0,0)).unwrap();cb.commit();cb.waitUntilCompleted();(output as *const u32).read()};
            assert_eq!(read(&selected.0),colors[6]+0x00010101);
            let replacement=cpu+9*page;for off in (0..4096).step_by(4) {((replacement+off) as *mut u32).write(0xffabcdef);}
            {let mut state=STATE.lock().unwrap();state.pools.insert(0xc5e0,super::super::objects::MemoryPool{cpu:replacement,size:4096,flags:0x10a});state.virtual_maps.get_mut(&0xc500).unwrap().last_mut().unwrap().2=0xc5e0;}
            super::super::objects::VIRTUAL_MAPPING_EPOCH.fetch_add(1,std::sync::atomic::Ordering::Relaxed);
            let (same_key,base)=textures.get(&view,false).unwrap();assert_eq!(key,same_key);let selected=textures.view(same_key,&view,&base,nxshader::TexKind::T2D).unwrap();assert_eq!(read(&selected.0),0xffabcdef);
            gfx::forget_pool(0xc5f0);let mut state=STATE.lock().unwrap();state.virtual_maps.remove(&0xc500);state.pools.retain(|id,_|!(0xc500..=0xc5ff).contains(id));
        });
    }

    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires a real Metal GPU"]
    fn gpu_strided_texture_roundtrip_and_vamm_cpu_uploads() {
        use super::*;
        objc2::rc::autoreleasepool(|_| unsafe {
            let page = crate::mem::PAGE as usize;
            let memory = libc::mmap(std::ptr::null_mut(), page * 8, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_PRIVATE | libc::MAP_ANON, -1, 0);
            assert_ne!(memory, libc::MAP_FAILED);
            let base = memory as u64;
            let src = base + 4096; let dst = base + page as u64 * 3;
            let texture_storage = base + page as u64 * 6;
            let pools = gfx::pool_memory(&[(0x9700, src, 8192), (0x9701, dst, 8192)]).unwrap();
            let words = [0xff030201u32, 0xff060504, 0xff090807, 0xff0c0b0a];
            for (i, off) in [0, 16, 64, 80].into_iter().enumerate() { ((src + off) as *mut u32).write(words[i]); }
            STATE.lock().unwrap().pools.insert(0x9702, super::super::objects::MemoryPool { cpu: texture_storage, size: page as u64, flags: 0x22 });
            let builder = |pool: u64, target: u32, width: u32, height: u32, depth: u32, stride: u32| {
                let mut b = [0u8; 0x68]; b[8..16].copy_from_slice(&pool.to_le_bytes());
                for (o, v) in [(0x18, 0x10u32), (0x1c, target), (0x20, width), (0x24, height), (0x28, depth), (0x2c, 1), (0x30, 37), (0x5c, stride)] { b[o..o + 4].copy_from_slice(&v.to_le_bytes()); }
                b
            };
            let b = builder(0x9702, 4, 1, 2, 2, 16);
            let mut e = Engine::default();
            let cb = gfx::gpu().queue.0.commandBuffer().unwrap();
            let mut r = Replay { e: &mut e, cb, pass: None, heap: None, pools, passes: 0, copies: HashMap::new(), fault:None, pool_table:None,resource_buffers:Default::default() };
            let region = [0, 0, 0, 1, 2, 2];
            r.copy_buffer_to_texture(src, &b, region, (16, 64)).unwrap();
            let (_, tex) = r.e.textures.get(&b, true).unwrap();
            copy::copy_to_buffer(&r.cb, &tex.0, (0, 0), region, 37, dst, &r.pools, (32, 128)).unwrap();
            r.commit_boundary(true);
            for (i, off) in [0, 32, 128, 160].into_iter().enumerate() { assert_eq!(((dst + off) as *const u32).read(), words[i]); }
            assert_eq!(((dst + 4) as *const u32).read(), 0);

            crate::kernel::KERNEL.aspace.get_or_init(|| crate::mem::AddressSpace::reserve().unwrap());
            let cpu = crate::kernel::aspace().alias() + 0x200000;
            crate::kernel::aspace().set(cpu, page as u64, crate::mem::Block { state: crate::mem::state::NORMAL, perm: crate::mem::perm::RW, attr: 0 }).unwrap();
            STATE.lock().unwrap().pools.insert(0x9703, super::super::objects::MemoryPool { cpu, size: page as u64, flags: 0x22 });
            let cpu_b = builder(0x9703, 1, 1, 1, 1, 4);
            for value in [0xff332211u32, 0xff665544] {
                (cpu as *mut u32).write(value);
                flush_memory(cpu, 4); r.e.forget_dead();
                let (_, t) = r.e.textures.get(&cpu_b, false).unwrap();
                copy::copy_to_buffer(&r.cb, &t.0, (0, 0), [0, 0, 0, 1, 1, 1], 37, dst, &r.pools, (0, 0)).unwrap();
                r.commit_boundary(true);
                assert_eq!((dst as *const u32).read(), value, "VAMM upload or same-frame flush lost a 4-byte color");
            }
            STATE.lock().unwrap().pools.remove(&0x9702); STATE.lock().unwrap().pools.remove(&0x9703);
            gfx::forget_pool(0x9700); gfx::forget_pool(0x9701);
        });
    }

    #[cfg(target_os="macos")]
    #[test]
    #[ignore="requires a real Metal GPU"]
    fn gpu_fragment_static_sampler_bank_crosses_direct_table_limit() {
        use super::*;
        objc2::rc::autoreleasepool(|_| unsafe {
            crate::kernel::KERNEL.aspace.get_or_init(||crate::mem::AddressSpace::reserve().unwrap());
            let page=crate::mem::PAGE;let cpu=crate::kernel::aspace().alias()+0x4400000;
            crate::kernel::aspace().set(cpu,page*20,crate::mem::Block{state:crate::mem::state::NORMAL,perm:crate::mem::perm::RW,attr:0}).unwrap();
            let positions=[[-1f32,-1.0,0.5,1.0],[3.0,-1.0,0.5,1.0],[-1.0,3.0,0.5,1.0]];
            std::ptr::copy_nonoverlapping(positions.as_ptr().cast::<u8>(),cpu as *mut u8,size_of_val(&positions));
            let make_shader=|stage:u32,words:&[u64]| {
                let mut b=vec![0u8;0x80];b[..4].copy_from_slice(&0x12345678u32.to_le_bytes());b[0x30..0x34].copy_from_slice(&(stage<<10).to_le_bytes());
                if stage==5 {b[0x78..0x7c].copy_from_slice(&15u32.to_le_bytes());}b.extend(maxwell_program(words));b
            };
            let vs=make_shader(1,&[0xefd800000007ff00|(3<<47)|(0x80<<20),0xeff000000007ff00|(3<<47)|(0x70<<20)]);
            let builder=|pool:u64,width:u32,height:u32| {
                let mut b=[0u8;0x68];b[8..16].copy_from_slice(&pool.to_le_bytes());
                for (o,v) in [(0x18,0x10u32),(0x1c,1),(0x20,width),(0x24,height),(0x28,1),(0x2c,1),(0x30,46),(0x5c,width*16)] {b[o..o+4].copy_from_slice(&v.to_le_bytes());}b
            };
            for i in 0..17u64 {
                let value=[(i+1) as f32,((i+1)*2) as f32,((i+1)*3) as f32,1.0];
                std::ptr::copy_nonoverlapping(value.as_ptr(),(cpu+(i+1)*page) as *mut f32,4);
                let pool=0xc100+i;let b=builder(pool,1,1);let mut s=STATE.lock().unwrap();
                s.pools.insert(pool,super::super::objects::MemoryPool{cpu:cpu+(i+1)*page,size:page,flags:0x22});
                s.registered_textures.insert((0xc180,0x100+i as u32),b);s.registered_samplers.insert((0xc190,i as u32),[0u8;0x60]);
            }
            let target=builder(0xc1a0,4,4);let output=cpu+19*page;
            STATE.lock().unwrap().pools.insert(0xc1a0,super::super::objects::MemoryPool{cpu:cpu+18*page,size:page,flags:0x22});
            STATE.lock().unwrap().pools.insert(0xc1b0,super::super::objects::MemoryPool{cpu:vs.as_ptr() as u64,size:vs.len() as u64,flags:0x22});
            let pools=gfx::pool_memory(&[(0xc1c0,cpu,page),(0xc1c1,output,page)]).unwrap();
            let mut e=Engine::default();e.colors=vec![Some(Box::new(target))];e.st.texture_pool=0xc180;e.st.sampler_pool=0xc190;
            e.st.attribs[0]=0x40|(46<<21);e.st.streams[0]=(16,0);e.st.vbufs[0]=(cpu,size_of_val(&positions) as u64);
            for i in 0..17 {e.st.textures[1][i]=(0x100+i as u64)|((i as u64)<<20);}
            let mut vc=vec![0u8;0x880];let mut fc=vc.clone();vc[0x700..0x704].copy_from_slice(&(vs.len() as u32).to_le_bytes());
            let cb=gfx::gpu().queue.0.commandBuffer().unwrap();let mut r=Replay{e:&mut e,cb,pass:None,heap:None,pools,passes:0,copies:HashMap::new(),fault:None,pool_table:None,resource_buffers:Default::default()};
            for count in [16u64,17] {
                let mut words=vec![mov32(0,0.5f32.to_bits()),mov32(1,0.5f32.to_bits()),mov32(20,0),mov32(21,0),mov32(22,0),mov32(23,0)];
                for i in 0..count {
                    words.push(mov_handle(2,0x20+i*8));
                    words.push((0xdeb80023a0270000u64&!(255u64|(255<<8)|(255<<20)|(15<<31)))|8|(2<<20)|(15<<31));
                    for c in 0..4u64 {words.push((0x5c58000000070000u64&!(255u64|(255<<8)|(255<<20)))|(20+c)|((20+c)<<8)|((8+c)<<20));}
                }
                for c in 0..4u64 {words.push((0x5c9807800ff70007u64&!(255u64|(255<<20)))|c|((20+c)<<20));}
                let fs=make_shader(5,&words);fc[0x700..0x704].copy_from_slice(&(fs.len() as u32).to_le_bytes());
                STATE.lock().unwrap().pools.insert(0xc1b1,super::super::objects::MemoryPool{cpu:fs.as_ptr() as u64,size:fs.len() as u64,flags:0x22});
                let translation=nxshader::translate(&fs).unwrap();assert_eq!(translation.interface.argument_textures,count>16);assert_eq!(translation.interface.textures.len(),count as usize);
                let (named,_)=translation.named("sampler_probe");assert!(!named.contains("struct StaticTextures {"));
                r.e.st.set_program(&[(vs.as_ptr() as u64,vc.as_ptr() as u64),(fs.as_ptr() as u64,fc.as_ptr() as u64)]);
                r.run(&[Cmd::ClearColor{index:0,rgba:[0.125;4]},Cmd::Draw(Draw{mode:4,index:None,first:0,count:3,base_vertex:0,base_instance:0,instances:1})],0);
                r.end_pass();let (_,texture)=r.e.textures.get(&target,true).unwrap();copy::copy_to_buffer(&r.cb,&texture.0,(0,0),[0,0,0,4,4,1],46,output,&r.pools,(0,0)).unwrap();r.commit_boundary(true);
                let sum=(count*(count+1)/2) as f32;let expected=[sum,sum*2.0,sum*3.0,count as f32];
                for pixel in std::slice::from_raw_parts(output as *const f32,64).chunks_exact(4) {assert_eq!(pixel,expected,"{count} statically resolved sampler pairs");}
            }
            for id in (0xc100..0xc111).chain([0xc1a0,0xc1b0,0xc1b1]) {STATE.lock().unwrap().pools.remove(&id);}
            STATE.lock().unwrap().registered_textures.retain(|(p,_),_|*p!=0xc180);STATE.lock().unwrap().registered_samplers.retain(|(p,_),_|*p!=0xc190);
            gfx::forget_pool(0xc1c0);gfx::forget_pool(0xc1c1);
        });
    }


    #[cfg(target_os="macos")]
    #[test]
    #[ignore="requires real Metal GPU, NX_BRX_CAPTURE and its independent pruned-dispatch reference"]
    fn gpu_captured_constant_jump_tables_preserve_all_original_output_bytes() {
        use super::*;
        let root=std::path::PathBuf::from(std::env::var("NX_BRX_CAPTURE").expect("set NX_BRX_CAPTURE"));
        let hash="36a74e3b93325d14";let capture=root.join(format!("work/compute-probes/{hash}-0"));
        let reference=root.parent().unwrap().parent().unwrap().join("outputs/p116-replay-pruned-reference");
        let metadata=std::fs::read_to_string(capture.join("metadata.txt")).unwrap();assert!(metadata.contains("groups=[9, 1, 1]"));
        let code=std::fs::read(root.join(format!("work/msl_rt/{hash}.bin"))).unwrap();let constants=std::fs::read(capture.join("cb1.bin")).unwrap();
        // Registration order varies with startup timing. Match the actual
        // instruction bytes rather than borrowing another run's ordinal.
        let control=std::fs::read_dir(root.join("work/shaders")).unwrap().filter_map(Result::ok).find_map(|entry| {
            let path=entry.path();if path.extension().is_none_or(|e|e!="code") {return None;}
            let bytes=std::fs::read(&path).ok()?;if !bytes.starts_with(&0x12345679u32.to_le_bytes()) {return None;}
            let ctl=std::fs::read(path.with_extension("control")).ok()?;
            let end=u32::from_le_bytes(ctl[0x700..0x704].try_into().unwrap()) as usize;let start=u32::from_le_bytes(ctl[0x708..0x70c].try_into().unwrap()) as usize;
            (bytes.get(start..end)==Some(code.as_slice())).then_some(ctl)
        }).expect("no captured control matches the original instructions");
        let word=|at:usize|u32::from_le_bytes(control[at..at+4].try_into().unwrap());let count=(word(0x704)-word(0x700)) as usize;assert_eq!(count,256);
        let program=nxshader::program::Program::decode_with_constants(&code,Some(&constants[..count])).unwrap();assert_eq!(program.indirect_constant_bytes,Some(256));assert!(program.targets.len()<32);
        objc2::rc::autoreleasepool(|_| unsafe {
            let compiled=shaders::compile_config_constants(&code,true,0,0,0,0xfc27b12,(0,0),Some(&constants[..count])).unwrap();
            let dev=&gfx::gpu().device.0;let pipeline=dev.newComputePipelineStateWithFunction_error(&compiled.func.0).unwrap();
            let make=|bytes:&[u8]|dev.newBufferWithBytes_length_options(std::ptr::NonNull::new(bytes.as_ptr() as *mut u8).unwrap().cast(),bytes.len(),MTLResourceOptions::StorageModeShared).unwrap();
            let mut buffers=Vec::new();let mut banks=std::collections::BTreeMap::new();
            for line in metadata.lines().filter(|l|l.starts_with("storage")) {
                let (name,range)=line.split_once('=').unwrap();let address=u64::from_str_radix(range.split(',').next().unwrap().trim_start_matches("0x"),16).unwrap();
                let bytes=std::fs::read(capture.join(format!("{name}.bin"))).unwrap();
                if let Some((_,old))=banks.get(&address) {assert_eq!(&bytes,old,"aliased snapshot changed");} else {banks.insert(address,(make(&bytes),bytes));}
            }
            let spans=banks.iter().flat_map(|(&address,(buffer,_))|[address,buffer.length() as u64,buffer.gpuAddress()]).collect::<Vec<_>>();let table=make(std::slice::from_raw_parts(spans.as_ptr().cast::<u8>(),spans.len()*8));
            let zero=&gfx::gpu().zeros.0;let mut res:state::Res=std::mem::zeroed();res.zero_gpu=zero.gpuAddress();res.cb.fill(zero.gpuAddress());res.pool_spans=table.gpuAddress();res.pool_span_count=banks.len() as u64;
            for bank in [0usize,1,3] {let bytes=std::fs::read(capture.join(format!("cb{bank}.bin"))).unwrap();let b=make(&bytes);res.cb[bank]=b.gpuAddress();buffers.push(b);}
            let fault=dev.newBufferWithLength_options(32,MTLResourceOptions::StorageModeShared).unwrap();std::ptr::write_bytes(fault.contents().as_ptr(),0,32);res.fault_gpu=fault.gpuAddress();res.shader_hash=u64::from_str_radix(hash,16).unwrap();
            let cb=gfx::gpu().queue.0.commandBuffer().unwrap();let enc=cb.computeCommandEncoder().unwrap();enc.setComputePipelineState(&pipeline);
            for b in buffers.iter().chain(banks.values().map(|(b,_)|b)).chain([&table,&fault,zero]) {enc.useResource_usage(ProtocolObject::from_ref(&**b),MTLResourceUsage::Read|MTLResourceUsage::Write);}
            enc.setBytes_length_atIndex(std::ptr::NonNull::from(&res).cast(),size_of_val(&res),0);enc.dispatchThreadgroups_threadsPerThreadgroup(MTLSize{width:9,height:1,depth:1},MTLSize{width:64,height:1,depth:1});enc.endEncoding();cb.commit();cb.waitUntilCompleted();assert_eq!(cb.status(),MTLCommandBufferStatus::Completed);
            assert_eq!(std::slice::from_raw_parts(fault.contents().as_ptr().cast::<u32>(),8),[0;8]);
            let mut modified=0;
            for (address,(buffer,initial)) in &banks {
                let actual=std::slice::from_raw_parts(buffer.contents().as_ptr().cast::<u8>(),buffer.length());let expected=std::fs::read(reference.join(format!("{address:x}.bin"))).unwrap();
                assert_eq!(actual,expected,"original output at {address:#x} differs from independently pruned control-flow reference");
                if actual!=initial {modified+=1;}
            }
            assert_eq!(modified,2,"both original geometry output streams must execute");
            // A genuinely shorter embedded bank must fault before reading
            // adjacent bytes, even though its first target remains legal.
            let truncated=shaders::compile_config_constants(&code,true,0,0,0,0xfc27b13,(0,0),Some(&constants[..4])).unwrap();
            let pipeline=dev.newComputePipelineStateWithFunction_error(&truncated.func.0).unwrap();std::ptr::write_bytes(fault.contents().as_ptr(),0,32);
            let cb=gfx::gpu().queue.0.commandBuffer().unwrap();let enc=cb.computeCommandEncoder().unwrap();enc.setComputePipelineState(&pipeline);
            for b in buffers.iter().chain(banks.values().map(|(b,_)|b)).chain([&table,&fault,zero]) {enc.useResource_usage(ProtocolObject::from_ref(&**b),MTLResourceUsage::Read|MTLResourceUsage::Write);}
            enc.setBytes_length_atIndex(std::ptr::NonNull::from(&res).cast(),size_of_val(&res),0);enc.dispatchThreadgroups_threadsPerThreadgroup(MTLSize{width:9,height:1,depth:1},MTLSize{width:64,height:1,depth:1});enc.endEncoding();cb.commit();cb.waitUntilCompleted();assert_eq!(cb.status(),MTLCommandBufferStatus::Completed);
            assert_eq!(fault.contents().as_ptr().cast::<u32>().read(),256,"truncated C1 must report its bounds violation");
        });
    }


    #[cfg(target_os="macos")]
    #[test]
    #[ignore="requires a real Metal GPU"]
    fn gpu_fragment_embedded_jump_tables_select_pixels_and_update_without_stale_cache() {
        use super::*;
        fn address(n:usize)->u32 {((n/3*4+1+n%3)*8) as u32}
        objc2::rc::autoreleasepool(|_| unsafe {
            crate::kernel::KERNEL.aspace.get_or_init(||crate::mem::AddressSpace::reserve().unwrap());let page=crate::mem::PAGE;let cpu=crate::kernel::aspace().alias()+0x4c00000;
            crate::kernel::aspace().set(cpu,page*4,crate::mem::Block{state:crate::mem::state::NORMAL,perm:crate::mem::perm::RW,attr:0}).unwrap();
            let positions=[[-1f32,-1.0,0.5,1.0],[3.0,-1.0,0.5,1.0],[-1.0,3.0,0.5,1.0]];std::ptr::copy_nonoverlapping(positions.as_ptr().cast::<u8>(),cpu as *mut u8,size_of_val(&positions));
            let make_shader=|stage:u32,words:&[u64]| {let mut b=vec![0u8;0x80];b[..4].copy_from_slice(&0x12345678u32.to_le_bytes());b[0x30..0x34].copy_from_slice(&(stage<<10).to_le_bytes());if stage==5 {b[0x78..0x7c].copy_from_slice(&15u32.to_le_bytes());}b.extend(maxwell_program(words));b};
            let vs=make_shader(1,&[0xefd800000007ff00|(3<<47)|(0x80<<20),0xeff000000007ff00|(3<<47)|(0x70<<20)]);
            let colors=[[0.25f32,0.5,0.75,2.0],[1.0,0.25,0.0,2.0],[0.0,0.75,0.5,2.0]];
            // The final alpha comes from two actual backward-loop iterations,
            // followed by a forward BRX and PBK/BRK convergence. This catches
            // a forward-dispatch implementation that loses a backedge or lets
            // a taken branch execute the old block's fallthrough assignment.
            let mut words=vec![0xe2a0000000000000u64,mov32(4,0),0x1c00000000170404,0x36538380002704ff,0xe24000000007000d,
                0x4c98078000070008|(3<<34),0x3848000000270808,0xef94001000070808,0xe25000000007080f];
            let delta=address(2) as i64-address(4) as i64-8;words[4]|=((delta as u64)&0xffffff)<<20;
            let mut targets=Vec::new();for color in colors {targets.push(address(words.len()));for (c,value) in color.iter().take(3).enumerate() {words.push(mov32(c as u64,value.to_bits()));}
                words.push(0x5cb8000000070000u64|3|(4<<20)|(2<<10)|(2<<8)); // I2F.F32.U32 R3,R4.
                words.push(0xe34000000007000f);}
            let delta=address(words.len()) as i64-address(0) as i64-8;words[0]|=((delta as u64)&0xffffff)<<20;
            let delta=-(address(8) as i64+8);words[8]|=((delta as u64)&0xffffff)<<20;
            let fs=make_shader(5,&words);let fs_cpu=cpu+page;std::ptr::copy_nonoverlapping(fs.as_ptr(),fs_cpu as *mut u8,fs.len());let constant_cpu=fs_cpu+fs.len() as u64;
            std::ptr::write_bytes(constant_cpu as *mut u8,0,256);std::ptr::copy_nonoverlapping(targets.as_ptr(),constant_cpu as *mut u32,3);
            let mut control=vec![0u8;0x880];let mut vc=control.clone();vc[0x700..0x704].copy_from_slice(&(vs.len() as u32).to_le_bytes());control[0x700..0x704].copy_from_slice(&(fs.len() as u32).to_le_bytes());control[0x704..0x708].copy_from_slice(&(fs.len() as u32+256).to_le_bytes());
            for (id,address,size) in [(0xc310,cpu,page),(0xc311,fs_cpu,page),(0xc312,cpu+2*page,page),(0xc31f,vs.as_ptr() as u64,vs.len() as u64)] {STATE.lock().unwrap().pools.insert(id,super::super::objects::MemoryPool{cpu:address,size,flags:0x22});}
            let mut target=[0u8;0x68];target[8..16].copy_from_slice(&0xc312u64.to_le_bytes());for (o,v) in [(0x1c,1u32),(0x20,4),(0x24,4),(0x28,1),(0x2c,1),(0x30,46)] {target[o..o+4].copy_from_slice(&v.to_le_bytes());}
            let output=cpu+3*page;let pools=gfx::pool_memory(&[(0xc310,cpu,page),(0xc311,fs_cpu,page),(0xc313,output,page)]).unwrap();
            let mut e=Engine::default();e.colors=vec![Some(Box::new(target))];e.st.attribs[0]=0x40|(46<<21);e.st.streams[0]=(16,0);e.st.vbufs[0]=(cpu,size_of_val(&positions) as u64);e.st.ubos[1][0]=(cpu+512,16);e.st.set_program(&[(vs.as_ptr() as u64,vc.as_ptr() as u64),(fs_cpu,control.as_ptr() as u64)]);
            let cb=gfx::gpu().queue.0.commandBuffer().unwrap();let mut r=Replay{e:&mut e,cb,pass:None,heap:None,pools,passes:0,copies:HashMap::new(),fault:None,pool_table:None,resource_buffers:Default::default()};
            let first=r.e.shaders.stage(fs_cpu,control.as_ptr() as u64).unwrap();assert_eq!(first.constants_len,256);
            for (selector,expected,swap) in [(0u32,colors[0],false),(1,colors[1],false),(2,colors[2],false),(0,colors[2],true)] {
                ((cpu+512) as *mut u32).write(selector);
                if swap {let a=(constant_cpu as *mut u32).read();(constant_cpu as *mut u32).write(targets[2]);((constant_cpu+8) as *mut u32).write(a);}
                r.run(&[Cmd::ClearColor{index:0,rgba:[0.125;4]},Cmd::Draw(Draw{mode:4,index:None,first:0,count:3,base_vertex:0,base_instance:0,instances:1})],0);
                r.end_pass();let (_,texture)=r.e.textures.get(&target,true).unwrap();copy::copy_to_buffer(&r.cb,&texture.0,(0,0),[0,0,0,4,4,1],46,output,&r.pools,(0,0)).unwrap();r.commit_boundary(true);
                for pixel in std::slice::from_raw_parts(output as *const f32,64).chunks_exact(4) {assert_eq!(pixel,expected,"selector{selector}, table swap{swap}");}
            }
            // GPU event writes may also modify a CPU-inspected embedded table
            // or live descriptor UBO. A shader barrier alone can be deferred
            // only if the next host read still observes these writes.
            for (address,value,expected) in [(constant_cpu,targets[1],colors[1]),(cpu+512,2,colors[0])] {
                r.run(&[Cmd::SignalEvent{address,mode:0,value},Cmd::Barrier(0x10),Cmd::Draw(Draw{mode:4,index:None,first:0,count:3,base_vertex:0,base_instance:0,instances:1})],0);
                r.end_pass();let (_,texture)=r.e.textures.get(&target,true).unwrap();copy::copy_to_buffer(&r.cb,&texture.0,(0,0),[0,0,0,4,4,1],46,output,&r.pools,(0,0)).unwrap();r.commit_boundary(true);
                for pixel in std::slice::from_raw_parts(output as *const f32,64).chunks_exact(4) {assert_eq!(pixel,expected,"GPU-written C1/UBO at{address:#x}");}
            }
            let changed=r.e.shaders.stage(fs_cpu,control.as_ptr() as u64).unwrap();assert_ne!(first.hash,changed.hash);assert_eq!(first.base_hash,changed.base_hash);
            match r.e.shaders.function(&changed,0,0,0,true) {shaders::Lookup::Ready(c)=>assert!(c.source.is_none(),"C1 specialization must not be persisted with the legacy code-only prewarm identity"),_=>panic!("fragment function not ready")}
            for id in [0xc310,0xc311,0xc312,0xc31f] {STATE.lock().unwrap().pools.remove(&id);}for id in [0xc310,0xc311,0xc313] {gfx::forget_pool(id);}
        });
    }


    #[cfg(target_os="macos")]
    #[test]
    #[ignore="requires a real Metal GPU and NX_GRAPHICS_REGIONS_CAPTURE"]
    fn gpu_graphics_original_embedded_tables_compile_with_region_dispatch() {
        use super::*;
        let root=std::path::PathBuf::from(std::env::var("NX_GRAPHICS_REGIONS_CAPTURE").expect("set NX_GRAPHICS_REGIONS_CAPTURE to an original native shader capture"));
        let dir=root.join("work/msl_rt");let mut files=std::fs::read_dir(&dir).unwrap().map(|p|p.unwrap().path()).filter(|p|p.file_name().unwrap().to_string_lossy().ends_with(".constants.bin")).collect::<Vec<_>>();files.sort();
        let mut count=0;let mut regional=0;
        objc2::rc::autoreleasepool(|_| {
            for path in files {
                let name=path.file_name().unwrap().to_string_lossy();let hash=name.strip_suffix(".constants.bin").unwrap();let code=std::fs::read(dir.join(format!("{hash}.bin"))).unwrap();
                if code.get(..4)!=Some(&0x12345678u32.to_le_bytes()) {continue;}
                let constants=std::fs::read(&path).unwrap();let extra=if nxshader::can_force_early_fragment_tests(&code) {nxshader::FRAGMENT_EARLY_TESTS}else{0};
                let translated=if std::env::var_os("NX_GRAPHICS_STRUCTURED").is_some() {nxshader::translate_graphics_structured(&code,extra,0,0,Some(&constants))} else {nxshader::translate_graphics_regions(&code,extra,0,0,Some(&constants))}.unwrap();
                let plan=translated.msl.lines().find(|line|line.contains("// graphics regions:") || line.contains("// structured graphics:"));regional+=usize::from(plan.is_some());
                assert!(shaders::compile_config_constants(&code,false,extra,0,0,0xfc27e000+count,(0,0),Some(&constants)).is_some(),"original C1 shader{hash} must compile");
                eprintln!("[verify] original graphics C1 {hash} {}",plan.unwrap_or("original dispatcher retained"));count+=1;
            }
        });
        assert!(count>0 && regional>0,"captured original graphics and an applicable region plan are required");
        eprintln!("[verify] original embedded-table graphics compiled{count}, regional plans{regional}");
    }

    #[cfg(target_os="macos")]
    #[test]
    #[ignore="requires a real Metal GPU"]
    fn gpu_fragment_multi_entry_loop_preserves_different_entry_values() {
        use super::*;
        fn address(n:usize)->u32 {((n/3*4+1+n%3)*8) as u32}
        objc2::rc::autoreleasepool(|_| unsafe {
            crate::kernel::KERNEL.aspace.get_or_init(||crate::mem::AddressSpace::reserve().unwrap());let page=crate::mem::PAGE;let cpu=crate::kernel::aspace().alias()+0x6400000;
            crate::kernel::aspace().set(cpu,page*4,crate::mem::Block{state:crate::mem::state::NORMAL,perm:crate::mem::perm::RW,attr:0}).unwrap();
            let positions=[[-1f32,-1.0,0.5,1.0],[3.0,-1.0,0.5,1.0],[-1.0,3.0,0.5,1.0]];std::ptr::copy_nonoverlapping(positions.as_ptr().cast::<u8>(),cpu as *mut u8,size_of_val(&positions));
            let make_shader=|stage:u32,words:&[u64]| {let mut b=vec![0u8;0x80];b[..4].copy_from_slice(&0x12345678u32.to_le_bytes());b[0x30..0x34].copy_from_slice(&(stage<<10).to_le_bytes());if stage==5 {b[0x78..0x7c].copy_from_slice(&15u32.to_le_bytes());}b.extend(maxwell_program(words));b};
            let vs=make_shader(1,&[0xefd800000007ff00|(3<<47)|(0x80<<20),0xeff000000007ff00|(3<<47)|(0x70<<20)]);
            // Enter the same A->B->A cycle at A or B. B increments R5 and
            // exits at3; entering at B must leave R4=2, entering at A R4=3.
            let i2f=|d:u64,s:u64|0x5cb8000000070000u64|d|(s<<20)|(2<<10)|(2<<8);
            let mut words=vec![0x4c98078000070008|(3<<34),mov32(4,0),mov32(5,0),0x36538380001708ff,0xe24000000007000d,
                0x1c00000000170404,0x1c00000000170505,0x36538380003705ff,0xe24000000007000d,i2f(0,4),i2f(1,5),mov32(2,0.25f32.to_bits()),mov32(3,1f32.to_bits())];
            for (at,target) in [(4,6),(8,5)] {let delta=address(target) as i64-address(at) as i64-8;words[at]|=((delta as u64)&0xffffff)<<20;}
            let fs=make_shader(5,&words);assert!(nxshader::translate_graphics_regions(&fs,0,0,0,None).unwrap().msl.contains("// graphics regions:"));
            let mut vc=vec![0u8;0x880];let mut fc=vc.clone();vc[0x700..0x704].copy_from_slice(&(vs.len() as u32).to_le_bytes());fc[0x700..0x704].copy_from_slice(&(fs.len() as u32).to_le_bytes());
            for (id,addr,size) in [(0xc810,cpu,page),(0xc811,cpu+page,page),(0xc81e,vs.as_ptr() as u64,vs.len() as u64),(0xc81f,fs.as_ptr() as u64,fs.len() as u64)] {STATE.lock().unwrap().pools.insert(id,super::super::objects::MemoryPool{cpu:addr,size,flags:0x22});}
            let mut target=[0u8;0x68];target[8..16].copy_from_slice(&0xc811u64.to_le_bytes());for (at,value) in [(0x1c,1u32),(0x20,4),(0x24,4),(0x28,1),(0x2c,1),(0x30,46)] {target[at..at+4].copy_from_slice(&value.to_le_bytes());}
            let output=cpu+page*2;let pools=gfx::pool_memory(&[(0xc810,cpu,page),(0xc812,output,page)]).unwrap();let mut e=Engine::default();e.colors=vec![Some(Box::new(target))];e.st.attribs[0]=0x40|(46<<21);e.st.streams[0]=(16,0);e.st.vbufs[0]=(cpu,size_of_val(&positions) as u64);e.st.ubos[1][0]=(cpu+512,16);e.st.set_program(&[(vs.as_ptr() as u64,vc.as_ptr() as u64),(fs.as_ptr() as u64,fc.as_ptr() as u64)]);
            let cb=gfx::gpu().queue.0.commandBuffer().unwrap();let mut r=Replay{e:&mut e,cb,pass:None,heap:None,pools,passes:0,copies:HashMap::new(),fault:None,pool_table:None,resource_buffers:Default::default()};
            for selector in [0u32,1,0,1] {
                ((cpu+512) as *mut u32).write(selector);r.run(&[Cmd::ClearColor{index:0,rgba:[0.125;4]},Cmd::Draw(Draw{mode:4,index:None,first:0,count:3,base_vertex:0,base_instance:0,instances:1})],0);r.end_pass();
                let (_,tex)=r.e.textures.get(&target,true).unwrap();copy::copy_to_buffer(&r.cb,&tex.0,(0,0),[0,0,0,4,4,1],46,output,&r.pools,(0,0)).unwrap();r.commit_boundary(true);
                for pixel in std::slice::from_raw_parts(output as *const f32,64).chunks_exact(4) {assert_eq!(pixel,[if selector==0 {2f32}else{3.0},3.0,0.25,1.0],"multi-entry loop selector{selector}");}
            }
            for id in [0xc810,0xc811,0xc81e,0xc81f] {STATE.lock().unwrap().pools.remove(&id);}gfx::forget_pool(0xc810);gfx::forget_pool(0xc812);
        });
    }

    #[cfg(target_os="macos")]
    #[test]
    #[ignore="requires a real Metal GPU"]
    fn gpu_fragment_structured_cross_edge_diamond_preserves_each_path() {
        use super::*;
        fn address(n:usize)->u32 {((n/3*4+1+n%3)*8) as u32}
        objc2::rc::autoreleasepool(|_| unsafe {
            crate::kernel::KERNEL.aspace.get_or_init(||crate::mem::AddressSpace::reserve().unwrap());let page=crate::mem::PAGE;let cpu=crate::kernel::aspace().alias()+0x6400000;
            crate::kernel::aspace().set(cpu,page*4,crate::mem::Block{state:crate::mem::state::NORMAL,perm:crate::mem::perm::RW,attr:0}).unwrap();
            let positions=[[-1f32,-1.0,0.5,1.0],[3.0,-1.0,0.5,1.0],[-1.0,3.0,0.5,1.0]];std::ptr::copy_nonoverlapping(positions.as_ptr().cast::<u8>(),cpu as *mut u8,size_of_val(&positions));
            let make_shader=|stage:u32,words:&[u64]| {let mut b=vec![0u8;0x80];b[..4].copy_from_slice(&0x12345678u32.to_le_bytes());b[0x30..0x34].copy_from_slice(&(stage<<10).to_le_bytes());if stage==5 {b[0x78..0x7c].copy_from_slice(&15u32.to_le_bytes());}b.extend(maxwell_program(words));b};
            let vs=make_shader(1,&[0xefd800000007ff00|(3<<47)|(0x80<<20),0xeff000000007ff00|(3<<47)|(0x70<<20)]);
            // Selector0 bypasses the common add, selector1 reaches it through
            // the inner branch, selector2 reaches it from the outer branch.
            // Both the cross edge and the unreachable100 assignment matter.
            let i2f=|d:u64,s:u64|0x5cb8000000070000u64|d|(s<<20)|(2<<10)|(2<<8);
            let less=|n:u64|(0x36538380001708ffu64&!(0x7ffff<<20))|(n<<20);
            let mut words=vec![0x4c98078000070008|(3<<34),mov32(4,1),less(2),0xe24000000007000d,
                mov32(4,10),0xe24000000007000f,mov32(4,100),less(1),0xe24000000007000d,
                0x1c00000000370404,0xe24000000007000f,0x1c00000000570404,i2f(0,4),mov32(1,2f32.to_bits()),mov32(2,0.25f32.to_bits()),mov32(3,1f32.to_bits())];
            for (at,target) in [(3,7),(5,11),(8,12),(10,11)] {let delta=address(target) as i64-address(at) as i64-8;words[at]|=((delta as u64)&0xffffff)<<20;}
            let fs=make_shader(5,&words);assert!(nxshader::translate_graphics_structured(&fs,0,0,0,None).unwrap().msl.contains("// structured graphics:"));
            let mut vc=vec![0u8;0x880];let mut fc=vc.clone();vc[0x700..0x704].copy_from_slice(&(vs.len() as u32).to_le_bytes());fc[0x700..0x704].copy_from_slice(&(fs.len() as u32).to_le_bytes());
            for (id,addr,size) in [(0xc810,cpu,page),(0xc811,cpu+page,page),(0xc81e,vs.as_ptr() as u64,vs.len() as u64),(0xc81f,fs.as_ptr() as u64,fs.len() as u64)] {STATE.lock().unwrap().pools.insert(id,super::super::objects::MemoryPool{cpu:addr,size,flags:0x22});}
            let mut target=[0u8;0x68];target[8..16].copy_from_slice(&0xc811u64.to_le_bytes());for (at,value) in [(0x1c,1u32),(0x20,4),(0x24,4),(0x28,1),(0x2c,1),(0x30,46)] {target[at..at+4].copy_from_slice(&value.to_le_bytes());}
            let output=cpu+page*2;let pools=gfx::pool_memory(&[(0xc810,cpu,page),(0xc812,output,page)]).unwrap();let mut e=Engine::default();e.colors=vec![Some(Box::new(target))];e.st.attribs[0]=0x40|(46<<21);e.st.streams[0]=(16,0);e.st.vbufs[0]=(cpu,size_of_val(&positions) as u64);e.st.ubos[1][0]=(cpu+512,16);e.st.set_program(&[(vs.as_ptr() as u64,vc.as_ptr() as u64),(fs.as_ptr() as u64,fc.as_ptr() as u64)]);
            let cb=gfx::gpu().queue.0.commandBuffer().unwrap();let mut r=Replay{e:&mut e,cb,pass:None,heap:None,pools,passes:0,copies:HashMap::new(),fault:None,pool_table:None,resource_buffers:Default::default()};
            for selector in [0u32,1,2,1,0,2] {
                ((cpu+512) as *mut u32).write(selector);r.run(&[Cmd::ClearColor{index:0,rgba:[0.125;4]},Cmd::Draw(Draw{mode:4,index:None,first:0,count:3,base_vertex:0,base_instance:0,instances:1})],0);r.end_pass();
                let (_,tex)=r.e.textures.get(&target,true).unwrap();copy::copy_to_buffer(&r.cb,&tex.0,(0,0),[0,0,0,4,4,1],46,output,&r.pools,(0,0)).unwrap();r.commit_boundary(true);
                for pixel in std::slice::from_raw_parts(output as *const f32,64).chunks_exact(4) {assert_eq!(pixel,[match selector {0=>1f32,1=>9.0,_=>15.0},2.0,0.25,1.0],"cross-edge diamond selector{selector}");}
            }
            for id in [0xc810,0xc811,0xc81e,0xc81f] {STATE.lock().unwrap().pools.remove(&id);}gfx::forget_pool(0xc810);gfx::forget_pool(0xc812);
        });
    }

    #[cfg(target_os="macos")]
    #[test]
    #[ignore="requires a real Metal GPU"]
    fn gpu_fragment_secondary_blend_source_and_mrt_switch_preserve_pixels() {
        use super::*;
        objc2::rc::autoreleasepool(|_| unsafe {
            crate::kernel::KERNEL.aspace.get_or_init(||crate::mem::AddressSpace::reserve().unwrap());
            let page=crate::mem::PAGE;let cpu=crate::kernel::aspace().alias()+0x5000000;
            crate::kernel::aspace().set(cpu,page*5,crate::mem::Block{state:crate::mem::state::NORMAL,perm:crate::mem::perm::RW,attr:0}).unwrap();
            let positions=[[-1f32,-1.0,0.5,1.0],[3.0,-1.0,0.5,1.0],[-1.0,3.0,0.5,1.0]];
            std::ptr::copy_nonoverlapping(positions.as_ptr().cast::<u8>(),cpu as *mut u8,size_of_val(&positions));
            let make_shader=|stage:u32,words:&[u64]| {
                let mut b=vec![0u8;0x80];b[..4].copy_from_slice(&0x12345678u32.to_le_bytes());b[0x30..0x34].copy_from_slice(&(stage<<10).to_le_bytes());
                if stage==5 {b[0x78..0x7c].copy_from_slice(&255u32.to_le_bytes());}b.extend(maxwell_program(words));b
            };
            let vs=make_shader(1,&[0xefd800000007ff00|(3<<47)|(0x80<<20),0xeff000000007ff00|(3<<47)|(0x70<<20)]);
            let primary=[32.0f32/255.0,96.0/255.0,160.0/255.0,224.0/255.0];let secondary=[0.25f32,0.5,0.75,0.2];
            let words=primary.into_iter().chain(secondary).enumerate().map(|(i,v)|mov32(i as u64,v.to_bits())).collect::<Vec<_>>();
            let fs=make_shader(5,&words);
            let ordinary=nxshader::translate(&fs).unwrap();let dual=nxshader::translate_with(&fs,nxshader::FRAGMENT_DUAL_SOURCE,0,0).unwrap();
            assert_eq!(ordinary.interface.color_targets,3);assert_eq!(dual.interface.color_targets,1);
            assert!(!ordinary.msl.contains("discard_fragment()"),"an opaque shader must retain Metal's opaque-fragment optimization opportunity");
            let tested=nxshader::translate_with(&fs,nxshader::FRAGMENT_ALPHA_TEST,0,0).unwrap();assert!(tested.msl.contains("discard_fragment()"));
            let mut missing=fs.clone();missing[0x78..0x7c].copy_from_slice(&15u32.to_le_bytes());assert!(nxshader::translate_with(&missing,nxshader::FRAGMENT_DUAL_SOURCE,0,0).is_err());
            for (id,address,size) in [(0xc410,cpu,page),(0xc411,cpu+page,page),(0xc412,cpu+2*page,page),(0xc41e,vs.as_ptr() as u64,vs.len() as u64),(0xc41f,fs.as_ptr() as u64,fs.len() as u64)] {
                STATE.lock().unwrap().pools.insert(id,super::super::objects::MemoryPool{cpu:address,size,flags:0x22});
            }
            let builder=|pool:u64| {let mut b=[0u8;0x68];b[8..16].copy_from_slice(&pool.to_le_bytes());for (o,v) in [(0x1c,1u32),(0x20,4),(0x24,4),(0x28,1),(0x2c,1),(0x30,37)] {b[o..o+4].copy_from_slice(&v.to_le_bytes());}b};
            let target0=builder(0xc411);let target1=builder(0xc412);let output=cpu+3*page;
            let pools=gfx::pool_memory(&[(0xc410,cpu,page),(0xc413,output,page*2)]).unwrap();
            let mut e=Engine::default();e.st.attribs[0]=0x40|(46<<21);e.st.streams[0]=(16,0);e.st.vbufs[0]=(cpu,size_of_val(&positions) as u64);
            let mut vc=vec![0u8;0x880];let mut fc=vc.clone();vc[0x700..0x704].copy_from_slice(&(vs.len() as u32).to_le_bytes());fc[0x700..0x704].copy_from_slice(&(fs.len() as u32).to_le_bytes());
            e.st.set_program(&[(vs.as_ptr() as u64,vc.as_ptr() as u64),(fs.as_ptr() as u64,fc.as_ptr() as u64)]);
            e.st.blend[0]=((16u64|(17<<8)|(18<<16)|(19<<24))<<32)|(1<<24)|(1<<28);
            let cb=gfx::gpu().queue.0.commandBuffer().unwrap();let mut r=Replay{e:&mut e,cb,pass:None,heap:None,pools,passes:0,copies:HashMap::new(),fault:None,pool_table:None,resource_buffers:Default::default()};
            // Return to dual blending after both ordinary and MRT draws: shader
            // functions and pipeline caches must retain distinct interpretations.
            for (mode,expected) in [(1,[56u8,112,168,249]),(0,[32,96,160,224]),(2,[32,96,160,224]),(1,[56,112,168,249]),(3,[64,128,192,255]),(4,[32,96,160,224]),(5,[64,128,192,255]),(0,[32,96,160,224]),(5,[64,128,192,255]),(7,[32,96,160,224])] {
                r.e.colors=vec![Some(Box::new(target0))];if mode==2 {r.e.colors.push(Some(Box::new(target1)));}
                r.e.st.color_state=if mode==1 || mode==3 {1<<8} else if mode==4 || mode==5 {2<<24} else if mode==7 {8<<24} else {0};r.e.st.alpha_ref=if mode==4 {1.0} else {0.5};r.e.st.channel_mask=if mode==3 {0} else {u32::MAX};
                r.run(&[Cmd::ClearColor{index:0,rgba:[64.0/255.0,128.0/255.0,192.0/255.0,1.0]},Cmd::Draw(Draw{mode:4,index:None,first:0,count:3,base_vertex:0,base_instance:0,instances:1})],0);
                r.end_pass();
                for (i,target) in [target0,target1].iter().enumerate().take(if mode==2 {2} else {1}) {
                    let (_,texture)=r.e.textures.get(target,true).unwrap();copy::copy_to_buffer(&r.cb,&texture.0,(0,0),[0,0,0,4,4,1],37,output+(i as u64)*page,&r.pools,(0,0)).unwrap();
                }
                r.commit_boundary(true);
                for (i,want) in std::iter::once(expected).chain(if mode==2 {Some([64,128,191,51])} else {None}).enumerate() {
                    for pixel in std::slice::from_raw_parts((output+i as u64*page) as *const u8,64).chunks_exact(4) {
                        for c in 0..4 {assert!((pixel[c] as i32-want[c] as i32).abs()<=1,"mode{mode} target{i}: {pixel:?}, expected{want:?}");}
                    }
                }
            }
            for id in [0xc410,0xc411,0xc412,0xc41e,0xc41f] {STATE.lock().unwrap().pools.remove(&id);}for id in [0xc410,0xc413] {gfx::forget_pool(id);}
        });
    }

}
