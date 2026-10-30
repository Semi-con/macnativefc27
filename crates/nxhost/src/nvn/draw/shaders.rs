// Mush8 - shaders.rs
// Constantin Clerc (c22dev) - 0.1
use super::super::objects::STATE;
use crate::gfx::{self, Shared};
use nxshader::{Interface, TexKind};
use objc2_foundation::NSString;
use objc2_metal::*;
use super::super::fx::{FastMap, FastSet};
use std::collections::HashMap;
use std::sync::{Arc, LazyLock, Mutex};

pub type Function = objc2::runtime::ProtocolObject<dyn MTLFunction>;

#[derive(Clone, Copy, Debug)]
pub struct Stage {
    pub hash: u64,
    pub base_hash:u64,
    pub code: u64,
    pub len: usize,
    pub constants: u64,
    pub constants_len:u32,
    pub fragment: bool,
    pub local_size: Option<[u32; 3]>,
    pub local_memory:u32,
    pub shared_memory:u32,
    pub global_writes:bool,
    pub early_fragment_tests:bool,
}

pub struct Compiled {
    pub func: Shared<Function>,
    pub iface: Interface,
    pub grid_write_footprint:Option<nxshader::GridWriteFootprint>,
    pub descriptor_write_footprint:Option<Vec<nxshader::DescriptorWriteFootprint>>,
    pub write_window_footprint:Option<Vec<nxshader::WriteWindowFootprint>>,
    pub source: Option<(u64, [u32; 3])>,
    pub(super) argument_encoders:super::bindless::EncoderCache,
    pub(super) vertex_formats:Option<super::vertex_formats::Library>,
    pub(super) dynamic_vertex:Option<super::dynamic_vertex::DynamicVertex>,
}

type Key = (u64, u32, u32, u32);

pub enum Lookup {
    Ready(Arc<Compiled>),
    Pending,
    Failed,
}

#[derive(Clone, Copy)]
struct Recent {
    live: bool,
    code: u64,
    ctl: u64,
    quick: [u8; 0x60],
    stage: Option<Stage>,
}

impl Default for Recent {
    fn default() -> Self {
        Self { live: false, code: 0, ctl: 0, quick: [0; 0x60], stage: None }
    }
}

#[derive(Default)]
pub struct Shaders {
    stages: FastMap<(u64, u64), Option<Stage>>,
    funcs: FastMap<Key, Option<Arc<Compiled>>>,
    pending: FastSet<Key>,
    recent: [Recent; 4],
    recent_i: usize,
}

static DONE: LazyLock<Mutex<Vec<(Key, Option<Arc<Compiled>>)>>> = LazyLock::new(Default::default);

pub fn spawn_job(f: Box<dyn FnOnce() + Send>) {
    static QUEUE: LazyLock<Mutex<std::sync::mpsc::Sender<Box<dyn FnOnce() + Send>>>> = LazyLock::new(|| {
        let (tx, rx) = std::sync::mpsc::channel::<Box<dyn FnOnce() + Send>>();
        let rx = Arc::new(Mutex::new(rx));
        let n = std::thread::available_parallelism().map_or(4, |n| n.get()).saturating_sub(2).max(2);
        for i in 0..n {
            let rx = rx.clone();
            std::thread::Builder::new()
                .name(format!("shader-{i}"))
                .spawn(move || loop {
                    let job = rx.lock().unwrap().recv();
                    match job {
                        Ok(job) => objc2::rc::autoreleasepool(|_| job()),
                        Err(_) => return,
                    }
                })
                .unwrap();
        }
        Mutex::new(tx)
    });
    let _ = QUEUE.lock().unwrap().send(f);
}

pub fn sync_mode() -> bool {
    static S: LazyLock<bool> = LazyLock::new(|| std::env::var_os("NX_ASYNC_SHADERS").is_none() || std::env::var_os("NX_SYNC_SHADERS").is_some());
    *S
}

fn fnv1a(b: &[u8]) -> u64 {
    b.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &x| (h ^ x as u64).wrapping_mul(0x100_0000_01b3))
}

fn rd32(addr: u64) -> u32 {
    unsafe { (addr as *const u32).read_unaligned() }
}

#[derive(Default)]
struct CodeAddressCache {epoch:u64,next:usize,spans:[(u64,u64,u64);4]}
thread_local!{static CODE_ADDRESSES:std::cell::RefCell<CodeAddressCache>=std::cell::RefCell::new(CodeAddressCache::default());}
fn cpu_code_address(code:u64,bytes:u64)->Option<u64> {
    if !super::super::gpu_domains::enabled() {return Some(code);}
    let epoch=super::super::objects::VIRTUAL_MAPPING_EPOCH.load(std::sync::atomic::Ordering::Relaxed);
    CODE_ADDRESSES.with(|cache| {
        let mut cache=cache.borrow_mut();if cache.epoch!=epoch {cache.epoch=epoch;cache.next=0;cache.spans=[(0,0,0);4];}
        for &(base,cpu,len) in &cache.spans {if code>=base && code-base<len && bytes<=len-(code-base) {return cpu.checked_add(code-base);}}
        if let Some(span)=STATE.lock().unwrap().gpu_cpu_span(code,bytes) {
            let n=cache.next;cache.spans[n]=span;cache.next=(n+1)%4;return span.1.checked_add(code-span.0);
        }
        (!(super::super::gpu_domains::START..super::super::gpu_domains::END).contains(&code)).then_some(code)
    })
}

pub(super) fn stage_bit(code:u64)->Option<u32> {
    let code=cpu_code_address(code,0x90)?;
    if code==0 {return None;}
    if rd32(code)!=0x1234_5678 {return Some(0x20);}
    // SPH encoding differs from NVNshaderStage: fragment=5 in SPH but
    // fragment=1 in NVN, geometry=4 in SPH but geometry=2 in NVN.
    match (rd32(code+0x30)>>10)&15 {0|1=>Some(1),2=>Some(8),3=>Some(16),4=>Some(4),5=>Some(2),_=>None}
}

const EMBEDDED_ID:u64=0x434f4e535431;
fn writes_global_memory(code:&[u8])->bool {
    use nxshader::table::{decode,InstName,Op};
    for (i,word) in code.chunks_exact(8).enumerate() {
        if i%4==0 {continue;}
        let raw=u64::from_le_bytes(word.try_into().unwrap());let (name,op,_)=decode(raw);
        if matches!(op,Op::Stg|Op::Atom|Op::AtomCas|Op::Red) {return true;}
        let relative=(((raw>>20) as i32)<<8)>>8;
        if name==InstName::Bra && (-32..=0).contains(&(relative+8)) && raw>>16&15==7 && raw&31==15 {break;}
    }
    false
}
pub(super) fn embedded_all_enabled()->bool {
    static ENABLED:LazyLock<bool>=LazyLock::new(||std::env::var_os("NX_EMBEDDED_C1").is_some());
    *ENABLED
}
pub(super) fn vector_loads_enabled()->bool {
    static ENABLED:LazyLock<bool>=LazyLock::new(||std::env::var_os("NX_VECTOR_GMEM").is_some());
    *ENABLED
}
fn has_indirect_branch(code:&[u8])->bool {
    code.chunks_exact(8).enumerate().any(|(i,w)|i%4!=0 && u64::from_le_bytes(w.try_into().unwrap())>>52==0xe25)
}
fn embedded_constants_len(code:u64,ctl:u64)->u32 {
    let len=rd32(ctl+0x700);
    rd32(ctl+0x704).checked_sub(len).filter(|n|*n!=0 && *n<=65536 && *n%4==0).filter(|n| {
        STATE.lock().unwrap().pool_at(code).is_some_and(|(_,p)|code.checked_add(len as u64+*n as u64).is_some_and(|end|end<=p.cpu+p.size))
    }).unwrap_or(0)
}

fn compute_stage(code: u64, ctl: u64) -> Option<Stage> {
    let local_size = [0x748, 0x74c, 0x750].map(|o| rd32(ctl + o));
    if local_size.iter().any(|&n| n == 0 || n > 1024) {
        return None;
    }
    let len = rd32(ctl + 0x700) as usize;
    if !(8..=0x10_0000).contains(&len) {
        return None;
    }
    // SDK 22's 0x12345679 compute container places instructions at the
    // control's code offset (0x100 in all captured FC27 startup programs).
    let offset = if rd32(code)==0x1234_5679 {rd32(ctl+0x708) as usize} else {0};
    if offset>=len || offset%32!=0 {return None;}
    let blob = unsafe { std::slice::from_raw_parts((code+offset as u64) as *const u8, len-offset) };
    let local_memory=rd32(ctl+0x710);let shared_memory=rd32(ctl+0x754);
    let mut metadata=Vec::new();for n in local_size.into_iter().chain([local_memory,shared_memory]) {metadata.extend_from_slice(&n.to_le_bytes());}
    let base_hash=fnv1a(blob)^fnv1a(&metadata)^0xC0;
    let constants_len=if embedded_all_enabled() || has_indirect_branch(blob) {embedded_constants_len(code,ctl)} else {0};
    let hash=if constants_len!=0 {base_hash^fnv1a(unsafe {std::slice::from_raw_parts((code+len as u64) as *const u8,constants_len as usize)})^EMBEDDED_ID} else {base_hash};
    Some(Stage { hash,base_hash, code:code+offset as u64, len:len-offset, constants: code + len as u64,constants_len, fragment: false, local_size: Some(local_size),local_memory,shared_memory,global_writes:writes_global_memory(blob),early_fragment_tests:false })
}

impl Shaders {
    pub fn stage(&mut self, code: u64, ctl: u64) -> Option<Stage> {
        let code=if super::super::gpu_domains::enabled() {
            let len=rd32(ctl+0x700) as u64;let end=rd32(ctl+0x704) as u64;
            let bytes=if end>=len && end-len<=65536 {end}else{len};cpu_code_address(code,bytes.clamp(0x90,0x110000))?
        } else {code};
        // Compute instruction arrays may be recycled with an unchanged first
        // cache line/control pointer. Hash their full body and memory config.
        if rd32(code)!=0x1234_5678 {return compute_stage(code,ctl);}
        let quick = unsafe { std::slice::from_raw_parts((code + 0x30) as *const u8, 0x60) };
        let control_key=ctl^fnv1a(unsafe {std::slice::from_raw_parts((ctl+0x700) as *const u8,0x70)});
        for slot in &self.recent {
            if slot.live && slot.code == code && slot.ctl == control_key && slot.quick.as_slice() == quick {
                if let Some(s)=slot.stage {
                    if s.constants_len==0 || s.hash==s.base_hash^fnv1a(unsafe {std::slice::from_raw_parts(s.constants as *const u8,s.constants_len as usize)})^EMBEDDED_ID {return Some(s);}
                } else {return None;}
            }
        }
        let qh = fnv1a(quick);
        let available_constants=embedded_constants_len(code,ctl);
        let constant_hash=if available_constants!=0 {fnv1a(unsafe {std::slice::from_raw_parts((code+rd32(ctl+0x700) as u64) as *const u8,available_constants as usize)})} else {0};
        let key = (code, control_key ^ qh ^ constant_hash);
        let s = if let Some(s) = self.stages.get(&key) {
            *s
        } else {
            let s = (|| {
                if rd32(code) != 0x1234_5678 {
                    return compute_stage(code, ctl);
                }
                let mut len = rd32(ctl + 0x700) as u64;
                let pool_end = STATE.lock().unwrap().pool_at(code).map(|(_, p)| p.cpu + p.size)?;
                len = len.clamp(0x100, 0x10_0000).min(pool_end - code);
                let blob = unsafe { std::slice::from_raw_parts(code as *const u8, len as usize) };
                let w0 = u32::from_le_bytes(blob[0x30..0x34].try_into().unwrap());
                let constants_len=if embedded_all_enabled() || has_indirect_branch(&blob[0x80..]) {available_constants} else {0};
                let base_hash=fnv1a(&blob[0x30..]);
                Some(Stage {
                    hash:if constants_len!=0 {base_hash^constant_hash^EMBEDDED_ID} else {base_hash},
                    base_hash,
                    code,
                    len: len as usize,
                    constants: code + rd32(ctl + 0x700) as u64,
                    constants_len,
                    fragment: (w0 >> 10) & 0xF == 5,
                    local_size: None,
                    local_memory:0,shared_memory:0,
                    global_writes:writes_global_memory(&blob[0x80..]),
                    early_fragment_tests:nxshader::can_force_early_fragment_tests(blob),
                })
            })();
            self.stages.insert(key, s);
            s
        };
        let mut bytes = [0u8; 0x60];
        bytes.copy_from_slice(quick);
        self.recent[self.recent_i] = Recent { live: true, code, ctl:control_key, quick: bytes, stage: s };
        self.recent_i = (self.recent_i + 1) % self.recent.len();
        s
    }

    pub fn function(&mut self, s: &Stage, extra_outputs: u32, uint_targets: u32, sint_targets: u32, wait: bool) -> Lookup {
        let key = (s.hash, extra_outputs, uint_targets, sint_targets);
        loop {
            if !self.pending.is_empty() {
                for (k, r) in std::mem::take(&mut *DONE.lock().unwrap()) {
                    self.pending.remove(&k);
                    self.funcs.insert(k, r);
                }
            }
            if let Some(f) = self.funcs.get(&key) {
                return f.clone().map_or(Lookup::Failed, Lookup::Ready);
            }
            if !self.pending.contains(&key) {
                break;
            }
            if !wait {
                return Lookup::Pending;
            }
            crate::kernel::blocking(|| std::thread::sleep(std::time::Duration::from_millis(1)));
        }
        let blob = unsafe { std::slice::from_raw_parts(s.code as *const u8, s.len) };
        dump_requested(blob, s.hash, extra_outputs, uint_targets, sint_targets);
        let compute = s.local_size.is_some();
        if !compute && s.constants_len==0 && !(super::write_windows::enabled() && s.global_writes) {
            if let Some(hit) = nxshader::corpus::key(blob).and_then(|gk| guest_compiled(gk, extra_outputs, uint_targets, sint_targets)) {
                self.funcs.insert(key, Some(hit.clone()));
                return Lookup::Ready(hit);
            }
        }
        let blob = blob.to_vec();
        let constants=if s.constants_len!=0 {Some(unsafe {std::slice::from_raw_parts(s.constants as *const u8,s.constants_len as usize)}.to_vec())} else {None};
        let hash = s.hash;
        let memory=(s.local_memory,s.shared_memory);
        if wait || sync_mode() {
            let t0 = std::time::Instant::now();
            let r = crate::kernel::blocking(|| compile_config_constants(&blob, compute, extra_outputs, uint_targets, sint_targets, hash,memory,constants.as_deref()));
            if wait && !sync_mode() && t0.elapsed().as_millis() >= 5 {
                eprintln!("[shader] {hash:016x}: compiled while the render thread waited (new render target), {} ms", t0.elapsed().as_millis());
            }
            self.funcs.insert(key, r.clone());
            return r.map_or(Lookup::Failed, Lookup::Ready);
        }
        self.pending.insert(key);
        spawn_job(Box::new(move || {
            let r = compile_config_constants(&blob, compute, extra_outputs, uint_targets, sint_targets, hash,memory,constants.as_deref());
            DONE.lock().unwrap().push((key, r));
        }));
        Lookup::Pending
    }
}

fn dump_requested(blob: &[u8], hash: u64, extra_outputs: u32, uint_targets: u32, sint_targets: u32) {
    static WANT: LazyLock<Option<u64>> = LazyLock::new(|| std::env::var("NX_DUMP_SHADER").ok().and_then(|v| u64::from_str_radix(&v, 16).ok()));
    let Some(want) = *WANT else { return };
    if hash != want && nxshader::corpus::key(blob) != Some(want) {
        return;
    }
    let _ = std::fs::create_dir_all("work/msl_rt");
    let _ = std::fs::write(format!("work/msl_rt/{want:016x}.bin"), blob);
    match nxshader::translate_with(blob, extra_outputs, uint_targets, sint_targets) {
        Ok(t) => drop(std::fs::write(format!("work/msl_rt/{want:016x}.metal"), &t.msl)),
        Err(e) => eprintln!("[shader] {want:016x}: translation failed: {e}"),
    }
    eprintln!("[shader] {want:016x} dumped to work/msl_rt");
}

pub(super) fn compile(blob: &[u8], compute: bool, extra_outputs: u32, uint_targets: u32, sint_targets: u32, hash: u64) -> Option<Arc<Compiled>> {
    compile_config(blob,compute,extra_outputs,uint_targets,sint_targets,hash,(0,0))
}
pub(super) fn compile_config(blob:&[u8],compute:bool,extra_outputs:u32,uint_targets:u32,sint_targets:u32,hash:u64,memory:(u32,u32))->Option<Arc<Compiled>> {
    compile_config_constants(blob,compute,extra_outputs,uint_targets,sint_targets,hash,memory,None)
}
pub(super) fn compile_config_constants(blob:&[u8],compute:bool,extra_outputs:u32,uint_targets:u32,sint_targets:u32,hash:u64,memory:(u32,u32),constants:Option<&[u8]>)->Option<Arc<Compiled>> {
    let t0 = std::time::Instant::now();
    let r = compile_inner(blob, compute, extra_outputs, uint_targets, sint_targets, hash,memory,constants);
    let ms = t0.elapsed().as_millis();
    if ms > 500 {
        eprintln!("[shader] {hash:016x}: compiling took {ms} ms");
    }
    r
}

fn compile_inner(blob: &[u8], compute: bool, extra_outputs: u32, uint_targets: u32, sint_targets: u32, hash: u64,memory:(u32,u32),constants:Option<&[u8]>) -> Option<Arc<Compiled>> {
    if !compute && constants.is_none() && !(super::write_windows::enabled() && blob.get(0x80..).is_some_and(writes_global_memory)) {
        if let Some(hit) = nxshader::corpus::key(blob).and_then(|gk| guest_compiled(gk, extra_outputs, uint_targets, sint_targets)) {
            return Some(hit);
        }
    }
    if std::env::var_os("NX_DUMP_MSL").is_some() {
        let _ = std::fs::create_dir_all("work/msl_rt");
        let _ = std::fs::write(format!("work/msl_rt/{hash:016x}.bin"), blob);
        if let Some(bytes)=constants {let _=std::fs::write(format!("work/msl_rt/{hash:016x}.constants.bin"),bytes);}
    }
    let graphics_dispatch=if std::env::var_os("NX_GRAPHICS_STRUCTURED").is_some() {nxshader::GraphicsDispatch::Structured}
        else if std::env::var_os("NX_GRAPHICS_REGIONS").is_some() {nxshader::GraphicsDispatch::Regions}
        else if std::env::var_os("NX_GRAPHICS_FORWARD_DISPATCH").is_some() {nxshader::GraphicsDispatch::Forward}
        else {nxshader::GraphicsDispatch::Switch};
    let config=nxshader::TranslationConfig{compute_memory:compute.then_some(memory),graphics_dispatch,vector_global_loads:vector_loads_enabled()};
    let t=nxshader::translate_configured(blob,extra_outputs,uint_targets,sint_targets,constants,config);
    let mut t = match t {
        Ok(t) => t,
        Err(e) => {
            eprintln!("[shader] {hash:016x}: translation failed: {e}");
            return None;
        }
    };
    t.msl=graphics_memory_source(&t.msl,compute,std::env::var_os("NX_NO_GMEM_CACHE").is_some(),
        super::address_table::enabled(),page_cache_enabled()).into_owned();
    if std::env::var_os("NX_DUMP_MSL").is_some() {
        let _ = std::fs::create_dir_all("work/msl_rt");
        let _ = std::fs::write(format!("work/msl_rt/{hash:016x}_{extra_outputs:x}.metal"), cbuf_source(&t.msl).as_bytes());
    }
    // The legacy prewarm identity has no embedded-bank fingerprint. Do not
    // persist a specialized function under that code-only identity.
    let source = if compute || constants.is_some() { None } else { nxshader::corpus::key(blob).map(|k| (k, [extra_outputs, uint_targets, sint_targets])) };
    let (key, src, entry) = cache_name(&t);
    if let Some(func) = aot_function(key, &entry) {
        let vertex_formats=aot_library(key).and_then(|lib|super::vertex_formats::retain(lib,&entry,&func.0));
        return Some(Arc::new(Compiled { vertex_formats,func, iface: t.interface, grid_write_footprint:t.grid_write_footprint, descriptor_write_footprint:t.descriptor_write_footprint,write_window_footprint:t.write_window_footprint, source, argument_encoders:Default::default(),dynamic_vertex:None }));
    }
    let msl = format!("{}{src}", cbuf_source(nxshader::prelude()));
    if let Some(vertex)=super::dynamic_vertex::compile(&t,key,&msl) {
        return Some(Arc::new(Compiled{func:vertex.func,iface:t.interface,grid_write_footprint:t.grid_write_footprint,descriptor_write_footprint:t.descriptor_write_footprint,write_window_footprint:t.write_window_footprint,source:None,argument_encoders:Default::default(),vertex_formats:None,dynamic_vertex:Some(vertex.state)}));
    }
    let opts = MTLCompileOptions::new();
    opts.setMathMode(MTLMathMode::Safe);
    opts.setPreserveInvariance(true);
    let dev = &gfx::gpu().device.0;
    let lib = match dev.newLibraryWithSource_options_error(&NSString::from_str(&msl), Some(&opts)) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("[shader] {hash:016x}: Metal compile failed: {}", e.localizedDescription());
            let _ = std::fs::create_dir_all("work/msl_fail");
            let _ = std::fs::write(format!("work/msl_fail/{hash:016x}.metal"), &msl);
            return None;
        }
    };
    let func = super::vertex_formats::generic(&lib,&entry)?;
    let vertex_formats=super::vertex_formats::retain(Shared(lib),&entry,&func.0);
    Some(Arc::new(Compiled { vertex_formats,func, iface: t.interface, grid_write_footprint:t.grid_write_footprint, descriptor_write_footprint:t.descriptor_write_footprint,write_window_footprint:t.write_window_footprint, source, argument_encoders:Default::default(),dynamic_vertex:None }))
}

// The opt-in indexed-page cache needs the translator's per-invocation cache
// object and calls. Keep defaults unchanged; an explicit no-cache wins. The
// old unconditional paged-path removal left the page-cache helper unused.
pub(super) fn graphics_memory_source(source:&str,compute:bool,no_cache:bool,paged:bool,page_cache:bool)->std::borrow::Cow<'_,str>{
    if !compute&&(no_cache||(paged&&!page_cache)){
        std::borrow::Cow::Owned(source.replace("    GmemCache global_memory_cache = {0ul, 0ul, 0ul};\n","")
            .replace("gmem_gpu_cached(global_memory_cache,res,","gmem_gpu(res,")
            .replace("gmem_cached(global_memory_cache,res,","gmem(res,"))
    }else{std::borrow::Cow::Borrowed(source)}
}

#[cfg(test)]mod memory_source_tests{
    use super::*;
    #[test]fn indexed_graphics_cache_opt_in_keeps_real_call_path_and_explicit_disable_wins(){
        let body="    GmemCache global_memory_cache = {0ul, 0ul, 0ul};\n    uint a=*gmem_cached(global_memory_cache,res,p);ulong b=gmem_gpu_cached(global_memory_cache,res,q);\n";
        let old="    uint a=*gmem(res,p);ulong b=gmem_gpu(res,q);\n";
        assert_eq!(graphics_memory_source(body,false,false,true,false),old);
        assert_eq!(graphics_memory_source(body,false,false,true,true),body);
        assert_eq!(graphics_memory_source(body,false,true,true,true),old);
        assert_eq!(graphics_memory_source(body,true,true,true,false),body);
        assert_eq!(graphics_memory_source(body,false,false,false,false),body);
    }
}

// The page index is built only when every mapping is nonoverlapping. A
// cached fallback span and indexed subpage therefore resolve identically.
// Apply to host compilation only, leaving original generated MSL unchanged.
pub(super) fn page_cached_source(source:&str)->std::borrow::Cow<'_,str> {
    const OLD:&str=r#"#if NX_PAGED_GMEM
    ulong page = gmem_page(res, ga);
    if (page != 0ul) return page;
#endif
    if (ga - cache.base < cache.size) return cache.gpu + ga - cache.base;"#;
    const NEW:&str=r#"    if (ga - cache.base < cache.size) return cache.gpu + ga - cache.base;
#if NX_PAGED_GMEM
    ulong page = gmem_page(res, ga);
    if (page != 0ul) {
        cache.base = ga & ~4095ul;
        cache.size = 4096ul;
        cache.gpu = page - (ga & 4095ul);
        return page;
    }
#endif"#;
    if source.contains(OLD) {std::borrow::Cow::Owned(source.replace(OLD,NEW))}else{std::borrow::Cow::Borrowed(source)}
}

pub(super) fn flat_source(source:&str)->std::borrow::Cow<'_,str> {
    const HEADER:&str="static inline ulong gmem_page(constant Res& res, ulong ga) {\n";
    const FAST:&str="static inline ulong gmem_page(constant Res& res, ulong ga) {\n    if ((res.pool_root_base & 0x8000000000000000ul) != 0ul && ga - 0x8000000000ul < 0x8000000000ul) {\n        ulong page=res.pool_pages[(ga-0x8000000000ul)>>16ul];\n        if(page!=0ul) return page+(ga&65535ul);\n    }\n";
    if source.contains(HEADER) {std::borrow::Cow::Owned(source.replace(HEADER,FAST).replace("ulong root = (ga >> 24ul) - res.pool_root_base;","ulong root = (ga >> 24ul) - (res.pool_root_base & 0x7ffffffffffffffful);"))}else{std::borrow::Cow::Borrowed(source)}
}

pub(super) fn buffer_read_enabled()->bool {
    static ON:LazyLock<bool>=LazyLock::new(||std::env::var_os("NX_BUFFER_READ_FAST").is_some());
    super::address_table::enabled() && *ON
}

// A nonzero gmem_page proves a whole 4KiB mapping, so a pixel contained
// within that page needs only one translation. Keep the original partial,
// cross-page and fault paths below this opt-in prefix.
pub(super) fn buffer_read_source(source:&str)->std::borrow::Cow<'_,str> {
    const OLD:&str="    ulong address=ulong(meta.x)|(ulong(meta.y)<<32ul);address+=ulong(x)*bytes;\n    if((address&3ul)==0ul && (bytes&3u)==0u) {";
    const NEW:&str=r#"    ulong address=ulong(meta.x)|(ulong(meta.y)<<32ul);address+=ulong(x)*bytes;
    bool outside_heap=res.heap_chunk_size==0ul || (address>=res.heap_base ? address-res.heap_base>=res.heap_size : res.heap_base-address>=ulong(bytes));
    if(outside_heap && bytes<=16u && (address&4095ul)+ulong(bytes)<=4096ul) {
        ulong gpu=gmem_page(res,address);
        if(gpu!=0ul && ((address&3ul)!=0ul || (bytes&3u)!=0u || (gpu&3ul)==0ul)) {
            if((gpu&3ul)==0ul && (bytes&3u)==0u) {
                switch(bytes) {
                case 4u: result.x=*reinterpret_cast<const device uint*>(gpu); break;
                case 8u: result.xy=uint2(*reinterpret_cast<const device packed_uint2*>(gpu)); break;
                case 12u: result.xyz=uint3(*reinterpret_cast<const device packed_uint3*>(gpu)); break;
                case 16u: result=uint4(*reinterpret_cast<const device packed_uint4*>(gpu)); break;
                }
            } else {
                const device uchar* pixel=reinterpret_cast<const device uchar*>(gpu);
                for(uint i=0u;i<bytes;i++) result[i/4u]|=uint(pixel[i])<<((i%4u)*8u);
            }
            return result;
        }
    }
    if((address&3ul)==0ul && (bytes&3u)==0u) {"#;
    // This prefix also appears in buffer_write_raw: change only the read.
    let Some(start)=source.find("inline uint4 buffer_read_raw(") else{return std::borrow::Cow::Borrowed(source);};
    let Some(end)=source[start..].find("inline void buffer_write_raw(").map(|e|start+e) else{return std::borrow::Cow::Borrowed(source);};
    if !source[start..end].contains(OLD){return std::borrow::Cow::Borrowed(source);}
    let mut out=source[..start].to_owned();out.push_str(&source[start..end].replacen(OLD,NEW,1));out.push_str(&source[end..]);
    std::borrow::Cow::Owned(out)
}

pub(super) fn direct_buffer_source(source:&str)->std::borrow::Cow<'_,str> {
    const OLD:&str="    if(outside_heap && bytes<=16u && (address&4095ul)+ulong(bytes)<=4096ul) {\n        ulong gpu=gmem_page(res,address);";
    const NEW:&str="    bool direct_buffer=(meta.w&0x80000000u)!=0u;\n    if(direct_buffer || (outside_heap && bytes<=16u && (address&4095ul)+ulong(bytes)<=4096ul)) {\n        ulong gpu=direct_buffer?address:gmem_page(res,address);";
    if source.contains(OLD){std::borrow::Cow::Owned(source.replacen(OLD,NEW,1))}else{std::borrow::Cow::Borrowed(source)}
}

pub(super) fn page_cache_enabled()->bool {
    super::address_table::enabled() && std::env::var_os("NX_GMEM_PAGE_CACHE").is_some()
}

pub(super) fn aot_dir() -> &'static str {
    static DIR: LazyLock<String> = LazyLock::new(|| {let mut directory=format!("{}/native-resources-v22{}{}{}{}{}{}{}", std::env::var("NX_SHADER_CACHE").unwrap_or_else(|_| "work/shader_cache.noindex".into()),if std::env::var_os("NX_NO_GMEM_CACHE").is_some() {"-nogmem"} else {""},if std::env::var_os("NX_CONSTANT_CBUF").is_some() {"-constant"} else {""},if std::env::var_os("NX_GRAPHICS_FORWARD_DISPATCH").is_some() {"-forward"} else {""},if std::env::var_os("NX_GRAPHICS_REGIONS").is_some() {"-regions"} else {""},if super::address_table::enabled() {"-paged"} else {""},if std::env::var_os("NX_GRAPHICS_STRUCTURED").is_some() {"-structured"} else {""},if std::env::var_os("NX_APPROX_MUFU").is_some() {"-mufu"} else {""});if page_cache_enabled() {directory.push_str("-pagecache");}if super::vertex_formats::enabled(){directory.push_str("-vformats1");}if super::vertex_formats::buffer_enabled(){directory.push_str("-bformats1");}if buffer_read_enabled(){directory.push_str("-bufferread1");}if super::buffer_direct::enabled(){directory.push_str("-directbuf1");}if super::address_table::flat_enabled(){directory.push_str("-flat64k");}if vector_loads_enabled() {directory.push_str("-wide");}if embedded_all_enabled() {directory.push_str("-c1");}directory});
    &DIR
}

fn aot_sdk() -> String {
    std::env::var("NX_SHADER_SDK").unwrap_or_else(|_| "macosx".into())
}

pub(super) const IOS_MIN: &str = "26.0";
const GUEST_FILE: &str = "guest.tsv";
const AOT_BATCH: usize = 64;
const AOT_FLAGS: [&str; 3] = ["-fmetal-math-mode=safe", "-fpreserve-invariance", "-w"];

type Library = objc2::runtime::ProtocolObject<dyn MTLLibrary>;

pub(super) fn cbuf_source(source:&str)->std::borrow::Cow<'_,str> {
    static ENABLED:LazyLock<bool>=LazyLock::new(||std::env::var_os("NX_CONSTANT_CBUF").is_some());
    let mut source=std::borrow::Cow::Borrowed(source);
    if *ENABLED {source=std::borrow::Cow::Owned(source.replace("    const device uint* cb[18];","    constant uint* cb[18];"));}
    if super::address_table::enabled() {source=std::borrow::Cow::Owned(source.replace("#define NX_PAGED_GMEM 0","#define NX_PAGED_GMEM 1"));}
    if super::address_table::flat_enabled() {source=std::borrow::Cow::Owned(flat_source(&source).into_owned());}
    if buffer_read_enabled() {source=std::borrow::Cow::Owned(buffer_read_source(&source).into_owned());}
    if super::buffer_direct::enabled() {source=std::borrow::Cow::Owned(direct_buffer_source(&source).into_owned());}
    if super::vertex_formats::enabled(){source=std::borrow::Cow::Owned(super::vertex_formats::source(&source).into_owned());}
    if super::vertex_formats::buffer_enabled(){source=std::borrow::Cow::Owned(super::vertex_formats::buffer_source(&source).into_owned());}
    if page_cache_enabled() {source=std::borrow::Cow::Owned(page_cached_source(&source).into_owned());}
    if std::env::var_os("NX_APPROX_MUFU").is_some() {source=std::borrow::Cow::Owned(source.replace("#define NX_APPROX_MUFU 0","#define NX_APPROX_MUFU 1"));}
    source
}

fn cache_name(t: &nxshader::Translated) -> (u64, String, String) {
    // Prelude helpers and the resource ABI are part of the compiled program.
    let key = fnv1a(cbuf_source(&t.msl).as_bytes()) ^ if super::vertex_formats::enabled(){0x76666f726d617431}else{0} ^ if super::vertex_formats::buffer_enabled(){0x62666f726d617431}else{0};
    let (src, entry) = t.named(&format!("{key:016x}"));
    (key, cbuf_source(&src).into_owned(), entry)
}

struct Aot {
    index: HashMap<u64, String>,
    libs: Mutex<HashMap<String, Option<Shared<Library>>>>,
}

static AOT: LazyLock<Aot> = LazyLock::new(|| {
    let index = std::fs::read_to_string(format!("{}/index.tsv", aot_dir()))
        .unwrap_or_default()
        .lines()
        .filter_map(|l| {
            let (k, f) = l.split_once('\t')?;
            Some((u64::from_str_radix(k, 16).ok()?, f.to_string()))
        })
        .collect();
    Aot { index, libs: Mutex::new(HashMap::new()) }
});

struct GuestShader {
    msl: u64,
    entry: String,
    iface: Interface,
}

static GUEST: LazyLock<HashMap<(u64, u32, u32, u32), GuestShader>> = LazyLock::new(load_guest);

fn kind_id(k: TexKind) -> u32 {
    match k {
        TexKind::T2D => 0,
        TexKind::T2DArray => 1,
        TexKind::T3D => 2,
        TexKind::Cube => 3,
        TexKind::CubeArray => 4,
        TexKind::Depth2D => 5,
        TexKind::Depth2DArray => 6,
        TexKind::DepthCube => 7,
    }
}

fn kind_from(n: u32) -> Option<TexKind> {
    Some(match n {
        0 => TexKind::T2D,
        1 => TexKind::T2DArray,
        2 => TexKind::T3D,
        3 => TexKind::Cube,
        4 => TexKind::CubeArray,
        5 => TexKind::Depth2D,
        6 => TexKind::Depth2DArray,
        7 => TexKind::DepthCube,
        _ => return None,
    })
}

fn pairs(map: &str) -> impl Iterator<Item = (u32, u32)> + '_ {
    map.split(',').filter(|s| !s.is_empty()).filter_map(|s| {
        let (a, b) = s.split_once(':')?;
        Some((u32::from_str_radix(a, 16).ok()?, u32::from_str_radix(b, 16).ok()?))
    })
}

fn parse_guest(line: &str) -> Option<((u64, u32, u32, u32), GuestShader)> {
    let f: Vec<&str> = line.split('\t').collect();
    if !matches!(f.len(),12|14|15|16|17|18|19) {
        return None;
    }
    let hex = |s: &str| u64::from_str_radix(s, 16).ok();
    let u = |s: &str| u32::from_str_radix(s, 16).ok();
    let mut iface = Interface::default();
    for (b, k) in pairs(f[10]) {
        iface.textures.insert(b, kind_from(k)?);
    }
    for (b, slot) in pairs(f[11]) {
        iface.samplers.insert(b, slot);
    }
    if f.len()>=14 {
        for (b,slot) in pairs(f[12]) {iface.texture_slots.insert(b,slot);}
        for (b,value) in pairs(f[13]) {iface.images.insert(b,nxshader::ImageBinding{kind:kind_from(value&255)?,raw_bytes:value>>8});}
    }
    let gk = hex(f[0])?;
    let extra = u(f[1])?;
    let uints = u(f[2])?;
    let sints = u(f[3])?;
    if iface.images.keys().any(|&b|b>=8) {return None;}
    let legacy_image_writes=iface.images.keys().fold(0,|mask,&b|mask|(1u32<<b));
    let g = GuestShader {
        msl: hex(f[4])?,
        entry: f[5].to_string(),
        iface: Interface {
            textures: iface.textures,
            samplers: iface.samplers,
            texture_slots:iface.texture_slots,
            images:iface.images,
            atomic_images:if f.len()>=16 {u(f[15])?} else {0},
            // Older manifests did not distinguish image readers/writers.
            // Preserve their conservative write classification.
            image_writes:if f.len()>=19 {u(f[18])?}else{legacy_image_writes},
            buffer_metadata:f.len()>=18 && u(f[17])?!=0,
            argument_textures:f.len()>=17 && u(f[16])?!=0,
            dynamic_textures:if f.len()>=15 {u(f[14])?} else {0},
            attributes: u(f[6])?,
            outputs: u(f[7])?,
            color_targets: u(f[8])?,
            writes_depth: f[9] == "1",
        },
    };
    (g.entry.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')).then(|| ((gk, extra, uints, sints), g))
}

fn load_guest() -> HashMap<(u64, u32, u32, u32), GuestShader> {
    std::fs::read_to_string(format!("{}/{GUEST_FILE}", aot_dir()))
        .unwrap_or_default()
        .lines()
        .filter_map(parse_guest)
        .collect()
}

fn guest_line(gk: u64, [extra, uints, sints]: [u32; 3], msl: u64, entry: &str, iface: &Interface) -> String {
    let list = |items: Vec<(u32, u32)>| items.into_iter().map(|(a, b)| format!("{a:x}:{b:x}")).collect::<Vec<_>>().join(",");
    let textures = list(iface.textures.iter().map(|(&b, &k)| (b, kind_id(k))).collect());
    let samplers = list(iface.samplers.iter().map(|(&b, &s)| (b, s)).collect());
    let slots=list(iface.texture_slots.iter().map(|(&b,&s)|(b,s)).collect());
    let images=list(iface.images.iter().map(|(&b,image)|(b,kind_id(image.kind)|(image.raw_bytes<<8))).collect());
    format!(
        "{gk:016x}\t{extra:x}\t{uints:x}\t{sints:x}\t{msl:016x}\t{entry}\t{:x}\t{:x}\t{:x}\t{}\t{textures}\t{samplers}\t{slots}\t{images}\t{:x}\t{:x}\t{:x}\t{:x}\t{:x}\n",
        iface.attributes,
        iface.outputs,
        iface.color_targets,
        iface.writes_depth as u8,
        iface.dynamic_textures,
        iface.atomic_images,
        iface.argument_textures as u8,
        iface.buffer_metadata as u8,
        iface.image_writes
    )
}

#[cfg(test)]
mod resource_interface_tests {
    use super::*;
    #[test]
    fn code_address_cache_follows_mapping_epoch_and_drops_finalized_storage() {
        if !super::super::super::gpu_domains::enabled(){return;}
        let first=Box::new([0u8;256]);let second=Box::new([0u8;256]);let mut object=Box::new([0u64;32]);let pool=object.as_mut_ptr() as u64;
        let call=|name:&str,args:[u64;8]|super::super::super::objects::host_impl(name).unwrap()(&crate::stubs::Args(args,[0;8]));
        let initialize=|cpu:u64|{let b=[0x12u64<<32,0,cpu,256];assert_eq!(call("nvnMemoryPoolInitialize",[pool,b.as_ptr() as u64,0,0,0,0,0,0]),1);call("nvnMemoryPoolGetBufferAddress",[pool,0,0,0,0,0,0,0])};
        let va=initialize(first.as_ptr() as u64);assert_ne!(va,first.as_ptr() as u64);
        for _ in 0..4{assert_eq!(cpu_code_address(va+16,128),Some(first.as_ptr() as u64+16));}assert_eq!(cpu_code_address(va+240,32),None);
        let new_va=initialize(second.as_ptr() as u64);assert_eq!(cpu_code_address(new_va+16,128),Some(second.as_ptr() as u64+16));
        call("nvnMemoryPoolFinalize",[pool,0,0,0,0,0,0,0]);assert_eq!(cpu_code_address(new_va+16,128),None);
    }
    #[test]
    fn branchless_embedded_banks_refresh_compute_and_graphics_recent_identity() {
        for compute in [false,true] {
            let mut code=vec![0u8;0x400];code[..4].copy_from_slice(&if compute {0x12345679u32}else{0x12345678u32}.to_le_bytes());code[0x30..0x34].copy_from_slice(&(5u32<<10).to_le_bytes());
            let mut control=vec![0u8;0x880];for (offset,value) in [(0x700,0x300u32),(0x704,0x400),(0x708,0x100),(0x748,1),(0x74c,1),(0x750,1)] {control[offset..offset+4].copy_from_slice(&value.to_le_bytes());}
            let pool=0xcf270040+u64::from(compute);STATE.lock().unwrap().pools.insert(pool,crate::nvn::objects::MemoryPool{cpu:code.as_ptr() as u64,size:code.len() as u64,flags:0x22});
            let mut shaders=Shaders::default();let first=shaders.stage(code.as_ptr() as u64,control.as_ptr() as u64).unwrap();
            assert_eq!(first.constants_len,if embedded_all_enabled(){256}else{0});
            code[0x300..0x304].copy_from_slice(&0x1234abcdu32.to_le_bytes());let changed=shaders.stage(code.as_ptr() as u64,control.as_ptr() as u64).unwrap();
            assert_eq!(first.base_hash,changed.base_hash);assert_eq!(first.hash!=changed.hash,embedded_all_enabled());
            assert_eq!(changed.hash,shaders.stage(code.as_ptr() as u64,control.as_ptr() as u64).unwrap().hash);
            STATE.lock().unwrap().pools.remove(&pool);
        }
    }
    #[test]
    fn sdk22_compute_container_keeps_code_and_constants_distinct() {
        let mut code=vec![0u8;0x300];code[..4].copy_from_slice(&0x12345679u32.to_le_bytes());
        let mut control=vec![0u8;0x880];
        for (offset,value) in [(0x700,0x300u32),(0x708,0x100),(0x748,8),(0x74c,8),(0x750,1)] {control[offset..offset+4].copy_from_slice(&value.to_le_bytes());}
        let s=compute_stage(code.as_ptr() as u64,control.as_ptr() as u64).unwrap();
        assert_eq!(s.code,code.as_ptr() as u64+0x100);assert_eq!(s.len,0x200);assert_eq!(s.constants,code.as_ptr() as u64+0x300);assert_eq!(s.local_size,Some([8,8,1]));
        control[0x708..0x70c].copy_from_slice(&0x300u32.to_le_bytes());
        assert!(compute_stage(code.as_ptr() as u64,control.as_ptr() as u64).is_none());
    }
    #[test]
    fn embedded_compute_constants_participate_in_runtime_shader_and_pipeline_identity() {
        let mut code=vec![0u8;0x400];code[..4].copy_from_slice(&0x12345679u32.to_le_bytes());code[0x108..0x110].copy_from_slice(&0xe25000000007040fu64.to_le_bytes());
        let mut control=vec![0u8;0x880];
        for (offset,value) in [(0x700,0x300u32),(0x704,0x400),(0x708,0x100),(0x748,64),(0x74c,1),(0x750,1)] {control[offset..offset+4].copy_from_slice(&value.to_le_bytes());}
        let pool=0xcf270001;STATE.lock().unwrap().pools.insert(pool,crate::nvn::objects::MemoryPool{cpu:code.as_ptr() as u64,size:code.len() as u64,flags:0x22});
        let first=compute_stage(code.as_ptr() as u64,control.as_ptr() as u64).unwrap();assert_eq!(first.constants_len,256);
        code[0x300..0x304].copy_from_slice(&0x80u32.to_le_bytes());let second=compute_stage(code.as_ptr() as u64,control.as_ptr() as u64).unwrap();
        assert_eq!(first.base_hash,second.base_hash);assert_ne!(first.hash,second.hash);
        control[0x704..0x708].copy_from_slice(&0x10000u32.to_le_bytes());let invalid=compute_stage(code.as_ptr() as u64,control.as_ptr() as u64).unwrap();assert_eq!(invalid.constants_len,0);assert_eq!(invalid.hash,invalid.base_hash);
        STATE.lock().unwrap().pools.remove(&pool);
    }
    #[test]
    fn graphics_recent_cache_rechecks_embedded_branch_table_identity() {
        let mut code=vec![0u8;0x400];code[..4].copy_from_slice(&0x12345678u32.to_le_bytes());code[0x30..0x34].copy_from_slice(&(5u32<<10).to_le_bytes());code[0x88..0x90].copy_from_slice(&0xe25000000007040fu64.to_le_bytes());
        let mut control=vec![0u8;0x880];for (offset,value) in [(0x700,0x300u32),(0x704,0x400)] {control[offset..offset+4].copy_from_slice(&value.to_le_bytes());}
        let pool=0xcf270002;STATE.lock().unwrap().pools.insert(pool,crate::nvn::objects::MemoryPool{cpu:code.as_ptr() as u64,size:code.len() as u64,flags:0x22});
        let mut shaders=Shaders::default();let first=shaders.stage(code.as_ptr() as u64,control.as_ptr() as u64).unwrap();assert_eq!(first.constants_len,256);assert!(first.fragment);
        assert_eq!(first.hash,shaders.stage(code.as_ptr() as u64,control.as_ptr() as u64).unwrap().hash);
        code[0x300..0x304].copy_from_slice(&0x80u32.to_le_bytes());let changed=shaders.stage(code.as_ptr() as u64,control.as_ptr() as u64).unwrap();
        assert_eq!(first.base_hash,changed.base_hash);assert_ne!(first.hash,changed.hash);assert_eq!(shaders.stages.len(),2);
        assert_eq!(changed.hash,shaders.stage(code.as_ptr() as u64,control.as_ptr() as u64).unwrap().hash);
        STATE.lock().unwrap().pools.remove(&pool);
    }
    #[test]
    fn cache_metadata_roundtrips_separate_texture_slots_and_raw_images() {
        let mut iface=Interface::default();
        iface.textures.insert(0x1001,TexKind::T2D);iface.samplers.insert(0x1001,0);iface.texture_slots.insert(0x1001,0);
        iface.images.insert(2,nxshader::ImageBinding{kind:TexKind::T2DArray,raw_bytes:8});
        iface.dynamic_textures=3;
        iface.atomic_images=4;iface.image_writes=4;
        iface.argument_textures=true;
        iface.buffer_metadata=true;
        let line=guest_line(11,[0,0,0],12,"cmain",&iface);
        let (_,shader)=parse_guest(line.trim_end()).unwrap();
        assert_eq!(shader.iface.textures,iface.textures);assert_eq!(shader.iface.texture_slots,iface.texture_slots);assert_eq!(shader.iface.images,iface.images);
        assert_eq!(shader.iface.dynamic_textures,iface.dynamic_textures);
        assert_eq!(shader.iface.atomic_images,iface.atomic_images);assert_eq!(shader.iface.image_writes,iface.image_writes);
        assert_eq!(shader.iface.argument_textures,iface.argument_textures);
        assert_eq!(shader.iface.buffer_metadata,iface.buffer_metadata);
        let fields=line.trim_end().split('\t').collect::<Vec<_>>();
        for count in [12,14,15,16] {
            let (_,old)=parse_guest(&fields[..count].join("\t")).unwrap();
            assert!(!old.iface.argument_textures);
            assert!(!old.iface.buffer_metadata);
            assert_eq!(old.iface.atomic_images,if count>=16 {4} else {0});assert_eq!(old.iface.image_writes,if count>=14 {4}else{0});
        }
    }
}

pub(super) fn aot_compiled((gk, [extra, uints, sints]): (u64, [u32; 3])) -> Option<Arc<Compiled>> {
    guest_compiled(gk, extra, uints, sints)
}

fn guest_compiled(gk: u64, extra: u32, uints: u32, sints: u32) -> Option<Arc<Compiled>> {
    let g = GUEST.get(&(gk, extra, uints, sints))?;
    // The code-only guest index has no dynamic argument ABI version. Derive
    // the full source identity for these few shaders so a legacy4096 library
    // cannot be paired with the wide host encoder. Source-keyed AOT still
    // avoids recompilation after translation; ordinary guest hits stay valid.
    if g.iface.dynamic_textures!=0 {return None;}
    let func = aot_function(g.msl, &g.entry)?;
    let vertex_formats=aot_library(g.msl).and_then(|lib|super::vertex_formats::retain(lib,&g.entry,&func.0));
    Some(Arc::new(Compiled { vertex_formats,func, iface: g.iface.clone(), grid_write_footprint:None,descriptor_write_footprint:None,write_window_footprint:None, source: Some((gk, [extra, uints, sints])), argument_encoders:Default::default(),dynamic_vertex:None }))
}

fn aot_function(key: u64, entry: &str) -> Option<Shared<Function>> {
    super::vertex_formats::generic(&aot_library(key)?.0,entry)
}

fn aot_library(key:u64)->Option<Shared<Library>> {
    let file = AOT.index.get(&key)?;
    let lib = AOT
        .libs
        .lock()
        .unwrap()
        .entry(file.clone())
        .or_insert_with(|| {
            let url = objc2_foundation::NSURL::fileURLWithPath(&NSString::from_str(&format!("{}/{file}", aot_dir())));
            gfx::gpu().device.0.newLibraryWithURL_error(&url).ok().map(Shared)
        })
        .clone()?;
    Some(lib)
}

pub fn precompile(romfs: nxfs::storage::DynStorage, modules: &[std::path::PathBuf]) -> anyhow::Result<()> {
    use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
    let t0 = std::time::Instant::now();
    let fs = nxfs::romfs::RomFs::open(romfs)?;
    let mut found = std::collections::BTreeMap::new();
    for f in fs.files() {
        if let Ok(data) = fs.storage().read_vec(f.offset, f.size as usize) {
            nxshader::corpus::scan(&data, &mut found);
        }
    }
    for m in modules {
        if let Ok(data) = std::fs::read(m) {
            nxshader::corpus::scan(&data, &mut found);
        }
    }
    let mut blobs: Vec<(u64, [u32; 3], &Vec<u8>)> = found.iter().map(|(k, v)| (*k, [0; 3], v)).collect();
    let variants = super::state::recorded_variants();
    blobs.extend(variants.iter().filter_map(|&(k, p)| Some((k, p, found.get(&k)?))));
    eprintln!("[shader] {} programs found in {:.1} s ({} recorded variants)", found.len(), t0.elapsed().as_secs_f32(), variants.len());
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());

    let sources = Mutex::new(std::collections::BTreeMap::new());
    let guest = Mutex::new(Vec::new());
    let next = AtomicUsize::new(0);
    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| loop {
                let Some(&(gk, p, blob)) = blobs.get(next.fetch_add(1, Relaxed)) else { return };
                let t = if p == [0; 3] { nxshader::translate(blob) } else { nxshader::translate_with(blob, p[0], p[1], p[2]) };
                match t {
                    Ok(t) => {
                        let (key, src, entry) = cache_name(&t);
                        guest.lock().unwrap().push(guest_line(gk, p, key, &entry, &t.interface));
                        sources.lock().unwrap().insert(key, src);
                    }
                    Err(e) => eprintln!("[shader] {:016x}: translation failed: {e}", fnv1a(&blob[0x30..])),
                }
            });
        }
    });
    let sources: Vec<(u64, String)> = sources.into_inner().unwrap().into_iter().collect();
    let mut guest = guest.into_inner().unwrap();
    guest.sort();
    guest.dedup();

    let have = |k: &u64| AOT.index.get(k).filter(|f| std::path::Path::new(&format!("{}/{f}", aot_dir())).exists());
    let kept: Vec<String> = sources.iter().filter_map(|(k, _)| have(k).map(|f| format!("{k:016x}\t{f}\n"))).collect();
    let sources: Vec<(u64, String)> = sources.into_iter().filter(|(k, _)| have(k).is_none()).collect();
    eprintln!("[shader] {} shaders already compiled, {} to compile", kept.len(), sources.len());
    std::fs::create_dir_all(aot_dir())?;
    std::fs::write(format!("{}/{GUEST_FILE}", aot_dir()), guest.concat())?;
    let used: std::collections::HashSet<&str> = kept.iter().filter_map(|l| l.trim_end().split('\t').nth(1)).collect();
    let mut first = 0;
    for e in std::fs::read_dir(aot_dir())?.flatten() {
        let name = e.file_name().to_string_lossy().into_owned();
        if let Some(n) = name.strip_suffix(".metallib") {
            if used.contains(name.as_str()) {
                first = first.max(n.parse::<usize>().map_or(0, |n| n + 1));
            } else {
                let _ = std::fs::remove_file(e.path());
            }
        }
    }

    let sdk = aot_sdk();
    let batches: Vec<_> = sources.chunks(AOT_BATCH).collect();
    let index = Mutex::new(kept.concat());
    let (next, done) = (AtomicUsize::new(0), AtomicUsize::new(0));
    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| loop {
                let i = next.fetch_add(1, Relaxed);
                let Some(batch) = batches.get(i) else { return };
                let n = first + i;
                let (metal, lib) = (format!("{}/{n:04}.metal", aot_dir()), format!("{n:04}.metallib"));
                let mut src = cbuf_source(nxshader::prelude()).into_owned();
                for (_, s) in batch.iter() {
                    src.push_str(s);
                }
                let ok = std::fs::write(&metal, &src).is_ok()
                    && std::process::Command::new("xcrun")
                        .args(["-sdk", &sdk, "metal"])
                        .args(AOT_FLAGS)
                        .args((sdk == "iphoneos").then(|| format!("-mios-version-min={IOS_MIN}")))
                        .args([metal.as_str(), "-o", &format!("{}/{lib}", aot_dir())])
                        .status()
                        .is_ok_and(|st| st.success());
                let _ = std::fs::remove_file(&metal);
                if ok {
                    let mut idx = index.lock().unwrap();
                    for (key, _) in batch.iter() {
                        idx.push_str(&format!("{key:016x}\t{lib}\n"));
                    }
                } else {
                    eprintln!("[shader] library {lib} failed to compile: its shaders will be compiled at runtime");
                }
                let n = done.fetch_add(batch.len(), Relaxed) + batch.len();
                if n / 1000 != (n - batch.len()) / 1000 {
                    eprintln!("[shader] {n}/{}", sources.len());
                }
            });
        }
    });
    let index = index.into_inner().unwrap();
    if index.is_empty() && !sources.is_empty() {
        eprintln!(
            "[shader] warning: no shader was precompiled, so the game compiles them while it plays (black or missing \
             graphics for a moment in new scenes). Install the Metal Toolchain with `xcodebuild -downloadComponent \
             MetalToolchain`, then Prepare again"
        );
    } else {
        eprintln!("[shader] precompiled {} shaders in {:.0} s", sources.len(), t0.elapsed().as_secs_f32());
    }
    std::fs::write(format!("{}/index.tsv", aot_dir()), index)?;
    if sdk == "iphoneos" || std::env::var_os("NX_PIPELINE_ARCHIVE").is_none() {
        return Ok(());
    }
    super::state::precompile_pipelines(&found)
}

pub fn integer_kind(f: MTLPixelFormat) -> u32 {
    use MTLPixelFormat as P;
    match f {
        P::R8Uint | P::R16Uint | P::R32Uint | P::RG8Uint | P::RG16Uint | P::RG32Uint | P::RGBA8Uint | P::RGBA16Uint | P::RGBA32Uint | P::RGB10A2Uint => 1,
        P::R8Sint | P::R16Sint | P::R32Sint | P::RG8Sint | P::RG16Sint | P::RG32Sint | P::RGBA8Sint | P::RGBA16Sint | P::RGBA32Sint => 2,
        _ => 0,
    }
}
