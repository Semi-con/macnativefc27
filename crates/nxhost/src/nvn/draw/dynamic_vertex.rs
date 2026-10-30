// Optional device-local cache of vertex-program bodies. Fragment/compute
// compilation stays unchanged. No translated instructions are removed.
use crate::gfx::{self,Shared};
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSArray,NSString,NSURL,NSProcessInfo};
use objc2_metal::*;
use std::{path::{Path,PathBuf},sync::{LazyLock,OnceLock,atomic::{AtomicU64,Ordering::Relaxed}}};

type Function=ProtocolObject<dyn MTLFunction>;
type Library=ProtocolObject<dyn MTLDynamicLibrary>;

pub(super) struct DynamicVertex {
    pub(super) library:Shared<Library>,
    normal_source:String,
    entry:String,
    fallback:OnceLock<Option<Shared<Function>>>,
}
pub(super) struct Product {
    pub(super) func:Shared<Function>,
    pub(super) state:DynamicVertex,
    disk_hit:bool,
    path:PathBuf,
}

fn options()->objc2::rc::Retained<MTLCompileOptions> {
    let o=MTLCompileOptions::new();o.setMathMode(MTLMathMode::Safe);o.setPreserveInvariance(true);o
}
impl DynamicVertex {
    pub(super) fn fallback_function(&self)->Option<Shared<Function>> {
        self.fallback.get_or_init(|| {
            let lib=gfx::gpu().device.0.newLibraryWithSource_options_error(&NSString::from_str(&self.normal_source),Some(&options())).ok()?;
            lib.newFunctionWithName(&NSString::from_str(&self.entry)).map(Shared)
        }).clone()
    }
}

fn input(t:&nxshader::Translated,key:u64)->Option<(String,String,String)> {
    let s=t.dynamic_vertex_sources(&format!("{key:016x}"))?;
    let library=format!("{}{}",super::shaders::cbuf_source(nxshader::prelude()),super::shaders::cbuf_source(&s.library));
    let wrapper=super::shaders::cbuf_source(&s.wrapper).into_owned();
    Some((library,wrapper,s.entry))
}
fn file_key(library:&str,wrapper:&str)->String {
    let mut bytes=b"dynamic-vertex-v1;math-safe;invariance=true\0".to_vec();
    bytes.extend_from_slice(&(library.len() as u64).to_le_bytes());bytes.extend_from_slice(library.as_bytes());bytes.extend_from_slice(wrapper.as_bytes());
    nxfs::crypto::sha256(&bytes)
}
fn url(path:&Path)->objc2::rc::Retained<NSURL> {NSURL::fileURLWithPath(&NSString::from_str(&path.to_string_lossy()))}

fn store(path:&Path,library:&Library)->bool {
    static NEXT:AtomicU64=AtomicU64::new(0);
    let temporary=path.with_extension(format!("{}.{}.tmp.metallib",std::process::id(),NEXT.fetch_add(1,Relaxed)));
    if std::fs::symlink_metadata(&temporary).is_ok_and(|m|!m.file_type().is_file()) {return false;}
    let result=library.serializeToURL_error(&url(&temporary)).is_ok()
        && std::fs::metadata(&temporary).is_ok_and(|m|m.len()<=64*1024*1024)
        && std::fs::rename(&temporary,path).is_ok();
    if !result {let _=std::fs::remove_file(&temporary);}
    result
}

fn compile_at(t:&nxshader::Translated,key:u64,normal_source:&str,directory:&Path)->Option<Product> {
    if !gfx::gpu().device.0.supportsRenderDynamicLibraries() {return None;}
    let (library_source,wrapper,entry)=input(t,key)?;
    if library_source.len()>32*1024*1024 {return None;}
    std::fs::create_dir_all(directory).ok()?;
    let path=directory.join(format!("{}.metallib",file_key(&library_source,&wrapper)));
    let mut disk_hit=false;
    let existing=if path.is_file() {
        gfx::gpu().device.0.newDynamicLibraryWithURL_error(&url(&path)).ok().filter(|library| {
            // Runtime-compiled render libraries require the original absolute
            // installName on this device. A relocated cache must be rebuilt.
            library.installName().to_string()==path.to_string_lossy()
        })
    } else {None};
    let library=if let Some(library)=existing {disk_hit=true;library} else {
        let used=std::fs::read_dir(directory).ok()?.flatten().filter_map(|e|e.metadata().ok()).filter(|m|m.is_file()).map(|m|m.len()).sum::<u64>();
        if used>=512*1024*1024 {return None;}
        let o=options();o.setLibraryType(MTLLibraryType::Dynamic);o.setInstallName(Some(&NSString::from_str(&path.to_string_lossy())));
        let lib=match gfx::gpu().device.0.newLibraryWithSource_options_error(&NSString::from_str(&library_source),Some(&o)) {Ok(v)=>v,Err(e)=>{eprintln!("[vertex-cache] body compile failed; ordinary compiler retained: {}",e.localizedDescription());return None;}};
        let library=match gfx::gpu().device.0.newDynamicLibrary_error(&lib) {Ok(v)=>v,Err(e)=>{eprintln!("[vertex-cache] body link failed; ordinary compiler retained: {}",e.localizedDescription());return None;}};
        if !store(&path,&library) {eprintln!("[vertex-cache] persistence failed; live library/preloaded or ordinary compiler remains available");}
        library
    };
    let o=options();o.setLibraries(Some(&NSArray::from_retained_slice(&[library.clone()])));
    let lib=match gfx::gpu().device.0.newLibraryWithSource_options_error(&NSString::from_str(&wrapper),Some(&o)) {Ok(v)=>v,Err(e)=>{eprintln!("[vertex-cache] entry compile failed; ordinary compiler retained: {}",e.localizedDescription());return None;}};
    let func=lib.newFunctionWithName(&NSString::from_str(&entry))?;
    Some(Product{func:Shared(func),state:DynamicVertex{library:Shared(library),normal_source:normal_source.into(),entry,fallback:OnceLock::new()},disk_hit,path})
}

pub(super) fn compile(t:&nxshader::Translated,key:u64,normal_source:&str)->Option<Product> {
    static ENABLED:LazyLock<bool>=LazyLock::new(||std::env::var_os("NX_DYNAMIC_VERTEX_CACHE").is_some());
    if !*ENABLED || t.stage!=nxshader::sph::Stage::Vertex {return None;}
    static DIRECTORY:LazyLock<PathBuf>=LazyLock::new(|| {
        let base=std::env::var_os("NX_DYNAMIC_VERTEX_DIR").map(PathBuf::from).unwrap_or_else(||PathBuf::from(super::shaders::aot_dir()));
        let version=NSProcessInfo::processInfo().operatingSystemVersionString().to_string();
        let os=nxfs::crypto::sha256(version.as_bytes());
        let directory=base.join(format!("vertex-dynamic-v1-{:x}-{}",gfx::gpu().device.0.registryID(),&os[..16]));
        eprintln!("[vertex-cache] vertex only, math-safe, namespace={}",directory.display());directory
    });
    let result=compile_at(t,key,normal_source,&DIRECTORY);
    static HITS:AtomicU64=AtomicU64::new(0);static BUILDS:AtomicU64=AtomicU64::new(0);static FALLBACKS:AtomicU64=AtomicU64::new(0);
    if let Some(product)=&result {
        let count=if product.disk_hit {HITS.fetch_add(1,Relaxed)+1} else {BUILDS.fetch_add(1,Relaxed)+1};
        if count<=8||count.is_power_of_two() {eprintln!("[vertex-cache] {}={count} hits={} builds={} file={}",if product.disk_hit {"disk-hit"} else {"built"},HITS.load(Relaxed),BUILDS.load(Relaxed),product.path.display());}
    } else {let count=FALLBACKS.fetch_add(1,Relaxed)+1;if count<=8||count.is_power_of_two() {eprintln!("[vertex-cache] ordinary-fallback={count} shader={key:016x}");}}
    result
}

#[cfg(all(test,target_os="macos"))]
mod tests;
