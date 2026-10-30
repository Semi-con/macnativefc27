// FC27 SDK 22 loader integration. Original AOT rtld builds and searches its
// own RoModule records; the host supplies the already verified native mappings.
use crate::{ipc::{Request,Response,Service,Stub},kernel,loader::Module};
use anyhow::{Result,ensure,Context};
use std::collections::{HashMap,HashSet};
use std::sync::{Arc,Mutex,OnceLock};

struct NativeModule {name:String,base:u64,size:u64,dynamic:u64,record:u64}
struct Runtime {rtld:u64,modules:Vec<NativeModule>,globals:Vec<(u64,u64)>,main:u64,overrides:HashMap<&'static str,u64>,binding_pages:Vec<u64>}
static RUNTIME: OnceLock<Runtime> = OnceLock::new();

pub fn configure(mods:&[Module]) -> Result<()> {
    let rtld = mods.iter().find(|m|m.name=="rtld").context("rtld missing")?.base;
    let mut modules = Vec::new();
    // The original rtld inserts each module at the list's front while scanning
    // ascending addresses. The renderer is staged in the address space, but
    // joins the original SDK's manual-load list only when the game loads it.
    for name in ["sdk","renderer","main","rtld"] {
        let m = mods.iter().find(|m|m.name==name).context("module missing")?;
        let rel = |at:u64| unsafe {(m.base+m.dynamic.mod0).wrapping_add(((m.base+m.dynamic.mod0+at) as *const i32).read_unaligned() as i64 as u64)};
        let record = rel(0x18);
        ensure!(record>=m.base+m.dynamic.bss.0 && record+0xd0<=m.base+m.dynamic.bss.1,"{name}: invalid original RoModule storage");
        modules.push(NativeModule {name:name.into(),base:m.base,size:(m.dynamic.bss.1+0xfff)&!0xfff,dynamic:rel(4),record});
    }
    let mut globals = Vec::new();
    for (name,value) in [
        ("_ZN2nn2ro6detail15g_pAutoLoadListE",rtld+0x5030),
        ("_ZN2nn2ro6detail17g_pManualLoadListE",rtld+0x5020),
        ("_ZN2nn2ro6detail14g_pRoDebugFlagE",rtld+0x5010),
        ("_ZN2nn2ro6detail34g_pLookupGlobalAutoFunctionPointerE",lookup_auto as *const () as usize as u64),
        ("_ZN2nn2ro6detail36g_pLookupGlobalManualFunctionPointerE",rtld+0x5018),
    ] {globals.push((crate::loader::symbol(mods,name).with_context(||name.to_owned())?,value));}
    let main = crate::loader::symbol(mods,"nnMain").context("nnMain missing")?;
    let renderer=mods.iter().find(|m|m.name=="renderer").unwrap();
    let definitions:HashSet<_>=renderer.dynamic.symbols.iter().filter(|s|s.defined()).map(|s|s.name.as_str()).collect();
    let mut binding_pages=Vec::new();
    for m in mods.iter().filter(|m|m.name!="renderer") {
        for relocation in &m.dynamic.relocs {
            if matches!(relocation.kind,nxfs::module::R_AARCH64_ABS64|nxfs::module::R_AARCH64_GLOB_DAT)
                && m.dynamic.symbols.get(relocation.sym as usize).is_some_and(|s|!s.defined() && definitions.contains(s.name.as_str())) {
                ensure!(relocation.offset>=m.data.0,"renderer import is not a data relocation");
                binding_pages.push((m.base+relocation.offset)&!(crate::mem::PAGE-1));
            }
        }
    }
    binding_pages.sort_unstable();binding_pages.dedup();
    RUNTIME.set(Runtime {rtld,modules,globals,main,overrides:crate::hle::table(),binding_pages}).map_err(|_|anyhow::anyhow!("ro already configured"))?;
    Ok(())
}

extern "C" fn bad_dynamic(tag:u64,value:u64) {
    eprintln!("[ro] original rtld rejected dynamic tag {tag:#x} value {value:#x}");
    crate::diag::report_and_exit("unsupported original module metadata");
}

pub extern "C" fn initialize_native() {
    let r = RUNTIME.get().expect("ro not configured");
    unsafe {
        let init:extern "C" fn(u64,u64,u64,u64,u32,u64) = std::mem::transmute((r.rtld+0x6f0) as usize);
        for m in r.modules.iter().filter(|m|m.name!="renderer") {
            init(m.record,m.base,m.size,m.dynamic,0,bad_dynamic as *const () as usize as u64);
            eprintln!("[ro] registered {} at {:#x}, original record {:#x}",m.name,m.base,m.record);
        }
        let head = r.rtld+0x5030;
        let nodes:Vec<_> = std::iter::once(head).chain(r.modules.iter().filter(|m|m.name!="renderer").map(|m|m.record)).collect();
        for (i,&node) in nodes.iter().enumerate() {
            (node as *mut u64).write(nodes[(i+1)%nodes.len()]);
            ((node+8) as *mut u64).write(nodes[(i+nodes.len()-1)%nodes.len()]);
        }
        let manual = r.rtld+0x5020;
        (manual as *mut u64).write(manual); ((manual+8) as *mut u64).write(manual);
        for &(address,value) in &r.globals {(address as *mut u64).write(value);}
        let lookup:extern "C" fn(*const u8)->u64 = std::mem::transmute((r.rtld+0x1070) as usize);
        if lookup(c"nnMain".as_ptr().cast()) != r.main {
            crate::diag::report_and_exit("original rtld global lookup disagrees with native loader");
        }
        // Auto lookup is a direct function pointer. Only manual lookup is a
        // pointer to a writable function-pointer slot, populated by the SDK.
        let configured:extern "C" fn(*const std::ffi::c_char)->u64=std::mem::transmute((r.globals[3].0 as *const u64).read() as usize);
        if configured(c"nnMain".as_ptr())!=r.main || configured(c"nvnBootstrapLoader".as_ptr())!=r.overrides["nvnBootstrapLoader"] {
            crate::diag::report_and_exit("SDK dynamic lookup ABI validation failed");
        }
        let ready:extern "C" fn() = std::mem::transmute((r.rtld+0x18c8) as usize);ready();
    }
    eprintln!("[ro] original global lookup verified; SDK loader pointers connected");
}

extern "C" fn lookup_auto(name:*const std::ffi::c_char)->u64 {
    let r=RUNTIME.get().unwrap();
    if let Ok(name)=unsafe {std::ffi::CStr::from_ptr(name)}.to_str() {
        if let Some(&address)=r.overrides.get(name){return address;}
    }
    let lookup:extern "C" fn(*const std::ffi::c_char)->u64=unsafe {std::mem::transmute((r.rtld+0x1070) as usize)};
    lookup(name)
}

fn load_with_bindings(module:u64,nro:u64,bss:u64,bss_size:u64,flags:i32,option:bool)->u32 {
    let r=RUNTIME.get().unwrap();let a=kernel::aspace();
    let mut restore=Vec::new();
    for &page in &r.binding_pages {
        let mut cursor=page;
        while cursor<page+crate::mem::PAGE {
            let (base,size,block)=a.map.lock().unwrap().query(cursor);
            let len=(base+size).min(page+crate::mem::PAGE)-cursor;
            if block.perm&crate::mem::perm::X!=0 || block.perm&crate::mem::perm::R==0 {
                crate::diag::report_and_exit("dynamic binding page is not readable data");
            }
            if block.perm&crate::mem::perm::W==0 {restore.push((cursor,len,block));}
            cursor+=len;
        }
    }
    for &(address,size,block) in &restore {
        if a.set(address,size,crate::mem::Block {perm:crate::mem::perm::RW,..block}).is_err(){crate::diag::report_and_exit("cannot open native relocation page");}
    }
    let sdk=r.modules.iter().find(|m|m.name=="sdk").unwrap().base;
    let load:extern "C" fn(u64,u64,u64,u64,i32,bool)->u32=unsafe {std::mem::transmute((sdk+0x156270) as usize)};
    let result=load(module,nro,bss,bss_size,flags,option);
    for &(address,size,block) in &restore {
        if a.set(address,size,block).is_err(){crate::diag::report_and_exit("cannot restore native relocation protection");}
    }
    eprintln!("[ro] original SDK LoadModule returned {result:#x}; {} relocation ranges restored",restore.len());
    result
}
pub extern "C" fn load_module(module:u64,nro:u64,bss:u64,bss_size:u64,flags:i32)->u32 {
    load_with_bindings(module,nro,bss,bss_size,flags,false)
}
pub extern "C" fn load_module_ex(module:u64,nro:u64,bss:u64,bss_size:u64,flags:i32,option:bool)->u32 {
    load_with_bindings(module,nro,bss,bss_size,flags,option)
}

const RENDERER_SHA:&str="487a417b97839e2e554a61caf3c4a61e0739078a63dbbd8b85c35a4cf1a1c714";
struct Mapping {base:u64,bss:u64,bss_size:u64}

#[derive(Default)]
struct State {registered:bool,nrrs:HashMap<u64,HashSet<[u8;32]>>,mapping:Option<Mapping>}
#[derive(Default)]
pub struct Ro {state:Mutex<State>}
pub fn open()->Arc<dyn Service> {Arc::new(Ro::default())}

fn nrr_hashes(b:&[u8])->Result<HashSet<[u8;32]>> {
    ensure!(b.len()>=0x350 && &b[..4]==b"NRR0","invalid NRR header");
    let u32at=|o|u32::from_le_bytes(b[o..o+4].try_into().unwrap()) as usize;
    ensure!(u64::from_le_bytes(b[0x330..0x338].try_into()?)==crate::game::TITLE_ID,"NRR title mismatch");
    ensure!(u32at(0x338)==b.len() && b[0x33c]==0,"NRR size/kind mismatch");
    let (off,count)=(u32at(0x340),u32at(0x344));
    ensure!(off>=0x350 && count>0 && off.checked_add(count.checked_mul(32).context("NRR hash overflow")?).is_some_and(|end|end<=b.len()),"invalid NRR hash table");
    Ok(b[off..off+count*32].chunks_exact(32).map(|s|s.try_into().unwrap()).collect())
}

impl Service for Ro {
    fn name(&self)->String {"ldr:ro".into()}
    fn call(&self,req:&Request,rsp:&mut Response)->u32 {
        let mut s = self.state.lock().unwrap();
        match req.cmd {
            0 => {
                if !s.registered || s.mapping.is_some(){return kernel::rc::INVALID_STATE;}
                let (source,size,bss,bss_size)=(req.u64(8),req.u64(16),req.u64(24),req.u64(32));
                let a=kernel::aspace();
                if source%0x1000!=0 || size!=0xee000 || bss_size!=0xd000 || !a.readable(source,size) || !a.readable(bss,bss_size){return kernel::rc::INVALID_ADDRESS;}
                let bytes=unsafe {std::slice::from_raw_parts(source as *const u8,size as usize)};
                let hash=nxfs::crypto::sha256(bytes);
                if hash!=RENDERER_SHA {
                    eprintln!("[ro] no offline-prepared native module for NRO SHA256 {hash}");return kernel::rc::NOT_FOUND;
                }
                let digest:[u8;32]=std::array::from_fn(|i|u8::from_str_radix(&hash[i*2..i*2+2],16).unwrap());
                if !s.nrrs.values().any(|hashes|hashes.contains(&digest)) {return kernel::rc::INVALID_STATE;}
                let renderer=RUNTIME.get().unwrap().modules.iter().find(|m|m.name=="renderer").unwrap();
                // Only mutable state is restored from the verified original NRO.
                // Executable pages keep the fully equivalence-tested AOT code,
                // keypage and native trampolines. The original SDK now owns
                // relocations, TLS, constructors and manual list insertion.
                if a.set(renderer.base+0xe4000,0x17000,crate::mem::Block {state:crate::mem::state::CODE_DATA,perm:crate::mem::perm::RW,attr:0}).is_err(){return kernel::rc::INVALID_ADDRESS;}
                unsafe {
                    std::ptr::copy_nonoverlapping(bytes[0xe4000..].as_ptr(),(renderer.base+0xe4000) as *mut u8,0xa000);
                    std::ptr::write_bytes((renderer.base+0xee000) as *mut u8,0,0xd000);
                }
                rsp.push_u64(renderer.base);s.mapping=Some(Mapping {base:renderer.base,bss,bss_size});
                eprintln!("[ro] matched registered renderer NRO; mapped AOT image at {:#x} for original SDK initialization",renderer.base);0
            }
            1 => {
                let Some(mapping)=&s.mapping else{return kernel::rc::NOT_FOUND};
                if req.u64(8)!=mapping.base{return kernel::rc::INVALID_ADDRESS;}
                if kernel::aspace().readable(mapping.bss,mapping.bss_size) {
                    unsafe {std::ptr::copy_nonoverlapping((mapping.base+0xee000) as *const u8,mapping.bss as *mut u8,mapping.bss_size as usize);}
                }
                s.mapping=None;0
            }
            4 => {
                if req.copy_handles.len()!=1 || !matches!(kernel::KERNEL.lock().get(req.copy_handles[0]),Some(kernel::Obj::Process)) {return kernel::rc::INVALID_HANDLE;}
                if s.registered {return kernel::rc::INVALID_STATE;} s.registered=true;0
            }
            2 => {
                if !s.registered {return kernel::rc::INVALID_STATE;}
                let (addr,size)=(req.u64(8),req.u64(16));
                if addr%0x1000!=0 || size==0 || size%0x1000!=0 || size>4*1024*1024 || !kernel::aspace().readable(addr,size) {return kernel::rc::INVALID_ADDRESS;}
                let bytes=unsafe {std::slice::from_raw_parts(addr as *const u8,size as usize)};
                match nrr_hashes(bytes) {
                    Ok(hashes)=>{s.nrrs.insert(addr,hashes);0},
                    Err(e)=>{eprintln!("[ro] registration rejected: {e}");kernel::rc::INVALID_STATE}
                }
            }
            3 => {if s.nrrs.remove(&req.u64(8)).is_some(){0}else{kernel::rc::NOT_FOUND}}
            _ => Stub(self.name()).call(req,rsp),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn registration_binds_title_and_checks_hash_table_extent() {
        let mut nrr=vec![0u8;0x1000];nrr[..4].copy_from_slice(b"NRR0");
        nrr[0x330..0x338].copy_from_slice(&crate::game::TITLE_ID.to_le_bytes());
        nrr[0x338..0x33c].copy_from_slice(&0x1000u32.to_le_bytes());
        nrr[0x340..0x344].copy_from_slice(&0x350u32.to_le_bytes());
        nrr[0x344..0x348].copy_from_slice(&1u32.to_le_bytes());nrr[0x350..0x370].fill(0xab);
        assert!(nrr_hashes(&nrr).unwrap().contains(&[0xab;32]));
        nrr[0x344..0x348].copy_from_slice(&u32::MAX.to_le_bytes());assert!(nrr_hashes(&nrr).is_err());
        nrr[0x344..0x348].copy_from_slice(&1u32.to_le_bytes());nrr[0x330]^=1;assert!(nrr_hashes(&nrr).is_err());
    }
}
