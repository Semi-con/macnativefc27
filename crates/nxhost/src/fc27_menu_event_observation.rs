//! Isolated, bounded observations of original menu entity event dispatch.
//! Delegates each void(const EntityEvent&) call once, preserving its arguments.
use std::sync::{OnceLock, atomic::{AtomicBool,AtomicU32, Ordering::{Acquire,Release,Relaxed}}};
use anyhow::{Result, ensure};
use crate::stubs::Args;

#[derive(Clone, Copy)]
struct Site {
    label: &'static str,
    name_at: u64,
    name: &'static [u8],
    vtable: u64,
    getter: u64,
    event: u64,
    words: [u32; 2],
}
const SITES: [Site; 4] = [
    Site {label:"sequence", name_at:0xbdd3ecd, name:b"CinematicMainMenuSequenceSelectionEntity\0",
        vtable:0xc8af5c8, getter:0x5e14ac0, event:0x5e14ad0, words:[0xb9400828,0x5294b5a9]},
    Site {label:"hub", name_at:0xbb6455d, name:b"FIFAHubEntity\0",
        vtable:0xc8b5018, getter:0x5e69b70, event:0x5e69df0, words:[0xd10143ff,0xa9027bfd]},
    Site {label:"cinematic-in", name_at:0xbdecd15, name:b"ClientCinematicInEntity\0",
        vtable:0xc5c3a20, getter:0x1ba53b0, event:0x1ba53c0, words:[0xd10183ff,0xa9027bfd]},
    Site {label:"platform", name_at:0xbbd78f8, name:b"PlatformSplitterEntity\0",
        vtable:0xcaf1ea8, getter:0x7068fb0, event:0x7068fc0, words:[0xa9be7bfd,0xf9000bf3]},
];
static ORIGINAL: OnceLock<[u64; 4]> = OnceLock::new();
static BASE: OnceLock<u64> = OnceLock::new();
static CALLS: [AtomicU32; 4] = [const {AtomicU32::new(0)}; 4];
static MANAGER_ORIGINAL:OnceLock<u64>=OnceLock::new();
static MANAGER_CALLS:AtomicU32=AtomicU32::new(0);
static NAMED_MANAGER_CALLS:AtomicU32=AtomicU32::new(0);
static MASTER_EVENT_CALLS:AtomicU32=AtomicU32::new(0);
static MASTER_PRIME_COMPLETE:AtomicBool=AtomicBool::new(false);
static MASTER_PREP_COMPLETE:AtomicBool=AtomicBool::new(false);
pub(crate) fn master_readiness()->(bool,bool){
    (MASTER_PRIME_COMPLETE.load(Acquire),MASTER_PREP_COMPLETE.load(Acquire))
}
pub(crate) fn reset_master_readiness(){
    MASTER_PRIME_COMPLETE.store(false,Release);MASTER_PREP_COMPLETE.store(false,Release);
}
// Original registered MainMenuInitializeData property interface: prepare,
// allocate a typed payload, then consume that payload. Observation only.
const INIT_METHODS:[(u64,u64,[u32;2]);3]=[
    (0xc895658,0x5cabfd0,[0xfc180fea,0x6d0123e9]),
    (0xc895670,0x5cb0770,[0xa9bf7bfd,0x910003fd]),
    (0xc895678,0x5cb07d0,[0xd10243ff,0xa9037bfd]),
];
static INIT_ORIGINAL:OnceLock<[u64;3]>=OnceLock::new();
static INIT_CALLS:[AtomicU32;3]=[const{AtomicU32::new(0)};3];
static POPULATE_ORIGINAL:OnceLock<u64>=OnceLock::new();
static POPULATE_CALLS:AtomicU32=AtomicU32::new(0);
// Names are verified by original diagnostic strings in the delegated methods.
// Slot100 is a general event receiver, not the playback request interface.
const PLAYBACK_METHODS:[(&str,u64,u64,[u32;2]);4]=[
    ("RequestPrime",0xc885898,0x5c19040,[0xa9bc7bfd,0xa9015ff8]),
    ("RequestPrep",0xc8858a0,0x1baa7b0,[0xa9bc7bfd,0xf9000bf7]),
    ("RequestStart",0xc8858a8,0x5c19180,[0xa9bd7bfd,0xa90157f6]),
    ("RequestUnprime",0xc8858b0,0x1baad60,[0xa9bc7bfd,0xf9000bf7]),
];
static PLAYBACK_ORIGINAL:OnceLock<[u64;4]>=OnceLock::new();
static PLAYBACK_CALLS:[AtomicU32;4]=[const{AtomicU32::new(0)};4];
static NAMED_PLAYBACK_CALLS:[AtomicU32;4]=[const{AtomicU32::new(0)};4];

fn dispatch_playback(tag:u64,args:&Args)->u64 {
    let index=tag as usize;
    let count=PLAYBACK_CALLS[index].fetch_add(1,Relaxed)+1;
    let mut bytes=[0;96];
    let name=(args.0[1]!=0&&snapshot(args.0[1],&mut bytes))
        .then(||crate::menu_cinematic_diagnostics::selected_pointer_name(&bytes)).flatten();
    let named_count=if name.is_some(){NAMED_PLAYBACK_CALLS[index].fetch_add(1,Relaxed)+1}else{0};
    if count<=16||(count<=4096&&count.is_power_of_two())||(named_count>0&&named_count<=32){
        eprintln!("[cinematic-playback] operation={} count={count} named_count={named_count} identity={} menu_name={}",
            PLAYBACK_METHODS[index].0,word(args.0[0])==Some(*BASE.get().unwrap()+0xc885828),
            name.map_or("null".into(),|n|format!("\"{n}\"")));
    }
    unsafe{delegate_manager(PLAYBACK_ORIGINAL.get().unwrap()[index],args)}
}

fn initialization_payload(address:u64)->String {
    let mut bytes=[0;64];
    if address==0||!snapshot(address,&mut bytes){return "null".into();}
    crate::menu_still_frame_diagnostics::row(&bytes,address,*BASE.get().unwrap())
        .unwrap_or_else(||"null".into())
}
fn dispatch_initialization(tag:u64,args:&Args)->u64 {
    let index=tag as usize;
    let count=INIT_CALLS[index].fetch_add(1,Relaxed)+1;
    let log=count<=16||(count<=4096&&count.is_power_of_two());
    if log&&index==0{
        eprintln!("[mainmenu-init-service] {}",crate::input::test_control::menu_playback::population_service(*BASE.get().unwrap()));
    }
    let before=if log&&index!=1 {initialization_payload(args.0[2])}else{"null".into()};
    let result=unsafe{delegate_manager(INIT_ORIGINAL.get().unwrap()[index],args)};
    if log {
        let after=initialization_payload(if index==1{result}else{args.0[2]});
        eprintln!("[mainmenu-init] operation={} count={count} before={before} after={after} returned_x0={result:#x}",
            ["prepare","allocate","consume"][index]);
    }
    result
}

fn population_identifiers(address:u64)->Option<[i32;11]>{
    let mut identifiers=[0;11];
    for (index,value)in identifiers.iter_mut().enumerate(){
        let mut bytes=[0;4];
        let at=address.checked_add(0x10+index as u64*0xd0).filter(|_|address!=0)?;
        if !snapshot(at,&mut bytes){return None;}
        *value=i32::from_le_bytes(bytes);
    }
    Some(identifiers)
}
fn dispatch_population(_:u64,args:&Args)->u64{
    let count=POPULATE_CALLS.fetch_add(1,Relaxed)+1;
    let log=count<=16||(count<=4096&&count.is_power_of_two());
    let base=*BASE.get().unwrap();
    let frontend_before=if log{word(base+0xdc557e8)}else{None};
    let result=unsafe{delegate_manager(*POPULATE_ORIGINAL.get().unwrap(),args)};
    if log{
        // The original a1c9df0 consumes an explicit X1 output pointer, returns
        // a boolean in W0, and copies eleven entries with stride d0. Only the
        // numeric sentinel identifiers used by the original consumer are read.
        eprintln!("[mainmenu-populate] count={count} provider_identity={} frontend_before_present={} frontend_after_present={} returned_w0={} identifiers_d0={:?}",
            word(args.0[0])==Some(base+0xccddf58),frontend_before.is_some_and(|v|v!=0),
            word(base+0xdc557e8).is_some_and(|v|v!=0),result as u32,population_identifiers(args.0[1]));
    }
    crate::fc27_menu_ambassador_source::apply_empty_result(base,args.0[1],result);
    result
}

fn snapshot(address:u64, bytes:&mut[u8])->bool {
    crate::input::test_control::config_inspection::snapshot(address, bytes)
}
fn word(address:u64)->Option<u64> {
    let mut b=[0;8];snapshot(address,&mut b).then(||u64::from_le_bytes(b))
}
fn event_id(address:u64)->String {
    let Some(at)=address.checked_add(8).filter(|_|address!=0)else{return "null".into();};
    let mut b=[0;4];if !snapshot(at,&mut b){return "null".into();}
    format!("\"{:#x}\"",u32::from_le_bytes(b))
}
fn menu_instance(data:u64)->String {
    let Some(at)=data.checked_add(0x30).filter(|_|data!=0)else{return "null".into();};
    let Some(pointer)=word(at).filter(|v|*v!=0)else{return "null".into();};
    let mut b=[0;96];
    if !snapshot(pointer,&mut b){return "null".into();}
    let Some(end)=b.iter().position(|v|*v==0)else{return "null".into();};
    let b=&b[..end];
    if !(4..=80).contains(&b.len()) || !b.iter().all(|v|v.is_ascii_alphanumeric()||*v==b'_') {
        return "null".into();
    }
    if ![b"f26_mm_".as_slice(),b"f27_mm_",b"metarewardspreview"].iter().any(|p|b.starts_with(p)) {
        return "null".into();
    }
    format!("\"{}\"",String::from_utf8_lossy(b))
}
unsafe fn delegate(address:u64,args:&Args) {
    let call:extern "C" fn(u64,u64)=unsafe{std::mem::transmute(address as usize)};
    call(args.0[0],args.0[1]);
}
unsafe fn delegate_manager(address:u64,args:&Args)->u64 {
    // Preserve all incoming GPR arguments and returned X0, as the existing
    // loader tracer does. This original interface consumes integers/pointers.
    let call:extern "C" fn(u64,u64,u64,u64,u64,u64,u64,u64)->u64=unsafe{std::mem::transmute(address as usize)};
    call(args.0[0],args.0[1],args.0[2],args.0[3],args.0[4],args.0[5],args.0[6],args.0[7])
}
fn manager_matches(base:u64,first:[u64;3],method:u64,words:[u32;2])->bool {
    first==[base+0x5c185c0,base+0x5c18680,base+0x9a4b50]
        &&method==base+0x5c19210&&words==[0xd10183ff,0xa9017bfd]
}
fn dispatch_manager(_:u64,args:&Args)->u64 {
    let count=MANAGER_CALLS.fetch_add(1,Relaxed)+1;
    let mut bytes=[0;96];
    let name=(args.0[3]!=0&&snapshot(args.0[3],&mut bytes))
        .then(||crate::menu_cinematic_diagnostics::selected_pointer_name(&bytes)).flatten();
    let named_count=if name.is_some(){NAMED_MANAGER_CALLS.fetch_add(1,Relaxed)+1}else{0};
    let master_count=if name==Some("f26_mm_master_01"){MASTER_EVENT_CALLS.fetch_add(1,Relaxed)+1}else{0};
    if count<=32||(count<=4096&&count.is_power_of_two())||(named_count>0&&named_count<=32)||(master_count>0&&master_count<=32) {
        let base=*BASE.get().unwrap();
        eprintln!("[cinematic-manager-event] count={count} named_count={named_count} identity={} event={:#x} instance_hash={:#x} menu_name={} condition={:#x}",
            word(args.0[0])==Some(base+0xc885828),args.0[1]as u32,args.0[2]as u32,
            name.map_or("null".into(),|n|format!("\"{n}\"")),args.0[4]);
    }
    let master=name==Some("f26_mm_master_01")&&args.0[2]as u32==610392389
        &&word(args.0[0])==Some(*BASE.get().unwrap()+0xc885828);
    let result=unsafe{delegate_manager(*MANAGER_ORIGINAL.get().unwrap(),args)};
    // Only completed original event dispatch marks readiness; the probe never
    // fabricates PrimeComplete or PrepComplete notifications.
    if master{
        match args.0[1]as u32{
            0xdc6d0daf=>MASTER_PRIME_COMPLETE.store(true,Release),
            0xae59cf3b=>MASTER_PREP_COMPLETE.store(true,Release),
            _=>{},
        }
        crate::fc27_menu_auto_lifecycle::master_event(args.0[1]as u32);
    }
    result
}
fn dispatch(tag:u64,args:&Args)->u64 {
    let slot=tag as usize;
    let count=CALLS[slot].fetch_add(1,Relaxed)+1;
    if count<=24 || (count<=4096&&count.is_power_of_two()) {
        let base=*BASE.get().unwrap();let site=SITES[slot];let object=args.0[0];
        let verified=word(object)==Some(base+site.vtable);
        let data=object.checked_add(0x38).and_then(word).unwrap_or(0);
        let extra=match slot {
            0=>{
                let mut b=[0;3];
                if data!=0&&data.checked_add(0x24).is_some_and(|at|snapshot(at,&mut b))&&b.iter().all(|v|*v<=1) {
                    format!(r#",asset_restrict_telstar={},asset_first_boot={},asset_still_frames={}"#,b[0],b[1],b[2])
                }else{String::new()}
            },
            1=>{
                let mut b=[0];
                if data!=0&&data.checked_add(0x24).is_some_and(|at|snapshot(at,&mut b))&&b[0]<=1 {
                    format!(",background_hub={}",b[0])
                }else{String::new()}
            },
            2=>format!(",menu_instance={}",menu_instance(data)),
            _=>String::new(),
        };
        eprintln!("[menu-event] class={} count={count} object={object:#x} identity={verified} event={}{}",site.label,event_id(args.0[1]),extra);
    }
    // Original Entity::handleEvent has two integer/pointer arguments and void
    // return. AAPCS64 callee-saved registers and argument forwarding follow the
    // same existing guest-to-host stubs used by libvpx observation.
    unsafe{delegate(ORIGINAL.get().unwrap()[slot],args);}
    if slot==1{
        let mut event=[0;4];
        if snapshot(args.0[1]+8,&mut event){crate::fc27_menu_auto_lifecycle::hub_event(*BASE.get().unwrap(),args.0[0],u32::from_le_bytes(event));}
    }
    0
}
fn matches(base:u64,site:Site,getter:u64,event:u64,words:[u32;2],name:&[u8])->bool {
    getter==base+site.getter&&event==base+site.event&&words==site.words&&name==site.name
}
pub(crate) fn apply(modules:&[crate::loader::Module])->Result<()> {
    if !std::env::var("NX_FC27_MENU_EVENT_TRACE").is_ok_and(|v|v=="1"){return Ok(());}
    ensure!(std::env::var_os("NX_TEST_CONTROL").is_some()&&std::env::var("NX_FC27_VIDEO_TRACE").is_ok_and(|v|v=="1"),"menu event observation requires isolated controls and video trace");
    ensure!(std::env::current_exe()?.file_name().is_some_and(|v|v=="FC27-test"),"menu event observation requires the test executable");
    let m=modules.iter().find(|m|m.name=="main").ok_or_else(||anyhow::anyhow!("menu event observation: missing main"))?;
    let id:String=m.module_id.iter().map(|v|format!("{v:02x}")).collect();
    ensure!(id=="fb81983aa1cc3a2e97402497e2ca652800000000000000000000000000000000","menu event observation: unknown game build");
    let mut original=[0;4];
    for (i,site)in SITES.into_iter().enumerate(){
        ensure!(site.event+8<=m.text&&site.vtable+0x38<=m.size&&site.name_at+site.name.len()as u64<=m.size,"menu event observation: bounds");
        let getter=unsafe{((m.base+site.vtable)as*const u64).read_unaligned()};
        let event=unsafe{((m.base+site.vtable+0x30)as*const u64).read_unaligned()};
        let words=unsafe{((m.base+site.event)as*const[u32;2]).read_unaligned()};
        let name=unsafe{std::slice::from_raw_parts((m.base+site.name_at)as*const u8,site.name.len())};
        ensure!(matches(m.base,site,getter,event,words,name),"menu event observation: {} original guard mismatch",site.label);
        original[i]=event;
    }
    ensure!(0xc885930<=m.size&&0x5c19218<=m.text,"cinematic request observation: bounds");
    let first=unsafe{((m.base+0xc885828)as*const[u64;3]).read_unaligned()};
    let manager_method=unsafe{((m.base+0xc885928)as*const u64).read_unaligned()};
    let manager_words=unsafe{((m.base+0x5c19210)as*const[u32;2]).read_unaligned()};
    ensure!(manager_matches(m.base,first,manager_method,manager_words)&&crate::menu_cinematic_diagnostics::version(m.base),"cinematic request observation: original interface/callsite mismatch");
    ensure!(crate::menu_still_frame_diagnostics::version(m.base),"menu initialization observation: original typed fields mismatch");
    let mut init_original=[0;3];
    for (index,(slot,method,words))in INIT_METHODS.into_iter().enumerate(){
        ensure!(slot+8<=m.size&&method+8<=m.text,"menu initialization observation: bounds");
        let pointer=unsafe{((m.base+slot)as*const u64).read_unaligned()};
        let instructions=unsafe{((m.base+method)as*const[u32;2]).read_unaligned()};
        ensure!(pointer==m.base+method&&instructions==words,"menu initialization observation: changed original method");
        init_original[index]=pointer;
    }
    let mut playback_original=[0;4];
    for (index,(_,slot,method,words))in PLAYBACK_METHODS.into_iter().enumerate(){
        ensure!(slot+8<=m.size&&method+8<=m.text,"cinematic playback observation: bounds");
        let pointer=unsafe{((m.base+slot)as*const u64).read_unaligned()};
        let instructions=unsafe{((m.base+method)as*const[u32;2]).read_unaligned()};
        ensure!(pointer==m.base+method&&instructions==words,"cinematic playback observation: changed original method");
        playback_original[index]=pointer;
    }
    ensure!(0xccddf90<=m.size&&0xa1caacc<=m.text,"menu population observation: bounds");
    let populate=unsafe{((m.base+0xccddf88)as*const u64).read_unaligned()};
    ensure!(populate==m.base+0xa1c9df0
        &&crate::diag::read_words::<1>(m.base+0xa1c9df0)==Some([0xa9016ffca9ba7bfd])
        &&crate::diag::read_words::<1>(m.base+0xa1c9e10)==Some([0xf943f5139001d468]),
        "menu population observation: original provider mismatch");
    ensure!(POPULATE_ORIGINAL.set(populate).is_ok(),"menu population observation already installed");
    crate::fc27_menu_ambassador_source::verify(m.base)?;
    ensure!(PLAYBACK_ORIGINAL.set(playback_original).is_ok(),"cinematic playback observation already installed");
    ensure!(INIT_ORIGINAL.set(init_original).is_ok(),"menu initialization observation already installed");
    ensure!(BASE.set(m.base).is_ok()&&ORIGINAL.set(original).is_ok()&&MANAGER_ORIGINAL.set(manager_method).is_ok(),"menu event observation already installed");
    // All guards pass before changes. Loader has not made rodata read-only or
    // started a guest thread. Only four event vtable slots are observed.
    for (i,site)in SITES.into_iter().enumerate(){
        let stub=crate::stubs::make(dispatch,i as u64);
        unsafe{((m.base+site.vtable+0x30)as*mut u64).write_volatile(stub);}
    }
    let stub=crate::stubs::make(dispatch_manager,0);
    unsafe{((m.base+0xc885928)as*mut u64).write_volatile(stub);}
    for (index,(slot,_,_))in INIT_METHODS.into_iter().enumerate(){
        let stub=crate::stubs::make(dispatch_initialization,index as u64);
        unsafe{((m.base+slot)as*mut u64).write_volatile(stub);}
    }
    for (index,(_,slot,_,_))in PLAYBACK_METHODS.into_iter().enumerate(){
        let stub=crate::stubs::make(dispatch_playback,index as u64);
        unsafe{((m.base+slot)as*mut u64).write_volatile(stub);}
    }
    let stub=crate::stubs::make(dispatch_population,0);
    unsafe{((m.base+0xccddf88)as*mut u64).write_volatile(stub);}
    eprintln!("[menu-event] installed4 entity event observers, cinematic manager event observer,3 menu initialization interface observers and4 original named playback API observers; bounded logs, all calls delegated unchanged");
    Ok(())
}

#[cfg(test)]mod tests{
    use super::*;
    #[test]fn manager_delegation_preserves_all_gprs_and_returned_x0_once(){
        use std::sync::atomic::AtomicU64;
        static N:AtomicU32=AtomicU32::new(0);
        static VALUES:[AtomicU64;8]=[const{AtomicU64::new(0)};8];
        extern "C" fn original(a:u64,b:u64,c:u64,d:u64,e:u64,f:u64,g:u64,h:u64)->u64{
            for (slot,value)in VALUES.iter().zip([a,b,c,d,e,f,g,h]){slot.store(value,Relaxed);}
            N.fetch_add(1,Relaxed);0xfedcba9876543210
        }
        let values=std::array::from_fn(|i|0x7100000000u64+i as u64*0x102030405);
        let args=Args(values,[0;8]);
        let result=unsafe{delegate_manager(original as*const()as usize as u64,&args)};
        assert_eq!(result,0xfedcba9876543210);assert_eq!(N.load(Relaxed),1);
        for (slot,expected)in VALUES.iter().zip(values){assert_eq!(slot.load(Relaxed),expected);}
    }
    #[test]fn manager_observer_rejects_changed_interface_and_entry_instructions(){
        let base=0x7100000000;let first=[base+0x5c185c0,base+0x5c18680,base+0x9a4b50];let method=base+0x5c19210;let words=[0xd10183ff,0xa9017bfd];
        assert!(manager_matches(base,first,method,words));
        assert!(!manager_matches(base,first,method+4,words));
        for i in 0..3{let mut changed=first;changed[i]+=8;assert!(!manager_matches(base,changed,method,words));}
        for i in 0..2{let mut changed=words;changed[i]^=1;assert!(!manager_matches(base,first,method,changed));}
    }
    #[test]fn event_delegation_calls_original_once_with_both_full_width_arguments(){
        use std::sync::atomic::AtomicU64;
        static N:AtomicU32=AtomicU32::new(0);
        static A:AtomicU64=AtomicU64::new(0);
        static B:AtomicU64=AtomicU64::new(0);
        extern "C" fn original(a:u64,b:u64){A.store(a,Relaxed);B.store(b,Relaxed);N.fetch_add(1,Relaxed);}
        let mut args=Args([0;8],[0;8]);args.0[0]=0x7123456788;args.0[1]=0x7298765430;
        unsafe{delegate(original as *const()as usize as u64,&args);}
        assert_eq!(N.load(Relaxed),1);assert_eq!(A.load(Relaxed),args.0[0]);assert_eq!(B.load(Relaxed),args.0[1]);
    }
    #[test]fn event_observer_requires_exact_original_callable_and_class_identity(){
        let base=0x7100000000;
        for site in SITES{
            assert!(matches(base,site,base+site.getter,base+site.event,site.words,site.name));
            assert!(!matches(base+4096,site,base+site.getter,base+site.event,site.words,site.name));
            for i in 0..2{let mut b=site.words;b[i]^=1;assert!(!matches(base,site,base+site.getter,base+site.event,b,site.name));}
            assert!(!matches(base,site,base+site.getter+4,base+site.event,site.words,site.name));
            assert!(!matches(base,site,base+site.getter,base+site.event+4,site.words,site.name));
            assert!(!matches(base,site,base+site.getter,base+site.event,site.words,b"DifferentClass\0"));
        }
    }
}
