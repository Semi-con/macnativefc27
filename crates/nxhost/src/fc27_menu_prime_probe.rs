//! Explicit bounded experiment: original Prime/Prep/Start/Unprime for the fixed master,
//! each once, gated by actual original completion events in the test process.
use std::{path::Path,sync::{OnceLock,atomic::{AtomicU32,Ordering::Relaxed}},time::{SystemTime,UNIX_EPOCH}};
use anyhow::{Result,ensure};
use serde_json::json;
static BASE:OnceLock<u64>=OnceLock::new();
static PHASE:AtomicU32=AtomicU32::new(0);
const MASTER:&str="f26_mm_master_01";
#[derive(Clone,Copy,Debug,PartialEq,Eq)]pub(crate) enum Operation{Prime,Prep,Start,Unprime}
impl Operation{
    fn phase(self)->u32{match self{Self::Prime=>0,Self::Prep=>1,Self::Start=>2,Self::Unprime=>3}}
    pub(crate) fn label(self)->&'static str{match self{Self::Prime=>"RequestPrime",Self::Prep=>"RequestPrep",Self::Start=>"RequestStart",Self::Unprime=>"RequestUnPrime"}}
    pub(crate) fn method(self)->u64{match self{Self::Prime=>0x5c19040,Self::Prep=>0x1baa7b0,Self::Start=>0x5c19180,Self::Unprime=>0x1baad60}}
}
fn ready(operation:Operation,phase:u32,prime_complete:bool,prep_complete:bool)->bool{
    match operation{
        Operation::Prime=>phase==0&&!prime_complete&&!prep_complete,
        Operation::Prep=>phase==1&&prime_complete&&!prep_complete,
        // AutomaticPrepSchematics can complete preparation after Prime without
        // a separate RequestPrep. The original completion event is required in
        // both paths; do not re-prepare an already prepared scene.
        Operation::Start=>matches!(phase,1|2)&&prime_complete&&prep_complete,
        Operation::Unprime=>phase==3&&prime_complete&&prep_complete,
    }
}
fn selected_allowed(operation:Operation,selected:u32,typed_menu_child:bool)->bool{
    selected==0||(operation!=Operation::Prime&&selected==610392389)
        ||(operation==Operation::Unprime&&typed_menu_child)
}

fn read(address:u64,bytes:&mut[u8])->bool{crate::input::test_control::config_inspection::snapshot(address,bytes)}
fn word(bytes:&[u8],at:usize)->Option<u64>{Some(u64::from_le_bytes(bytes.get(at..at+8)?.try_into().ok()?))}
fn pointer(address:u64)->Option<u64>{let mut bytes=[0;8];read(address,&mut bytes).then(||u64::from_le_bytes(bytes))}
fn parse(text:&str,now:u128)->Option<(Operation,u64,u64)>{
    let mut parts=text.split_whitespace();
    let operation=match parts.next()?{"prime"=>Operation::Prime,"prep"=>Operation::Prep,"start"=>Operation::Start,"unprime"=>Operation::Unprime,_=>return None};
    let sequence=parts.next()?.parse::<u64>().ok()?;
    let entity=u64::from_str_radix(parts.next()?.strip_prefix("0x")?,16).ok()?;
    (sequence>0&&entity!=0&&entity%8==0&&parts.next().is_none()
        &&now.checked_sub(sequence as u128).is_some_and(|age|age<=15_000_000_000))
        .then_some((operation,sequence,entity))
}
fn master_fields(base:u64,entity:&[u8],data:&[u8],name:&[u8])->Option<(u64,u64)>{
    if !(word(entity,0)==Some(base+0xc5c3a20)&&word(data,0)==Some(base+0xc5c59b8)
        &&word(data,8)==Some(base+0xd44d6f8)
        &&data.get(0x28..0x2c)==Some(610392389u32.to_le_bytes().as_slice())
        &&crate::menu_cinematic_diagnostics::selected_pointer_name(name)==Some(MASTER)){return None;}
    Some((word(entity,0x30)?,word(data,0x30)?))
}
unsafe fn queue_prime(method:u64,manager:u64,name:u64){
    let call:extern "C" fn(u64,u64,u64,u64,u32)=unsafe{std::mem::transmute(method as usize)};
    call(manager,name,0,0,0);
}
pub(crate) unsafe fn queue_operation(operation:Operation,method:u64,manager:u64,name:u64){
    match operation{
        Operation::Prime=>unsafe{queue_prime(method,manager,name);},
        Operation::Prep|Operation::Unprime=>{
            let call:extern "C" fn(u64,u64,u32)=unsafe{std::mem::transmute(method as usize)};
            call(manager,name,0);
        },
        Operation::Start=>{
            let call:extern "C" fn(u64,u64,u64,u32)=unsafe{std::mem::transmute(method as usize)};
            call(manager,name,0,0);
        },
    }
}
pub(crate) fn method_versions(base:u64)->bool{
    [(0x5c19040,0xa9015ff8a9bc7bfd),(0x1baa7b0,0xf9000bf7a9bc7bfd),(0x5c19180,0xa90157f6a9bd7bfd),(0x1baad60,0xf9000bf7a9bc7bfd)]
        .into_iter().all(|(at,words)|crate::diag::read_words::<1>(base+at)==Some([words]))
}
pub(crate) fn apply(modules:&[crate::loader::Module])->Result<()>{
    if !std::env::var("NX_FC27_MENU_PRIME_PROBE").is_ok_and(|v|v=="1"){return Ok(());}
    ensure!(std::env::var_os("NX_TEST_CONTROL").is_some()
        &&std::env::var("NX_FC27_VIDEO_TRACE").is_ok_and(|v|v=="1")
        &&std::env::var("NX_FC27_MENU_EVENT_TRACE").is_ok_and(|v|v=="1")
        &&std::env::var("NX_FC27_MENU_SEQUENCE").is_ok_and(|v|v=="g4"),"menu prime probe requires isolated observed G4 experiment");
    ensure!(std::env::current_exe()?.file_name().is_some_and(|v|v=="FC27-test"),"menu prime probe requires FC27-test");
    let m=modules.iter().find(|m|m.name=="main").ok_or_else(||anyhow::anyhow!("missing main"))?;
    let id:String=m.module_id.iter().map(|v|format!("{v:02x}")).collect();
    ensure!(id=="fb81983aa1cc3a2e97402497e2ca652800000000000000000000000000000000","menu prime probe build mismatch");
    ensure!(crate::menu_cinematic_diagnostics::version(m.base)
        &&method_versions(m.base),"menu probe original method guard mismatch");
    ensure!(BASE.set(m.base).is_ok(),"menu prime probe already installed");
    eprintln!("[menu-prime-probe] armed fixed master Prime/Prep/Start/Unprime once each; completion-event gates, original queued methods, no raw field edits");
    Ok(())
}
pub(crate) fn poll(control:&Path){
    if crate::fc27_menu_auto_lifecycle::enabled(){return;}
    let Some(&base)=BASE.get()else{return;};
    let request=control.with_extension("menu-prime-probe");
    let Some(text)=crate::input::test_control::read(&request)else{return;};
    let _=std::fs::remove_file(request);
    let now=SystemTime::now().duration_since(UNIX_EPOCH).map_or(0,|v|v.as_nanos());
    let Some((operation,sequence,entity_address))=parse(&text,now)else{return;};
    let result=(||->Result<(u64,u64)>{
        ensure!(crate::guest::try_current().is_some(),"requires existing guest thread context");
        ensure!(crate::menu_cinematic_diagnostics::version(base)
            &&method_versions(base),"original version changed");
        let mut entity=[0;64];ensure!(read(entity_address,&mut entity),"entity unreadable");
        let data_address=word(&entity,0x38).ok_or_else(||anyhow::anyhow!("missing data"))?;
        let mut data=[0;64];ensure!(read(data_address,&mut data),"data unreadable");
        let name_address=word(&data,0x30).ok_or_else(||anyhow::anyhow!("missing instance name"))?;
        let mut name=[0;96];ensure!(read(name_address,&mut name),"instance name unreadable");
        let (owner,name_address)=master_fields(base,&entity,&data,&name).ok_or_else(||anyhow::anyhow!("fixed master entity/data/hash/name guard mismatch"))?;
        ensure!(pointer(owner)==Some(base+0xc6641f8),"original scene owner mismatch");
        let manager=pointer(base+0xd44cac0).ok_or_else(||anyhow::anyhow!("missing manager"))?;
        let asset=crate::menu_cinematic_diagnostics::registered_master_asset(manager,base).ok_or_else(||anyhow::anyhow!("master not in original typed manager index"))?;
        let mut selected=[0;4];ensure!(read(manager+0x1884,&mut selected),"manager selection unreadable");
        let selected=u32::from_le_bytes(selected);
        // A running master selects its authored menu children in1884. A
        // finite Unprime of our previously started master may clean up those
        // exact typed registered menu assets, while other cinematics still
        // block this probe. No broader selection allowance for Prime/Start.
        let child=operation==Operation::Unprime&&crate::menu_cinematic_diagnostics::registered_menu_asset(manager,base,selected).is_some();
        ensure!(selected_allowed(operation,selected,child),"another cinematic already selected");
        let (prime_complete,prep_complete)=crate::fc27_menu_event_observation::master_readiness();
        let phase=PHASE.load(Relaxed);
        ensure!(ready(operation,phase,prime_complete,prep_complete),"original completion event/sequence prerequisite unmet");
        ensure!(PHASE.compare_exchange(phase,operation.phase()+1,Relaxed,Relaxed).is_ok(),"operation already submitted");
        // Null optional listener/options, false queues each original request;
        // only the original manager changes cinematic state.
        eprintln!("[menu-prime-probe] queue original {} for {MASTER}; guest={:?}",operation.label(),crate::guest::try_current().map(|v|v.handle));
        unsafe{queue_operation(operation,base+operation.method(),manager,name_address);}
        Ok((manager,asset))
    })();
    let report=match result{
        Ok((manager,asset))=>json!({"sequence":sequence,"accepted":true,"operation":operation.label(),"instance_name":MASTER,"manager":format!("{manager:#x}"),"registered_asset":format!("{asset:#x}"),"scope":"One queued original operation per stage; later stages require actual original completion events. No raw guest field edits or playback success claim. Isolated copied profile only."}),
        Err(error)=>json!({"sequence":sequence,"accepted":false,"error":error.to_string()}),
    };
    let _=std::fs::write(control.with_extension(format!("menu-prime-probe-{sequence}.json")),report.to_string());
}
#[cfg(test)]mod tests{
    use super::*;
    #[test]fn queue_uses_exact_manager_name_and_zero_optional_arguments_once(){
        use std::sync::atomic::{AtomicU32,AtomicU64};
        static COUNT:AtomicU32=AtomicU32::new(0);
        static MANAGER:AtomicU64=AtomicU64::new(0);
        static NAME:AtomicU64=AtomicU64::new(0);
        extern "C" fn original(manager:u64,name:u64,listener:u64,options:u64,synchronous:u32){
            assert_eq!((listener,options,synchronous),(0,0,0));MANAGER.store(manager,Relaxed);NAME.store(name,Relaxed);COUNT.fetch_add(1,Relaxed);
        }
        unsafe{queue_prime(original as*const()as usize as u64,0x7502cfdd80,0x750800f080);}
        assert_eq!(COUNT.load(Relaxed),1);assert_eq!(MANAGER.load(Relaxed),0x7502cfdd80);assert_eq!(NAME.load(Relaxed),0x750800f080);
    }
    #[test]fn only_recent_bounded_exact_transport_is_accepted(){
        let now=20_000_000_000u128;
        for (name,operation)in [("prime",Operation::Prime),("prep",Operation::Prep),("start",Operation::Start),("unprime",Operation::Unprime)]{
            assert_eq!(parse(&format!("{name} 20000000000 0x1000"),now),Some((operation,20_000_000_000,0x1000)));
        }
        assert!(parse("prime 5000000000 0x1000",now).is_some());
        for bad in ["prime 0 0x1000","prime 4999999999 0x1000","prime 20000000001 0x1000","prime 20000000000 0x1001","prime 20000000000 0x0","prime 20000000000 1000","prime 20000000000 0x1000 extra","20000000000 0x1000","activate 20000000000 0x1000"]{assert!(parse(bad,now).is_none(),"{bad}");}
    }
    #[test]fn original_completion_notifications_are_required_in_order(){
        assert!(ready(Operation::Prime,0,false,false));
        assert!(ready(Operation::Prep,1,true,false));assert!(ready(Operation::Start,2,true,true));
        assert!(!ready(Operation::Prep,0,true,false));assert!(!ready(Operation::Prep,1,false,false));
        assert!(ready(Operation::Start,1,true,true)); // Observed original automatic preparation.
        assert!(!ready(Operation::Start,1,true,false));assert!(!ready(Operation::Start,2,true,false));
        assert!(!ready(Operation::Prep,1,true,true));
        assert!(!ready(Operation::Start,2,false,true));assert!(!ready(Operation::Start,3,true,true));
        assert!(!ready(Operation::Prime,0,true,false));assert!(!ready(Operation::Prime,1,false,false));
        assert!(ready(Operation::Unprime,3,true,true));assert!(!ready(Operation::Unprime,2,true,true));assert!(!ready(Operation::Unprime,4,true,true));assert!(!ready(Operation::Unprime,3,false,true));
    }
    #[test]fn unprime_allows_only_verified_menu_descendants_and_keeps_other_stage_guards(){
        let child=2190595582;
        assert!(selected_allowed(Operation::Unprime,child,true));assert!(!selected_allowed(Operation::Unprime,child,false));
        for op in [Operation::Prime,Operation::Prep,Operation::Start]{assert!(!selected_allowed(op,child,true));}
        assert!(!selected_allowed(Operation::Unprime,0xfedcba98,false));assert!(selected_allowed(Operation::Unprime,610392389,false));assert!(selected_allowed(Operation::Unprime,0,false));
    }
    #[test]fn prep_and_start_use_original_integer_argument_positions(){
        static CALLS:AtomicU32=AtomicU32::new(0);
        extern "C" fn prep(manager:u64,name:u64,synchronous:u32){
            assert_eq!((manager,name,synchronous),(0x7502cfdd80,0x750800f080,0));CALLS.fetch_add(1,Relaxed);
        }
        extern "C" fn start(manager:u64,name:u64,options:u64,synchronous:u32){
            assert_eq!((manager,name,options,synchronous),(0x7502cfdd80,0x750800f080,0,0));CALLS.fetch_add(1,Relaxed);
        }
        unsafe{
            queue_operation(Operation::Prep,prep as*const()as usize as u64,0x7502cfdd80,0x750800f080);
            queue_operation(Operation::Start,start as*const()as usize as u64,0x7502cfdd80,0x750800f080);
            queue_operation(Operation::Unprime,prep as*const()as usize as u64,0x7502cfdd80,0x750800f080);
        }
        assert_eq!(CALLS.load(Relaxed),3);
    }
    #[test]fn mismatched_entity_data_hash_or_name_never_yields_a_target(){
        let base=0x7100000000u64;let mut entity=[0;64];let mut data=[0;64];
        entity[..8].copy_from_slice(&(base+0xc5c3a20).to_le_bytes());entity[0x30..0x38].copy_from_slice(&0x1000u64.to_le_bytes());
        data[..8].copy_from_slice(&(base+0xc5c59b8).to_le_bytes());data[8..16].copy_from_slice(&(base+0xd44d6f8).to_le_bytes());data[0x28..0x2c].copy_from_slice(&610392389u32.to_le_bytes());data[0x30..0x38].copy_from_slice(&0x2000u64.to_le_bytes());
        let name=b"f26_mm_master_01\0";
        assert_eq!(master_fields(base,&entity,&data,name),Some((0x1000,0x2000)));
        assert!(master_fields(base+4096,&entity,&data,name).is_none());
        for at in [0usize,8,0x28]{let mut changed=data;changed[at]^=1;assert!(master_fields(base,&entity,&changed,name).is_none());}
        assert!(master_fields(base,&entity[..32],&data,name).is_none());
        assert!(master_fields(base,&entity,&data,b"f26_mm_master_01").is_none());
        assert!(master_fields(base,&entity,&data,b"f26_mm_mlkr_plyr_01\0").is_none());
    }
}
