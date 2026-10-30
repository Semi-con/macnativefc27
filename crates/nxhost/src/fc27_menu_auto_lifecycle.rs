//! Opt-in isolated integration of the original 3D menu's playback lifecycle.
//! Original frontend events supply intent; only original queued manager methods
//! advance playback. No timers, fabricated completion events or guest field edits.
use anyhow::{ensure,Result};
use std::sync::{OnceLock,Mutex};
use crate::fc27_menu_prime_probe::Operation;
static BASE:OnceLock<u64>=OnceLock::new();
static STATE:Mutex<State>=Mutex::new(State::new());
#[derive(Clone,Copy,Debug,PartialEq,Eq)]enum Stage{Idle,PrimeQueued,StartQueued,UnprimeQueued}
struct State{wanted:bool,stage:Stage,cycles:u32,manager:u64,asset:u64,unprime_dispatched:bool,unprime_returned:bool}
impl State{
    const fn new()->Self{Self{wanted:false,stage:Stage::Idle,cycles:0,manager:0,asset:0,unprime_dispatched:false,unprime_returned:false}}
    fn next(&mut self,selected_allowed:bool,prime:bool,prep:bool,clean:bool)->Option<Operation>{
        match self.stage{
            Stage::Idle if self.wanted&&selected_allowed&&self.cycles<4=>{
                self.cycles+=1;self.stage=Stage::PrimeQueued;Some(Operation::Prime)
            },
            Stage::PrimeQueued if prime&&prep=>{
                if self.wanted{self.stage=Stage::StartQueued;Some(Operation::Start)}
                else{self.stage=Stage::UnprimeQueued;self.unprime_dispatched=false;self.unprime_returned=false;Some(Operation::Unprime)}
            },
            Stage::StartQueued if !self.wanted&&selected_allowed=>{
                self.stage=Stage::UnprimeQueued;self.unprime_dispatched=false;self.unprime_returned=false;Some(Operation::Unprime)
            },
            Stage::UnprimeQueued if self.unprime_dispatched&&self.unprime_returned&&clean=>{
                self.stage=Stage::Idle;None
            },
            _=>None,
        }
    }
}
pub(crate)fn enabled()->bool{BASE.get().is_some()}
fn copy(address:u64,bytes:&mut[u8])->bool{crate::input::test_control::config_inspection::snapshot(address,bytes)}
fn word(address:u64)->Option<u64>{let mut bytes=[0;8];copy(address,&mut bytes).then(||u64::from_le_bytes(bytes))}
fn intent(wanted:bool,source:&str){
    if !enabled(){return;}
    let mut state=STATE.lock().unwrap();
    if state.wanted!=wanted{state.wanted=wanted;eprintln!("[menu-auto] desired={wanted} source={source} stage={:?}",state.stage);}
}
pub(crate)fn hub_event(base:u64,object:u64,event:u32){
    if BASE.get()!=Some(&base)||word(object)!=Some(base+0xc8b5018){return;}
    let Some(data)=word(object+0x38)else{return;};let mut background=[0];
    if !copy(data+0x24,&mut background)||background!=[0]{return;}
    match event{
        0x9000e8bd=>intent(true,"original foreground FIFAHub initialization"),
        0x6c69d606=>intent(false,"original foreground FIFAHub exit"),
        _=>{},
    }
}
pub(crate)fn command(index:usize){
    match index{0=>intent(true,"original loadMainMenuScreen"),1=>intent(false,"original unloadMainMenuStadium"),_=>{}}
}
pub(crate)fn master_event(event:u32){
    if enabled()&&event==0x55c23f4c{
        let mut state=STATE.lock().unwrap();
        if state.stage==Stage::UnprimeQueued{state.unprime_dispatched=true;}
    }
}
pub(crate)fn unprime_returned(manager:u64,asset:u64){
    if !enabled(){return;}
    let mut state=STATE.lock().unwrap();
    if state.stage==Stage::UnprimeQueued&&state.manager==manager&&state.asset==asset{state.unprime_returned=true;}
}
pub(crate)fn apply(modules:&[crate::loader::Module])->Result<()>{
    if !std::env::var("NX_FC27_MENU_AUTO_LIFECYCLE").is_ok_and(|v|v=="1"){return Ok(());}
    ensure!(std::env::current_exe()?.file_name().is_some_and(|v|v=="FC27-test")&&std::env::var_os("NX_TEST_CONTROL").is_some(),"automatic menu experiment requires isolated test process");
    for variable in ["NX_FC27_MENU_PRIME_PROBE","NX_FC27_MENU_EVENT_TRACE","NX_FC27_MENU_SHADOW_PARENT_BINDINGS","NX_FC27_MENU_AMBASSADOR_SOURCE"]{
        ensure!(std::env::var(variable).is_ok_and(|v|v=="1"),"automatic menu experiment missing {variable}");
    }
    let m=modules.iter().find(|m|m.name=="main").ok_or_else(||anyhow::anyhow!("missing main"))?;
    ensure!(m.module_id.iter().map(|v|format!("{v:02x}")).collect::<String>()=="fb81983aa1cc3a2e97402497e2ca652800000000000000000000000000000000"
        &&crate::menu_cinematic_diagnostics::version(m.base)&&crate::fc27_menu_prime_probe::method_versions(m.base),"automatic menu original build/method mismatch");
    ensure!(BASE.set(m.base).is_ok(),"automatic menu already installed");
    eprintln!("[menu-auto] armed original frontend intent and queued master lifecycle; max4 cycles; real Prime/Prep, original instance UnPrime return and cleared selection gates; manual playback transport disabled");Ok(())
}
pub(crate)fn poll(){
    let Some(&base)=BASE.get()else{return;};
    if crate::guest::try_current().is_none(){return;}
    let Some(manager)=word(base+0xd44cac0).filter(|v|word(*v)==Some(base+0xc885828))else{return;};
    let Some((asset,name))=crate::menu_cinematic_diagnostics::master_name(manager,base)else{return;};
    // The isolated resource candidate explicitly enables the original automatic
    // preparation path. Do not silently start an asset needing another route.
    let mut automatic=[0];if !copy(asset+0x9d,&mut automatic)||automatic!=[1]{return;}
    let mut selected=[0;4];if !copy(manager+0x1884,&mut selected){return;}
    let selected=u32::from_le_bytes(selected);
    let menu_selected=selected==0||crate::menu_cinematic_diagnostics::registered_menu_asset(manager,base,selected).is_some();
    let (prime,prep)=crate::fc27_menu_event_observation::master_readiness();
    let mut state=STATE.lock().unwrap();
    if state.stage!=Stage::Idle&&(state.manager!=manager||state.asset!=asset){return;}
    if state.stage==Stage::PrimeQueued&&selected!=0&&selected!=610392389{return;}
    if state.stage==Stage::Idle{
        if selected!=0{return;}
        state.manager=manager;state.asset=asset;
    }
    let previous=state.stage;
    let clean=selected==0;
    let operation=state.next(menu_selected,prime,prep,clean);
    if previous==Stage::UnprimeQueued&&state.stage==Stage::Idle{
        crate::fc27_menu_event_observation::reset_master_readiness();
        eprintln!("[menu-auto] original master UnPrime dispatched and instance method returned; selection empty; cycle{} ready for later original Prime. No GPU/IO resource-drain claim.",state.cycles);
    }
    if operation==Some(Operation::Prime){
        crate::fc27_menu_event_observation::reset_master_readiness();
    }
    let cycle=state.cycles;drop(state);
    if let Some(operation)=operation{
        eprintln!("[menu-auto] cycle={cycle} queue original {} selected={selected:#x}",operation.label());
        unsafe{crate::fc27_menu_prime_probe::queue_operation(operation,base+operation.method(),manager,name);}
    }
}
#[cfg(test)]mod tests{
    use super::*;
    #[test]fn original_completion_and_dispatched_cleanup_gate_every_cycle(){
        let mut s=State::new();assert_eq!(s.next(true,false,false,true),None);
        s.wanted=true;assert_eq!(s.next(false,false,false,true),None);
        assert_eq!(s.next(true,false,false,true),Some(Operation::Prime));
        assert_eq!(s.next(true,true,false,false),None);
        assert_eq!(s.next(true,false,true,false),None);
        assert_eq!(s.next(true,true,true,false),Some(Operation::Start));
        assert_eq!(s.next(true,true,true,false),None);
        s.wanted=false;assert_eq!(s.next(false,true,true,false),None);
        assert_eq!(s.next(true,true,true,false),Some(Operation::Unprime));
        assert_eq!(s.next(true,true,true,true),None);assert_eq!(s.stage,Stage::UnprimeQueued);
        s.unprime_dispatched=true;assert_eq!(s.next(true,true,true,true),None);assert_eq!(s.stage,Stage::UnprimeQueued);
        s.unprime_returned=true;assert_eq!(s.next(true,true,true,false),None);assert_eq!(s.stage,Stage::UnprimeQueued);
        s.wanted=true;assert_eq!(s.next(true,true,true,true),None);assert_eq!(s.stage,Stage::Idle);
        assert_eq!(s.next(true,false,false,true),Some(Operation::Prime));assert_eq!(s.cycles,2);
    }
    #[test]fn leaving_during_preparation_waits_for_real_completion_then_only_unprimes(){
        let mut s=State::new();s.wanted=true;assert_eq!(s.next(true,false,false,true),Some(Operation::Prime));
        s.wanted=false;assert_eq!(s.next(true,true,false,false),None);
        assert_eq!(s.next(true,true,true,false),Some(Operation::Unprime));
        s.unprime_dispatched=true;s.unprime_returned=true;s.next(true,true,true,true);
        s.cycles=4;s.wanted=true;assert_eq!(s.next(true,false,false,true),None);
    }
}
