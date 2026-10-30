// Explicit local tests only: three reflected fields, bounded values/duration,
// version and object identity checks, and independent conditional rollback.
use std::{path::Path,sync::{LazyLock,Mutex,Condvar,OnceLock},time::{Duration,Instant}};
#[derive(Clone,Copy,Debug,PartialEq)]
enum Field {MaxSim,ForceSim,Interval}
impl Field {
    fn parse(s:&str)->Option<Self> {match s {"sim-max-fps"=>Some(Self::MaxSim),"sim-force-rate"=>Some(Self::ForceSim),"display-interval"=>Some(Self::Interval),_=>None}}
    fn bits(self,s:&str)->Option<u32> {match (self,s) {(Self::MaxSim,"30")=>Some(30f32.to_bits()),(Self::MaxSim,"60")=>Some(60f32.to_bits()),(Self::ForceSim,"30")=>Some(30),(Self::ForceSim,"60")=>Some(60),(Self::Interval,"1")=>Some(1),(Self::Interval,"2")=>Some(2),_=>None}}
    fn offset(self)->u64 {match self {Self::MaxSim=>0x34,Self::ForceSim=>0x38,Self::Interval=>0x78}}
    fn identity(self,base:u64)->(u64,u64) {match self {Self::Interval=>(base+0xcc415c8,base+0xda01b88),_=>(base+0xc5dd148,base+0xdda3268)}}
}
enum Command {Edit{sequence:u64,field:Field,object:u64,old:u32,new:u32,hold:Duration},Reset(u64)}
fn parse(s:&str)->Option<Command> {
    let mut p=s.split_whitespace();let command=p.next()?;let sequence=p.next()?.parse::<u64>().ok()?;if sequence==0 {return None;}
    if command=="reset" {return p.next().is_none().then_some(Command::Reset(sequence));}
    if command!="edit" {return None;}
    let field=Field::parse(p.next()?)?;let object=u64::from_str_radix(p.next()?.strip_prefix("0x")?,16).ok()?;
    let old=field.bits(p.next()?)?;let new=field.bits(p.next()?)?;let ms=p.next()?.parse::<u64>().ok()?;
    if object%8!=0 || object==0 || object.checked_add(0x140).is_none() || old==new || !(100..=120000).contains(&ms) || p.next().is_some() {return None;}
    Some(Command::Edit{sequence,field,object,old,new,hold:Duration::from_millis(ms)})
}
fn copy(address:u64,bytes:&mut[u8])->bool {super::super::config_inspection::snapshot(address,bytes)}
fn write32(address:u64,value:u32)->bool {
    if address%4!=0 {return false;}
    unsafe extern "C" {static mach_task_self_:u32;fn mach_vm_write(task:u32,address:u64,data:u64,count:u32)->i32;}
    let bytes=value.to_le_bytes();unsafe {mach_vm_write(mach_task_self_,address,bytes.as_ptr() as u64,4)==0}
}
struct Patch {sequence:u64,object:u64,field:Field,header:[u8;32],old:u32,new:u32,until:Instant}
fn matches(p:&Patch,value:u32)->bool {
    let mut header=[0u8;32];let mut field=[0u8;4];
    copy(p.object,&mut header)&&header==p.header&&copy(p.object+p.field.offset(),&mut field)&&u32::from_le_bytes(field)==value
}
fn restore(p:&Patch)->bool {matches(p,p.new)&&write32(p.object+p.field.offset(),p.old)&&matches(p,p.old)}
fn apply(sequence:u64,object:u64,field:Field,base:u64,old:u32,new:u32,hold:Duration)->Result<Patch,&'static str> {
    let mut header=[0u8;32];if !copy(object,&mut header) {return Err("object read failed");}
    let (vtable,type_info)=field.identity(base);
    if super::word(&header,0)!=Some(vtable)||super::word(&header,8)!=Some(type_info)||super::uint(&header,0x18)!=Some(11) {return Err("object vtable or type identity mismatch");}
    let patch=Patch{sequence,object,field,header,old,new,until:Instant::now()+hold};
    if !matches(&patch,old) {return Err("expected field value changed");}
    if !write32(object+field.offset(),new) {return Err("OS field write failed");}
    if !matches(&patch,new) {let _=restore(&patch);return Err("field write verification failed");}
    Ok(patch)
}
#[derive(Default)]
struct State {last:u64,accepted:u32,patches:Vec<Patch>}
static JOURNAL:LazyLock<(Mutex<State>,Condvar)>=LazyLock::new(||(Mutex::new(State::default()),Condvar::new()));
fn worker()->bool {
    static STARTED:OnceLock<bool>=OnceLock::new();
    *STARTED.get_or_init(||std::thread::Builder::new().name("frame-setting-expiry".into()).spawn(|| {
        let (mutex,changed)=&*JOURNAL;let mut state=mutex.lock().unwrap();
        loop {
            let now=Instant::now();let mut at=0;
            while at<state.patches.len() {
                if state.patches[at].until<=now {let patch=state.patches.remove(at);let restored=restore(&patch);eprintln!("[frame-settings-edit] expired sequence={} field={:?} restored={restored}",patch.sequence,patch.field);} else {at+=1;}
            }
            if let Some(until)=state.patches.iter().map(|p|p.until).min() {state=changed.wait_timeout(state,until.saturating_duration_since(Instant::now())).unwrap().0;} else {state=changed.wait(state).unwrap();}
        }
    }).is_ok())
}
fn report(control:&Path,sequence:u64,accepted:bool,message:&str) {
    let text=format!(r#"{{"sequence":{sequence},"accepted":{accepted},"message":"{message}","scope":"Test process memory only; no game files, keys, clocks or saves edited. Conditional rollback checks object header and current value."}}"#);
    let _=std::fs::write(control.with_extension(format!("edit-{sequence}.json")),text);
    eprintln!("[frame-settings-edit] sequence={sequence} accepted={accepted} {message}");
}
pub(super) fn poll(control:&Path) {
    let path=control.with_extension("setting-edit");let Some(text)=super::super::read(&path) else {return;};let _=std::fs::remove_file(path);
    let Some(command)=parse(&text) else {eprintln!("[frame-settings-edit] rejected invalid bounded command");return;};
    let sequence=match command {Command::Edit{sequence,..}|Command::Reset(sequence)=>sequence};
    let (mutex,changed)=&*JOURNAL;let mut state=mutex.lock().unwrap();
    if sequence<=state.last {report(control,sequence,false,"stale sequence");return;};state.last=sequence;
    match command {
        Command::Reset(_)=>{
            let mut ok=true;for patch in state.patches.drain(..) {let restored=restore(&patch);eprintln!("[frame-settings-edit] reset sequence={} field={:?} restored={restored}",patch.sequence,patch.field);ok&=restored;}
            changed.notify_all();report(control,sequence,ok,if ok {"restored original values"}else{"some values changed or object disappeared; rollback skipped to preserve newer state"});
        },
        Command::Edit{field,object,old,new,hold,..}=>{
            if state.accepted>=16||state.patches.len()>=3||state.patches.iter().any(|p|p.object==object&&p.field==field) {report(control,sequence,false,"mutation limit or field already overridden");return;}
            let Some(base)=super::version_base() else {report(control,sequence,false,"original build signatures mismatch");return;};
            let space=crate::kernel::aspace();let (start,size,block)=space.map.lock().unwrap().query(object);
            if block.state!=crate::mem::state::NORMAL||block.perm&(crate::mem::perm::R|crate::mem::perm::W)!=(crate::mem::perm::R|crate::mem::perm::W)||object+0x140>start+size {report(control,sequence,false,"object is not inside one writable guest NORMAL region");return;}
            if !worker() {report(control,sequence,false,"expiry worker unavailable; no write performed");return;}
            match apply(sequence,object,field,base,old,new,hold) {Ok(patch)=>{state.patches.push(patch);state.accepted+=1;changed.notify_all();report(control,sequence,true,"typed field temporarily changed and verified; independent rollback armed");},Err(message)=>report(control,sequence,false,message)}
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn only_three_typed_fields_and_bounded_values_addresses_and_durations_are_accepted() {
        assert!(parse("edit 1 sim-max-fps 0x7200000000 30 60 45000").is_some());assert!(parse("edit 2 display-interval 0x7200000000 2 1 1000").is_some());assert!(parse("reset 3").is_some());
        for bad in ["edit 0 sim-max-fps 0x7200000000 30 60 1000","edit 1 arbitrary 0x7200000000 30 60 1000","edit 1 sim-max-fps 0x7200000001 30 60 1000","edit 1 sim-max-fps 0xfffffffffffffff8 30 60 1000","edit 1 sim-force-rate 0x7200000000 30 120 1000","edit 1 sim-max-fps 0x7200000000 NaN 60 1000","edit 1 display-interval 0x7200000000 2 0 1000","edit 1 display-interval 0x7200000000 2 1 120001","edit 1 display-interval 0x7200000000 2 1 99","edit 1 sim-max-fps 0x7200000000 30 30 1000","reset 3 extra"] {assert!(parse(bad).is_none(),"{bad}");}
    }
    #[test]
    fn actual_os_field_write_and_conditional_restore_keep_neighbor_guards_and_newer_values() {
        let base=0x7100000000;let mut storage=[0xfeedfaceu32;0x60];let object=unsafe {storage.as_mut_ptr().add(8)} as u64;let field=Field::MaxSim;let (vtable,kind)=field.identity(base);
        let mut header=[0u8;32];header[..8].copy_from_slice(&vtable.to_le_bytes());header[8..16].copy_from_slice(&kind.to_le_bytes());header[0x18..0x1c].copy_from_slice(&11u32.to_le_bytes());
        unsafe {std::ptr::copy_nonoverlapping(header.as_ptr(),object as *mut u8,32)};
        let address=object+field.offset();assert!(write32(address,30f32.to_bits()));let before=storage;
        let patch=apply(1,object,field,base,30f32.to_bits(),60f32.to_bits(),Duration::from_secs(1)).unwrap();assert!(matches(&patch,60f32.to_bits()));assert_eq!(&storage[..8],&before[..8]);assert_eq!(&storage[22..],&before[22..]);assert!(restore(&patch));assert_eq!(storage,before);
        let patch=apply(2,object,field,base,30f32.to_bits(),60f32.to_bits(),Duration::from_secs(1)).unwrap();assert!(write32(address,45f32.to_bits()));assert!(!restore(&patch));let mut value=[0;4];assert!(copy(address,&mut value));assert_eq!(u32::from_le_bytes(value),45f32.to_bits());
        assert!(!write32(0x121300000500,1));assert!(!write32(address+1,1));
    }
}
