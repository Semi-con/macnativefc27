// Original frame limiter: main+99e3ef0 reads a double frequency at object+8.
// Explicit tests only. The object comes from the verified original global,
// never from an arbitrary caller address. No system clock or file is changed.
use std::{path::Path,sync::{LazyLock,Mutex},time::Duration};
use crate::engine_frame_target::{ROOT,read,write,base,snapshot,Snapshot};
#[derive(Clone,Copy)]
struct Patch {sequence:u64,root_slot:u64,object:u64,callback:u64,old:u64,new:u64}
fn matches(p:&Patch,value:u64)->bool {read(p.root_slot)==Some(p.object)&&read(p.object+0x4f8)==Some(p.callback)&&read(p.object+8)==Some(value)}
fn restore(p:&Patch)->bool {matches(p,p.new)&&write(p.object+8,p.old)&&matches(p,p.old)}
enum Command {Read(u64),Reset(u64),Edit{sequence:u64,old:u64,new:u64,hold:Duration}}
fn parse(text:&str)->Option<Command> {
    let mut p=text.split_whitespace();let command=p.next()?;let sequence=p.next()?.parse::<u64>().ok()?;if sequence==0 {return None;}
    if command=="read"||command=="reset" {return p.next().is_none().then_some(if command=="read" {Command::Read(sequence)}else{Command::Reset(sequence)});}
    if command!="edit" {return None;}
    let convert=|s:&str|match s {"30"=>Some(30f64.to_bits()),"60"=>Some(60f64.to_bits()),_=>None};
    let old=convert(p.next()?)?;let new=convert(p.next()?)?;let ms=p.next()?.parse::<u64>().ok()?;
    if old==new||!(100..=120000).contains(&ms)||p.next().is_some() {return None;}
    Some(Command::Edit{sequence,old,new,hold:Duration::from_millis(ms)})
}
#[derive(Default)]
struct State {last:u64,count:u32,patch:Option<Patch>}
static STATE:LazyLock<Mutex<State>>=LazyLock::new(||Mutex::new(State::default()));
fn report(control:&Path,sequence:u64,accepted:bool,message:&str,details:Option<(u64,Snapshot)>) {
    let extra=details.map_or(String::new(),|(base,s)|format!(r#", "object":"{:#x}","target_fps":{},"clock_callback_rva":"{:#x}""#,s.object,f64::from_bits(s.fps),s.callback-base));
    let text=format!(r#"{{"sequence":{sequence},"accepted":{accepted},"message":"{message}"{extra},"scope":"Test process original frame-frequency field only. Guest system-clock counters, game files and saves are unchanged. Root pointer/current field/callback guard conditional rollback."}}"#);
    let _=std::fs::write(control.with_extension(format!("limiter-{sequence}.json")),text);eprintln!("[engine-limiter] sequence={sequence} accepted={accepted} {message}");
}
pub(super) fn poll(control:&Path) {
    let path=control.with_extension("limiter");let Some(text)=super::read(&path) else {return;};let _=std::fs::remove_file(path);let Some(command)=parse(&text) else {return;};
    let sequence=match command {Command::Read(s)|Command::Reset(s)|Command::Edit{sequence:s,..}=>s};
    let mut state=STATE.lock().unwrap();if sequence<=state.last {report(control,sequence,false,"stale request",None);return;};state.last=sequence;
    let Some(base)=base() else {report(control,sequence,false,"original limiter signatures mismatch",None);return;};
    match command {
        Command::Read(_)=>{let value=snapshot(base);report(control,sequence,value.is_some(),if value.is_some(){"verified original limiter object"}else{"limiter object unavailable"},value.map(|s|(base,s)));},
        Command::Reset(_)=>{let ok=state.patch.take().is_none_or(|p|restore(&p));report(control,sequence,ok,if ok{"original target restored"}else{"object or value changed; rollback skipped"},snapshot(base).map(|s|(base,s)));},
        Command::Edit{old,new,hold,..}=>{
            if state.count>=8||state.patch.is_some() {report(control,sequence,false,"mutation limit or already overridden",None);return;}
            let Some(s)=snapshot(base).filter(|s|s.fps==old) else {report(control,sequence,false,"expected original target or object mismatch",None);return;};
            let p=Patch{sequence,root_slot:base+ROOT,object:s.object,callback:s.callback,old,new};
            if !matches(&p,old)||!write(p.object+8,new)||!matches(&p,new) {let _=restore(&p);report(control,sequence,false,"target write verification failed",None);return;}
            state.patch=Some(p);state.count+=1;
            if std::thread::Builder::new().name("engine-target-expiry".into()).spawn(move|| {
                std::thread::sleep(hold);let mut state=STATE.lock().unwrap();if state.patch.is_some_and(|v|v.sequence==sequence) {let restored=restore(&p);state.patch=None;eprintln!("[engine-limiter] expired sequence={sequence} restored={restored}");}
            }).is_err() {let _=restore(&p);state.patch=None;report(control,sequence,false,"expiry thread unavailable; target reverted",None);return;}
            report(control,sequence,true,"temporary engine target verified; rollback armed",snapshot(base).map(|s|(base,s)));
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn engine_target_transport_accepts_only_30_60_and_bounded_expiry() {
        assert!(parse("read 1").is_some());assert!(parse("reset 2").is_some());assert!(parse("edit 3 30 60 45000").is_some());
        for s in ["read 0","reset 1 extra","edit 1 30 120 1000","edit 1 30 30 1000","edit 1 NaN 60 1000","edit 1 30 60 99","edit 1 30 60 120001","edit 1 30 60 1000 extra"] {assert!(parse(s).is_none());}
    }
    #[test]
    fn actual_64bit_target_write_restores_conditionally_without_touching_guards() {
        let mut object=[0xfeedfacecafebabeu64;0xb0];let object_address=object.as_mut_ptr() as u64;let mut root=object_address;object[1]=30f64.to_bits();object[0x4f8/8]=0x7100000100;
        let p=Patch{sequence:1,root_slot:&mut root as *mut u64 as u64,object:object_address,callback:0x7100000100,old:30f64.to_bits(),new:60f64.to_bits()};let before=object;
        assert!(matches(&p,p.old));assert!(write(object_address+8,p.new));assert!(matches(&p,p.new));assert_eq!(object[0],before[0]);assert_eq!(&object[2..],&before[2..]);assert!(restore(&p));assert_eq!(object,before);
        assert!(write(object_address+8,p.new));root=0;assert!(!restore(&p));assert!(write(p.root_slot,object_address));assert!(restore(&p));
        assert!(write(object_address+8,45f64.to_bits()));assert!(!restore(&p));assert_eq!(object[1],45f64.to_bits());assert!(!write(0x121300000500,1));assert!(!write(object_address+1,1));
    }
}
