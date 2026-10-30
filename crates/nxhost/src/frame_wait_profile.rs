// Short opt-in observation of original guest wait calls. Never changes waits,
// clocks, scheduler decisions, rendering or game data.
use std::{path::Path,sync::{LazyLock,Mutex,atomic::{AtomicBool,AtomicU32,Ordering::{Relaxed,Acquire,Release}}},time::{Instant,Duration}};
static ACTIVE:AtomicBool=AtomicBool::new(false);
const LIMIT:usize=64;
fn request(text:&str)->Option<(u64,Duration)> {
    let mut p=text.split_whitespace();let sequence=p.next()?.parse::<u64>().ok()?;let ms=p.next()?.parse::<u64>().ok()?;
    (sequence>0&&(1000..=5000).contains(&ms)&&p.next().is_none()).then_some((sequence,Duration::from_millis(ms)))
}
fn timeout(svc:u32,x:&[u64;31])->Option<i64> {match svc {0x0b=>Some(x[0] as i64),0x18|0x1c|0x34=>Some(x[3] as i64),0x1a=>Some(-2),_=>None}}
fn bucket(ns:i64)->i64 {if ns<0 {ns} else {ns/1_000_000}}
struct Row {thread:u32,svc:u32,bucket:i64,min_requested:i64,max_requested:i64,started:u64,completed:u64,total_ns:u64,max_ns:u64,last_result:u64,trace:Vec<String>}
#[derive(Default)]
struct State {sequence:u64,until:Option<Instant>,rows:Vec<Row>,overflow:u64}
impl State {
    fn begin(&mut self,thread:u32,svc:u32,requested:i64)->Option<usize> {
        let key=bucket(requested);
        let at=if let Some(at)=self.rows.iter().position(|r|(r.thread,r.svc,r.bucket)==(thread,svc,key)) {at} else {
            if self.rows.len()==LIMIT {self.overflow+=1;return None;}
            self.rows.push(Row{thread,svc,bucket:key,min_requested:requested,max_requested:requested,started:0,completed:0,total_ns:0,max_ns:0,last_result:0,trace:Vec::new()});self.rows.len()-1
        };
        let row=&mut self.rows[at];row.started+=1;row.min_requested=row.min_requested.min(requested);row.max_requested=row.max_requested.max(requested);Some(at)
    }
    fn finish(&mut self,sequence:u64,at:usize,ns:u64,result:u64) {
        if self.sequence!=sequence||!self.until.is_some_and(|until|Instant::now()<until) {return;}
        let row=&mut self.rows[at];row.completed+=1;row.total_ns=row.total_ns.saturating_add(ns);row.max_ns=row.max_ns.max(ns);row.last_result=result;
    }
}
static STATE:LazyLock<Mutex<State>>=LazyLock::new(||Mutex::new(State::default()));
pub(crate) struct Guard {sequence:u64,index:usize,started:Instant}
impl Guard {
    pub(crate) fn finish(self,result:u64) {let ns=self.started.elapsed().as_nanos().min(u64::MAX as u128) as u64;STATE.lock().unwrap().finish(self.sequence,self.index,ns,result);}
}
pub(super) fn start(frame:&crate::guest::Frame)->Option<Guard> {
    // One atomic read is the only overhead outside an explicitly armed window.
    if !ACTIVE.load(Acquire) {return None;}
    let svc=frame.x[16] as u32;let requested=timeout(svc,&frame.x)?;let thread=crate::guest::try_current()?.handle;
    let mut state=STATE.lock().unwrap();if !state.until.is_some_and(|u|Instant::now()<u) {return None;}
    let index=state.begin(thread,svc,requested)?;
    if state.rows[index].trace.is_empty() {
        let mut trace=vec![crate::diag::symbolize(frame.x[30])];trace.extend(crate::diag::backtrace_lines(frame.x[29]).into_iter().take(10));state.rows[index].trace=trace;
    }
    Some(Guard{sequence:state.sequence,index,started:Instant::now()})
}
fn finish_report(sequence:u64,duration:Duration,output:&Path) {
    std::thread::sleep(duration);let mut state=STATE.lock().unwrap();if state.sequence!=sequence {return;}
    ACTIVE.store(false,Release);state.until=None;
    let quote=super::config_inspection::quote_ascii;
    let rows=state.rows.iter().map(|r|format!(r#"{{"thread":{},"svc":{},"timeout_bucket_ms":{},"requested_ns_min":{},"requested_ns_max":{},"started":{},"completed_in_window":{},"inflight_at_end":{},"total_completed_ms":{},"max_completed_ms":{},"last_result":{},"representative_stack":[{}]}}"#,r.thread,r.svc,r.bucket,r.min_requested,r.max_requested,r.started,r.completed,r.started-r.completed,r.total_ns as f64/1e6,r.max_ns as f64/1e6,r.last_result,r.trace.iter().map(|s|quote(s)).collect::<Vec<_>>().join(","))).collect::<Vec<_>>();
    let text=format!(r#"{{"sequence":{sequence},"window_ms":{},"overflow_calls":{},"rows":[{}],"limits":"At most4 explicitly requested1–5second windows/process,64thread/SVC/timeout buckets. Representative stack per bucket, not proof all calls share its origin. Wait calls that start before/finish after window are not completed samples. Instrumentation can perturb timing; validate FPS outside profiling. No wait or clock semantics changed."}}"#,duration.as_millis(),state.overflow,rows.join(","));
    let _=std::fs::write(output,text);eprintln!("[frame-waits] completed sequence={sequence} rows={} output={}",rows.len(),output.display());
}
pub(super) fn poll(control:&Path) {
    let path=control.with_extension("wait-profile");let Some(text)=super::read(&path) else {return;};let _=std::fs::remove_file(path);
    let Some((sequence,duration))=request(&text) else {return;};static COUNT:AtomicU32=AtomicU32::new(0);
    if ACTIVE.load(Relaxed)||COUNT.fetch_add(1,Relaxed)>=4 {return;}
    let mut state=STATE.lock().unwrap();if sequence<=state.sequence {return;};*state=State{sequence,until:Some(Instant::now()+duration),..Default::default()};ACTIVE.store(true,Release);drop(state);
    let output=control.with_extension(format!("wait-profile-{sequence}.json"));
    if std::thread::Builder::new().name("frame-waits".into()).spawn(move||finish_report(sequence,duration,&output)).is_err() {ACTIVE.store(false,Release);STATE.lock().unwrap().until=None;}
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn windows_and_svc_timeout_arguments_are_bounded_and_clocks_are_excluded() {
        assert_eq!(request("1 2000"),Some((1,Duration::from_secs(2))));for bad in ["0 2000","1 999","1 5001","1 2000 extra"] {assert!(request(bad).is_none());}
        let mut x=[0u64;31];x[0]=123;x[3]=u64::MAX;assert_eq!(timeout(0xb,&x),Some(123));for svc in [0x18,0x1c,0x34] {assert_eq!(timeout(svc,&x),Some(-1));}assert_eq!(timeout(0x1a,&x),Some(-2));for svc in [0x1e,0x27,0x1d] {assert_eq!(timeout(svc,&x),None);}
    }
    #[test]
    fn bounded_histogram_distinguishes_inflight_completed_and_stale_windows() {
        let mut s=State{sequence:7,until:Some(Instant::now()+Duration::from_secs(1)),..Default::default()};let at=s.begin(0x10,0xb,32_100_000).unwrap();assert_eq!(s.begin(0x10,0xb,32_900_000),Some(at));s.finish(6,at,123,0);assert_eq!(s.rows[at].completed,0);s.finish(7,at,31_000_000,0);
        assert_eq!((s.rows[at].started,s.rows[at].completed,s.rows[at].min_requested,s.rows[at].max_requested),(2,1,32_100_000,32_900_000));assert_eq!(s.rows[at].total_ns,31_000_000);
        for thread in 1..=LIMIT as u32 {s.begin(thread,0x18,-1);}assert_eq!(s.rows.len(),LIMIT);assert_eq!(s.overflow,1);s.until=None;s.finish(7,at,999,0);assert_eq!(s.rows[at].completed,1);
    }
}
