//! Explicitly armed local-test observation of original RomFS reads. Metadata only.
use std::{path::Path,sync::{LazyLock,Mutex,atomic::{AtomicU32,AtomicU64,Ordering::Relaxed}},time::{Duration,Instant}};
static ENABLED:LazyLock<bool>=LazyLock::new(||std::env::var("NX_FC27_VIDEO_TRACE").is_ok_and(|v|v=="1") && std::env::var_os("NX_TEST_CONTROL").is_some());
static REQUESTS:AtomicU32=AtomicU32::new(0);
static ACTIVE:AtomicU64=AtomicU64::new(0);
struct Read {offset:u64,size:u64,start_ms:f64,elapsed_ms:f64,ok:bool,thread:Option<u32>}
struct Window {sequence:u64,start:Instant,until:Instant,trace_start_ms:f64,rows:Vec<Read>,dropped:u64}
static WINDOW:Mutex<Option<Window>>=Mutex::new(None);
pub(crate) struct Ticket {sequence:u64,start:Instant,thread:Option<u32>}
pub(crate) fn begin()->Option<Ticket> {
    let sequence=ACTIVE.load(Relaxed);if sequence==0 {return None;}
    Some(Ticket{sequence,start:Instant::now(),thread:crate::guest::try_current().map(|v|v.handle)})
}
pub(crate) fn end(ticket:Option<Ticket>,offset:u64,size:u64,ok:bool) {
    let Some(ticket)=ticket else{return;};let end=Instant::now();let mut guard=WINDOW.lock().unwrap();let Some(w)=guard.as_mut()else{return;};
    if ticket.sequence!=w.sequence || ticket.start<w.start || ticket.start>=w.until {return;}
    if w.rows.len()==512 {w.dropped+=1;return;}
    w.rows.push(Read{offset,size,start_ms:ticket.start.duration_since(w.start).as_secs_f64()*1000.,elapsed_ms:end.duration_since(ticket.start).as_secs_f64()*1000.,ok,thread:ticket.thread});
}
fn parse(text:&str)->Option<(u64,Duration)> {
    let mut p=text.split_whitespace();let sequence=p.next()?.parse::<u64>().ok()?;let ms=p.next()?.parse::<u64>().ok()?;
    (sequence!=0 && (1000..=30000).contains(&ms) && p.next().is_none()).then_some((sequence,Duration::from_millis(ms)))
}
pub(crate) fn poll(control:&Path) {
    if !*ENABLED {return;}
    let request=control.with_extension("menu-asset-io");let Some(text)=crate::input::test_control::read(&request)else{return;};let _=std::fs::remove_file(request);
    let Some((sequence,duration))=parse(&text)else{return;};
    let Some(trace_start_ms)=crate::video_diagnostics::trace_time_ms()else{return;};
    let mut guard=WINDOW.lock().unwrap();if guard.is_some() || REQUESTS.fetch_add(1,Relaxed)>=2 {return;}
    let start=Instant::now();*guard=Some(Window{sequence,start,until:start+duration,trace_start_ms,rows:Vec::new(),dropped:0});ACTIVE.store(sequence,Relaxed);drop(guard);
    let output=control.with_extension(format!("menu-asset-io-{sequence}.json"));
    if std::thread::Builder::new().name("menu-asset-io".into()).spawn(move||{
        std::thread::sleep(duration);ACTIVE.store(0,Relaxed);let mut guard=WINDOW.lock().unwrap();let Some(w)=guard.take()else{return;};drop(guard);
        let rows=w.rows.iter().map(|r|format!(r#"{{"offset":{},"size":{},"start_ms":{},"elapsed_ms":{},"ok":{},"thread":{}}}"#,r.offset,r.size,r.start_ms,r.elapsed_ms,r.ok,r.thread.map_or("null".into(),|v|v.to_string()))).collect::<Vec<_>>();
        let _=std::fs::write(output,format!(r#"{{"sequence":{},"video_trace_start_ms":{},"duration_ms":{},"recorded":{},"dropped":{},"reads":[{}],"limits":"Original virtual RomFS read metadata only;512 rows,30sec maximum,2 requests per isolated video-trace process. No payload, guest memory, game-file writes or clock changes. Reads crossing window end may be incomplete."}}"#,w.sequence,w.trace_start_ms,duration.as_millis(),rows.len(),w.dropped,rows.join(",")));
    }).is_err(){ACTIVE.store(0,Relaxed);*WINDOW.lock().unwrap()=None;}
}
