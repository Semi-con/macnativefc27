// Optional local test transport. No listener or behavior change without an
// explicit per-run path; every held input expires independently of its writer.
use std::path::PathBuf;
use std::sync::{LazyLock,Mutex};
use std::time::{Duration,Instant};
use crate::remote::Pad;
#[path="input_config_inspection.rs"]
pub(crate) mod config_inspection;
#[path="input_frame_settings.rs"]
mod frame_settings;
#[path="input_render_quality.rs"]
mod render_quality;
#[path="input_menu_playback.rs"]
pub(crate) mod menu_playback;
#[path="frame_wait_profile.rs"]
mod wait_profile;
#[path="engine_limiter_control.rs"]
mod engine_limiter;
pub(crate) fn profile_wait(frame:&crate::guest::Frame)->Option<wait_profile::Guard> {wait_profile::start(frame)}

static PATH:LazyLock<Option<PathBuf>>=LazyLock::new(||std::env::var_os("NX_TEST_CONTROL").map(PathBuf::from));
static BUFFER_FORMATS_MODE:std::sync::atomic::AtomicBool=std::sync::atomic::AtomicBool::new(true);
static BUFFER_FORMATS_SEQUENCE:std::sync::atomic::AtomicU64=std::sync::atomic::AtomicU64::new(0);
static BUFFER_FORMATS_CONTROL:Mutex<()>=Mutex::new(());
// A host-only exact-output A/B selector. No guest memory/settings are changed.
pub(crate) fn buffer_formats_active()->bool {BUFFER_FORMATS_MODE.load(std::sync::atomic::Ordering::Relaxed)}
fn parse_buffer_formats(text:&str)->Option<(u64,bool,Duration)> {
    let mut p=text.split_whitespace();let sequence=p.next()?.parse::<u64>().ok()?;let enabled=p.next()?.parse::<u32>().ok()?;let hold=p.next()?.parse::<u64>().ok()?;
    (sequence>0 && enabled<=1 && (1000..=120000).contains(&hold) && p.next().is_none()).then_some((sequence,enabled!=0,Duration::from_millis(hold)))
}
fn poll_buffer_formats(control:&std::path::Path) {
    use std::sync::atomic::Ordering::Relaxed;
    static ENABLED:LazyLock<bool>=LazyLock::new(||std::env::var_os("NX_BUFFER_FORMATS").is_some());
    if !*ENABLED {return;}
    let request=control.with_extension("buffer-formats");let Some(text)=read(&request)else{return;};let _=std::fs::remove_file(request);
    let Some((sequence,enabled,hold))=parse_buffer_formats(&text)else{return;};
    let _guard=BUFFER_FORMATS_CONTROL.lock().unwrap();
    static COUNT:std::sync::atomic::AtomicU32=std::sync::atomic::AtomicU32::new(0);
    let accepted=sequence>BUFFER_FORMATS_SEQUENCE.load(Relaxed) && COUNT.fetch_add(1,Relaxed)<8;
    let accepted=if accepted {
        BUFFER_FORMATS_SEQUENCE.store(sequence,Relaxed);BUFFER_FORMATS_MODE.store(enabled,Relaxed);
        let spawned=std::thread::Builder::new().name("buffer-format-ab".into()).spawn(move||{
            std::thread::sleep(hold);let _guard=BUFFER_FORMATS_CONTROL.lock().unwrap();if BUFFER_FORMATS_SEQUENCE.load(Relaxed)==sequence {BUFFER_FORMATS_MODE.store(true,Relaxed);}
        }).is_ok();
        if !spawned {BUFFER_FORMATS_MODE.store(true,Relaxed);}spawned
    }else{false};
    let output=control.with_extension(format!("buffer-formats-{sequence}.json"));
    let _=std::fs::write(output,format!(r#"{{"sequence":{sequence},"accepted":{accepted},"specialized":{enabled},"hold_ms":{},"scope":"At most8 host-only original/specialized pipeline choices,1-120seconds, independent expiry to specialized default. No guest memory, clocks, files or saves changed."}}"#,hold.as_millis()));
    eprintln!("[buffer-formats-ab] sequence={sequence} accepted={accepted} specialized={enabled} hold_ms={}",hold.as_millis());
}
static CPU_TEXTURE_CHECKS_MODE:std::sync::atomic::AtomicBool=std::sync::atomic::AtomicBool::new(false);
static CPU_TEXTURE_CHECKS_SEQUENCE:std::sync::atomic::AtomicU64=std::sync::atomic::AtomicU64::new(0);
static CPU_TEXTURE_CHECKS_CONTROL:Mutex<()>=Mutex::new(());
pub(crate) fn cpu_texture_checks_active()->bool {CPU_TEXTURE_CHECKS_MODE.load(std::sync::atomic::Ordering::Relaxed)}
fn poll_cpu_texture_checks(control:&std::path::Path) {
    use std::sync::atomic::Ordering::Relaxed;
    let request=control.with_extension("cpu-texture-checks");let Some(text)=read(&request)else{return;};let _=std::fs::remove_file(request);
    let Some((sequence,enabled,hold))=parse_buffer_formats(&text)else{return;};
    let _guard=CPU_TEXTURE_CHECKS_CONTROL.lock().unwrap();
    static COUNT:std::sync::atomic::AtomicU32=std::sync::atomic::AtomicU32::new(0);
    let accepted=sequence>CPU_TEXTURE_CHECKS_SEQUENCE.load(Relaxed) && COUNT.fetch_add(1,Relaxed)<8;
    let accepted=if accepted {
        CPU_TEXTURE_CHECKS_SEQUENCE.store(sequence,Relaxed);CPU_TEXTURE_CHECKS_MODE.store(enabled,Relaxed);
        let spawned=std::thread::Builder::new().name("cpu-texture-ab".into()).spawn(move||{
            std::thread::sleep(hold);let _guard=CPU_TEXTURE_CHECKS_CONTROL.lock().unwrap();
            if CPU_TEXTURE_CHECKS_SEQUENCE.load(Relaxed)==sequence {CPU_TEXTURE_CHECKS_MODE.store(false,Relaxed);}
        }).is_ok();
        if !spawned {CPU_TEXTURE_CHECKS_MODE.store(false,Relaxed);}spawned
    }else{false};
    let _=std::fs::write(control.with_extension(format!("cpu-texture-checks-{sequence}.json")),format!(r#"{{"sequence":{sequence},"accepted":{accepted},"full_checks":{enabled},"hold_ms":{},"scope":"At most8 host CPU texture checksum choices,1-120seconds, independent expiry to default. No guest memory, clocks or saves changed."}}"#,hold.as_millis()));
    eprintln!("[cpu-texture-ab] sequence={sequence} accepted={accepted} full_checks={enabled} hold_ms={}",hold.as_millis());
}
fn poll_menu_waits(control:&std::path::Path) {
    let request=control.with_extension("menu-waits");let Some(text)=read(&request)else{return;};let _=std::fs::remove_file(request);
    let mut parts=text.split_whitespace();let Some(sequence)=parts.next().and_then(|s|s.parse::<u64>().ok()).filter(|v|*v>0)else{return;};
    let Some(milliseconds)=parts.next().and_then(|s|s.parse::<u64>().ok()).filter(|v|(1000..=30000).contains(v))else{return;};if parts.next().is_some(){return;}
    use std::sync::atomic::{AtomicU32,Ordering::Relaxed};static COUNT:AtomicU32=AtomicU32::new(0);
    let accepted=COUNT.fetch_add(1,Relaxed)<2 && crate::kernel::arm_menu_wait_observation(milliseconds);
    let _=std::fs::write(control.with_extension(format!("menu-waits-{sequence}.json")),format!(r#"{{"sequence":{sequence},"accepted":{accepted},"milliseconds":{milliseconds},"scope":"Bounded read-only host wait timing after manual scene verification; original results, scheduling, timeouts, clocks, game memory and save data unchanged"}}"#));
}
struct State {sequence:u64,pad:Pad,until:Instant,poll:Instant,presentation:Option<(u32,Instant)>}
static STATE:LazyLock<Mutex<State>>=LazyLock::new(||Mutex::new(State{sequence:0,pad:Pad::default(),until:Instant::now(),poll:Instant::now()-Duration::from_secs(1),presentation:None}));
pub(crate) fn read(path:&std::path::Path)->Option<String> {
    use std::io::Read;
    let mut text=String::new();std::fs::File::open(path).ok()?.take(256).read_to_string(&mut text).ok()?;Some(text)
}
fn parse(text:&str)->Option<(u64,Pad,Duration)> {
    let mut parts=text.split_whitespace();let sequence=parts.next()?.parse().ok()?;
    let buttons=u64::from_str_radix(parts.next()?.trim_start_matches("0x"),16).ok()?;
    let hold=parts.next()?.parse::<u64>().ok()?;
    if sequence==0 || buttons>0xffff || !(1..=5000).contains(&hold) {return None;}
    let mut sticks=[0f32;4];for value in &mut sticks {*value=parts.next()?.parse().ok()?;if !value.is_finite() || !(-1.0..=1.0).contains(value) {return None;}}
    if parts.next().is_some() {return None;}
    Some((sequence,Pad{buttons,sticks,motion:None},Duration::from_millis(hold)))
}
fn parse_presentation(text:&str)->Option<(u64,u32,Duration)> {
    let mut parts=text.split_whitespace();if parts.next()? != "present" {return None;}
    let sequence=parts.next()?.parse::<u64>().ok()?;let interval=parts.next()?.parse::<u32>().ok()?;let hold=parts.next()?.parse::<u64>().ok()?;
    if sequence==0 || interval>2 || !(100..=120000).contains(&hold) || parts.next().is_some() {return None;}
    Some((sequence,interval,Duration::from_millis(hold)))
}
fn choose_presentation(original:u32,override_until:Option<(u32,Instant)>,now:Instant)->u32 {
    override_until.filter(|(_,until)|now<*until).map_or(original,|(interval,_)|interval)
}
// This bounded diagnostic affects only host presentation pacing. Guest clocks,
// display events, logic and saved settings retain their original code paths.
// Without the explicitly enabled local transport there is no override.
pub fn presentation_interval(original:u32)->u32 {
    if PATH.is_none() {return original;}
    choose_presentation(original,STATE.lock().unwrap().presentation,Instant::now())
}
pub fn pad()->Option<Pad> {
    let path=PATH.as_ref()?;let now=Instant::now();let mut state=STATE.lock().unwrap();
    if now.duration_since(state.poll)>=Duration::from_millis(10) {
        state.poll=now;
        if let Some(text)=read(path) {
            if let Some((sequence,pad,hold))=parse(&text) {
                if sequence>state.sequence {state.sequence=sequence;state.pad=pad;state.until=now+hold;eprintln!("[test-input] sequence {sequence}, buttons {:#x}, sticks {:?}, hold {}ms",pad.buttons,pad.sticks,hold.as_millis());}
            } else if let Some((sequence,interval,hold))=parse_presentation(&text) {
                if sequence>state.sequence {state.sequence=sequence;state.presentation=Some((interval,now+hold));eprintln!("[test-present] sequence {sequence}, interval {interval}, expires {}ms; guest clocks unchanged",hold.as_millis());}
            }
        }
    }
    (now<state.until).then_some(state.pad)
}
pub fn frame()->Option<u64> {
    crate::fc27_menu_auto_lifecycle::poll();
    let control=PATH.as_ref()?;
    poll_buffer_formats(control);
    poll_cpu_texture_checks(control);
    config_inspection::poll(control);
    frame_settings::poll(control);
    render_quality::poll(control);
    menu_playback::poll(control);
    crate::fc27_menu_prime_probe::poll(control);
    poll_menu_waits(control);
    crate::menu_asset_io::poll(control);
    wait_profile::poll(control);
    engine_limiter::poll(control);
    let path=control.with_extension("frame");
    let sequence=read(&path)?.trim().parse().ok()?;std::fs::remove_file(path).ok()?;Some(sequence)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn host_buffer_format_ab_transport_requires_bounded_expiring_modes() {
        assert_eq!(parse_buffer_formats("1 0 1000"),Some((1,false,Duration::from_secs(1))));
        assert_eq!(parse_buffer_formats("2 1 120000"),Some((2,true,Duration::from_secs(120))));
        for bad in ["0 1 1000","1 2 1000","1 -1 1000","1 0 999","1 0 120001","1 1 1000 extra"] {assert!(parse_buffer_formats(bad).is_none());}
    }
    #[test]
    fn test_transport_rejects_unbounded_holds_and_nonfinite_sticks() {
        let (id,pad,hold)=parse("8 0x4001 160 0.5 -1 0 0").unwrap();assert_eq!(id,8);assert_eq!(pad.buttons,0x4001);assert_eq!(hold,Duration::from_millis(160));assert_eq!(pad.sticks,[0.5,-1.0,0.0,0.0]);
        for bad in ["0 1 160 0 0 0 0","1 1 5001 0 0 0 0","1 1 0 0 0 0 0","1 1 160 NaN 0 0 0","1 1 160 2 0 0 0","1 10000 160 0 0 0 0"] {assert!(parse(bad).is_none());}
    }
    #[test]
    fn presentation_transport_bounds_and_expiry_restore_original_interval() {
        let now=Instant::now();
        for interval in 0..=2 {let (id,value,hold)=parse_presentation(&format!("present 8 {interval} 120000")).unwrap();assert_eq!((id,value),(8,interval));assert_eq!(hold,Duration::from_secs(120));assert_eq!(choose_presentation(2,Some((value,now+hold)),now),value);assert_eq!(choose_presentation(3,Some((value,now)),now),3);}
        assert_eq!(choose_presentation(2,None,now),2);
        for bad in ["present 0 1 1000","present 1 3 1000","present 1 -1 1000","present 1 1 120001","present 1 1 99","present 1 1 1000 extra","1 1 160 0 0 0 0"] {assert!(parse_presentation(bad).is_none());}
        assert!(parse("present 1 1 1000").is_none());
    }
}
