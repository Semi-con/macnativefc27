// Version-checked inspection of two original settings classes.
// No listener or scan exists without the per-process test control transport.
use std::{path::Path,sync::atomic::{AtomicU32,Ordering::Relaxed},time::{Duration,Instant}};
#[path="input_frame_settings_edit.rs"]
mod edit;
const CHUNK:usize=1<<20;
const TAIL:usize=0x140;
fn chunk_shape(remaining:u64)->(usize,usize) {
    let step=remaining.min((CHUNK-TAIL) as u64) as usize;
    (step,remaining.min((step+TAIL) as u64) as usize)
}
#[derive(Clone,Copy)]
enum Kind {Simulation,Display}
fn word(b:&[u8],at:usize)->Option<u64> {Some(u64::from_le_bytes(b.get(at..at+8)?.try_into().ok()?))}
fn uint(b:&[u8],at:usize)->Option<u32> {Some(u32::from_le_bytes(b.get(at..at+4)?.try_into().ok()?))}
fn float(b:&[u8],at:usize)->Option<String> {let v=f32::from_bits(uint(b,at)?);Some(if v.is_finite(){v.to_string()}else{"null".into()})}
fn row(b:&[u8],address:u64,base:u64,kind:Kind)->Option<String> {
    let (name,size,vtable,canonical)=match kind {Kind::Simulation=>("SimulationTimeSettings",0x40,0xc5dd148,0xdda3268),Kind::Display=>("BaseDisplaySettings",0x130,0xcc415c8,0xda01b88)};
    if b.len()<size || word(b,0)?!=base+vtable {return None;}
    let type_pointer=word(b,8)?;let flags=uint(b,0x18)?;
    let fields=match kind {
        Kind::Simulation=>format!(r#""MaxSimFps":{},"ForceSimRate":{},"ForceUseSleepTimer":{},"TimeScale":{},"ForceDeltaTime":{},"MaxVirtualTicks":{}"#,float(b,0x34)?,uint(b,0x38)?,*b.get(0x3d)?,float(b,0x28)?,float(b,0x2c)?,uint(b,0x24)?),
        Kind::Display=>format!(r#""PresentInterval":{},"FullscreenRefreshRate":{},"PresentEnable":{},"ScreenRenderEnable":{}"#,uint(b,0x78)?,float(b,0x70)?,*b.get(0x120)?,*b.get(0x119)?),
    };
    Some(format!(r#"{{"class":"{name}","address":"{address:#x}","type_pointer":"{type_pointer:#x}","canonical_type_match":{},"flags":{flags},"fields":{{{fields}}}}}"#,type_pointer==base+canonical))
}
fn version_base()->Option<u64> {
    let main=crate::diag::MODULES.get()?.iter().find(|m|m.name=="main")?;
    let base=main.base;
    // Original constructor and reflected field descriptors identify this build.
    let constructor=crate::diag::read_words::<1>(base+0x1cf8d48);
    let simulation=crate::diag::read_words::<4>(base+0xc5dd1a0);
    let display=crate::diag::read_words::<4>(base+0xcc41780);
    let valid=constructor==Some([0xb9003808528003c8])
        && simulation.is_some_and(|w|w[0]==base+0xbd7e4a6&&w[1]==0x871828d5&&w[2]==0x34)
        && display.is_some_and(|w|w[0]==base+0xbe2bd32&&w[1]==0xa181f00c&&w[2]==0x78);
    valid.then_some(base)
}
fn inspect(output:&Path) {
    let Some(base)=version_base() else {let _=std::fs::write(output,r#"{"version_verified":false,"objects":[],"error":"Original constructor/field signature mismatch; no scan performed"}"#);return;};
    let started=Instant::now();let space=crate::kernel::aspace();let first=space.heap();
    let bound=first+space.layout.heap_size.min(1536u64<<20);let mut cursor=first;let mut scanned=0u64;let mut failures=0;let mut objects=Vec::new();let mut buffer=vec![0;CHUNK+TAIL];
    while cursor<bound && objects.len()<64 && started.elapsed()<Duration::from_secs(15) {
        let (region,size,block)=space.map.lock().unwrap().query(cursor);
        let Some(end)=region.checked_add(size).map(|v|v.min(bound)).filter(|v|*v>cursor) else {break;};
        if block.state!=crate::mem::state::NORMAL || block.perm&crate::mem::perm::R==0 {cursor=end;continue;}
        // The shared OS-copy helper accepts at most CHUNK bytes, including
        // the forward overlap needed for objects spanning a scan boundary.
        let (step,amount)=chunk_shape(end-cursor);
        if super::config_inspection::snapshot(cursor,&mut buffer[..amount]) {
            scanned+=step as u64;
            for at in (0..step.saturating_sub(7)).step_by(8) {
                let kind=match word(&buffer[..amount],at) {Some(v) if v==base+0xc5dd148=>Kind::Simulation,Some(v) if v==base+0xcc415c8=>Kind::Display,_=>continue};
                if let Some(value)=row(&buffer[at..amount],cursor+at as u64,base,kind) {objects.push(value);if objects.len()==64 {break;}}
            }
        } else {failures+=1;}
        cursor+=step as u64;
    }
    let report=format!(r#"{{"version_verified":true,"bytes_scanned":{scanned},"seconds":{},"failed_chunks":{failures},"range_complete":{},"objects":[{}],"limits":"Read-only OS copies of guest NORMAL readable heap; first1536MiB,15seconds,64objects. Exact vtable and original field signatures; active vs default/unused objects is NOT established. Concurrent changes are not a coherent snapshot. No raw heap or keys retained."}}"#,started.elapsed().as_secs_f64(),cursor>=bound,objects.join(","));
    let _=std::fs::write(output,report);eprintln!("[frame-settings] inspected objects={} bytes={scanned} output={}",objects.len(),output.display());
}
pub(super) fn poll(control:&Path) {
    edit::poll(control);
    let path=control.with_extension("settings");let Some(text)=super::read(&path) else {return;};let _=std::fs::remove_file(path);
    let Ok(sequence)=text.trim().parse::<u64>() else {return;};static COUNT:AtomicU32=AtomicU32::new(0);
    if sequence==0 || COUNT.fetch_add(1,Relaxed)>=2 {return;}
    let output=control.with_extension(format!("settings-{sequence}.json"));let _=std::thread::Builder::new().name("frame-settings".into()).spawn(move||inspect(&output));
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn scan_overlap_fits_actual_shared_snapshot_limit() {
        let (step,amount)=chunk_shape(2*CHUNK as u64);assert_eq!(amount,CHUNK);assert_eq!(amount-step,TAIL);assert_eq!(step%8,0);
        let source=vec![0x5au8;CHUNK+TAIL];let mut copy=vec![0;CHUNK+TAIL];
        assert!(super::super::config_inspection::snapshot(source.as_ptr() as u64,&mut copy[..amount]));assert_eq!(&copy[..amount],&source[..amount]);
        assert!(!super::super::config_inspection::snapshot(source.as_ptr() as u64,&mut copy[..CHUNK+1]));
        for left in [1,7,8,0x130,CHUNK-TAIL,CHUNK,CHUNK+1] {let (step,amount)=chunk_shape(left as u64);assert!(step>0&&step<=amount&&amount<=CHUNK&&amount<=left);}
    }
    #[test]
    fn reflected_offsets_decode_typed_values_and_reject_wrong_vtable_and_short_objects() {
        let base=0x7100000000u64;let mut bytes=[0u8;0x140];bytes[..8].copy_from_slice(&(base+0xc5dd148).to_le_bytes());bytes[8..16].copy_from_slice(&(base+0xdda3268).to_le_bytes());
        bytes[0x34..0x38].copy_from_slice(&30f32.to_bits().to_le_bytes());bytes[0x38..0x3c].copy_from_slice(&30u32.to_le_bytes());bytes[0x3d]=1;
        let text=row(&bytes,0x7200000000,base,Kind::Simulation).unwrap();assert!(text.contains("\"MaxSimFps\":30"));assert!(text.contains("\"ForceSimRate\":30"));assert!(text.contains("\"ForceUseSleepTimer\":1"));assert!(text.contains("\"canonical_type_match\":true"));
        assert!(row(&bytes[..0x3f],0,base,Kind::Simulation).is_none());assert!(row(&bytes,0,base,Kind::Display).is_none());
        bytes[..8].copy_from_slice(&(base+0xcc415c8).to_le_bytes());bytes[0x78..0x7c].copy_from_slice(&2u32.to_le_bytes());assert!(row(&bytes,0,base,Kind::Display).unwrap().contains("\"PresentInterval\":2"));
    }
}
