// FC27 1.8c.cab1 process-local frame target. Opt-in, version/identity checked.
// The original limiter consumes object+8 as a frequency. Guest clocks, game
// files, saved settings and simulation delta are never altered here.
use std::sync::{LazyLock,atomic::{AtomicBool,AtomicU32,Ordering::Relaxed}};
pub(crate) const ROOT:u64=0xda19e38;
pub(crate) fn read(address:u64)->Option<u64> {crate::diag::read_words::<1>(address).map(|x|x[0])}
pub(crate) fn write(address:u64,value:u64)->bool {
    if address%8!=0 {return false;}
    unsafe extern "C" {static mach_task_self_:u32;fn mach_vm_write(task:u32,address:u64,data:u64,count:u32)->i32;}
    let bytes=value.to_le_bytes();unsafe {mach_vm_write(mach_task_self_,address,bytes.as_ptr() as u64,8)==0}
}
pub(crate) fn base()->Option<u64> {
    let base=crate::diag::MODULES.get()?.iter().find(|m|m.name=="main")?.base;
    // LDR D0,[X0,#8], FMOV D1,#1, followed by FDIV D1,D1,D0.
    (read(base+0x99e3f0c)==Some(0x1e6e1001fd400400)
        && crate::diag::read_words::<1>(base+0x99e3f24).is_some_and(|v|v[0] as u32==0x1e601821)
        && read(base+0x99e233c)==Some(0xfc4307e8a940fbfd)).then_some(base)
}
pub(crate) struct Snapshot {pub object:u64,pub callback:u64,pub fps:u64}
pub(crate) fn snapshot(base:u64)->Option<Snapshot> {
    let object=read(base+ROOT)?;if object%8!=0||object.checked_add(0x510).is_none() {return None;}
    let space=crate::kernel::aspace();let (start,size,block)=space.map.lock().unwrap().query(object);
    if block.state!=crate::mem::state::NORMAL||block.perm&crate::mem::perm::RW!=crate::mem::perm::RW||object+0x510>start+size {return None;}
    let fps=read(object+8)?;let rate=f64::from_bits(fps);let callback=read(object+0x4f8)?;
    if !rate.is_finite()||!(1.0..=240.0).contains(&rate)||callback%4!=0||!(base..base+0xb99a7c0).contains(&callback) {return None;}
    Some(Snapshot{object,callback,fps})
}

fn requested(value:Option<&str>)->bool {value==Some("60")}
fn enabled()->bool {
    static ENABLED:LazyLock<bool>=LazyLock::new(||requested(std::env::var("NX_FC27_RENDER_FPS").ok().as_deref()));
    *ENABLED
}

pub(crate) fn presentation_interval(original:u32)->u32 {
    if !enabled() {return original;}
    static TICK:AtomicU32=AtomicU32::new(0);
    // Scene changes may recreate/reset the original limiter. Revalidate its
    // complete identity at most once per64 presents; the hot path is atomic.
    if TICK.fetch_add(1,Relaxed)%64==0 { refresh(); }
    1
}

fn refresh() {
    static SIGNATURE_REPORTED:AtomicBool=AtomicBool::new(false);
    static APPLIED_REPORTED:AtomicBool=AtomicBool::new(false);
    let Some(base)=base() else {
        if !SIGNATURE_REPORTED.swap(true,Relaxed) {eprintln!("[frame-target] original signatures differ; original engine target retained");}
        return;
    };
    let Some(s)=snapshot(base) else {return;};
    if s.callback!=base+0x65760f0 {return;}
    if s.fps==60f64.to_bits() {return;}
    if s.fps!=30f64.to_bits() {return;}
    if read(base+ROOT)!=Some(s.object) || read(s.object+0x4f8)!=Some(s.callback) || read(s.object+8)!=Some(s.fps) {return;}
    if write(s.object+8,60f64.to_bits()) && read(s.object+8)==Some(60f64.to_bits()) {
        if !APPLIED_REPORTED.swap(true,Relaxed) {eprintln!("[frame-target] FC27 original engine target30->60 verified; host present interval1; guest clocks unchanged");}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn frame_target_is_explicit_and_accepts_only_supported_frequency() {
        assert!(requested(Some("60")));
        for value in [None,Some("30"),Some("120"),Some("60 "),Some("1"),Some("")] {assert!(!requested(value));}
    }
}
