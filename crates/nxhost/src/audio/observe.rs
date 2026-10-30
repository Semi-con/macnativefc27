// Optional, bounded evidence from actual post-spatial-mixer output callbacks.
// The realtime callback only updates counters; a separate thread writes once.
use std::sync::{Arc,atomic::{AtomicU32,AtomicU64,AtomicI32,Ordering::Relaxed}};
use std::time::{Duration,Instant};

#[derive(Default)]
pub(super) struct CallbackStats {
    callbacks:AtomicU32,nonzero_callbacks:AtomicU32,frames:AtomicU64,
    nonfinite_samples:AtomicU64,errors:AtomicU32,first_error:AtomicI32,peak:AtomicU32,
}
impl CallbackStats {
    fn done(&self)->bool {self.callbacks.load(Relaxed)>=4096 || self.nonzero_callbacks.load(Relaxed)>=256}
    pub(super) fn record(&self,status:i32,frames:u32,channels:&[&mut [f32]]) {
        if self.done() {return;}
        self.frames.fetch_add(frames as u64,Relaxed);
        if status!=0 {self.errors.fetch_add(1,Relaxed);let _=self.first_error.compare_exchange(0,status,Relaxed,Relaxed);}
        let mut peak=0f32;let mut invalid=0;
        for channel in channels {for &value in channel.iter() {if value.is_finite() {peak=peak.max(value.abs());} else {invalid+=1;}}}
        self.nonfinite_samples.fetch_add(invalid,Relaxed);self.peak.fetch_max(peak.to_bits(),Relaxed);
        if peak>0.0 {self.nonzero_callbacks.fetch_add(1,Relaxed);}
        self.callbacks.fetch_add(1,Relaxed);
    }
}
pub(super) fn start(rate:u32)->Option<Arc<CallbackStats>> {
    let directory=std::env::var_os("NX_AUDIO_CALLBACK_STATS")?;
    static NEXT:AtomicU32=AtomicU32::new(0);let index=NEXT.fetch_add(1,Relaxed);
    let stats=Arc::new(CallbackStats::default());let observed=stats.clone();
    std::thread::Builder::new().name("audio-observation".into()).spawn(move || {
        let start=Instant::now();while !observed.done() && start.elapsed()<Duration::from_secs(120) {std::thread::sleep(Duration::from_millis(100));}
        let directory=std::path::PathBuf::from(directory);let _=std::fs::create_dir_all(&directory);
        let result=format!("{{\"source\":\"actual post-spatial-mixer stereo render callback after app fader\",\"rate\":{rate},\"channels\":2,\"callbacks\":{},\"frames\":{},\"nonzero_callbacks\":{},\"nonfinite_samples\":{},\"render_errors\":{},\"first_render_error\":{},\"peak\":{},\"seconds\":{:.3},\"system_volume_modified\":false}}\n",observed.callbacks.load(Relaxed),observed.frames.load(Relaxed),observed.nonzero_callbacks.load(Relaxed),observed.nonfinite_samples.load(Relaxed),observed.errors.load(Relaxed),observed.first_error.load(Relaxed),f32::from_bits(observed.peak.load(Relaxed)),start.elapsed().as_secs_f64());
        if std::fs::write(directory.join(format!("postmix-{index}.json")),result).is_ok() {eprintln!("[audio-observation] stereo output callback evidence saved for stream{index}");}
    }).expect("audio observation thread");Some(stats)
}
