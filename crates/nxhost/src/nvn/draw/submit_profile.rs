// Bounded encoder-boundary timings. This path neither splits original passes,
// commits extra command buffers, waits for counters, nor downloads textures.
use crate::gfx;
use objc2::{runtime::ProtocolObject};
use objc2_metal::*;
use std::{collections::BTreeMap,sync::{LazyLock,Mutex,atomic::{AtomicBool,AtomicU32,Ordering::Relaxed}}};

type DrawKey=(u32,u32,u32,u64,u64);
#[derive(Default)] struct Workload{draws:u64,vertices:u64,indirect:u64}
static DRAW_WORK:LazyLock<Mutex<BTreeMap<DrawKey,Workload>>>=LazyLock::new(||Mutex::new(BTreeMap::new()));

static ACTIVE:AtomicBool=AtomicBool::new(false);
static LEFT:AtomicU32=AtomicU32::new(0);
static SUBMIT:AtomicU32=AtomicU32::new(0);

pub(super) fn active()->bool {ACTIVE.load(Relaxed)}

pub(super) fn begin(submit:u32)->Option<std::time::Instant> {
    if submit%32==0 {
        if let Ok(value)=std::fs::read_to_string("work/profile-submit") {
            let _=std::fs::remove_file("work/profile-submit");
            LEFT.store(value.trim().parse::<u32>().unwrap_or(1).clamp(1,4),Relaxed);
        }
    }
    let enabled=LEFT.load(Relaxed)>0;ACTIVE.store(enabled,Relaxed);SUBMIT.store(submit,Relaxed);
    if enabled {crate::nvn::summary();}
    if enabled {DRAW_WORK.lock().unwrap().clear();LEFT.fetch_sub(1,Relaxed);eprintln!("[submit-profile] begin submit={submit}");Some(std::time::Instant::now())} else {None}
}

pub(super) fn finish(start:Option<std::time::Instant>) {
    ACTIVE.store(false,Relaxed);
    if let Some(start)=start {
        let elapsed_ms=start.elapsed().as_secs_f64()*1000.0;
        let work=std::mem::take(&mut *DRAW_WORK.lock().unwrap());let mut rows:Vec<_>=work.into_iter().collect();rows.sort_by_key(|(_,w)|std::cmp::Reverse(w.vertices));
        for ((pass,width,height,vertex,fragment),w) in rows.into_iter().take(24) {eprintln!("[submit-profile] submit={} workload pass={pass} target={width}x{height} vertex={vertex:016x} fragment={fragment:016x} draws={} vertices={} indirect={}",SUBMIT.load(Relaxed),w.draws,w.vertices,w.indirect);}
        eprintln!("[submit-profile] end submit={} cpu_replay_ms={:.6}",SUBMIT.load(Relaxed),elapsed_ms);}
}

pub(super) fn watch(cb:&gfx::CommandBuffer,counter:gfx::Shared<ProtocolObject<dyn MTLCounterSampleBuffer>>,label:String,count:usize) {
    let submit=SUBMIT.load(Relaxed);
    let handler=block2::RcBlock::new(move |_:std::ptr::NonNull<gfx::CommandBuffer>| {
        let Some(data)=(unsafe {counter.0.resolveCounterRange(objc2_foundation::NSRange{location:0,length:count})}) else {return;};
        if data.length()!=count*8 {return;}
        let mut values=[0u64;4];unsafe {data.getBytes_length(std::ptr::NonNull::from(&mut values).cast(),count*8);}
        if values[..count].iter().any(|v|*v==0 || *v==u64::MAX) {return;}
        if count==2 && values[1]>=values[0] {eprintln!("[submit-profile] submit={submit} {label} start_ns={} end_ns={} gpu_ms={:.6}",values[0],values[1],(values[1]-values[0]) as f64/1e6);}
        if count==4 && values[1]>=values[0] && values[3]>=values[2] {eprintln!("[submit-profile] submit={submit} {label} vertex_start_ns={} vertex_end_ns={} fragment_start_ns={} fragment_end_ns={} vertex_ms={:.6} fragment_ms={:.6}",values[0],values[1],values[2],values[3],(values[1]-values[0]) as f64/1e6,(values[3]-values[2]) as f64/1e6);}
    });
    unsafe {cb.addCompletedHandler(block2::RcBlock::as_ptr(&handler));}
}

pub(super) fn context(pass:u32,size:(u32,u32),viewport:Option<[f32;4]>,vertex:u64,fragment:u64) {
    if active() {eprintln!("[submit-profile] submit={} context pass={pass} target={}x{} viewport={viewport:?} vertex={vertex:016x} fragment={fragment:016x}",SUBMIT.load(Relaxed),size.0,size.1);}
}

// Diagnostic CPU counts only; indirect geometry remains unknown. Never read GPU arguments.
pub(super) fn workload(pass:u32,size:(u32,u32),vertex:u64,fragment:u64,count:u32,instances:u32,indirect:bool) {
    if !active(){return;}let mut rows=DRAW_WORK.lock().unwrap();if rows.len()>=4096 {return;}
    let w=rows.entry((pass,size.0,size.1,vertex,fragment)).or_default();w.draws+=1;if indirect{w.indirect+=1;}else{w.vertices=w.vertices.saturating_add(count as u64*instances as u64);}
}
