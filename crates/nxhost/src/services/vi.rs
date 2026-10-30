// Mush8 - vi.rs
// Constantin Clerc (c22dev) - 0.1
use crate::ipc::{Act, Request, Response, Service, Stub, Table, new_event, write_buffer};
use std::sync::{Arc, LazyLock, Mutex,OnceLock};
use std::collections::{BTreeMap,VecDeque};

static NATIVE_API:OnceLock<(u64,u64)>=OnceLock::new();
static NATIVE_LAYERS:LazyLock<Mutex<BTreeMap<u64,u64>>>=LazyLock::new(Default::default);
pub fn configure_native_api(get_window:u64,get_layer_id:u64) {NATIVE_API.set((get_window,get_layer_id)).expect("native VI API configured twice");}
pub extern "C" fn get_native_window(out:*mut u64,layer:u64)->u32 {
    let &(get,id)=NATIVE_API.get().expect("native VI API unavailable");
    let get:extern "C" fn(*mut u64,u64)->u32=unsafe {std::mem::transmute(get as usize)};
    let id:extern "C" fn(u64)->u64=unsafe {std::mem::transmute(id as usize)};
    let result=get(out,layer);
    if result==0 && !out.is_null() {
        let native=unsafe {out.read_unaligned()};let layer=id(layer);
        if native!=0 && LAYERS.lock().unwrap().contains_key(&layer) {
            NATIVE_LAYERS.lock().unwrap().insert(native,layer);
            eprintln!("[vi] original native window {native:#x} belongs to layer {layer}");
        }
    }
    result
}
pub fn native_layer(window:u64)->Option<u64> {NATIVE_LAYERS.lock().unwrap().get(&window).copied()}

#[derive(Clone,Copy,Debug)]
pub struct FrameRecord {pub frame:u64,pub queued_ns:u64,pub presented_ns:u64,pub state:u32}
pub fn timestamp_ns()->u64 {(crate::svc::system_tick() as u128*1_000_000_000/nxabi::GUEST_TICK_HZ as u128) as u64}
pub fn queue_frame(layer:u64)->u64 {
    let mut layers=LAYERS.lock().unwrap();let layer=layers.get_mut(&layer).expect("presenting an unknown VI layer");
    layer.frame+=1;let frame=layer.frame;
    layer.history.push_front(FrameRecord{frame,queued_ns:timestamp_ns(),presented_ns:0,state:2});
    layer.history.truncate(32);frame
}
pub fn frame_presented(layer:u64,frame:u64,time_ns:u64) {
    if let Some(l)=LAYERS.lock().unwrap().get_mut(&layer) {
        if let Some(f)=l.history.iter_mut().find(|f|f.frame==frame) {
            f.presented_ns=time_ns.max(f.queued_ns);f.state=3;
        }
    }
}
pub fn frame_history(layer:u64,count:usize)->Option<Vec<FrameRecord>> {
    let layers=LAYERS.lock().unwrap();let l=layers.get(&layer).filter(|l|l.opened)?;
    Some(l.history.iter().take(count.min(32)).copied().collect())
}

pub fn open(name: &str) -> Arc<dyn Service> {
    Arc::new(Table {
        name: match name {
            "vi:m" => "vi:m",
            "vi:s" => "vi:s",
            _ => "vi:u",
        },
        cmds: &[
            (0, Act::Obj(|| Arc::new(DisplayService))),
            (1, Act::Obj(|| Arc::new(DisplayService))),
            (2, Act::Obj(|| Arc::new(DisplayService))),
        ],
    })
}

pub fn docked() -> bool {
    static D: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| std::env::var("NX_HANDHELD").map_or(true, |v| v == "0"));
    *D
}

pub const UHD_HEAP: u64 = 0xFFE0_0000;

pub fn uhd() -> bool {
    static U: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        if !docked() || !std::env::var("NX_4K").is_ok_and(|v| v == "1") {
            return false;
        }
        let fits = crate::kernel::aspace().layout.heap_size >= UHD_HEAP && crate::gfx::max_buffer_len() >= UHD_HEAP;
        if !fits {
            eprintln!("[vi] 4K needs a {UHD_HEAP:#x}-byte guest heap the GPU can map: staying at 1080p");
        }
        fits
    });
    *U
}

fn display_resolution(docked:bool,uhd:bool,native1080:bool)->(u32,u32) {
    if native1080 {(1920,1080)}else if uhd {(3840,2160)}else if docked {(1920,1080)}else{(1280,720)}
}
pub fn resolution() -> (u32, u32) {
    static FULL_HD:LazyLock<bool>=LazyLock::new(||std::env::var("NX_NATIVE_1080").is_ok_and(|v|v=="1"));
    if *FULL_HD {
        static ONCE:std::sync::Once=std::sync::Once::new();
        ONCE.call_once(||eprintln!("[vi] explicit native1920x1080 display; originaloperationmode docked={} remainsindependent",docked()));
        return display_resolution(docked(),false,true);
    }
    if uhd() {
        (3840, 2160)
    } else if docked() {
        (1920, 1080)
    } else {
        (1280, 720)
    }
}

const DISPLAY_ID: u64 = 0;

#[derive(Clone)]
struct Layer {
    frame:u64,
    history:VecDeque<FrameRecord>,
    aruid: u64,
    opened: bool,
    size: (u64, u64),
    position: (f32, f32),
    z: i64,
    scaling: u32,
}
static LAYERS: LazyLock<Mutex<BTreeMap<u64, Layer>>> = LazyLock::new(Default::default);

fn create_layer(aruid: u64) -> u64 {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
    let id = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let (w, h) = resolution();
    LAYERS.lock().unwrap().insert(id, Layer { frame:0,history:VecDeque::new(),aruid, opened: false, size: (w as u64, h as u64), position: (0.0, 0.0), z: 0, scaling: 2 });
    id
}

pub fn create_managed_layers() -> (u64, u64) { (create_layer(0x100), create_layer(0x100)) }

fn is_default_name(bytes: &[u8]) -> bool { bytes.split(|b| *b == 0).next() == Some(b"Default".as_slice()) }

struct ManagerDisplayService;
impl Service for ManagerDisplayService {
    fn name(&self) -> String { "vi::IManagerDisplayService".into() }
    fn call(&self, req: &Request, rsp: &mut Response) -> u32 {
        match req.cmd {
            2010 => { if req.u64(8) != DISPLAY_ID { return crate::kernel::rc::NOT_FOUND; } rsp.push_u64(create_layer(req.u64(16))); }
            2011 => { if LAYERS.lock().unwrap().remove(&req.u64(0)).is_none() { return crate::kernel::rc::NOT_FOUND; } }
            _ => return Stub(self.name()).call(req, rsp),
        }
        0
    }
}

struct SystemDisplayService;
impl Service for SystemDisplayService {
    fn name(&self) -> String { "vi::ISystemDisplayService".into() }
    fn call(&self, req: &Request, rsp: &mut Response) -> u32 {
        match req.cmd {
            1200 | 1202 => { if req.u64(0) != DISPLAY_ID { return crate::kernel::rc::NOT_FOUND; } rsp.push_u64(if req.cmd == 1200 { 0 } else { 127 }); }
            1203 => { if req.u64(0) != DISPLAY_ID { return crate::kernel::rc::NOT_FOUND; } let (w, h) = resolution(); rsp.push_u32(w); rsp.push_u32(h); }
            2201 | 2203 | 2205 => {
                let id = req.u64(if req.cmd == 2201 { 8 } else { 0 });
                let mut layers = LAYERS.lock().unwrap();
                let Some(layer) = layers.get_mut(&id) else { return crate::kernel::rc::NOT_FOUND; };
                match req.cmd {
                    2201 => {
                        let (x, y) = (f32::from_bits(req.u32(0)), f32::from_bits(req.u32(4)));
                        if !x.is_finite() || !y.is_finite() { return crate::kernel::rc::INVALID_ENUM; }
                        if x != 0.0 || y != 0.0 { return crate::ipc::rc::UNKNOWN_COMMAND; }
                        layer.position = (x, y);
                    }
                    2203 => { let (w, h) = (req.u64(8), req.u64(16)); if w == 0 || h == 0 || w > 16384 || h > 16384 { return crate::kernel::rc::INVALID_SIZE; } layer.size = (w, h); }
                    _ => layer.z = req.u64(8) as i64,
                }
            }
            _ => return Stub(self.name()).call(req, rsp),
        }
        0
    }
}

struct DisplayService;

impl Service for DisplayService {
    fn name(&self) -> String {
        "vi::IApplicationDisplayService".into()
    }
    fn call(&self, req: &Request, rsp: &mut Response) -> u32 {
        match req.cmd {
            100 | 103 => rsp.objects.push(Arc::new(super::binder::HosBinderDriver)),
            101 => rsp.objects.push(Arc::new(SystemDisplayService)),
            102 => rsp.objects.push(Arc::new(ManagerDisplayService)),
            1000 => {
                if let Some(b) = req.b.first() {
                    let mut info = vec![0u8; 0x60];
                    info[..7].copy_from_slice(b"Default");
                    info[0x40] = 1;
                    info[0x48..0x50].copy_from_slice(&1u64.to_le_bytes());
                    let (w, h) = resolution();
                    info[0x50..0x58].copy_from_slice(&(w as u64).to_le_bytes());
                    info[0x58..0x60].copy_from_slice(&(h as u64).to_le_bytes());
                    write_buffer(b, &info);
                }
                rsp.push_u64(1);
            }
            1010 => { if !is_default_name(&req.data) { return crate::kernel::rc::NOT_FOUND; } rsp.push_u64(DISPLAY_ID); }
            1011 => rsp.push_u64(DISPLAY_ID),
            1020 | 1101 => { if req.u64(0) != DISPLAY_ID { return crate::kernel::rc::NOT_FOUND; } }
            2021 => { let mut layers = LAYERS.lock().unwrap(); let Some(l) = layers.get_mut(&req.u64(0)) else { return crate::kernel::rc::NOT_FOUND; }; l.opened = false; }
            2031 => { if LAYERS.lock().unwrap().remove(&req.u64(0)).is_none() { return crate::kernel::rc::NOT_FOUND; } }
            2101 => {
                let mut layers = LAYERS.lock().unwrap(); let Some(l) = layers.get_mut(&req.u64(8)) else { return crate::kernel::rc::NOT_FOUND; };
                if !matches!(req.u32(0), 0 | 2) { return crate::ipc::rc::UNKNOWN_COMMAND; } l.scaling = req.u32(0);
            }
            1102 => {
                if req.u64(0) != DISPLAY_ID { return crate::kernel::rc::NOT_FOUND; }
                let (w, h) = resolution();
                rsp.push_u64(w as u64);
                rsp.push_u64(h as u64);
            }
            2020 | 2030 => {
                let id = if req.cmd == 2030 {
                    if req.u64(8) != DISPLAY_ID { return crate::kernel::rc::NOT_FOUND; }
                    let id = create_layer(0x100); rsp.push_u64(id); id
                } else {
                    if req.data.len() < 80 || !is_default_name(&req.data[..64]) { return crate::kernel::rc::NOT_FOUND; }
                    req.u64(64)
                };
                let mut layers = LAYERS.lock().unwrap();
                let Some(l) = layers.get_mut(&id) else { return crate::kernel::rc::NOT_FOUND; };
                if req.cmd == 2020 && req.u64(72) != l.aruid { return crate::kernel::rc::INVALID_HANDLE; }
                let parcel = native_window_parcel(id as u32);
                let Some(b) = req.out_buffer().filter(|b| b.size >= parcel.len() as u64) else { return crate::kernel::rc::INVALID_SIZE; };
                write_buffer(&b, &parcel); l.opened = true;
                rsp.push_u64(parcel.len() as u64);
            }
            2102 => rsp.push_u64(req.u64(0)),
            5202 => {
                let handle = vsync_event();
                let mut k = crate::kernel::KERNEL.lock();
                let Some(crate::kernel::Obj::ReadableEvent(event)) = k.get(handle) else { return crate::kernel::rc::INVALID_HANDLE; };
                rsp.copy_handles.push(k.add(crate::kernel::Obj::ReadableEvent(event)));
            }
            _ => return Stub(self.name()).call(req, rsp),
        }
        0
    }
}

fn native_window_parcel(binder_id: u32) -> Vec<u8> {
    let mut data = Vec::new();
    for w in [2u32, 1, binder_id, 0, 0, 0] {
        data.extend_from_slice(&w.to_le_bytes());
    }
    data.extend_from_slice(b"dispdrv\0");
    data.extend_from_slice(&[0u8; 8]);
    let mut p = Vec::new();
    for w in [data.len() as u32, 0x10, 4, 0x10 + data.len() as u32] {
        p.extend_from_slice(&w.to_le_bytes());
    }
    p.extend(data);
    p.extend_from_slice(&0u32.to_le_bytes());
    p
}

fn vsync_event() -> u32 {
    use std::sync::OnceLock;
    static EVENT: OnceLock<u32> = OnceLock::new();
    *EVENT.get_or_init(|| {
        let h = new_event(false);
        std::thread::Builder::new()
            .name("vsync".into())
            .spawn(move || {
                let period = std::time::Duration::from_nanos(16_666_667);
                let mut next = std::time::Instant::now();
                loop {
                    next += if crate::gfx::turbo() { period / 8 } else { period };
                    let now = std::time::Instant::now();
                    if next > now {
                        std::thread::sleep(next - now);
                    } else if now - next > period * 4 {
                        next = now;
                    }
                    let e = crate::kernel::KERNEL.lock().get(h);
                    if let Some(crate::kernel::Obj::ReadableEvent(e)) = e {
                        crate::kernel::signal_event(&e);
                    }
                }
            })
            .unwrap();
        h
    })
}

#[cfg(test)]
mod layer_tests {
    use super::*;
    #[test]
    fn frame_history_parcel_is_empty_before_queue_then_tracks_only_actual_notifications() {
        let (layer,other)=create_managed_layers();
        LAYERS.lock().unwrap().get_mut(&layer).unwrap().opened=true;
        LAYERS.lock().unwrap().get_mut(&other).unwrap().opened=true;
        let query=|count:i32| {
            let mut input=Vec::new();for value in [4u32,16,0,20] {input.extend_from_slice(&value.to_le_bytes());}input.extend_from_slice(&count.to_le_bytes());
            let mut output=[0u8;1024];let mut data=Vec::new();for value in [layer as u32,17,0] {data.extend_from_slice(&value.to_le_bytes());}
            let req=Request{cmd:3,data,a:vec![crate::ipc::Buffer{addr:input.as_ptr() as u64,size:input.len() as u64}],b:vec![crate::ipc::Buffer{addr:output.as_mut_ptr() as u64,size:output.len() as u64}],..Default::default()};
            assert_eq!(super::super::binder::HosBinderDriver.call(&req,&mut Response::default()),0);
            let len=u32::from_le_bytes(output[..4].try_into().unwrap()) as usize;output[16..16+len].to_vec()
        };
        assert_eq!(query(8),[0u8;8]);assert_eq!(query(0)[..4],(-22i32).to_le_bytes());
        let first=queue_frame(layer);let second=queue_frame(layer);
        let queued=frame_history(layer,8).unwrap()[1].queued_ns;frame_presented(layer,first,queued+1234);
        let data=query(8);assert_eq!(data.len(),8+2*28);assert_eq!(&data[4..8],&2i32.to_le_bytes());
        assert_eq!(u64::from_le_bytes(data[8..16].try_into().unwrap()),second);
        assert_eq!(u32::from_le_bytes(data[32..36].try_into().unwrap()),2);
        assert_eq!(u64::from_le_bytes(data[36..44].try_into().unwrap()),first);
        assert_eq!(u64::from_le_bytes(data[52..60].try_into().unwrap()),queued+1234);
        assert_eq!(u32::from_le_bytes(data[60..64].try_into().unwrap()),3);
        assert!(frame_history(other,8).unwrap().is_empty());
        for _ in 0..40 {queue_frame(layer);}
        assert_eq!(frame_history(layer,100).unwrap().len(),32);assert_eq!(frame_history(layer,3).unwrap().len(),3);
        LAYERS.lock().unwrap().remove(&layer);LAYERS.lock().unwrap().remove(&other);
    }
    #[test]
    fn layer_parcels_identify_actual_layer_and_lifecycle_rejects_stale_ids() {
        let (display, recording) = create_managed_layers();
        assert_ne!(display, recording);
        for id in [display, recording] {
            let mut data = vec![0; 80]; data[..7].copy_from_slice(b"Default");
            data[64..72].copy_from_slice(&id.to_le_bytes()); data[72..80].copy_from_slice(&0x100u64.to_le_bytes());
            let mut parcel = [0u8; 256];
            let req = Request { cmd:2020, data, b:vec![crate::ipc::Buffer {addr:parcel.as_mut_ptr() as u64, size:256}], ..Default::default() };
            let mut out = Response::default();
            assert_eq!(DisplayService.call(&req, &mut out), 0);
            let n = u64::from_le_bytes(out.data[..8].try_into().unwrap()) as usize;
            assert!(n <= 256);
            let offset = u32::from_le_bytes(parcel[4..8].try_into().unwrap()) as usize;
            assert_eq!(u32::from_le_bytes(parcel[offset + 8..offset + 12].try_into().unwrap()), id as u32);
            assert!(LAYERS.lock().unwrap()[&id].opened);
            assert_eq!(DisplayService.call(&Request {cmd:2021, data:id.to_le_bytes().to_vec(), ..Default::default()}, &mut Response::default()), 0);
            assert!(!LAYERS.lock().unwrap()[&id].opened);
            assert_eq!(ManagerDisplayService.call(&Request {cmd:2011, data:id.to_le_bytes().to_vec(), ..Default::default()}, &mut Response::default()), 0);
            assert_ne!(DisplayService.call(&req, &mut Response::default()), 0);
        }
    }

    #[test]
    fn display_vsync_handles_can_be_closed_independently() {
        let call = || { let mut r = Response::default(); assert_eq!(DisplayService.call(&Request {cmd:5202, ..Default::default()}, &mut r), 0); r.copy_handles[0] };
        let a = call(); let b = call(); assert_ne!(a, b);
        let closed = {crate::kernel::KERNEL.lock().close(a)}; drop(closed);
        assert!(matches!(crate::kernel::KERNEL.lock().get(b), Some(crate::kernel::Obj::ReadableEvent(_))));
        let closed = {crate::kernel::KERNEL.lock().close(b)}; drop(closed);
    }
}
