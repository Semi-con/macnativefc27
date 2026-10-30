// Mush8 - hid.rs
// Constantin Clerc (c22dev) - 0.1
use crate::ipc::{Request, Response, Service, Stub};
use crate::kernel::{KERNEL, Obj};
use std::sync::{Arc, OnceLock};

pub const SHARED_MEMORY_SIZE: u64 = 0x40000;

pub fn shared_memory() -> &'static Arc<crate::kernel::SharedMemory> {
    static SHM: OnceLock<Arc<crate::kernel::SharedMemory>> = OnceLock::new();
    SHM.get_or_init(|| Arc::new(crate::shm::create(SHARED_MEMORY_SIZE).expect("hid shared memory")))
}

const NPAD: u64 = 0x9A00;
const NPAD_SIZE: u64 = 0x5000;
const LIFO_ENTRIES: u64 = 17;
const GENERIC_ENTRY: u64 = 0x30;
const LIFO_FULLKEY: u64 = 0x28;
const LIFO_HANDHELD: u64 = 0x378;
const SIXAXIS_ENTRY: u64 = 0x68;
const SIXAXIS_LIFOS: [u64; 6] = [0x1758, 0x1E60, 0x2568, 0x2C70, 0x3378, 0x3A80];
const DEVICE_TYPE: u64 = 0x4188;
const TOUCH_LIFO: u64 = 0x400;
const TOUCH_ENTRY: u64 = 0x298;
const TOUCH_STATE: u64 = 0x28;
static TOUCH_RANGE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1280 | 720 << 32);
static TOUCH_ACTIVE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
static TOUCH_CONFIG: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static GESTURE_RANGE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1280 | 720 << 32);
static GESTURE_ACTIVE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
const GESTURE_LIFO: u64 = 0x3ba00;
const GESTURE_ENTRY: u64 = 0x68;

#[repr(C)]
#[derive(Default, Clone, Copy)]
struct GestureState {
    sampling: u64, context: u64,
    kind: u32, direction: u32, x: u32, y: u32,
    dx: i32, dy: i32, vx: f32, vy: f32,
    attributes: u32, scale: f32, rotation: f32, point_count: i32,
    points: [[u32; 2]; 4],
}
const _: () = assert!(size_of::<GestureState>() == 0x60);

#[derive(Default)]
struct GestureRecognizer {
    context: u64, last: Option<[u32; 2]>, origin: [u32; 2],
    elapsed: f32, dragged: bool, pending_complete: bool,
}

impl GestureRecognizer {
    fn update(&mut self, touches: &[(u32, f32, f32, u32)], dt: f32) -> GestureState {
        let range = GESTURE_RANGE.load(std::sync::atomic::Ordering::Relaxed);
        let (w, h) = (range as u32 as f32, (range >> 32) as u32 as f32);
        let points: Vec<_> = touches.iter().filter(|t| t.3 & 2 == 0).take(4).map(|t| [(t.1 * w).clamp(0.0, w - 1.0) as u32, (t.2 * h).clamp(0.0, h - 1.0) as u32]).collect();
        let mut s = GestureState { context: self.context, scale: 1.0, point_count: points.len() as i32, ..Default::default() };
        for (to, point) in s.points.iter_mut().zip(&points) { *to = *point; }
        if points.len() > 1 {
            // This Mac input path provides one pointer. Publish an explicit
            // cancellation instead of fabricating pinch/rotation measurements.
            s.kind = 2; self.last = None; self.pending_complete = false; return s;
        }
        let direction = |dx: i32, dy: i32| if dx.abs() >= dy.abs() { if dx < 0 {1} else {3} } else if dy < 0 {2} else {4};
        if let Some(&p) = points.first() {
            s.x = p[0]; s.y = p[1];
            if let Some(last) = self.last {
                self.elapsed += dt;
                s.dx = p[0] as i32 - last[0] as i32; s.dy = p[1] as i32 - last[1] as i32;
                s.vx = s.dx as f32 / dt.max(0.001); s.vy = s.dy as f32 / dt.max(0.001);
                self.dragged |= p[0].abs_diff(self.origin[0]) > 3 || p[1].abs_diff(self.origin[1]) > 3;
                s.kind = if self.dragged {6} else if self.elapsed >= 0.5 {4} else {3};
                if s.dx != 0 || s.dy != 0 { s.direction = direction(s.dx, s.dy); }
            } else {
                self.context += 1; self.origin = p; self.elapsed = 0.0; self.dragged = false;
                self.pending_complete = false; s.kind = 3; s.attributes = 1 << 4;
            }
            self.last = Some(p); s.context = self.context;
        } else if let Some(last) = self.last.take() {
            s.x = last[0]; s.y = last[1];
            let (dx, dy) = (last[0] as i32 - self.origin[0] as i32, last[1] as i32 - self.origin[1] as i32);
            s.kind = if !self.dragged && self.elapsed < 0.5 {5} else if self.dragged && self.elapsed < 0.3 {7} else {1};
            if self.dragged { s.direction = direction(dx, dy); }
            self.pending_complete = s.kind != 1;
        } else if std::mem::take(&mut self.pending_complete) { s.kind = 1; }
        s
    }
}

fn push_gesture(lifo: u64, sampling: u64, state: GestureState) {
    let (tail, e) = lifo_slot(lifo, GESTURE_ENTRY);
    lifo_write_begin(e,sampling);
    unsafe { std::ptr::copy_nonoverlapping((&state as *const GestureState as *const u8).add(8), (e + 0x10) as *mut u8, 0x58); }
    std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
    write(e + 8, sampling);
    lifo_write_end(e,sampling);
    lifo_commit(lifo, sampling, tail);
}

const STYLE_FULLKEY: u32 = 1;
const DEVICE_FULLKEY: u32 = 1;
const ATTR_CONNECTED: u32 = 1;
const ATTR_WIRED: u32 = 2;

#[derive(Default, Clone, Copy)]
pub struct PadState {
    pub buttons: u64,
    pub left: (i32, i32),
    pub right: (i32, i32),
}

pub static BUTTONS: [std::sync::atomic::AtomicU64; crate::input::MAX_PLAYERS] = [const { std::sync::atomic::AtomicU64::new(0) }; crate::input::MAX_PLAYERS];

static SUPPORTED_STYLES: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(u32::MAX);
static HOLD_TYPE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
static NPAD_ACTIVE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
static SUPPORTED_IDS: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0x3ff);
static DISCONNECTED: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
static HANDHELD_MODE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static CENTER_CLAMP: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);
static HOME_PROTECTION: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

fn enabled(index: usize) -> bool {
    use std::sync::atomic::Ordering::Relaxed;
    NPAD_ACTIVE.load(Relaxed) && SUPPORTED_STYLES.load(Relaxed) & STYLE_FULLKEY != 0
        && SUPPORTED_IDS.load(Relaxed) & (1 << index) != 0 && DISCONNECTED.load(Relaxed) & (1 << index) == 0
}

static STYLE_EVENTS: std::sync::Mutex<Vec<(usize, u32)>> = std::sync::Mutex::new(Vec::new());

fn npad_index(id: u32) -> Option<usize> {
    match id {
        0..=7 => Some(id as usize),
        0x20 => Some(8),
        0x10 => Some(9),
        _ => None,
    }
}

fn signal_style_events(index: usize) {
    let handles: Vec<u32> = STYLE_EVENTS.lock().unwrap().iter().filter(|(i, _)| *i == index).map(|(_, h)| *h).collect();
    for h in handles {
        let o = KERNEL.lock().get(h);
        if let Some(Obj::ReadableEvent(e)) = o {
            crate::kernel::signal_event(&e);
        }
    }
}

fn write<T: Copy>(addr: u64, v: T) {
    unsafe { (addr as *mut T).write_volatile(v) }
}
fn read<T: Copy>(addr: u64) -> T {
    unsafe { (addr as *const T).read_volatile() }
}

fn lifo_slot(lifo: u64, entry: u64) -> (u64, u64) {
    let tail = (read::<u64>(lifo + 0x10) + 1) % LIFO_ENTRIES;
    (tail, lifo + 0x20 + tail * entry)
}

fn lifo_write_begin(entry:u64,sampling:u64) {
    // SDK AtomicStorage uses a sequence lock, separate from payload sampling.
    // Odd means being modified; stable entries must have an even version.
    unsafe { std::sync::atomic::AtomicU64::from_ptr(entry as *mut u64) }.store(sampling.wrapping_mul(2)|1,std::sync::atomic::Ordering::SeqCst);
    std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
}
fn lifo_write_end(entry:u64,sampling:u64) {
    unsafe { std::sync::atomic::AtomicU64::from_ptr(entry as *mut u64) }.store(sampling.wrapping_mul(2),std::sync::atomic::Ordering::Release);
}

fn lifo_commit(lifo: u64, sampling: u64, tail: u64) {
    std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
    write(lifo, sampling);
    write(lifo + 0x08, LIFO_ENTRIES);
    write(lifo + 0x10, tail);
    write(lifo + 0x18, (read::<u64>(lifo + 0x18) + 1).min(LIFO_ENTRIES));
}

fn push_generic(lifo: u64, sampling: u64, p: &PadState, attributes: u32) {
    let (tail, e) = lifo_slot(lifo, GENERIC_ENTRY);
    lifo_write_begin(e,sampling);
    write(e + 0x10, p.buttons);
    write(e + 0x18, p.left.0);
    write(e + 0x1c, p.left.1);
    write(e + 0x20, p.right.0);
    write(e + 0x24, p.right.1);
    write(e + 0x28, attributes);
    std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
    write(e + 0x08, sampling);
    lifo_write_end(e,sampling);
    lifo_commit(lifo, sampling, tail);
}

fn push_touch(lifo: u64, sampling: u64, touches: &[(u32, f32, f32, u32)]) {
    let range = TOUCH_RANGE.load(std::sync::atomic::Ordering::Relaxed);
    let (width, height) = (range as u32 as f32, (range >> 32) as u32 as f32);
    let (tail, e) = lifo_slot(lifo, TOUCH_ENTRY);
    let touches = &touches[..touches.len().min(16)];
    lifo_write_begin(e,sampling);
    write(e + 0x10, touches.len() as i32);
    write(e + 0x14, 0u32);
    for (i, &(finger, x, y, attr)) in touches.iter().enumerate() {
        let t = e + 0x18 + i as u64 * TOUCH_STATE;
        write(t, 0u64);
        write(t + 0x08, attr);
        write(t + 0x0c, finger);
        write(t + 0x10, (x * width).clamp(0.0, width - 1.0) as u32);
        write(t + 0x14, (y * height).clamp(0.0, height - 1.0) as u32);
        write(t + 0x18, 15u32);
        write(t + 0x1c, 15u32);
        write(t + 0x20, 0u32);
        write(t + 0x24, 0u32);
    }
    std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
    write(e + 0x08, sampling);
    lifo_write_end(e,sampling);
    lifo_commit(lifo, sampling, tail);
}

static AT_REST: [std::sync::atomic::AtomicBool; crate::input::MAX_PLAYERS] = [const { std::sync::atomic::AtomicBool::new(true) }; crate::input::MAX_PLAYERS];

fn push_sixaxis(lifo: u64, sampling: u64, m: Option<&crate::input::Motion>, fused: &crate::motion::Orientation) {
    let (tail, e) = lifo_slot(lifo, SIXAXIS_ENTRY);
    let (accel, gyro) = m.map_or(([0.0, 0.0, -1.0], [0.0; 3]), |m| (m.accel, fused.gyro(m.gyro.map(|r| r / std::f32::consts::TAU))));
    lifo_write_begin(e,sampling);
    write(e + 0x08, 5_000_000u64);
    write(e + 0x18, accel);
    write(e + 0x24, gyro);
    write(e + 0x30, fused.rotation());
    write(e + 0x3c, fused.orientation());
    write(e + 0x60, m.is_some() as u32);
    std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
    write(e + 0x10, sampling);
    lifo_write_end(e,sampling);
    lifo_commit(lifo, sampling, tail);
}

fn start_input_thread() {
    static STARTED: OnceLock<()> = OnceLock::new();
    STARTED.get_or_init(|| {
        let base = shared_memory().host;
        crate::input::SCREEN_ACTIVE.store(true, std::sync::atomic::Ordering::Relaxed);
        std::thread::Builder::new()
            .name("hid".into())
            .spawn(move || {
                let mut pads = crate::input::Pads::new();
                let mut present = [false; crate::input::MAX_PLAYERS];
                let mut sampling = 0u64;
                let mut fused = [crate::motion::Orientation::default(); crate::input::MAX_PLAYERS];
                let mut moving = [false; crate::input::MAX_PLAYERS];
                let mut last = std::time::Instant::now();
                let mut gestures = GestureRecognizer::default();
                loop {
                    let dt = last.elapsed().as_secs_f32().min(0.1);
                    last = std::time::Instant::now();
                    sampling += 1;
                    let states = pads.poll();
                    for (i, input) in states.iter().enumerate() {
                        let state = input.filter(|_| enabled(i));
                        let st = &state;
                        let npad = base + NPAD + i as u64 * NPAD_SIZE;
                        if st.is_some() != present[i] {
                            present[i] = st.is_some();
                            let (style, device) = if present[i] { (STYLE_FULLKEY, DEVICE_FULLKEY) } else { (0, 0) };
                            write(npad, style);
                            write(npad + DEVICE_TYPE, device);
                            if !present[i] {
                                push_generic(npad + LIFO_FULLKEY, sampling, &PadState::default(), 0);
                            }
                            signal_style_events(i);
                        }
                        if let Some(p) = st {
                            push_generic(npad + LIFO_FULLKEY, sampling, p, ATTR_CONNECTED | ATTR_WIRED);
                        }
                        let sample = if present[i] { crate::input::MOTION.lock().unwrap()[i] } else { None };
                        let (motion, from_pad) = (sample.map(|(m, _)| m), sample.is_some_and(|(_, pad)| pad));
                        let still = motion.is_none_or(|m| m.gyro.iter().all(|r| r.abs() < 0.05));
                        AT_REST[i].store(still, std::sync::atomic::Ordering::Relaxed);
                        if motion.is_some() && !std::mem::replace(&mut moving[i], true) {
                            eprintln!("[hid] motion data for player {}", i + 1);
                        }
                        match motion {
                            Some(m) => fused[i].update(m.accel, m.gyro.map(|r| r / std::f32::consts::TAU), from_pad, dt),
                            None => fused[i] = crate::motion::Orientation::default(),
                        }
                        if present[i] {
                            for l in SIXAXIS_LIFOS {
                                push_sixaxis(npad + l, sampling, motion.as_ref(), &fused[i]);
                            }
                        }
                        BUTTONS[i].store(st.map_or(0, |p| p.buttons), std::sync::atomic::Ordering::Relaxed);
                    }
                    push_generic(base + NPAD + 8 * NPAD_SIZE + LIFO_HANDHELD, sampling, &PadState::default(), 0);
                    let touches = crate::input::screen_sample();
                    push_touch(base + TOUCH_LIFO, sampling, if TOUCH_ACTIVE.load(std::sync::atomic::Ordering::Relaxed) { &touches } else { &[] });
                    if GESTURE_ACTIVE.load(std::sync::atomic::Ordering::Relaxed) { push_gesture(base + GESTURE_LIFO, sampling, gestures.update(&touches, dt)); }
                    std::thread::sleep(std::time::Duration::from_millis(5));
                }
            })
            .unwrap();
    });
}

fn vibration_target(handle: u32) -> Option<(usize, usize)> {
    let player = npad_index(handle >> 8 & 0xff).filter(|&i| i < crate::input::MAX_PLAYERS).unwrap_or(0);
    Some((player, (handle >> 16 & 0xff == 1) as usize))
}

fn vibration_value(b: &[u8]) -> crate::haptics::Value {
    let f = |i: usize| b.get(i * 4..i * 4 + 4).map_or(0.0, |b| f32::from_le_bytes(b.try_into().unwrap()));
    crate::haptics::Value { amp_low: f(0), freq_low: f(1), amp_high: f(2), freq_high: f(3) }
}

pub struct Hid;

impl Service for Hid {
    fn name(&self) -> String {
        "hid".into()
    }
    fn call(&self, req: &Request, rsp: &mut Response) -> u32 {
        match req.cmd {
            0 => {
                start_input_thread();
                rsp.objects.push(Arc::new(AppletResource));
            }
            11 => TOUCH_ACTIVE.store(true, std::sync::atomic::Ordering::Relaxed),
            91 => { if req.u32(0) != 1 { return crate::kernel::rc::INVALID_ENUM; } GESTURE_ACTIVE.store(true, std::sync::atomic::Ordering::Relaxed); }
            92 => {
                let (width, height) = (req.u32(0), req.u32(4));
                if width == 0 || height == 0 || width > 16384 || height > 16384 { return crate::kernel::rc::INVALID_SIZE; }
                GESTURE_RANGE.store(width as u64 | (height as u64) << 32, std::sync::atomic::Ordering::Relaxed);
            }
            106 => {
                let Some(i) = npad_index(req.u32(0)) else { return crate::kernel::rc::INVALID_ENUM; };
                let mut events = STYLE_EVENTS.lock().unwrap();
                let h = if let Some((_, h)) = events.iter().find(|(index, _)| *index == i) { *h } else {
                    let h = crate::ipc::new_event(true); events.push((i, h)); h
                };
                drop(events);
                let mut k = KERNEL.lock();
                let Some(Obj::ReadableEvent(event)) = k.get(h) else { return crate::kernel::rc::INVALID_HANDLE; };
                rsp.copy_handles.push(k.add(Obj::ReadableEvent(event)));
            }
            66 | 67 | 69 | 70 | 72 | 79 | 81 => {}
            68 => rsp.push_u32(0),
            71 => {
                rsp.push_u32(0);
                rsp.push_u32(0);
            }
            80 => rsp.push_u32(0),
            82 => rsp.push_u32(vibration_target(req.u32(0)).is_none_or(|(p, _)| AT_REST[p].load(std::sync::atomic::Ordering::Relaxed)) as u32),
            100 => SUPPORTED_STYLES.store(req.u32(0), std::sync::atomic::Ordering::Relaxed),
            101 => rsp.push_u32(SUPPORTED_STYLES.load(std::sync::atomic::Ordering::Relaxed)),
            102 => {
                let Some(buffer) = req.in_buffer() else { return crate::kernel::rc::INVALID_SIZE; };
                if buffer.size > 40 || buffer.size % 4 != 0 { return crate::kernel::rc::INVALID_SIZE; }
                let mut mask = 0;
                for i in 0..buffer.size / 4 {
                    let id = unsafe { ((buffer.addr + i * 4) as *const u32).read_unaligned() };
                    let Some(index) = npad_index(id) else { return crate::kernel::rc::INVALID_ENUM; };
                    mask |= 1 << index;
                }
                SUPPORTED_IDS.store(mask, std::sync::atomic::Ordering::Relaxed);
            }
            103 | 109 => {
                if req.cmd == 109 && !matches!(req.u32(0), 0 | 1 | 2 | 3 | 5) { return crate::kernel::rc::INVALID_ENUM; }
                DISCONNECTED.store(0, std::sync::atomic::Ordering::Relaxed);
                NPAD_ACTIVE.store(true, std::sync::atomic::Ordering::Relaxed);
                eprintln!("[hid] Npad activated, revision {}", if req.cmd == 109 { req.u32(0) } else { 0 });
            }
            104 => NPAD_ACTIVE.store(false, std::sync::atomic::Ordering::Relaxed),
            107 => {
                let Some(i) = npad_index(req.u32(0)) else { return crate::kernel::rc::INVALID_ENUM; };
                DISCONNECTED.fetch_or(1 << i, std::sync::atomic::Ordering::Relaxed);
            }
            108 => {
                let Some(i) = npad_index(req.u32(0)) else { return crate::kernel::rc::INVALID_ENUM; };
                rsp.push_u64([1, 3, 7, 15, 9, 5, 13, 6, 0, 0][i]);
            }
            120 => HOLD_TYPE.store(req.u64(8), std::sync::atomic::Ordering::Relaxed),
            121 => rsp.push_u64(HOLD_TYPE.load(std::sync::atomic::Ordering::Relaxed)),
            // FullKey controllers remain whole pads; Joy-Con assignment mode
            // changes do not split an SDL/GCController into synthetic halves.
            124 => { if npad_index(req.u32(0)).is_none() { return crate::kernel::rc::INVALID_ENUM; } }
            128 => { if req.u64(8) > 2 { return crate::kernel::rc::INVALID_ENUM; } HANDHELD_MODE.store(req.u64(8), std::sync::atomic::Ordering::Relaxed); }
            129 => rsp.push_u64(HANDHELD_MODE.load(std::sync::atomic::Ordering::Relaxed)),
            131 => { let Some(i) = npad_index(req.u32(0)) else { return crate::kernel::rc::INVALID_ENUM; }; rsp.data.push((HOME_PROTECTION.load(std::sync::atomic::Ordering::Relaxed) & (1 << i) != 0) as u8); }
            132 => {
                let Some(i) = npad_index(req.u32(4)) else { return crate::kernel::rc::INVALID_ENUM; };
                if req.data.first().copied().unwrap_or(0) != 0 { HOME_PROTECTION.fetch_or(1 << i, std::sync::atomic::Ordering::Relaxed); }
                else { HOME_PROTECTION.fetch_and(!(1 << i), std::sync::atomic::Ordering::Relaxed); }
            }
            134 => CENTER_CLAMP.store(req.data.first().copied().unwrap_or(0) != 0, std::sync::atomic::Ordering::Relaxed),
            200 => {
                rsp.push_u32(1);
                rsp.push_u32(match req.u32(0) >> 16 & 0xff {
                    0 => 1,
                    1 => 2,
                    _ => 0,
                });
            }
            201 => {
                if let Some((p, side)) = vibration_target(req.u32(0)) {
                    crate::haptics::send(p, side, vibration_value(&req.data[4.min(req.data.len())..]));
                }
            }
            202 => {
                let v = vibration_target(req.u32(0)).map_or(crate::haptics::Value::IDLE, |(p, side)| crate::haptics::last(p, side));
                for f in [v.amp_low, v.freq_low, v.amp_high, v.freq_high] {
                    rsp.push_u32(f.to_bits());
                }
            }
            203 => rsp.objects.push(Arc::new(crate::ipc::Table { name: "hid::IActiveVibrationDeviceList", cmds: &[(0, crate::ipc::Act::Ok)] })),
            204 | 207 | 209 | 210 => {}
            205 | 211 => rsp.push_u32(1),
            206 => {
                if let (Some(handles), Some(values)) = (req.x.first(), req.x.get(1)) {
                    let n = (handles.size / 4).min(values.size / 16);
                    for i in 0..n {
                        let handle: u32 = read(handles.addr + i * 4);
                        let raw: [u8; 16] = read(values.addr + i * 16);
                        if let Some((p, side)) = vibration_target(handle) {
                            crate::haptics::send(p, side, vibration_value(&raw));
                        }
                    }
                }
            }
            208 => rsp.push_u64(0),
            // Since Horizon 2.x these communication-mode commands are fixed
            // no-op/default-mode operations; they don't configure host Bluetooth.
            1000 => {},
            1001 => rsp.push_u64(3),
            1002 => TOUCH_CONFIG.store(req.u64(0), std::sync::atomic::Ordering::Relaxed),
            1003 => rsp.data.push(0),
            1004 => {
                let (width, height) = (req.u32(0), req.u32(4));
                if width == 0 || height == 0 || width > 16384 || height > 16384 { return crate::kernel::rc::INVALID_SIZE; }
                TOUCH_RANGE.store(width as u64 | (height as u64) << 32, std::sync::atomic::Ordering::Relaxed);
            }
            _ => return Stub(self.name()).call(req, rsp),
        }
        0
    }
}

struct AppletResource;

impl Service for AppletResource {
    fn name(&self) -> String {
        "hid::IAppletResource".into()
    }
    fn call(&self, req: &Request, rsp: &mut Response) -> u32 {
        match req.cmd {
            0 => {
                let h = KERNEL.lock().add(Obj::SharedMemory(shared_memory().clone()));
                rsp.copy_handles.push(h);
                0
            }
            _ => Stub(self.name()).call(req, rsp),
        }
    }
}

#[cfg(test)]
mod npad_tests {
    use super::*;
    use std::sync::atomic::Ordering::Relaxed;

    #[test]
    fn revision_policy_id_filter_disconnect_and_style_event_lifetime() {
        let old = (NPAD_ACTIVE.load(Relaxed), SUPPORTED_IDS.load(Relaxed), SUPPORTED_STYLES.load(Relaxed), DISCONNECTED.load(Relaxed));
        let call = |cmd, data: Vec<u8>| { let mut r = Response::default(); let rc = Hid.call(&Request { cmd, data, ..Default::default() }, &mut r); (rc, r) };
        assert_eq!(call(109, 4u32.to_le_bytes().to_vec()).0, crate::kernel::rc::INVALID_ENUM);
        assert_eq!(call(109, 5u32.to_le_bytes().to_vec()).0, 0);
        assert_eq!(call(100, 1u32.to_le_bytes().to_vec()).0, 0);
        let ids = [1u32];
        assert_eq!(Hid.call(&Request {cmd:102, x:vec![crate::ipc::Buffer {addr:ids.as_ptr() as u64, size:4}], ..Default::default()}, &mut Response::default()), 0);
        assert!(!enabled(0)); assert!(enabled(1));
        assert_eq!(call(107, 1u32.to_le_bytes().to_vec()).0, 0);
        assert!(!enabled(1));
        let a = call(106, 1u32.to_le_bytes().to_vec()).1.copy_handles[0];
        let b = call(106, 1u32.to_le_bytes().to_vec()).1.copy_handles[0];
        assert_ne!(a, b);
        let event = match KERNEL.lock().get(b).unwrap() { Obj::ReadableEvent(e) => e, _ => panic!("event") };
        event.signaled.store(false, Relaxed);
        let closed = { KERNEL.lock().close(a) }; drop(closed);
        signal_style_events(1); assert!(event.signaled.load(Relaxed));
        let closed = { KERNEL.lock().close(b) }; drop(closed);
        NPAD_ACTIVE.store(old.0, Relaxed); SUPPORTED_IDS.store(old.1, Relaxed); SUPPORTED_STYLES.store(old.2, Relaxed); DISCONNECTED.store(old.3, Relaxed);
    }

    #[test]
    fn generic_input_ring_wraps_with_complete_buttons_axes_and_versions() {
        let mut storage = vec![0u64; 0x350 / 8 * 2];
        let base = storage.as_mut_ptr() as u64;
        for n in 1..=50u64 {
            let p = PadState { buttons: n << 16, left: (n as i32, -(n as i32)), right: (123, -456) };
            push_generic(base, n, &p, ATTR_CONNECTED | ATTR_WIRED);
            let tail = read::<u64>(base + 0x10);
            let e = base + 0x20 + tail * GENERIC_ENTRY;
            assert_eq!((read::<u64>(e), read::<u64>(e + 8)), (2*n, n));
            assert_eq!(read::<u64>(e + 0x10), n << 16);
            assert_eq!((read::<i32>(e + 0x18), read::<i32>(e + 0x1c)), (n as i32, -(n as i32)));
            assert_eq!(read::<u32>(e + 0x28), 3);
            assert_eq!(read::<u64>(base + 0x18), n.min(17));
        }
        assert_eq!(read::<u64>(base + 0x10), 50 % 17);
        assert!(storage[0x350 / 8..].iter().all(|w| *w == 0), "input ring wrote into the next controller's area");
    }

    #[test]
    fn touch_output_ranges_scale_and_clamp_published_coordinates() {
        let old = TOUCH_RANGE.load(Relaxed);
        let mut data = Vec::new(); data.extend_from_slice(&1920u32.to_le_bytes()); data.extend_from_slice(&1080u32.to_le_bytes()); data.extend_from_slice(&0x100u64.to_le_bytes());
        assert_eq!(Hid.call(&Request {cmd:1004, data, ..Default::default()}, &mut Response::default()), 0);
        let mut storage = vec![0u64; (0x20 + TOUCH_ENTRY as usize * 17) / 8];
        let base = storage.as_mut_ptr() as u64;
        push_touch(base, 1, &[(5, 1.1, 0.5, 1)]);
        let e = base + 0x20 + TOUCH_ENTRY;
        assert_eq!(read::<u64>(e + 8), 1);
        assert_eq!(read::<u32>(e + 0x10), 1);
        assert_eq!(read::<u32>(e + 0x18 + 0x0c), 5);
        assert_eq!(read::<u32>(e + 0x18 + 0x10), 1919);
        assert_eq!(read::<u32>(e + 0x18 + 0x14), 540);
        assert_eq!(Hid.call(&Request {cmd:1004, data:vec![0;16], ..Default::default()}, &mut Response::default()), crate::kernel::rc::INVALID_SIZE);
        TOUCH_RANGE.store(old, Relaxed);
    }

    #[test]
    fn gestures_publish_touch_tap_pan_swipe_press_and_completion() {
        let mut g = GestureRecognizer::default();
        assert_eq!(g.update(&[], 0.005).kind, 0);
        let start = g.update(&[(1, 0.25, 0.5, 1)], 0.005);
        assert_eq!((start.kind, start.attributes, start.x, start.y, start.context), (3, 16, 320, 360, 1));
        assert_eq!(g.update(&[], 0.05).kind, 5);
        assert_eq!(g.update(&[], 0.005).kind, 1);
        assert_eq!(g.update(&[], 0.005).kind, 0);
        g.update(&[(2, 0.1, 0.1, 1)], 0.005);
        let pan = g.update(&[(2, 0.6, 0.1, 0)], 0.05);
        assert_eq!((pan.kind, pan.direction, pan.dx, pan.context), (6, 3, 640, 2));
        assert_eq!(g.update(&[], 0.005).kind, 7);
        assert_eq!(g.update(&[], 0.005).kind, 1);
        g.update(&[(3, 0.5, 0.5, 1)], 0.005);
        assert_eq!(g.update(&[(3, 0.5, 0.5, 0)], 0.6).kind, 4);
        assert_eq!(g.update(&[], 0.005).kind, 1);
        assert_eq!(g.update(&[(1, 0.1, 0.1, 1), (2, 0.2, 0.2, 1)], 0.005).kind, 2);
        let mut buffer = vec![0u64; (0x20 + GESTURE_ENTRY as usize * 17) / 8];
        let base = buffer.as_mut_ptr() as u64;
        push_gesture(base, 99, pan);
        let e = base + 0x20 + GESTURE_ENTRY;
        let payload: GestureState = read(e + 8);
        assert_eq!(read::<u64>(e), 2*payload.sampling);
        assert_eq!((payload.sampling, payload.kind, payload.context, payload.dx), (99, 6, 2, 640));
    }

    #[test]
    fn sdk_atomic_storage_reader_accepts_full_gesture_history() {
        let mut storage = vec![0u64;(0x20+17*GESTURE_ENTRY) as usize/8];
        let base = storage.as_mut_ptr() as u64;
        for n in 1..=35 { push_gesture(base,n,GestureState::default()); }
        let tail = read::<u64>(base+0x10);
        for i in 0..17 {
            let entry = base+0x20+((tail+17-i)%17)*GESTURE_ENTRY;
            let version = read::<u64>(entry);
            assert_eq!(version&1,0,"SDK retries forever on odd stable history entries");
            let s:GestureState = read(entry+8);
            std::sync::atomic::fence(std::sync::atomic::Ordering::Acquire);
            assert_eq!(read::<u64>(entry),version);
            assert_eq!(s.sampling,35-i);
        }
    }
}
