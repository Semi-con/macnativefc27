// Mush8 - mod.rs
// Constantin Clerc (c22dev) - 0.1
mod cmd;
mod draw;
mod fx;
mod events;
pub use draw::{precompile, prewarm};
mod objects;
mod gpu_domains;
mod virtual_lookup;
mod tex;

use crate::stubs::{self, Args};
use std::collections::HashMap;
use std::ffi::{CStr, c_char};
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::{Mutex, OnceLock};

struct DriverTables {
    module_id: &'static str,
    name_offsets: u64,
    names: u64,
    procs: u64,
    count: usize,
    property_helper: u64,
    format_table: u64,
    format_stride: u64,
}

const DRIVERS: &[DriverTables] = &[DriverTables {
    module_id: "fed212d6e0963706f9f4b16ea5d65879c4ec6c4d",
    name_offsets: 0x9d9548, names: 0x9d5be2, procs: 0xbabbd0,
    count: 534, property_helper: 0x4b2920,
    format_table: 0xba9898, format_stride: 0x38,
}];

struct Driver {
    procs: HashMap<String, u64>,
    property_helper: u64,
    format_table: u64,
    format_stride: u64,
}

static DRIVER: OnceLock<Driver> = OnceLock::new();

pub fn init(sdk_base: u64, module_id: &[u8]) -> anyhow::Result<()> {
    let id = hex_id(module_id);
    let t = DRIVERS.iter().find(|t| t.module_id == id).ok_or_else(|| anyhow::anyhow!("unknown sdk build {id}"))?;
    let mut procs = HashMap::new();
    for i in 0..t.count as u64 {
        let off = unsafe { *((sdk_base + t.name_offsets + 4 * i) as *const u32) } as u64;
        let name = unsafe { CStr::from_ptr((sdk_base + t.names + off) as *const c_char) }.to_string_lossy();
        let addr = unsafe { *((sdk_base + t.procs + 8 * i) as *const u64) };
        procs.insert(format!("nvn{name}"), addr);
    }
    let format_table = sdk_base + t.format_table;
    let _ = DRIVER.set(Driver { procs, property_helper: sdk_base + t.property_helper, format_table, format_stride: t.format_stride });
    Ok(())
}

fn hex_id(id: &[u8]) -> String {
    id.iter().take(20).map(|b| format!("{b:02x}")).collect()
}

fn is_driver_passthrough(name: &str) -> bool {
    const STATES: [&str; 8] = [
        "BlendState", "ChannelMaskState", "ColorState", "DepthStencilState", "MultisampleState", "PolygonState",
        "VertexAttribState", "VertexStreamState",
    ];
    let Some(rest) = name.strip_prefix("nvn") else { return false };
    STATES.iter().any(|s| rest.starts_with(s))
        // SDK 22 implementations are pure integer packing (4b8f20..4b8f48),
        // independent of its Tegra device object or driver state.
        || matches!(rest,"DeviceGetSeparateTextureHandle"|"DeviceGetSeparateSamplerHandle"|"DeviceGetTexelFetchHandle")
        // Sampler storage matches this SDK's packed builder, so its read-only
        // accessors (including floating-point returns) can execute unchanged.
        || rest.starts_with("SamplerGet") || rest == "SamplerCompare"
        || rest.starts_with("TextureView")
        || (rest.contains("Builder") && (rest.contains("BuilderSet") || rest.contains("BuilderGet")))
}

type HostFn = fn(&Args) -> u64;

fn host_impl(name: &str) -> Option<HostFn> {
    if let Some(f) = events::host_impl(name) { return Some(f); }
    if let Some(f) = objects::host_impl(name) {
        return Some(f);
    }
    Some(match name {
        "nvnDeviceInitialize" => |_| 1,
        "nvnDeviceGetInteger" => device_get_integer,
        "nvnTextureBuilderGetStorageSize" => |a| texture_builder_storage(canonical_texture_builder(a.0[0]).as_ptr() as u64).0,
        "nvnTextureBuilderGetStorageAlignment" => |a| texture_builder_storage(canonical_texture_builder(a.0[0]).as_ptr() as u64).1,
        "nvnTextureBuilderGetZCullStorageSize" => |_| 0,
        "nvnTextureBuilderGetStorageClass" => |a| {
            let b = canonical_texture_builder(a.0[0]);
            texture_storage_class(&b)
        },
        _ => return None,
    })
}

pub(super) fn texture_storage_class(bytes:&[u8;0x68])->u64 {
    let b=TextureBuilder::read(bytes.as_ptr() as u64);
    // SDK55.15 encodes an explicit page kind with bit8, including pitched
    // kind0. This backend has flat pool mappings and no Tegra compression,
    // so all block-linear texture formats use its uncompressed kind FE.
    0x100|if b.target==TARGET_BUFFER || b.flags&(FLAG_LINEAR|FLAG_LINEAR_RENDER_TARGET)!=0 {0} else {0xfe}
}

pub(crate) fn gfx_integer_block_format(format: u32) -> bool {
    matches!(format, 23 | 24 | 47 | 48)
}

fn format_info(format: u32) -> (u32, u32, u32) {
    // Common public NVN formats are independent of driver initialization.
    // These entries match the pinned SDK's format table (RGBA8 = 37).
    let plain=match format {1..=4=>Some(1),5..=9=>Some(2),10..=12=>Some(4),13..=16=>Some(2),17..=21=>Some(4),22..=24=>Some(8),37..=40=>Some(4),41..=45=>Some(8),46..=48=>Some(16),_=>None};
    if let Some(bytes)=plain {return (1,1,bytes);}
    if (61..=64).contains(&format) {return (1,1,4);}
    if format==52 {return (1,1,4);}
    if (153..=156).contains(&format) {return (1,1,4);}
    if format == 65 { return (4, 4, 8); }
    let d = DRIVER.get().unwrap();
    let e = d.format_table + d.format_stride * format as u64;
    let b = |o: u64| unsafe { *((e + o) as *const u8) } as u32;
    (b(8).max(1), b(9).max(1), b(0x10).max(1))
}

// SDK 22 packs and extends public builders differently. The Metal backend
// uses a private canonical layout; never expose that layout to guest setters.
pub(crate) fn canonical_texture_builder(raw: u64) -> [u8; 0x68] {
    let mut b = [0u8; 0x68];
    let bytes = unsafe { std::slice::from_raw_parts(raw as *const u8, 0x68) };
    b[..0x50].copy_from_slice(&bytes[8..0x58]);
    b[0x50..0x58].copy_from_slice(&bytes[0x60..0x68]);
    b[0x58..0x5a].copy_from_slice(&bytes[0x54..0x56]);
    b[0x5c..0x60].copy_from_slice(&bytes[0x58..0x5c]);
    b
}

pub(crate) fn canonical_sampler_builder(raw: u64) -> [u8; 0x60] {
    let bytes = unsafe { std::slice::from_raw_parts(raw as *const u8, 0x30) };
    let packed = u32::from_le_bytes(bytes[0x2c..0x30].try_into().unwrap());
    let mut b = [0u8; 0x60];
    for (to, v) in [(8, (packed >> 16) & 15), (12, (packed >> 20) & 15),
        (16, packed & 15), (20, (packed >> 4) & 15), (24, (packed >> 8) & 15),
        (0x28, (packed >> 24) & 3), (0x2c, packed >> 26), (0x44, (packed >> 12) & 15)] {
        b[to..to+4].copy_from_slice(&v.to_le_bytes());
    }
    b[0x1c..0x28].copy_from_slice(&bytes[8..0x14]);
    b[0x30..0x40].copy_from_slice(&bytes[0x1c..0x2c]);
    b[0x40..0x44].copy_from_slice(&bytes[0x14..0x18]);
    b
}

struct TextureBuilder {
    flags: u32,
    target: u32,
    width: u32,
    height: u32,
    depth: u32,
    levels: u32,
    format: u32,
    samples: u32,
    packaged: u16,
    stride: u32,
}

impl TextureBuilder {
    fn read(b: u64) -> Self {
        let u = |o: u64| unsafe { *((b + o) as *const u32) };
        Self {
            flags: u(0x18),
            target: u(0x1c),
            width: u(0x20),
            height: u(0x24),
            depth: u(0x28),
            levels: u(0x2c),
            format: u(0x30),
            samples: u(0x34),
            packaged: unsafe { *((b + 0x58) as *const u16) },
            stride: u(0x5c),
        }
    }
}

const TARGET_1D_ARRAY: u32 = 3;
const TARGET_2D_ARRAY: u32 = 4;
const TARGET_2D_MS_ARRAY: u32 = 6;
const TARGET_3D: u32 = 2;
const TARGET_CUBE: u32 = 8;
const TARGET_CUBE_ARRAY: u32 = 9;
const TARGET_BUFFER: u32 = 10;
const FLAG_LINEAR: u32 = 0x10;
const FLAG_LINEAR_RENDER_TARGET: u32 = 0x40;

pub(crate) fn texture_layout(builder: u64) -> Option<(tex::Shape, tex::Layout)> {
    let b = TextureBuilder::read(builder);
    let (bw, bh, bpb) = format_info(b.format);
    if b.target == TARGET_BUFFER || b.flags & (FLAG_LINEAR | FLAG_LINEAR_RENDER_TARGET) != 0 {
        return None;
    }
    let (sx, sy) = match b.samples {
        2 => (2, 1),
        4 => (2, 2),
        8 => (4, 2),
        16 => (4, 4),
        _ => (1, 1),
    };
    let (height, depth, layers) = match b.target {
        TARGET_1D_ARRAY => (1, 1, b.height),
        TARGET_2D_ARRAY | TARGET_2D_MS_ARRAY | TARGET_CUBE_ARRAY => (b.height, 1, b.depth),
        TARGET_CUBE => (b.height, 1, 6),
        TARGET_3D => (b.height, b.depth, 1),
        _ => (b.height, 1, 1),
    };
    let shape = tex::Shape {
        width: b.width * sx,
        height: height * sy,
        depth,
        levels: b.levels.max(1),
        layers: layers.max(1),
        block_w: bw,
        block_h: bh,
        bytes_per_block: bpb,
    };
    let (gy, gz) = if b.packaged & 1 != 0 {
        (((b.packaged >> 4) & 0xF) as u32, ((b.packaged >> 8) & 0xF) as u32)
    } else {
        tex::default_texture_gobs(shape.height.div_ceil(bh),depth,b.target==TARGET_3D)
    };
    Some((shape, tex::block_linear(&shape, gy, gz)))
}

pub(crate) fn texture_builder_storage(builder: u64) -> (u64, u64) {
    let b = TextureBuilder::read(builder);
    let (_, bh, bpb) = format_info(b.format);
    if b.target == TARGET_BUFFER {
        return (b.width as u64 * bpb as u64, if bpb == 12 { 4 } else { bpb as u64 });
    }
    if b.flags & (FLAG_LINEAR | FLAG_LINEAR_RENDER_TARGET) != 0 {
        let align = if b.flags & FLAG_LINEAR_RENDER_TARGET != 0 { 0x80 } else { 0x20 };
        return (b.stride as u64 * b.height.div_ceil(bh) as u64, align);
    }
    (texture_layout(builder).map_or(0, |(_, l)| l.total_size), 0x200)
}

fn device_get_integer(a: &Args) -> u64 {
    let (prop, out) = (a.0[1] as i32, a.0[2] as *mut i32);
    let v = match prop {
        0 => 55,
        1 => 15,
        38=>1,
        // M1 Max supports at most 1024 threads in a compute threadgroup;
        // the original Tegra helper advertises 1536 in X and in total.
        28|29|31=>1024,
        30=>64,
        41 | 42 | 46 | 47 => 1,
        // These optional NVN features have no translation in this backend yet.
        // Report them unavailable so the engine can choose its supported path.
        40 | 50 | 66..=69 | 71..=76 | 87 | 97 => 0,
        48 => 14,
        49 => 5,
        90 => 0,
        56 => 32,
        24 => 0,
        59 => 0,
        // Buffer textures use addressable pool storage rather than a Metal
        // 2D image; their capacity is independent of the 2D width limit.
        62 => 128*1024*1024,
        61 | 64 | 65 => 16384,
        63 | 45 => 2048,
        58 => 0x1000,
        // Metal allocates shader scratch internally and exposes no configurable
        // Tegra scratch-scale or L2-cache allocation requirement to this host.
        23 | 43 | 55 => 0,
        _ => {
            let d = DRIVER.get().unwrap();
            let helper: extern "C" fn(i32, *mut i32) -> i32 = unsafe { std::mem::transmute(d.property_helper as usize) };
            let mut v = 0;
            if helper(prop, &mut v)==0 {
                eprintln!("[nvn] unknown DeviceGetInteger property {prop}");
                crate::diag::report_and_exit("unsupported NVN capability query");
            }
            v
        }
    };
    unsafe { *out = v };
    if std::env::var_os("NX_NVN_LOG_PROPERTIES").is_some() && (0..128).contains(&prop) {
        static SEEN:[AtomicU64;2]=[const {AtomicU64::new(0)};2];
        let mask=1u64<<(prop%64);
        if SEEN[prop as usize/64].fetch_or(mask,Relaxed)&mask==0 {eprintln!("[nvn] property {prop} = {v}");}
    }
    0
}

struct Proc {
    name: String,
    calls: AtomicU64,
    host: Option<HostFn>,
}

static PROCS: Mutex<Vec<&'static Proc>> = Mutex::new(Vec::new());
static BY_NAME: Mutex<Option<HashMap<String, u64>>> = Mutex::new(None);
static SEQ: AtomicU64 = AtomicU64::new(0);

fn log_limit() -> u64 {
    static L: OnceLock<u64> = OnceLock::new();
    *L.get_or_init(|| std::env::var("NX_NVN_LOG").ok().and_then(|v| v.parse().ok()).unwrap_or(400))
}

fn call(tag: u64, a: &Args) -> u64 {
    let p = unsafe { &*(tag as *const Proc) };
    let n = p.calls.fetch_add(1, Relaxed);
    crate::video_diagnostics::texel_write(&p.name,n,a);
    if let Some(f) = p.host {
        return f(a);
    }
    let seq = SEQ.fetch_add(1, Relaxed);
    if n == 0 || (seq < log_limit() && n < 4) {
        eprintln!("[nvn] #{seq} STUB {}({:#x}, {:#x}, {:#x}, {:#x})", p.name, a.0[0], a.0[1], a.0[2], a.0[3]);
    }
    let fp:u64; unsafe {std::arch::asm!("mov {}, x29",out(reg)fp)};
    crate::diag::backtrace(fp);
    crate::diag::report_and_exit(&format!("unimplemented NVN function {}", p.name))
}

fn proc_for(name: &str) -> u64 {
    if name == "nvnDeviceGetProcAddress" {
        return device_get_proc_address as *const () as usize as u64;
    }
    if host_impl(name).is_none() && is_driver_passthrough(name) {
        if let Some(&a) = DRIVER.get().and_then(|d| d.procs.get(name)) {
            return a;
        }
    }
    let mut m = BY_NAME.lock().unwrap();
    let m = m.get_or_insert_with(HashMap::new);
    if let Some(&a) = m.get(name) {
        return a;
    }
    let p: &'static Proc = Box::leak(Box::new(Proc { name: name.to_string(), calls: AtomicU64::new(0), host: host_impl(name) }));
    PROCS.lock().unwrap().push(p);
    let addr = stubs::make(call, p as *const Proc as u64);
    m.insert(name.to_string(), addr);
    addr
}

pub extern "C" fn bootstrap_loader(name: *const c_char) -> u64 {
    let n = unsafe { CStr::from_ptr(name) }.to_string_lossy();
    proc_for(&n)
}

extern "C" fn device_get_proc_address(_device: u64, name: *const c_char) -> u64 {
    if name.is_null() {
        return 0;
    }
    let n = unsafe { CStr::from_ptr(name) }.to_string_lossy();
    proc_for(&n)
}

pub fn summary() {
    let Ok(procs) = PROCS.try_lock() else {return};
    let called: Vec<_> = procs.iter().filter(|p| p.calls.load(Relaxed) > 0).collect();
    if called.is_empty() {
        return;
    }
    eprintln!("[nvn] host/stub functions called:");
    for p in called {
        let kind = if p.host.is_some() { "host" } else { "STUB" };
        eprintln!("[nvn]   {:6}  {kind}  {}", p.calls.load(Relaxed), p.name);
    }
    let Ok(state)=objects::STATE.try_lock() else {return};
    for (object,pool) in state.pools.iter().take(32) {
        let direct=crate::gfx::heap().is_some_and(|h|pool.cpu>=h.base && pool.cpu.checked_add(pool.size).is_some_and(|end|end<=h.base+h.size));
        eprintln!("[nvn] pool {object:#x}: cpu {:#x}+{:#x}, flags {:#x}, contiguous GPU heap {direct}",pool.cpu,pool.size,pool.flags);
    }
}
