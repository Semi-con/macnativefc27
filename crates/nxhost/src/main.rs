// Mush8 - main.rs
// Constantin Clerc (c22dev) - 0.1
mod aspect;
mod audio;
mod diag;
mod dnssd;
mod engine_frame_target;
mod video_diagnostics;
mod menu_asset_io;
mod shared_type_diagnostics;
mod menu_config_diagnostics;
mod menu_still_frame_diagnostics;
mod menu_cinematic_diagnostics;
mod fc27_menu_sequence_control;
mod game;
mod gfx;
mod guest;
mod haptics;
mod hle;
mod input;
mod ipc;
mod kernel;
mod loader;
mod mem;
mod mods;
mod fc27_menu_event_observation;
mod fc27_menu_lifecycle_observation;
mod fc27_menu_auto_lifecycle;
mod fc27_menu_prime_probe;
mod fc27_menu_time_observation;
mod fc27_menu_shadow_bindings;
mod fc27_menu_ambassador_source;
mod motion;
mod movie;
mod remote;
mod ro;
mod services;
#[cfg(target_os = "macos")]
mod sdl;
mod shm;
mod stubs;
mod nvn;
mod svc;
mod webpad;

use anyhow::{Context, Result};
use clap::Parser;
use std::path::PathBuf;
use std::sync::OnceLock;

static INITS: OnceLock<Vec<u64>> = OnceLock::new();
static FINIS: OnceLock<Vec<u64>> = OnceLock::new();

extern "C" fn init_done() {ro::initialize_native();}

extern "C" fn init_modules() {
    for &f in INITS.get().unwrap() {
        eprintln!("[init] ENTER {}", diag::symbolize(f));
        let f: extern "C" fn() = unsafe { std::mem::transmute(f as usize) };
        f();
        eprintln!("[init] RETURN");
    }
}

extern "C" fn fini_modules() {
    for &f in FINIS.get().unwrap().iter().rev() {
        let f: extern "C" fn() = unsafe { std::mem::transmute(f as usize) };
        f();
    }
}

struct Npdm {
    priority: u8,
    core: u8,
    stack_size: u32,
    system_resource_size: u32,
    core_mask:u64,
    priority_mask:u64,
}

fn read_npdm(path: &PathBuf) -> Result<Npdm> {
    let b = std::fs::read(path).with_context(|| path.display().to_string())?;
    anyhow::ensure!(b.len() >= 0x20 && &b[..4] == b"META", "not an NPDM");
    anyhow::ensure!(nxfs::crypto::sha256(&b) == "70c5c693bcd38cfdadbfe41d3271e703734b9e1d08d776bf52b2bc2318cd49f3", "NPDM differs from FC27 1.8c.cab1");
    let read32=|at:usize|->Result<u32>{Ok(u32::from_le_bytes(b.get(at..at+4).context("truncated NPDM")?.try_into()?))};
    let aci=read32(0x70)? as usize;
    let (off,len)=(read32(aci+0x30)? as usize,read32(aci+0x34)? as usize);
    let caps=b.get(aci+off..aci+off+len).context("NPDM capabilities out of bounds")?;
    let info:Vec<_>=caps.chunks_exact(4).map(|s|u32::from_le_bytes(s.try_into().unwrap())).filter(|w|w&15==7).collect();
    anyhow::ensure!(len%4==0 && info.len()==1,"NPDM thread capability missing or ambiguous");
    let w=info[0];let (low,high,first,last)=(w>>4&63,w>>10&63,w>>16&255,w>>24&255);
    anyhow::ensure!(high<=low && first<=last && last<4,"unsupported NPDM thread permissions");
    let mask=|first:u32,last:u32|((u64::MAX>>(63-last))>>first)<<first;
    Ok(Npdm { priority: b[0xE], core: b[0xF], stack_size: u32::from_le_bytes(b[0x1C..0x20].try_into()?),
        system_resource_size: u32::from_le_bytes(b[0x14..0x18].try_into()?),core_mask:mask(first,last),priority_mask:mask(high,low) })
}

fn update_file(p: &std::path::Path) -> PathBuf {
    if !p.exists() && p == std::path::Path::new("update.nsp") && std::path::Path::new("upd4-0-1.nsp").exists() {
        return PathBuf::from("upd4-0-1.nsp");
    }
    p.to_path_buf()
}

fn program(n: &nxfs::Nsp) -> Result<&nxfs::nca::Nca> {
    n.find(nxfs::nca::ContentType::Program).context("no Program NCA")
}

fn load_dlc(args: &Args, keys: &nxfs::keys::Keys) -> Vec<(u64, nxfs::storage::DynStorage)> {
    let default = PathBuf::from("dlc.nsp");
    let paths = if args.dlc.is_empty() && default.exists() { vec![default] } else { args.dlc.clone() };
    let mut out = Vec::new();
    for p in paths {
        let found = (|| -> Result<Vec<(u64, nxfs::storage::DynStorage)>> {
            let nsp = nxfs::Nsp::open(&p, keys)?;
            let mut v = Vec::new();
            for nca in nsp.ncas.iter().filter(|n| n.content_type == nxfs::nca::ContentType::PublicData) {
                let sec = nca.sections.iter().find(|s| s.fs_type == nxfs::nca::FsType::RomFs).context("no RomFS")?;
                v.push((nca.program_id, nca.fs_storage(sec)?));
            }
            Ok(v)
        })();
        match found {
            Ok(v) => {
                for (id, _) in &v {
                    eprintln!("[host] add-on content {id:016x} ({})", p.display());
                }
                out.extend(v);
            }
            Err(e) => eprintln!("[host] {}: {e:#}", p.display()),
        }
    }
    out
}

#[derive(Parser)]
#[command(about = "Run rewritten Switch modules natively")]
struct Args {
    #[arg(long, default_value = "work/v-upd/patched")]
    patched: PathBuf,
    #[arg(long, default_value = "work/v-upd/exefs/main.npdm")]
    npdm: PathBuf,
    #[arg(long, default_value = "prod.keys")]
    keys: PathBuf,
    #[arg(long, default_value = "base.nsp")]
    base: PathBuf,
    #[arg(long, default_value = "update.nsp")]
    update: PathBuf,
    #[arg(long, default_value = "work/save")]
    save: PathBuf,
    #[arg(long, default_value = "work/sdmc")]
    sdmc: PathBuf,
    #[arg(long, default_value = "system")]
    system: PathBuf,
    #[arg(long, default_value = "Firmware")]
    firmware: PathBuf,
    #[arg(long)]
    dlc: Vec<PathBuf>,
    #[arg(long = "mod")]
    mods: Vec<PathBuf>,
    #[arg(long)]
    precompile: bool,
    #[arg(long)]
    movies: bool,
}

#[cfg(target_os = "macos")]
fn main() -> Result<()> {
    start(Args::parse())?;
    gfx::run_app("EA SPORTS FC 27 — native port test", 1280.0, 720.0)
}

#[cfg(target_os = "ios")]
fn main() {
    let home = PathBuf::from(std::env::var_os("HOME").expect("HOME"));
    let docs = home.join("Documents");
    let _ = std::env::set_current_dir(&docs);
    let _ = std::fs::create_dir_all("work");
    let _ = std::fs::rename("work/nxhost.log", "work/nxhost.prev.log");
    if let Ok(log) = std::fs::File::create("work/nxhost.log") {
        use std::os::fd::AsRawFd;
        unsafe { libc::dup2(log.as_raw_fd(), 2) };
    }
    let bundle = std::env::current_exe().ok().and_then(|e| e.parent().map(PathBuf::from)).unwrap_or_default();
    let mut env: Vec<(String, String)> = vec![
        ("NX_SHADER_CACHE".into(), bundle.join("shaders").display().to_string()),
        ("NX_PIPELINE_LIST".into(), bundle.join("shaders/pipelines.tsv").display().to_string()),
    ];
    for line in std::fs::read_to_string("env.txt").unwrap_or_default().lines() {
        if let Some((k, v)) = line.trim().split_once('=').filter(|(k, _)| !k.is_empty() && !k.starts_with('#')) {
            env.push((k.trim().into(), v.trim().into()));
        }
    }
    for (k, v) in env {
        unsafe { std::env::set_var(k, v) };
    }
    eprintln!("[host] iOS: files in {}, bundle {}", docs.display(), bundle.display());
    let args = Args::parse_from(["nxhost", "--patched", &bundle.join("modules").display().to_string(), "--npdm", &bundle.join("modules/main.npdm").display().to_string()]);
    gfx::run_app(move || {
        apply_launcher_settings();
        let missing: Vec<&str> = ["prod.keys", "base.nsp", "update.nsp"].into_iter().filter(|f| !update_file(std::path::Path::new(f)).exists()).collect();
        anyhow::ensure!(
            missing.is_empty(),
            "Missing {}.\n\nCopy prod.keys, base.nsp, the update .nsp (and optionally dlc.nsp and the Firmware folder) into this app's folder: Finder → your iPhone or iPad → Files → Mush8, or the Files app → On My iPhone → Mush8.",
            missing.join(", ")
        );
        start(args)
    })
}

#[cfg(target_os = "ios")]
fn apply_launcher_settings() {
    let config = std::fs::read_to_string("config.json").unwrap_or_default();
    let field = |key: &str| {
        let rest = config.split(&format!("\"{key}\"")).nth(1)?;
        let rest = rest.trim_start().strip_prefix(':')?.trim_start();
        if let Some(rest) = rest.strip_prefix('"') {
            let mut out = String::new();
            let mut chars = rest.chars();
            while let Some(c) = chars.next() {
                if c == '\\' {
                    if let Some(n) = chars.next() {
                        out.push(n);
                    }
                } else if c == '"' {
                    break;
                } else {
                    out.push(c);
                }
            }
            Some(out)
        } else {
            Some(rest.split([',', '\n', '}']).next()?.trim().to_string())
        }
    };
    let network = field("network").unwrap_or_else(|| if field("lan").as_deref() == Some("true") { "lan".into() } else { "off".into() });
    unsafe { std::env::set_var("NX_NET", &network) };
    if network == "lan" {
        unsafe { std::env::set_var("NX_LAN", "1") };
    }
    if let Some(server) = field("lanplay_server").filter(|s| !s.is_empty()) {
        unsafe { std::env::set_var("NX_LANPLAY_SERVER", server) };
    }
    if field("prewarm_pipelines").as_deref() != Some("false") {
        unsafe { std::env::set_var("NX_PREWARM", "1") };
    }
    if field("background_compile").as_deref() == Some("false") {
        unsafe { std::env::set_var("NX_SYNC_SHADERS", "1") };
    }
    if field("rumble").as_deref() == Some("false") {
        unsafe { std::env::set_var("NX_RUMBLE", "0") };
    }
    if field("gamepad_driver").as_deref() == Some("sdl") {
        unsafe { std::env::set_var("NX_PAD_DRIVER", "sdl") };
    }
    if field("spatial_audio").as_deref() == Some("false") {
        unsafe { std::env::set_var("NX_SPATIAL", "0") };
    }
    if field("motion_controls").as_deref() == Some("true") {
        unsafe { std::env::set_var("NX_MOTION", "1") };
    }
    if field("handheld").as_deref() != Some("false") {
        unsafe { std::env::set_var("NX_HANDHELD", "1") };
    }
    if field("phone_pads").as_deref() == Some("true") {
        unsafe { std::env::set_var("NX_PHONE_PADS", "1") };
    }
    if field("aspect").as_deref() != Some("16:9") {
        unsafe { std::env::set_var("NX_ASPECT", "auto") };
    }
    if field("hud").as_deref() == Some("center") {
        unsafe { std::env::set_var("NX_HUD", "center") };
    }
    if let Some(scale) = field("hud_scale") {
        unsafe { std::env::set_var("NX_HUD_SCALE", scale) };
    }
}

fn start(args: Args) -> Result<()> {
    let dir = args.patched.clone();
    let npdm = read_npdm(&args.npdm)?;
    let _ = svc::SYSTEM_RESOURCE_SIZE.set(npdm.system_resource_size as u64);
    let _ = svc::THREAD_PERMISSIONS.set((npdm.core_mask,npdm.priority_mask));
    let _ = svc::DEFAULT_CORE.set(npdm.core as i32);

    let keys = nxfs::keys::Keys::load(&args.keys)?;
    let base = nxfs::Nsp::open(&args.base, &keys)?;
    let update = nxfs::Nsp::open(&update_file(&args.update), &keys)?;
    let version = game::identify(&update, &dir)?;
    let save_policy=services::fs::save_policy::Policy::from_nacp(&game::control_data(&update)?)?;
    let romfs = mods::romfs(update.program(true)?.patched_romfs(base.program(false)?)?, &args.mods)?;
    eprintln!("[host] RomFS: {:.2} GB", nxfs::storage::Storage::size(&*romfs) as f64 / 1e9);
    if args.movies {
        movie::webm::convert_all(romfs)?;
        std::process::exit(0);
    }
    if args.precompile {
        let modules: Vec<PathBuf> = ["main", "renderer", "sdk", "rtld"].iter().map(|m| dir.join(format!("{m}.bin"))).collect();
        nvn::precompile(romfs, &modules)?;
        std::process::exit(0);
    }
    let _ = services::fs::CONFIG.set(services::fs::Config {
        romfs,
        save_policy,
        save_dir: if args.save == std::path::Path::new("work/save") { PathBuf::from(format!("work/fc27-test-save-{version}")) } else { args.save.clone() },
        sd_dir: args.sdmc.clone(),
        system_dir: args.system.clone(),
        firmware_dir: args.firmware.clone(),
        dlc: load_dlc(&args, &keys),
        keys,
    });

    if remote::enabled() {
        remote::host();
        if webpad::enabled() {
            webpad::host();
        }
    }
    diag::install_handlers();
    guest::init_tsd();
    let _ = kernel::KERNEL.aspace.set(mem::AddressSpace::reserve()?);
    let a = kernel::aspace();
    eprintln!("[host] guest address space at {:#x}..{:#x}", a.base, a.base + a.layout.total());

    // The renderer is a DT_NEEDED dependency of main, including strong data
    // imports. Load and relocate all dependencies before running constructors.
    let mods = loader::load(&dir, &["main", "renderer", "sdk", "rtld"], &hle::table())?;
    for (name, expected) in [
        ("main", game::MAIN_BUILD_ID),
        ("renderer", "d924da0d0673a87b550159a839f5f80700000000000000000000000000000000"),
        ("sdk", "fed212d6e0963706f9f4b16ea5d65879c4ec6c4d000000000000000000000000"),
        ("rtld", "56533df3bb579c77f5b2f78ea047a44a5c4b4ff6000000000000000000000000"),
    ] {
        let m = mods.iter().find(|m| m.name == name).context("missing module")?;
        let id: String = m.module_id.iter().map(|b| format!("{b:02x}")).collect();
        anyhow::ensure!(id == expected, "{name}: mixed or unsupported module build {id}");
    }
    let _ = diag::MODULES.set(
        mods.iter()
            .map(|m| {
                let mut symbols: Vec<(u64, String)> = m.dynamic.symbols.iter()
                    .filter(|s| s.defined() && !s.name.is_empty())
                    .map(|s| (s.value, nxfs::nso::demangle(&s.name)))
                    .collect();
                symbols.sort();
                diag::ModInfo { name: m.name.clone(), base: m.base, size: m.size, symbols }
            })
            .collect(),
    );
    diag::start_watch(&mods.iter().map(|m| (m.name.clone(), m.base)).collect::<Vec<_>>());
    kernel::start_wait_diagnostics();
    // Renderer code is staged to resolve main's NVN function-pointer data,
    // but its constructors/TLS belong to the game's later nn::ro::LoadModule.
    let ordered: Vec<_> = ["sdk", "main"].iter().filter_map(|n| mods.iter().find(|m| &m.name == n)).collect();
    let _ = INITS.set(ordered.iter().filter_map(|m| m.dynamic.init.map(|i| m.base + i)).collect());
    let _ = FINIS.set(ordered.iter().filter_map(|m| m.dynamic.fini.map(|i| m.base + i)).collect());
    ro::configure(&mods)?;
    let start = loader::symbol(&mods, "_ZN2nn4init5StartEmmPFvvES2_S2_").context("SDK 22 nn::init::Start not found")?;
    let sdk = mods.iter().find(|m| m.name == "sdk").context("no sdk module")?;
    nvn::init(sdk.base, &sdk.module_id)?;
    video_diagnostics::install(&mods)?;
    services::vi::configure_native_api(
        loader::symbol(&mods,"_ZN2nn2vi15GetNativeWindowEPPvPNS0_5LayerE").context("SDK native window API missing")?,
        loader::symbol(&mods,"_ZN2nn2vi24GetLayerIdWithValidationEPKNS0_5LayerE").context("SDK layer identity API missing")?);
    if std::env::var_os("NX_PREWARM").is_some() {
        nvn::prewarm();
    }
    if let Some(v) = loader::symbol(&mods, "_ZNK2nn3mii6detail11CharInfoRaw6VerifyEv") {
        let _ = services::mii::VERIFY.set(v);
    }

    let stack_size = (npdm.stack_size as u64).next_multiple_of(mem::PAGE);
    a.set(a.stack(), stack_size, mem::Block { state: mem::state::STACK, perm: mem::perm::RW, attr: 0 })?;
    let h = kernel::create_thread(start, 0, a.stack() + stack_size, npdm.priority as i32, npdm.core as i32)?;
    let t = kernel::thread(h).unwrap();
    eprintln!("[host] main thread {h:#x}: prio {} core {} stack {stack_size:#x}; entering nn::init::Start", npdm.priority, npdm.core);
    let j = kernel::start_thread_with_args(t, [h as u64, 0, init_done as *const () as usize as u64,
        init_modules as *const () as usize as u64, fini_modules as *const () as usize as u64, 0, 0, 0])?;
    std::thread::spawn(move || {
        j.join().ok();
        eprintln!("[host] main thread returned");
        std::process::exit(0);
    });
    Ok(())
}
