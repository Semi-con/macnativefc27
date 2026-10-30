// Mush8 - game.rs
// Constantin Clerc (c22dev) - 0.1
use anyhow::{Context, Result, bail};
use std::path::Path;
use std::sync::OnceLock;

pub const TITLE_ID: u64 = 0x0100_C490_25D3_E000;
pub const TITLE: &str = "EA SPORTS FC 27";
pub const MAIN_BUILD_ID: &str = "fb81983aa1cc3a2e97402497e2ca652800000000000000000000000000000000";
const VERSIONS: &[(&str, &str)] = &[(MAIN_BUILD_ID, "1.8c.cab1")];

static VERSION: OnceLock<&'static str> = OnceLock::new();

pub fn version() -> &'static str {
    VERSION.get().copied().unwrap_or("1.8c.cab1")
}

pub fn control_data(package:&nxfs::Nsp)->Result<Vec<u8>> {
    for nca in package.ncas.iter().filter(|n|n.content_type==nxfs::nca::ContentType::Control) {
        let section=nca.sections.iter().find(|s|s.fs_type==nxfs::nca::FsType::RomFs).context("control has no RomFS")?;
        let rom=nxfs::romfs::RomFs::open(nca.fs_storage(section)?)?;
        let f=rom.find("/control.nacp").context("control.nacp is missing")?;
        anyhow::ensure!(f.size>=0x3170 && f.size<=0x10000,"invalid NACP size");
        let data=rom.storage().read_vec(f.offset,f.size as usize)?;
        let display=&data[0x3060..0x3070];let end=display.iter().position(|v|*v==0).unwrap_or(display.len());
        if &display[..end]==version().as_bytes() {return Ok(data);}
    }
    bail!("no control data matching the verified game version")
}

fn named(id: &str) -> Option<&'static str> {
    VERSIONS.iter().find(|(p, _)| id.starts_with(p)).map(|(_, v)| *v)
}

fn update_main_id(update: &nxfs::Nsp) -> Result<String> {
    let prog = update.program(true)?;
    anyhow::ensure!(prog.program_id == TITLE_ID, "input is not FC27");
    let exefs = prog.sections.iter().find(|s| s.fs_type == nxfs::nca::FsType::PartitionFs).context("the update has no ExeFS")?;
    let exefs = nxfs::pfs0::Pfs0::open(prog.fs_storage(exefs)?)?;
    for (name, hash) in [
        ("main", "4c65f206f68d07af5487886552ae749ba4dc5426b30f06a048fdce4cbac9de29"),
        ("sdk", "78a5c1861da411802714080f4f74017a94cd839f8fe5b447dabfce81fc9abb6f"),
        ("rtld", "ee8fd522d4961802e7f3c356ffee80a3f5eec3c3eb8e1f3a1155aeea4d4e091e"),
    ] {
        let file = exefs.file(name).context("missing pinned source module")?;
        anyhow::ensure!(file.size() <= 256*1024*1024, "oversized source module");
        let bytes = file.read_vec(0,file.size() as usize)?;
        anyhow::ensure!(nxfs::crypto::sha256(&bytes) == hash, "{name}: source module differs from the inspected FC27 update");
    }
    let header = exefs.file("main").context("the update has no main module")?.read_vec(0, 0x60)?;
    anyhow::ensure!(&header[..4] == b"NSO0", "the update's main is not an NSO");
    Ok(header[0x40..0x60].iter().map(|b| format!("{b:02x}")).collect())
}

fn prepared_main_id(patched: &Path) -> Result<String> {
    let text = std::fs::read_to_string(patched.join("main.layout")).context("no prepared game code")?;
    text.lines().find_map(|l| l.strip_prefix("module_id ")).map(str::to_owned).context("main.layout has no module_id")
}

pub fn identify(update: &nxfs::Nsp, patched: &Path) -> Result<&'static str> {
    let id = update_main_id(update)?;
    let Some(version) = named(&id) else {
        bail!("this FC27 update isn't supported (main {id}); expected 1.8c.cab1");
    };
    let prepared = prepared_main_id(patched)?;
    if !prepared.starts_with(&id) {
        let built = named(&prepared).unwrap_or("another update");
        if cfg!(target_os = "ios") {
            bail!("this build is for update {built}, but the update file is {version}");
        }
        bail!("the game code was prepared for update {built}, but the update file is {version}: Prepare again");
    }
    let _ = VERSION.set(version);
    eprintln!("[host] {TITLE} {version}");
    Ok(version)
}
