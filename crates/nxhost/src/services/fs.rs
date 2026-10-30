// Mush8 - fs.rs
// Constantin Clerc (c22dev) - 0.1
use crate::ipc::{Request, Response, Service, Stub, write_buffer};
use nxfs::storage::DynStorage;
use std::fs::{File, OpenOptions};
use std::os::unix::fs::FileExt;
use std::path::{Component, Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

pub mod save_policy;

mod rc {
    pub const PATH_NOT_FOUND: u32 = 0x202;
    pub const PATH_ALREADY_EXISTS: u32 = 0x402;
    pub const OUT_OF_RANGE: u32 = 0x3E02;
    pub const INVALID_PATH: u32 = 0x1A202;
    pub const TARGET_NOT_FOUND: u32 = 0x7D402;
}

pub struct Config {
    pub romfs: DynStorage,
    pub save_policy:save_policy::Policy,
    pub save_dir: PathBuf,
    pub sd_dir: PathBuf,
    pub system_dir: PathBuf,
    pub firmware_dir: PathBuf,
    pub keys: nxfs::keys::Keys,
    pub dlc: Vec<(u64, DynStorage)>,
}

fn is_data(n: &nxfs::nca::Nca) -> bool {
    matches!(n.content_type, nxfs::nca::ContentType::Data | nxfs::nca::ContentType::PublicData)
}

fn firmware_index(cfg: &Config) -> &'static std::collections::HashMap<u64, PathBuf> {
    static INDEX: OnceLock<std::collections::HashMap<u64, PathBuf>> = OnceLock::new();
    INDEX.get_or_init(|| {
        let mut m = std::collections::HashMap::new();
        let (mut unreadable, mut missing_key) = (0, None);
        for e in std::fs::read_dir(&cfg.firmware_dir).into_iter().flatten().flatten() {
            let p = e.path();
            if p.extension().is_none_or(|x| x != "nca") {
                continue;
            }
            match nxfs::open_nca(&p, &cfg.keys) {
                Ok(n) if is_data(&n) => {
                    m.insert(n.program_id, p);
                }
                Ok(_) => {}
                Err(err) => {
                    unreadable += 1;
                    missing_key.get_or_insert(err.to_string());
                }
            }
        }
        if unreadable > 0 {
            eprintln!("[fs] firmware: {unreadable} NCAs unreadable ({})", missing_key.unwrap_or_default());
        }
        m
    })
}

pub fn system_data(cfg: &Config, id: u64) -> Option<DynStorage> {
    static CACHE: Mutex<Option<std::collections::HashMap<u64, Option<DynStorage>>>> = Mutex::new(None);
    let mut c = CACHE.lock().unwrap();
    c.get_or_insert_with(Default::default)
        .entry(id)
        .or_insert_with(|| {
            let path = cfg.system_dir.join(format!("{id:016x}.nsp"));
            let romfs = |nca: &nxfs::nca::Nca| -> anyhow::Result<DynStorage> {
                let sec = nca.sections.iter().find(|s| s.fs_type == nxfs::nca::FsType::RomFs).ok_or_else(|| anyhow::anyhow!("no RomFS"))?;
                nca.fs_storage(sec)
            };
            let from_nsp = || -> anyhow::Result<DynStorage> {
                let nsp = nxfs::Nsp::open(&path, &cfg.keys)?;
                let nca = nsp.ncas.iter().find(|n| is_data(n)).ok_or_else(|| anyhow::anyhow!("no Data NCA"))?;
                romfs(nca)
            };
            if path.exists() {
                match from_nsp() {
                    Ok(s) => return Some(s),
                    Err(e) => eprintln!("[fs] {}: {e}", path.display()),
                }
            }
            if let Some(nca) = firmware_index(cfg).get(&id).and_then(|p| nxfs::open_nca(p, &cfg.keys).ok()) {
                match romfs(&nca) {
                    Ok(s) => return Some(s),
                    Err(e) => eprintln!("[fs] firmware title {id:016x}: {e}"),
                }
            }
            eprintln!(
                "[fs] system data {id:016x} unavailable: not in {} (needs keys for its key generation?) nor {}",
                cfg.firmware_dir.display(),
                path.display()
            );
            None
        })
        .clone()
}

pub static CONFIG: OnceLock<Config> = OnceLock::new();

pub struct FspSrv;

impl Service for FspSrv {
    fn name(&self) -> String {
        "fsp-srv".into()
    }
    fn call(&self, req: &Request, rsp: &mut Response) -> u32 {
        let cfg = CONFIG.get().expect("fs not configured");
        match req.cmd {
            1 | 1003 | 1006 | 22 => {}
            18 => rsp.objects.push(HostFs::open("sdmc", &cfg.sd_dir)),
            51 | 52 => {
                let attr_type = req.data.get(0x08 + 0x20).copied().unwrap_or(0);
                let index = req.data.get(0x08 + 0x22).copied().unwrap_or(0);
                let dir = cfg.save_dir.join(format!("type{attr_type}-index{index}"));
                rsp.objects.push(HostFs::open("save", &dir));
            }
            61 | 60 => rsp.objects.push(Arc::new(EmptyInfoReader)),
            200 | 203 => rsp.objects.push(Arc::new(Storage { name: "romfs", s: cfg.romfs.clone() })),
            202 => {
                let id = req.u64(8);
                if let Some((_, s)) = cfg.dlc.iter().find(|(t, _)| *t == id) {
                    rsp.objects.push(Arc::new(Storage { name: "add-on content", s: s.clone() }));
                    return 0;
                }
                match system_data(cfg, id) {
                    Some(s) => rsp.objects.push(Arc::new(Storage { name: "system data", s })),
                    None => return rc::TARGET_NOT_FOUND,
                }
            }
            1005 => rsp.push_u32(0),
            1011 => rsp.push_u32(0),
            _ => return Stub(self.name()).call(req, rsp),
        }
        0
    }
}

struct EmptyInfoReader;
impl Service for EmptyInfoReader {
    fn name(&self) -> String {
        "fs::ISaveDataInfoReader".into()
    }
    fn call(&self, req: &Request, rsp: &mut Response) -> u32 {
        match req.cmd {
            0 => rsp.push_u64(0),
            _ => return Stub(self.name()).call(req, rsp),
        }
        0
    }
}

struct Storage {
    name: &'static str,
    s: DynStorage,
}

impl Service for Storage {
    fn name(&self) -> String {
        format!("fs::IStorage({})", self.name)
    }
    fn call(&self, req: &Request, rsp: &mut Response) -> u32 {
        match req.cmd {
            0 => {
                let (off, size) = (req.u64(0), req.u64(8));
                let Some(b) = req.b.first() else { return rc::OUT_OF_RANGE };
                let size = size.min(b.size);
                if off + size > self.s.size() {
                    return rc::OUT_OF_RANGE;
                }
                let dst = unsafe { std::slice::from_raw_parts_mut(b.addr as *mut u8, size as usize) };
                let observation=if self.name=="romfs" {crate::menu_asset_io::begin()}else{None};
                let result=crate::kernel::blocking(|| self.s.read_at(off, dst));
                crate::menu_asset_io::end(observation,off,size,result.is_ok());
                if let Err(e) = result {
                    eprintln!("[fs] {} read {off:#x}+{size:#x}: {e}", self.name);
                    return rc::OUT_OF_RANGE;
                }
            }
            4 => rsp.push_u64(self.s.size()),
            _ => return Stub(self.name()).call(req, rsp),
        }
        0
    }
}

fn req_path(req: &Request) -> Option<PathBuf> {
    let b = req.x.first()?;
    if b.addr==0 || b.size==0 || b.size>0x301 {return None;}
    let raw = unsafe { std::slice::from_raw_parts(b.addr as *const u8, b.size as usize) };
    let end=raw.iter().position(|&c| c==0)?;
    let s = std::str::from_utf8(&raw[..end]).ok()?;
    let p = Path::new(s.trim_start_matches('/'));
    p.components().all(|c| match c {Component::Normal(n)=>n.len()<=255,Component::CurDir=>true,_=>false}).then(|| p.to_path_buf())
}

struct HostFs {
    name: &'static str,
    root: PathBuf,
}

impl HostFs {
    fn open(name: &'static str, root: &Path) -> Arc<dyn Service> {
        let _ = std::fs::create_dir_all(root);
        Arc::new(HostFs { name, root: root.to_path_buf() })
    }
}

fn io_rc(e: &std::io::Error) -> u32 {
    match e.kind() {
        std::io::ErrorKind::NotFound => rc::PATH_NOT_FOUND,
        std::io::ErrorKind::AlreadyExists => rc::PATH_ALREADY_EXISTS,
        _ => rc::INVALID_PATH,
    }
}

fn sync_tree(root:&Path)->std::io::Result<()> {
    for entry in std::fs::read_dir(root)? {
        let entry=entry?;let kind=entry.file_type()?;
        if kind.is_dir(){sync_tree(&entry.path())?;} else if kind.is_file(){File::open(entry.path())?.sync_all()?;}
    }
    File::open(root)?.sync_all()
}

fn filesystem_attribute()->[u8;0xc0] {
    let mut data=[0;0xc0];data[..4].fill(1);
    // UTF-8 limits enforced by req_path. UTF-16-specific limits are absent,
    // as the host exposes byte limits for the local save directory.
    for (i,limit) in [255u32,255,0x300,0x300].iter().enumerate() {data[0x28+i*4..0x2c+i*4].copy_from_slice(&limit.to_le_bytes());}
    data
}

impl Service for HostFs {
    fn name(&self) -> String {
        format!("fs::IFileSystem({})", self.name)
    }
    fn call(&self, req: &Request, rsp: &mut Response) -> u32 {
        if req.cmd==16 {rsp.data.extend_from_slice(&filesystem_attribute());return 0;}
        if matches!(req.cmd, 10 | 11 | 12) {
            match req.cmd {
                10 => if let Err(e)=crate::kernel::blocking(||sync_tree(&self.root)){return io_rc(&e)},
                _ => {
                    use std::os::unix::ffi::OsStrExt;
                    let Ok(path)=std::ffi::CString::new(self.root.as_os_str().as_bytes()) else{return rc::INVALID_PATH};
                    let mut st:libc::statvfs=unsafe {std::mem::zeroed()};
                    if unsafe {libc::statvfs(path.as_ptr(),&mut st)}!=0 {return io_rc(&std::io::Error::last_os_error());}
                    let blocks=if req.cmd==11{st.f_bavail}else{st.f_blocks};
                    rsp.push_u64((blocks as u64).saturating_mul(st.f_frsize as u64).min(i64::MAX as u64));
                }
            }
            return 0;
        }
        if !matches!(req.cmd,0..=9|13|14) {return Stub(self.name()).call(req,rsp);}
        let Some(rel) = req_path(req) else { return rc::INVALID_PATH };
        let p = self.root.join(&rel);
        let r: std::io::Result<()> = crate::kernel::blocking(|| {
            match req.cmd {
                0 => {
                    if p.exists() {
                        return Err(std::io::ErrorKind::AlreadyExists.into());
                    }
                    File::create(&p)?.set_len(req.u64(8))?
                }
                1 => std::fs::remove_file(&p)?,
                2 => std::fs::create_dir(&p)?,
                3 => std::fs::remove_dir(&p)?,
                4 => std::fs::remove_dir_all(&p)?,
                5 | 6 => {
                    let to = req.x.get(1).and_then(|_| {
                        let r2 = Request { x: req.x[1..].to_vec(), ..Default::default() };
                        req_path(&r2)
                    });
                    let Some(to) = to else { return Err(std::io::ErrorKind::InvalidInput.into()) };
                    std::fs::rename(&p, self.root.join(to))?
                }
                7 => rsp.push_u32(if std::fs::metadata(&p)?.is_dir() { 0 } else { 1 }),
                8 => {
                    let mode = req.u32(0);
                    let f = OpenOptions::new().read(true).write(mode & 2 != 0).append(false).open(&p)?;
                    if f.metadata()?.is_dir() {
                        return Err(std::io::ErrorKind::NotFound.into());
                    }
                    rsp.objects.push(Arc::new(HostFile { f: Mutex::new(f), allow_extend: mode & 4 != 0 }));
                }
                9 => {
                    let mode = req.u32(0);
                    let mut entries = Vec::new();
                    for e in std::fs::read_dir(&p)? {
                        let e = e?;
                        let md = e.metadata()?;
                        let is_dir = md.is_dir();
                        if (is_dir && mode & 1 != 0) || (!is_dir && mode & 2 != 0) {
                            entries.push((e.file_name().to_string_lossy().into_owned(), is_dir, md.len()));
                        }
                    }
                    let total=entries.len() as u64;
                    rsp.objects.push(Arc::new(HostDir { entries: Mutex::new(entries.into()),total }));
                }
                13 => {
                    for e in std::fs::read_dir(&p)? {
                        let e = e?.path();
                        if e.is_dir() { std::fs::remove_dir_all(e)? } else { std::fs::remove_file(e)? }
                    }
                }
                14 => {
                    let md=std::fs::metadata(&p)?;
                    let seconds=|time:std::io::Result<std::time::SystemTime>|time.ok().and_then(|t|t.duration_since(std::time::UNIX_EPOCH).ok()).map_or(0,|d|d.as_secs());
                    for time in [md.created(),md.modified(),md.accessed()]{rsp.push_u64(seconds(time));}
                    rsp.data.extend_from_slice(&[1,0,0,0,0,0,0,0]);
                }
                _ => unreachable!(),
            }
            Ok(())
        });
        match r {
            Ok(()) => 0,
            Err(e) => {
                if e.kind() != std::io::ErrorKind::NotFound {
                    eprintln!("[fs] {} cmd {} {}: {e}", self.name, req.cmd, rel.display());
                }
                io_rc(&e)
            }
        }
    }
}

struct HostFile {
    f: Mutex<File>,
    // Horizon OpenMode_AllowAppend permits extending a file. Every Write still
    // uses its explicit offset, including writes that overwrite existing data.
    allow_extend: bool,
}

impl Service for HostFile {
    fn name(&self) -> String {
        "fs::IFile".into()
    }
    fn call(&self, req: &Request, rsp: &mut Response) -> u32 {
        if req.cmd>4 {return Stub(self.name()).call(req,rsp);}
        let r: std::io::Result<u32> = crate::kernel::blocking(|| {
            let f = self.f.lock().unwrap();
            match req.cmd {
                0 => {
                    let (off, size) = (req.u64(8), req.u64(16));
                    let Some(b) = req.b.first() else { return Ok(0) };
                    let len = size.min(b.size) as usize;
                    let dst = unsafe { std::slice::from_raw_parts_mut(b.addr as *mut u8, if b.addr == 0 { 0 } else { len }) };
                    let n = (|| -> std::io::Result<usize> {
                        let mut n = 0;
                        while n < dst.len() {
                            let k = f.read_at(&mut dst[n..], off + n as u64)?;
                            if k == 0 {
                                break;
                            }
                            n += k;
                        }
                        Ok(n)
                    })()?;
                    rsp.push_u64(n as u64);
                }
                1 => {
                    let (off, size) = (req.u64(8), req.u64(16));
                    let Some(a) = req.a.first() else { return Ok(0) };
                    let Some(end)=off.checked_add(size).filter(|&end|end<=i64::MAX as u64) else{return Ok(rc::OUT_OF_RANGE)};
                    if size>a.size || size>isize::MAX as u64 || (!self.allow_extend && end>f.metadata()?.len()) {return Ok(rc::OUT_OF_RANGE);}
                    let src = unsafe { std::slice::from_raw_parts(a.addr as *const u8, size as usize) };
                    f.write_all_at(src, off)?;
                    if req.u32(0)&1 != 0 {f.sync_data()?;}
                }
                2 => f.sync_all()?,
                3 => f.set_len(req.u64(0))?,
                4 => rsp.push_u64(f.metadata()?.len()),
                _ => unreachable!(),
            }
            Ok(0)
        });
        match r {
            Ok(result) => result,
            Err(e) => {
                eprintln!("[fs] file cmd {}: {e}", req.cmd);
                io_rc(&e)
            }
        }
    }
}

struct HostDir {
    entries: Mutex<std::collections::VecDeque<(String, bool, u64)>>,
    total:u64,
}

impl Service for HostDir {
    fn name(&self) -> String {
        "fs::IDirectory".into()
    }
    fn call(&self, req: &Request, rsp: &mut Response) -> u32 {
        let mut entries = self.entries.lock().unwrap();
        match req.cmd {
            0 => {
                let Some(b) = req.b.first() else { return 0 };
                let max = (b.size / 0x310) as usize;
                let mut out = Vec::new();
                let mut n = 0;
                while n < max {
                    let Some((name, is_dir, size)) = entries.pop_front() else { break };
                    let mut e = vec![0u8; 0x310];
                    let nb = name.as_bytes();
                    e[..nb.len().min(0x300)].copy_from_slice(&nb[..nb.len().min(0x300)]);
                    e[0x304] = if is_dir { 0 } else { 1 };
                    e[0x308..0x310].copy_from_slice(&size.to_le_bytes());
                    out.extend(e);
                    n += 1;
                }
                write_buffer(b, &out);
                rsp.push_u64(n as u64);
            }
            1 => rsp.push_u64(self.total),
            _ => return Stub(self.name()).call(req, rsp),
        }
        0
    }
}

#[cfg(test)]
mod local_persistence_tests {
    use super::*;
    use crate::ipc::Buffer;

    #[test]
    fn allow_append_honors_offsets_without_doubling_saved_files() {
        let root=std::env::temp_dir().join(format!("fc27-fs-offset-test-{}",std::process::id()));
        assert!(!root.exists());
        let fs=HostFs::open("offset-test",&root);
        let path=b"/offset.dat\0";
        let request=|cmd,data:Vec<u8>|Request {cmd,data,x:vec![Buffer {addr:path.as_ptr() as u64,size:path.len() as u64}],..Default::default()};
        let mut create=vec![0;16];create[8..].copy_from_slice(&16u64.to_le_bytes());
        assert_eq!(fs.call(&request(0,create),&mut Response::default()),0);
        let open=|mode:u32| {
            let mut rsp=Response::default();assert_eq!(fs.call(&request(8,mode.to_le_bytes().to_vec()),&mut rsp),0);
            rsp.objects.pop().unwrap()
        };
        let write=|file:&Arc<dyn Service>,offset:u64,payload:&[u8]| {
            let mut data=vec![0;24];data[..4].copy_from_slice(&1u32.to_le_bytes());
            data[8..16].copy_from_slice(&offset.to_le_bytes());data[16..].copy_from_slice(&(payload.len() as u64).to_le_bytes());
            file.call(&Request {cmd:1,data,a:vec![Buffer {addr:payload.as_ptr() as u64,size:payload.len() as u64}],..Default::default()},&mut Response::default())
        };
        let file=open(7); // read, write, and allow file extension
        assert_eq!(write(&file,4,b"FC27"),0);
        // The game preallocates, replaces the container, then overwrites small
        // regions. None of these offset writes may append another container.
        for _ in 0..3 {
            assert_eq!(file.call(&Request {cmd:3,data:16u64.to_le_bytes().to_vec(),..Default::default()},&mut Response::default()),0);
            assert_eq!(write(&file,4,b"SAVE"),0);
            assert_eq!(write(&file,5,b"!"),0);
            assert_eq!(std::fs::metadata(root.join("offset.dat")).unwrap().len(),16);
        }
        assert_eq!(write(&file,16,b"TAIL"),0);
        assert_eq!(write(&file,u64::MAX-1,b"BAD"),rc::OUT_OF_RANGE);
        assert_eq!(write(&file,i64::MAX as u64,b"BAD"),rc::OUT_OF_RANGE);
        let fixed=open(3);
        assert_eq!(write(&fixed,19,b"BAD"),rc::OUT_OF_RANGE);
        assert_eq!(write(&fixed,0,b"HEAD"),0);
        assert_eq!(fs.call(&Request {cmd:10,..Default::default()},&mut Response::default()),0);
        drop((file,fixed));
        let reopened=open(1);
        let mut bytes=[0u8;20];let mut data=vec![0;24];data[16..].copy_from_slice(&20u64.to_le_bytes());
        let mut rsp=Response::default();
        assert_eq!(reopened.call(&Request {cmd:0,data,b:vec![Buffer {addr:bytes.as_mut_ptr() as u64,size:20}],..Default::default()},&mut rsp),0);
        assert_eq!(u64::from_le_bytes(rsp.data[..8].try_into().unwrap()),20);
        assert_eq!(&bytes,b"HEADS!VE\0\0\0\0\0\0\0\0TAIL");
        assert_eq!(std::fs::metadata(root.join("offset.dat")).unwrap().len(),20);
        drop(reopened);drop(fs);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn saves_commit_reopen_timestamp_and_report_real_attributes() {
        let root=std::env::temp_dir().join(format!("fc27-fs-test-{}",std::process::id()));
        assert!(!root.exists());
        let fs=HostFs::open("test",&root);
        let path=b"/save.dat\0";
        let request=|cmd,data:Vec<u8>|Request {cmd,data,x:vec![Buffer {addr:path.as_ptr() as u64,size:path.len() as u64}],..Default::default()};
        let mut rsp=Response::default();
        assert_eq!(fs.call(&Request {cmd:16,..Default::default()},&mut rsp),0);
        assert_eq!(rsp.data.len(),0xc0);assert_eq!(&rsp.data[..4],&[1,1,1,1]);
        assert_eq!(fs.call(&Request {cmd:999,..Default::default()},&mut Response::default()),crate::ipc::rc::UNKNOWN_COMMAND);
        let mut create=vec![0;16];create[8..].copy_from_slice(&4u64.to_le_bytes());
        assert_eq!(fs.call(&request(0,create),&mut Response::default()),0);
        let mut opened=Response::default();assert_eq!(fs.call(&request(8,3u32.to_le_bytes().to_vec()),&mut opened),0);
        let file=opened.objects.pop().unwrap();let payload=*b"FC27";
        let mut write=vec![0;24];write[..4].copy_from_slice(&1u32.to_le_bytes());write[16..].copy_from_slice(&4u64.to_le_bytes());
        assert_eq!(file.call(&Request {cmd:1,data:write,a:vec![Buffer {addr:payload.as_ptr() as u64,size:4}],..Default::default()},&mut Response::default()),0);
        assert_eq!(fs.call(&Request {cmd:10,..Default::default()},&mut Response::default()),0);
        let mut timestamp=Response::default();assert_eq!(fs.call(&request(14,vec![]),&mut timestamp),0);
        assert!(u64::from_le_bytes(timestamp.data[8..16].try_into().unwrap())>1_700_000_000);
        assert_eq!(timestamp.data[24],1);
        drop((file,fs));
        let fs=HostFs::open("reopened",&root);let mut opened=Response::default();
        assert_eq!(fs.call(&request(8,1u32.to_le_bytes().to_vec()),&mut opened),0);
        let mut read=[0u8;4];let mut data=vec![0;24];data[16..].copy_from_slice(&4u64.to_le_bytes());
        assert_eq!(opened.objects[0].call(&Request {cmd:0,data,b:vec![Buffer {addr:read.as_mut_ptr() as u64,size:4}],..Default::default()},&mut Response::default()),0);
        assert_eq!(&read,b"FC27");
        assert_eq!(opened.objects[0].call(&Request {cmd:99,..Default::default()},&mut Response::default()),crate::ipc::rc::UNKNOWN_COMMAND);
        std::fs::remove_dir_all(root).unwrap();
    }
}
