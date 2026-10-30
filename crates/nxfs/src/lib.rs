// Mush8 - lib.rs
// Constantin Clerc (c22dev) - 0.1
pub mod bflan;
pub mod bflyt;
pub mod bntx;
pub mod crypto;
pub mod keys;
mod ivfc;
pub mod module;
pub mod msbt;
pub mod nca;
pub mod nso;
pub mod pfs0;
pub mod romfs;
pub mod sarc;
pub mod storage;

use anyhow::{Context, Result, ensure};
use nca::{Nca, TitleKeys};
use pfs0::Pfs0;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;
use storage::{DynStorage, FileRange, Sub};

pub struct Nsp {
    pub ncas: Vec<Nca>,
}

impl Nsp {
    pub fn open(path: &Path, keys: &keys::Keys) -> Result<Self> {
        let f = Arc::new(File::open(path).with_context(|| format!("opening {}", path.display()))?);
        let size = f.metadata()?.len();
        let source: DynStorage = Arc::new(FileRange::new(f, 0, size));
        let pfs = if source.read_vec(0, 4)? == b"PFS0" {
            Pfs0::open(source)?
        } else {
            ensure!(size >= 0x138 && source.read_vec(0x100, 4)? == b"HEAD", "expected NSP or XCI");
            let root = u64::from_le_bytes(source.read_vec(0x130, 8)?.try_into().unwrap());
            ensure!(root < size, "XCI root outside source");
            let roots = Pfs0::open_hfs(Arc::new(Sub::new(source, root, size - root)))?;
            let secure = roots.file("secure").context("XCI has no secure partition")?;
            Pfs0::open_hfs(secure)?
        };

        let mut title_keys = TitleKeys::new();
        for e in pfs.entries.iter().filter(|e| e.name.ends_with(".tik")) {
            let tik = pfs.storage_of(e).read_vec(0, e.size as usize)?;
            let (rights, key) = nca::ticket_title_key(keys, &tik).with_context(|| e.name.clone())?;
            title_keys.insert(rights, key);
        }

        let mut ncas = Vec::new();
        for e in pfs.entries.iter().filter(|e| e.name.ends_with(".nca")) {
            let s: DynStorage = pfs.storage_of(e);
            ncas.push(Nca::open(&e.name, s, keys, &title_keys)?);
        }
        Ok(Self { ncas })
    }

    pub fn find(&self, t: nca::ContentType) -> Option<&Nca> {
        self.ncas.iter().find(|n| n.content_type == t)
    }

    pub fn program(&self, patch: bool) -> Result<&Nca> {
        let mut programs = self.ncas.iter().filter(|n| n.content_type == nca::ContentType::Program &&
            n.sections.iter().any(|s| s.encryption == nca::Encryption::CtrEx) == patch);
        let program = programs.next().context("selected base/update Program NCA is missing")?;
        ensure!(programs.next().is_none(), "ambiguous Program NCAs in input");
        Ok(program)
    }
}

pub fn open_nca(path: &Path, keys: &keys::Keys) -> Result<Nca> {
    let f = Arc::new(File::open(path).with_context(|| format!("opening {}", path.display()))?);
    let size = f.metadata()?.len();
    let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    Nca::open(&name, Arc::new(FileRange::new(f, 0, size)), keys, &TitleKeys::new())
}
