// Mush8 - pfs0.rs
// Constantin Clerc (c22dev) - 0.1
use crate::storage::{DynStorage, Sub};
use anyhow::{Result, bail, ensure};
use std::sync::Arc;

pub struct Entry {
    pub name: String,
    pub offset: u64,
    pub size: u64,
}

pub struct Pfs0 {
    storage: DynStorage,
    pub entries: Vec<Entry>,
}

impl Pfs0 {
    pub fn open(storage: DynStorage) -> Result<Self> {
        Self::open_format(storage, b"PFS0", 24)
    }

    pub fn open_hfs(storage: DynStorage) -> Result<Self> {
        Self::open_format(storage, b"HFS0", 64)
    }

    fn open_format(storage: DynStorage, magic: &[u8; 4], stride: usize) -> Result<Self> {
        ensure!(storage.size() >= 16, "truncated partition header");
        let hdr = storage.read_vec(0, 16)?;
        if &hdr[..4] != magic {
            bail!("unexpected partition magic");
        }
        let n = u32::from_le_bytes(hdr[4..8].try_into()?) as usize;
        let st = u32::from_le_bytes(hdr[8..12].try_into()?) as usize;
        ensure!(n <= 4096 && st <= 1024 * 1024, "oversized partition table");
        let table_size = n.checked_mul(stride).and_then(|s| s.checked_add(st)).ok_or_else(|| anyhow::anyhow!("partition size overflow"))?;
        ensure!(16 + table_size as u64 <= storage.size(), "partition table outside storage");
        let table = storage.read_vec(16, table_size)?;
        let strs = &table[n * stride..];
        let data = 16 + table_size as u64;
        let mut entries = Vec::with_capacity(n);
        for i in 0..n {
            let e = &table[i * stride..];
            let off = u64::from_le_bytes(e[0..8].try_into()?);
            let size = u64::from_le_bytes(e[8..16].try_into()?);
            let so = u32::from_le_bytes(e[16..20].try_into()?) as usize;
            ensure!(so < strs.len(), "partition name outside string table");
            let end = strs[so..].iter().position(|&c| c == 0).ok_or_else(|| anyhow::anyhow!("unterminated partition name"))?;
            let name = std::str::from_utf8(&strs[so..so + end])?.to_owned();
            ensure!(!name.is_empty() && name != "." && name != ".." && !name.contains(['/', '\\']), "unsafe partition name");
            ensure!(!entries.iter().any(|e: &Entry| e.name == name), "duplicate partition name");
            ensure!(off.checked_add(size).and_then(|e| e.checked_add(data)).is_some_and(|e| e <= storage.size()), "partition entry outside storage");
            entries.push(Entry { name, offset: data + off, size });
        }
        let mut ranges: Vec<_> = entries.iter().map(|e| (e.offset, e.offset + e.size)).collect();
        ranges.sort_unstable();
        let mut end = data;
        for (lo, hi) in ranges {
            ensure!(lo >= end, "overlapping partition entries");
            end = hi;
        }
        Ok(Self { storage, entries })
    }

    pub fn file(&self, name: &str) -> Option<DynStorage> {
        self.entries.iter().find(|e| e.name == name).map(|e| self.storage_of(e))
    }

    pub fn storage_of(&self, e: &Entry) -> DynStorage {
        Arc::new(Sub::new(self.storage.clone(), e.offset, e.size))
    }
}
