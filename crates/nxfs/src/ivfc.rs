//! Verify each RomFS block through every IVFC parent, including BKTR overlays.
use crate::storage::{DynStorage, Storage};
use anyhow::{Result, ensure};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

#[derive(Clone, Copy)]
struct Level { offset: u64, size: u64, block: u64 }

struct Ivfc {
    source: DynStorage,
    levels: Vec<Level>,
    master: [u8; 32],
    cache: Mutex<(usize, HashMap<(usize,u64), Arc<[u8]>>)>,
}

pub(crate) fn open(source: DynStorage, header: &[u8]) -> Result<DynStorage> {
    ensure!(header.len() == 0x200 && &header[8..12] == b"IVFC", "unsupported RomFS integrity format");
    let u32_at = |o| u32::from_le_bytes(header[o..o+4].try_into().unwrap());
    let u64_at = |o| u64::from_le_bytes(header[o..o+8].try_into().unwrap());
    ensure!(u32_at(12) == 0x20000 && u32_at(16) == 32 && u32_at(20) == 7, "unsupported IVFC version or levels");
    let mut levels = Vec::new();
    for i in 0..6 {
        let at = 0x18 + 24*i;
        let (offset,size,power) = (u64_at(at),u64_at(at+8),u32_at(at+16));
        ensure!((12..=24).contains(&power) && u32_at(at+20) == 0 && size != 0 && offset.checked_add(size).is_some_and(|e| e <= source.size()), "IVFC level outside storage");
        levels.push(Level { offset,size,block: 1u64 << power });
    }
    ensure!(levels[0].size <= levels[0].block, "multiple IVFC master hashes unsupported");
    for i in 1..6 {
        ensure!(levels[i].size.div_ceil(levels[i].block)*32 <= levels[i-1].size, "IVFC parent table too small");
    }
    Ok(Arc::new(Ivfc { source,levels,master:header[0xc8..0xe8].try_into()?,cache:Mutex::new((0,HashMap::new())) }))
}

impl Ivfc {
    fn block(&self, level: usize, index: u64) -> Result<Arc<[u8]>> {
        if let Some(data) = self.cache.lock().unwrap().1.get(&(level,index)).cloned() { return Ok(data); }
        let l = self.levels[level];
        let at = index.checked_mul(l.block).ok_or_else(|| anyhow::anyhow!("IVFC block overflow"))?;
        ensure!(at < l.size, "IVFC block outside level");
        let len = l.block.min(l.size-at) as usize;
        let mut padded = vec![0u8; l.block as usize];
        self.source.read_at(l.offset+at,&mut padded[..len])?;
        let expected = if level == 0 { self.master } else {
            let mut hash = [0;32]; self.read_level(level-1,index*32,&mut hash)?; hash
        };
        ensure!(Sha256::digest(&padded)[..] == expected, "IVFC hash mismatch at level {level} block {index}");
        padded.truncate(len);
        let data: Arc<[u8]> = padded.into();
        let mut c = self.cache.lock().unwrap();
        if c.0 + len > 64*1024*1024 { c.1.clear(); c.0 = 0; }
        if c.1.insert((level,index),data.clone()).is_none() { c.0 += len; }
        Ok(data)
    }
    fn read_level(&self, level: usize, mut offset: u64, mut out: &mut [u8]) -> Result<()> {
        let l = self.levels[level];
        ensure!(offset.checked_add(out.len() as u64).is_some_and(|e| e <= l.size), "read outside IVFC level");
        while !out.is_empty() {
            let (index,within) = (offset/l.block,(offset%l.block) as usize);
            let data = self.block(level,index)?;
            let n = out.len().min(data.len()-within);
            out[..n].copy_from_slice(&data[within..within+n]);
            out = &mut out[n..]; offset += n as u64;
        }
        Ok(())
    }
}
impl Storage for Ivfc {
    fn size(&self) -> u64 { self.levels[5].size }
    fn read_at(&self, offset: u64, out: &mut [u8]) -> Result<()> { self.read_level(5,offset,out) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::Mem;
    fn fixture(corrupt: bool) -> (DynStorage, Vec<u8>) {
        let mut image = vec![0u8;6*4096]; let mut header = vec![0u8;512];
        header[8..12].copy_from_slice(b"IVFC");
        for (at,v) in [(12,0x20000u32),(16,32),(20,7)] { header[at..at+4].copy_from_slice(&v.to_le_bytes()); }
        image[5*4096..5*4096+5].copy_from_slice(b"hello");
        for i in (0..5).rev() {
            let hash = Sha256::digest(&image[(i+1)*4096..(i+2)*4096]);
            image[i*4096..i*4096+32].copy_from_slice(&hash);
        }
        header[0xc8..0xe8].copy_from_slice(&Sha256::digest(&image[..4096]));
        for i in 0..6 {
            let at = 0x18+i*24;
            header[at..at+8].copy_from_slice(&(i as u64*4096).to_le_bytes());
            header[at+8..at+16].copy_from_slice(&(if i == 5 {5u64} else {32u64}).to_le_bytes());
            header[at+16..at+20].copy_from_slice(&12u32.to_le_bytes());
        }
        if corrupt { image[5*4096] ^= 1; }
        (Arc::new(Mem(image)), header)
    }
    #[test]
    fn verifies_partial_final_block_through_all_parents() {
        let (s,h) = fixture(false); assert_eq!(open(s,&h).unwrap().read_vec(0,5).unwrap(),b"hello");
    }
    #[test]
    fn rejects_payload_corruption() {
        let (s,h) = fixture(true); assert!(open(s,&h).unwrap().read_vec(0,5).unwrap_err().to_string().contains("hash mismatch"));
    }
}
