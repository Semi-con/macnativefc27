// Mush8 - module.rs
// Constantin Clerc (c22dev) - 0.1
use anyhow::{Result, bail, ensure};

pub const R_AARCH64_ABS64: u32 = 257;
pub const R_AARCH64_GLOB_DAT: u32 = 1025;
pub const R_AARCH64_JUMP_SLOT: u32 = 1026;
pub const R_AARCH64_RELATIVE: u32 = 1027;
pub const R_AARCH64_TLSDESC: u32 = 1031;

#[derive(Debug, Clone, Copy)]
pub struct Rela {
    pub offset: u64,
    pub kind: u32,
    pub sym: u32,
    pub addend: i64,
}

#[derive(Debug, Clone)]
pub struct Sym {
    pub name: String,
    pub value: u64,
    pub info: u8,
    pub visibility: u8,
    pub shndx: u16,
}

impl Sym {
    pub fn defined(&self) -> bool {
        self.shndx != 0
    }
    pub fn weak(&self) -> bool {
        self.info >> 4 == 2
    }
}

#[derive(Debug, Default)]
pub struct Dynamic {
    pub mod0: u64,
    pub bss: (u64, u64),
    pub relocs: Vec<Rela>,
    pub symbols: Vec<Sym>,
    pub init: Option<u64>,
    pub fini: Option<u64>,
    pub init_array: Option<(u64, u64)>,
    pub fini_array: Option<(u64, u64)>,
}

fn u32_at(b: &[u8], o: u64) -> Result<u32> {
    let o = o as usize;
    ensure!(o + 4 <= b.len(), "read past image at {o:#x}");
    Ok(u32::from_le_bytes(b[o..o + 4].try_into()?))
}
fn u64_at(b: &[u8], o: u64) -> Result<u64> {
    let o = o as usize;
    ensure!(o + 8 <= b.len(), "read past image at {o:#x}");
    Ok(u64::from_le_bytes(b[o..o + 8].try_into()?))
}

pub fn parse(image: &[u8]) -> Result<Dynamic> {
    let mod0 = u32_at(image, 4)? as u64;
    if u32_at(image, mod0)? != u32::from_le_bytes(*b"MOD0") {
        bail!("no MOD0 header");
    }
    let rel = |o: u64| -> Result<u64> { Ok((mod0 as i64 + u32_at(image, mod0 + o)? as i32 as i64) as u64) };
    let dynamic = rel(4)?;
    let mut d = Dynamic { mod0, bss: (rel(8)?, rel(12)?), ..Default::default() };

    let (mut rela, mut relasz, mut jmprel, mut pltrelsz) = (0, 0, 0, 0);
    let (mut symtab, mut strtab, mut strsz, mut syment, mut hash, mut gnu_hash) = (0, 0, 0, 24, 0, 0);
    let (mut relr, mut relrsz, mut relrent) = (0, 0, 8);
    let (mut init_array, mut init_arraysz, mut fini_array, mut fini_arraysz) = (None, 0, None, 0);
    let mut o = dynamic;
    loop {
        let (tag, val) = (u64_at(image, o)? as i64, u64_at(image, o + 8)?);
        o += 16;
        match tag {
            0 => break,
            2 => pltrelsz = val,
            4 => hash = val,
            5 => strtab = val,
            6 => symtab = val,
            7 => rela = val,
            8 => relasz = val,
            10 => strsz = val,
            11 => syment = val,
            12 => d.init = Some(val),
            13 => d.fini = Some(val),
            23 => jmprel = val,
            25 => init_array = Some(val),
            26 => fini_array = Some(val),
            27 => init_arraysz = val,
            28 => fini_arraysz = val,
            35 => relrsz = val,
            36 => relr = val,
            37 => relrent = val,
            0x6ffffef5 => gnu_hash = val,
            _ => {}
        }
    }
    d.init_array = init_array.map(|a| (a, init_arraysz));
    d.fini_array = fini_array.map(|a| (a, fini_arraysz));

    for (start, size) in [(rela, relasz), (jmprel, pltrelsz)] {
        ensure!(size % 24 == 0 && start.checked_add(size).is_some_and(|e| e <= image.len() as u64), "invalid RELA table");
        let mut r = start;
        while r < start + size {
            let info = u64_at(image, r + 8)?;
            d.relocs.push(Rela {
                offset: u64_at(image, r)?,
                kind: info as u32,
                sym: (info >> 32) as u32,
                addend: u64_at(image, r + 16)? as i64,
            });
            r += 24;
        }
    }

    if relrsz != 0 {
        ensure!(relr != 0 && relrent == 8 && relrsz % 8 == 0, "invalid RELR layout");
        ensure!(relr.checked_add(relrsz).is_some_and(|e| e <= image.len() as u64), "RELR table outside image");
        let mut cursor = 0u64;
        let mut append = |where_: u64| -> Result<()> {
            ensure!(where_ % 8 == 0, "unaligned RELR target");
            let end = where_.checked_add(8).ok_or_else(|| anyhow::anyhow!("RELR target overflow"))?;
            let addend = if end <= image.len() as u64 {
                u64_at(image, where_)?
            } else {
                ensure!(where_ >= d.bss.0 && end <= d.bss.1, "RELR target outside image and BSS");
                0
            };
            d.relocs.push(Rela { offset: where_, kind: R_AARCH64_RELATIVE, sym: 0, addend: addend as i64 });
            Ok(())
        };
        for offset in (0..relrsz).step_by(8) {
            let entry = u64_at(image, relr + offset)?;
            if entry & 1 == 0 {
                append(entry)?;
                cursor = entry.checked_add(8).ok_or_else(|| anyhow::anyhow!("RELR cursor overflow"))?;
            } else {
                ensure!(cursor != 0, "RELR bitmap without address");
                for bit in 1..64 {
                    if entry & (1u64 << bit) != 0 {
                        append(cursor.checked_add((bit - 1) * 8).ok_or_else(|| anyhow::anyhow!("RELR bitmap overflow"))?)?;
                    }
                }
                cursor = cursor.checked_add(63 * 8).ok_or_else(|| anyhow::anyhow!("RELR cursor overflow"))?;
            }
        }
    }

    ensure!(symtab != 0 && strtab != 0 && syment == 24, "invalid dynamic symbol layout");
    let nsyms = if hash != 0 {
        u32_at(image, hash + 4)? as u64
    } else {
        ensure!(gnu_hash != 0, "no ELF symbol hash table");
        let buckets = u32_at(image, gnu_hash)? as u64;
        let first = u32_at(image, gnu_hash + 4)? as u64;
        let bloom = u32_at(image, gnu_hash + 8)? as u64;
        ensure!(buckets <= image.len() as u64 / 4 && bloom <= image.len() as u64 / 8, "oversized GNU hash");
        let bucket_at = gnu_hash + 16 + bloom * 8;
        let chain_at = bucket_at + buckets * 4;
        let mut max = 0;
        for i in 0..buckets {
            max = max.max(u32_at(image, bucket_at + i * 4)? as u64);
        }
        if max == 0 { first } else {
            ensure!(max >= first, "invalid GNU hash bucket");
            loop {
                let chain = u32_at(image, chain_at + (max - first) * 4)?;
                max += 1;
                if chain & 1 != 0 { break; }
            }
            max
        }
    };
    ensure!(nsyms <= 1_000_000 && symtab.checked_add(nsyms * syment).is_some_and(|e| e <= image.len() as u64), "dynamic symbols outside image");
    let string_end = if strsz == 0 { image.len() as u64 } else { strtab.checked_add(strsz).ok_or_else(|| anyhow::anyhow!("string table overflow"))? };
    ensure!(string_end <= image.len() as u64, "dynamic string table outside image");
    for i in 0..nsyms {
        let s = symtab + i * syment;
        let name_off = strtab + u32_at(image, s)? as u64;
        ensure!(name_off < string_end, "symbol name outside string table");
        let end = image[name_off as usize..string_end as usize].iter().position(|&c| c == 0).ok_or_else(|| anyhow::anyhow!("unterminated symbol name"))?;
        d.symbols.push(Sym {
            name: String::from_utf8_lossy(&image[name_off as usize..name_off as usize + end]).into_owned(),
            info: image[s as usize + 4],
            visibility: image[s as usize + 5] & 3,
            shndx: u16::from_le_bytes([image[s as usize + 6], image[s as usize + 7]]),
            value: u64_at(image, s + 8)?,
        });
    }
    Ok(d)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> Vec<u8> {
        let mut b = vec![0u8; 0x600];
        let put32 = |b: &mut [u8], o, v: u32| b[o..o+4].copy_from_slice(&v.to_le_bytes());
        let put64 = |b: &mut [u8], o, v: u64| b[o..o+8].copy_from_slice(&v.to_le_bytes());
        put32(&mut b, 4, 0x40);
        b[0x40..0x44].copy_from_slice(b"MOD0");
        put32(&mut b, 0x44, 0x40);
        put32(&mut b, 0x48, 0x5c0);
        put32(&mut b, 0x4c, 0x6c0);
        let tags = [(5,0x300), (10,5), (6,0x200), (11,24),
            (0x6ffffef5,0x180), (35,24), (36,0x380), (37,8), (0,0)];
        for (i,(tag,v)) in tags.into_iter().enumerate() {
            put64(&mut b, 0x80+16*i, tag); put64(&mut b, 0x88+16*i, v);
        }
        for (i,v) in [1,1,1,0].into_iter().enumerate() { put32(&mut b, 0x180+4*i,v); }
        put32(&mut b, 0x198, 1); // one bucket points at symbol 1
        put32(&mut b, 0x19c, 1); // final hash chain
        put32(&mut b, 0x218, 1); b[0x21c] = 0x12; b[0x21e] = 1;
        put64(&mut b, 0x220, 0x400);
        b[0x300..0x305].copy_from_slice(b"\0foo\0");
        put64(&mut b, 0x380, 0x480);
        put64(&mut b, 0x388, 3); // next word is a relative relocation
        put64(&mut b, 0x390, 0x608); // target in zero initialized BSS
        put64(&mut b, 0x480, 0x1234); put64(&mut b, 0x488, 0x5678);
        b
    }
    #[test]
    fn gnu_hash_relr_and_bss_addends() {
        let d = parse(&fixture()).unwrap();
        assert_eq!(d.symbols.len(), 2); assert_eq!(d.symbols[1].name, "foo");
        assert_eq!(d.relocs.iter().map(|r| (r.offset,r.addend)).collect::<Vec<_>>(),
            vec![(0x480,0x1234),(0x488,0x5678),(0x608,0)]);
    }
    #[test]
    fn relr_bitmap_requires_an_address_and_bounded_targets() {
        let mut b = fixture(); b[0x380..0x388].copy_from_slice(&3u64.to_le_bytes());
        assert!(parse(&b).unwrap_err().to_string().contains("bitmap without address"));
        let mut b = fixture(); b[0x390..0x398].copy_from_slice(&0x800u64.to_le_bytes());
        assert!(parse(&b).unwrap_err().to_string().contains("outside image and BSS"));
    }
}
