// Mush8 - nso.rs
// Constantin Clerc (c22dev) - 0.1
use anyhow::{Result, ensure};

#[derive(Debug, Clone, Copy)]
pub struct Segment {
    pub mem_offset: u32,
    pub size: u32,
}

pub struct Nso {
    pub module_id: [u8; 32],
    pub text: Segment,
    pub ro: Segment,
    pub data: Segment,
    pub bss_size: u32,
    pub image: Vec<u8>,
    dynstr: (u32, u32),
    dynsym: (u32, u32),
}

#[derive(Debug, Clone)]
pub struct Symbol {
    pub name: String,
    pub value: u64,
    pub size: u64,
    pub info: u8,
    pub defined: bool,
}

fn rd(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes(b[o..o + 4].try_into().unwrap())
}

impl Nso {
    pub fn parse(file: &[u8]) -> Result<Self> {
        if file.get(0x10..0x14) == Some(b"NRO0") {
            return Self::parse_nro(file);
        }
        ensure!(file.len() >= 0x100 && &file[..4] == b"NSO0", "not an NSO");
        let flags = rd(file, 0xC);
        ensure!(flags & 0x80 == 0, "ZBIC NSO: normalize with the verified offline decoder first");
        let mut segs = [Segment { mem_offset: 0, size: 0 }; 3];
        let mut end = 0usize;
        let mut parts = Vec::new();
        for i in 0..3 {
            let h = 0x10 + i * 0x10;
            let (foff, moff, size) = (rd(file, h) as usize, rd(file, h + 4), rd(file, h + 8));
            let csize = rd(file, 0x60 + i * 4) as usize;
            let len = if flags & (1 << i) != 0 { csize } else { size as usize };
            let raw = file.get(foff..foff.checked_add(len).ok_or_else(|| anyhow::anyhow!("segment overflow"))?).ok_or_else(|| anyhow::anyhow!("segment {i} outside file"))?;
            let bytes = if flags & (1 << i) != 0 { lz4_flex::block::decompress(raw, size as usize)? } else { raw.to_vec() };
            ensure!(bytes.len() == size as usize, "segment {i} size mismatch");
            segs[i] = Segment { mem_offset: moff, size };
            end = end.max((moff + size) as usize);
            parts.push(bytes);
        }
        let mut image = vec![0; end];
        for (s, b) in segs.iter().zip(&parts) {
            image[s.mem_offset as usize..][..b.len()].copy_from_slice(b);
        }
        Ok(Self {
            module_id: file[0x40..0x60].try_into()?,
            text: segs[0],
            ro: segs[1],
            data: segs[2],
            bss_size: rd(file, 0x3C),
            image,
            dynstr: (rd(file, 0x90), rd(file, 0x94)),
            dynsym: (rd(file, 0x98), rd(file, 0x9C)),
        })
    }

    fn parse_nro(file: &[u8]) -> Result<Self> {
        ensure!(file.len() >= 0x80, "truncated NRO");
        ensure!(rd(file, 0x1c) & !1 == 0, "unsupported NRO flags");
        let size = rd(file, 0x18) as usize;
        ensure!(size >= 0x80 && size <= file.len(), "invalid NRO image size");
        let segment = |at| Segment { mem_offset: rd(file, at), size: rd(file, at + 4) };
        let (text, ro, data) = (segment(0x20), segment(0x28), segment(0x30));
        for s in [text, ro, data] {
            ensure!((s.mem_offset as u64 + s.size as u64) <= size as u64, "NRO segment outside image");
        }
        ensure!(text.mem_offset == 0, "unsupported NRO text offset");
        let image = file[..size].to_vec();
        // NRO segments retain their original virtual offsets, including the NRO header.
        // BSS addends are interpreted by module::parse without inventing file bytes.
        crate::module::parse(&image)?;
        let dynstr = (rd(file, 0x70).saturating_sub(ro.mem_offset), rd(file, 0x74));
        let dynsym = (rd(file, 0x78).saturating_sub(ro.mem_offset), rd(file, 0x7c));
        Ok(Self { module_id: file[0x40..0x60].try_into()?, text, ro, data,
            bss_size: rd(file, 0x38), image, dynstr, dynsym })
    }

    pub fn symbols(&self) -> Vec<Symbol> {
        if let Ok(dynamic) = crate::module::parse(&self.image) {
            return dynamic.symbols.into_iter().filter(|s| !s.name.is_empty()).map(|s| Symbol {
                name: s.name, value: s.value, size: 0, info: s.info, defined: s.shndx != 0,
            }).collect();
        }
        let ro = self.ro.mem_offset as usize;
        let strs = &self.image[ro + self.dynstr.0 as usize..][..self.dynstr.1 as usize];
        let syms = &self.image[ro + self.dynsym.0 as usize..][..self.dynsym.1 as usize];
        syms.chunks_exact(24)
            .skip(1)
            .filter_map(|s| {
                let n = rd(s, 0) as usize;
                let end = strs[n..].iter().position(|&c| c == 0)?;
                let name = String::from_utf8_lossy(&strs[n..n + end]).into_owned();
                if name.is_empty() {
                    return None;
                }
                Some(Symbol {
                    name,
                    info: s[4],
                    defined: u16::from_le_bytes([s[6], s[7]]) != 0,
                    value: u64::from_le_bytes(s[8..16].try_into().unwrap()),
                    size: u64::from_le_bytes(s[16..24].try_into().unwrap()),
                })
            })
            .collect()
    }

    pub fn text_bytes(&self) -> &[u8] {
        &self.image[self.text.mem_offset as usize..][..self.text.size as usize]
    }
}

pub fn demangle(name: &str) -> String {
    cpp_demangle::Symbol::new(name)
        .ok()
        .and_then(|s| s.demangle().ok())
        .unwrap_or_else(|| name.to_string())
}
