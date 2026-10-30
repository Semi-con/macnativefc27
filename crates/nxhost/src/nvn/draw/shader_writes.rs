use super::{Replay,GState,state::Res};
use std::sync::LazyLock;
use objc2_metal::{MTLBuffer,MTLResource,MTLStorageMode};

pub(super) fn enabled()->bool {
    static ENABLED:LazyLock<bool>=LazyLock::new(||std::env::var_os("NX_DEFER_SHADER_WAITS").is_some());
    *ENABLED
}

impl Replay<'_> {
    // Inspect only an immutable overlay or the very same shared mapping bound
    // to this dispatch. Shared UBO contents retain the existing API lifetime:
    // the CPU must not mutate an in-flight GPU input. No new snapshot/freeze,
    // stale shadow, or diagnostic GPU wait is introduced by this proof.
    pub(super) fn grid_write_ranges(&mut self,proof:nxshader::GridWriteFootprint,st:&GState,res:&Res,groups:[u32;3])->Option<Vec<(u64,u64,u64)>> {
        if !enabled() || self.e.gpu_buffer_dirty {return None;}
        let bank=proof.row_bank;let offset=(proof.row_word as u64).checked_mul(4)?;
        let &(addr,size)=st.ubos[2].get(bank.checked_sub(3)?)?;
        if addr==0 || offset.checked_add(4)?>size {return None;}
        let word=addr.checked_add(offset)?;
        let overlay=self.e.overlays.range(..=addr).next_back().and_then(|(&base,&(len,gpu,ptr))|
            (addr-base<len && word-base<=len && len-(word-base)>=4 && res.cb[bank]==gpu+addr-base)
                .then_some(ptr as u64+word-base));
        let (row_cpu,live)=if let Some(ptr)=overlay {(ptr,false)} else {
            let (chunk,off)=self.buffer_range(word,4)?;
            if chunk.buffer.0.storageMode()!=MTLStorageMode::Shared {return None;}
            if res.cb[bank].checked_add(offset)?!=chunk.gpu+off as u64 {return None;}
            (chunk.buffer.0.contents().as_ptr() as u64+off as u64,true)
        };
        if live && (self.pending_cpu_write(word,4) || self.pending_cpu_write(row_cpu,4)) {return None;}
        let row=unsafe {(row_cpu as *const u32).read_unaligned()};
        let base=st.storage[2].get(proof.storage_slot)?.0;
        let (base,len)=proof.bytes(base,row,groups)?;
        if len>16*1024*1024 {return None;}
        let mut ranges=Vec::new();let mut at=base;let end=base.checked_add(len)?;
        while at<end {
            let (chunk,off)=self.buffer_range(at,1)?;
            let n=(chunk.size.checked_sub(at.checked_sub(chunk.base)?)?).min(end-at);
            if n==0 || ranges.len()>=256 {return None;}
            if chunk.buffer.0.storageMode()!=MTLStorageMode::Shared {return None;}
            let cpu=chunk.buffer.0.contents().as_ptr() as u64+off as u64;
            ranges.push((at,cpu,n));at=at.checked_add(n)?;
        }
        // A writer may legally address outside the declared SSBO length.
        // Prove those actual mappings too, and reject self-modifying live row
        // inputs (including partial physical/virtual aliases).
        if live {
            for &(gpu,cpu,n) in &ranges {
                if self.aliases_for_write(gpu,cpu,n).iter().any(|&(a,m)|
                    (a<word.saturating_add(4) && word<a.saturating_add(m)) ||
                    (a<row_cpu.saturating_add(4) && row_cpu<a.saturating_add(m))) {return None;}
            }
        }
        Some(ranges)
    }
}

impl Replay<'_> {
    pub(super) fn descriptor_write_ranges(&mut self,stores:&[nxshader::DescriptorWriteFootprint],st:&GState)->Option<Vec<(u64,u64,u64)>> {
        if !enabled() {return None;}
        let mut ranges=Vec::new();
        for &proof in stores {
            let base=st.storage[2].get(proof.storage_slot)?.0;
            let (mut at,len)=proof.range(base)?;let end=at.checked_add(len)?;
            while at<end {
                let (chunk,off)=self.buffer_range(at,1)?;
                let n=chunk.size.checked_sub(at.checked_sub(chunk.base)?)?.min(end-at);
                if n==0 || ranges.len()>=256 || chunk.buffer.0.storageMode()!=MTLStorageMode::Shared {return None;}
                ranges.push((at,chunk.buffer.0.contents().as_ptr() as u64+off as u64,n));at=at.checked_add(n)?;
            }
        }
        ranges.sort_unstable();ranges.dedup();Some(ranges)
    }
}

impl Replay<'_> {
    pub(super) fn proven_write_buffers(&self,ranges:&[(u64,u64,u64)])->Vec<super::gfx::Shared<super::gfx::Buffer>> {
        if ranges.is_empty() {return Vec::new();}
        let mut out=Vec::new();
        // Include every overlapping physical buffer object, not only the
        // virtual address selected by the store; Metal tracks object hazards.
        let chunks=self.pools.iter().flat_map(|p|p.chunks.iter()).chain(self.heap.into_iter().flat_map(|h|h.chunks.iter()));
        for chunk in chunks {
            let cpu=chunk.buffer.0.contents().as_ptr() as u64+chunk.offset as u64;
            if ranges.iter().any(|&(_,at,len)|cpu<at.saturating_add(len) && at<cpu.saturating_add(chunk.size)) && !out.iter().any(|b:&super::gfx::Shared<super::gfx::Buffer>|objc2::rc::Retained::as_ptr(&b.0)==objc2::rc::Retained::as_ptr(&chunk.buffer.0)) {out.push(chunk.buffer.clone());}
        }
        out
    }
}
