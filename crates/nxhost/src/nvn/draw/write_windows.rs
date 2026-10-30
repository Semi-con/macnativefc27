// Conservative domains for data-dependent STG low words. Only immutable CB0
// high roots are inspected. Resolve every mapped physical/virtual alias; actual
// CPU reads keep the existing CpuEvents guard. Unmapped domains do not write.
use super::{Replay,GState};use objc2_metal::*;use std::sync::LazyLock;
pub(super) fn enabled()->bool {static E:LazyLock<bool>=LazyLock::new(||std::env::var_os("NX_DEFER_WINDOW_WAITS").is_some());*E}
fn merged(mut ranges:Vec<(u64,u64)>)->Vec<(u64,u64)> {
    ranges.sort_unstable();let mut out:Vec<(u64,u64)>=Vec::new();
    for (at,len) in ranges {if let Some(last)=out.last_mut() {let end=last.0 as u128+last.1 as u128;if at as u128<=end {last.1=((at as u128+len as u128).max(end)-last.0 as u128) as u64;continue;}}out.push((at,len));}out
}
// Join only identical GPU-to-CPU translations with touching mapped ranges.
// Preserve holes and distinct physical aliases. Apply the writer cap after
// this exact union, so many small adjacent pools do not force unknown writes.
fn mapped_union(mut ranges:Vec<(u64,u64,u64)>)->Vec<(u64,u64,u64)> {
    ranges.sort_unstable_by_key(|&(gpu,cpu,_)|(cpu as i128-gpu as i128,gpu));
    let mut out:Vec<(u64,u64,u64)>=Vec::new();
    for (gpu,cpu,len) in ranges {
        if let Some(last)=out.last_mut() {
            if cpu as i128-gpu as i128==last.1 as i128-last.0 as i128 && gpu as u128<=last.0 as u128+last.2 as u128 {
                last.2=((last.0 as u128+last.2 as u128).max(gpu as u128+len as u128)-last.0 as u128) as u64;continue;
            }
        }
        out.push((gpu,cpu,len));
    }
    out.sort_unstable();out
}
impl Replay<'_> {
    fn mapped_write_ranges(&self,base:u64,len:u64)->Option<Vec<(u64,u64,u64)>> {
        let end=base.checked_add(len)?;let mut at=base;let mut out=Vec::new();
        while at<end {
            let (chunk,off)=self.buffer_range(at,1)?;let n=(chunk.size-(at-chunk.base)).min(end-at);
            if n==0 || out.len()>=256 || chunk.buffer.0.storageMode()!=MTLStorageMode::Shared {return None;}
            out.push((at,chunk.buffer.0.contents().as_ptr() as u64+off as u64,n));at=at.checked_add(n)?;
        }
        Some(out)
    }
    // Buffer image helpers in the unchanged MSL check x<width. Include the
    // complete aligned words touched by subword atomic replacement too.
    fn image_write_ranges(&mut self,st:&GState,stage:usize,c:&super::shaders::Compiled)->Option<Vec<(u64,u64,u64)>> {
        if c.iface.image_writes==0 {return Some(Vec::new());}
        if !enabled() || stage>=3 {return None;}
        let descriptors={let state=super::STATE.lock().unwrap();c.iface.images.keys().filter(|&&b|c.iface.image_writes&(1u32<<b)!=0).map(|&b|{
            let handle=*st.images[stage].get(b as usize)?;state.registered_textures.get(&(st.texture_pool,(handle&0xfffff) as u32)).copied()
        }).collect::<Option<Vec<_>>>()?};
        let mut out=Vec::new();
        for builder in descriptors {
            if u32::from_le_bytes(builder[0x1c..0x20].try_into().unwrap())!=10 {continue;}
            let (format,meta)=super::textures::buffer_descriptor(&builder).ok()?;
            let base=meta[0] as u64|((meta[1] as u64)<<32);let bytes=(format[0] as u64).checked_mul(format[1] as u64)?.checked_div(8)?.checked_mul(meta[2] as u64)?;
            let start=base&!3;let end=base.checked_add(bytes)?.checked_add(3)?&!3;
            out.extend(self.mapped_write_ranges(start,end.checked_sub(start)?)?);
        }
        Some(out)
    }
    pub(super) fn shared_write_ranges(&mut self,st:&GState,stage:usize,c:&super::shaders::Compiled,global:Option<Vec<(u64,u64,u64)>>)->Option<Vec<(u64,u64,u64)>> {
        let mut out=global?;out.extend(self.image_write_ranges(st,stage,c)?);self.preflight_write_ranges(out)
    }
    pub(super) fn window_write_ranges(&mut self,proofs:&[nxshader::WriteWindowFootprint],st:&GState,stage:usize)->Option<Vec<(u64,u64,u64)>> {
        if !enabled() || stage>=3 {return None;}
        let mut windows=Vec::new();for &p in proofs {if p.storage_slot>=16 || !matches!(p.bytes,4|8|16) {return None;}windows.extend(p.windows(st.storage[stage][p.storage_slot].0));}
        let windows=merged(windows);let mut out=Vec::new();
        for chunk in self.pools.iter().flat_map(|p|&p.chunks).chain(self.heap.into_iter().flat_map(|h|&h.chunks)) {
            for &(at,len) in &windows {
                let start=(at as u128).max(chunk.base as u128);let end=(at as u128+len as u128).min(chunk.base as u128+chunk.size as u128);
                if start>=end {continue;}
                if chunk.buffer.0.storageMode()!=MTLStorageMode::Shared {return None;}
                let offset=(chunk.offset as u64).checked_add(start as u64-chunk.base)?;
                let cpu=(chunk.buffer.0.contents().as_ptr() as u64).checked_add(offset)?;
                out.push((start as u64,cpu,(end-start) as u64));if out.len()>1024 {if super::submit_profile::active(){eprintln!("[cpu-write-window] fallback raw_mapping_cap count={}",out.len());}return None;}
            }
        }
        let raw=out.len();let out=mapped_union(out);if super::submit_profile::active(){eprintln!("[cpu-write-window] mapped raw={raw} union={} cap=256",out.len());}
        self.preflight_write_ranges(out)
    }
    pub(super) fn preflight_write_ranges(&mut self,ranges:Vec<(u64,u64,u64)>)->Option<Vec<(u64,u64,u64)>> {
        let out=mapped_union(ranges);
        if out.len()>256 {return None;}
        // Reject the whole proof before recording any partial range when a
        // uniform snapshot needs the original conservative ordering path.
        for &(gpu,cpu,len) in &out {
            let aliases=self.aliases_for_write(gpu,cpu,len);
            if self.e.overlays.iter().any(|(&a,&(n,_,_))|aliases.iter().any(|&(p,m)|a<p.saturating_add(m) && p<a.saturating_add(n))) {
                if super::submit_profile::active(){eprintln!("[cpu-write-window] fallback preflight_overlay");}return None;
            }
        }
        Some(out)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn mapped_union_preserves_holes_distinct_aliases_and_unsigned_endpoints() {
        assert_eq!(mapped_union(vec![(100,1000,10),(110,1010,10),(105,1005,8),(125,1025,5),(100,2000,10),(u64::MAX-3,u64::MAX-3,4)]),vec![(100,1000,20),(100,2000,10),(125,1025,5),(u64::MAX-3,u64::MAX-3,4)]);
        let fragmented=(0..512).map(|i|(0x3500000000+i*32,0x7c00000000+i*32,32)).collect();
        assert_eq!(mapped_union(fragmented),vec![(0x3500000000,0x7c00000000,16384)]);
    }
    #[test]
    fn window_union_keeps_unsigned_wrap_segments_and_joins_overlapping_domains() {
        assert_eq!(merged(vec![(10,10),(15,20),(35,5),(0,3),(u64::MAX-2,3)]),vec![(0,3),(10,30),(u64::MAX-2,3)]);
    }
}
