// Known event/transfer writes can stay on the GPU until a host read needs
// them. Unknown shader writes retain the existing conservative barrier.
use super::{Engine, Replay, GState, STATE, TRACE_DRAWS};
use std::sync::{LazyLock, atomic::Ordering::Relaxed};
use objc2_metal::*;

#[derive(Default)]
pub(super) struct CpuEvents {
    writes: Vec<(u64,u64,u64)>, // GPU address, canonical shared-buffer CPU address, bytes
    aliases: Option<Vec<(u64,u64)>>,
    pool_aliases:Option<Vec<(u64,u64,u64)>>,
    texture_ranges:Option<(u64,super::super::fx::FastMap<super::TexBuilder,Vec<(u64,u64)>>)>,
}

pub(super) fn enabled()->bool {
    static ENABLED:LazyLock<bool>=LazyLock::new(||std::env::var_os("NX_DEFER_EVENT_WAITS").is_some());
    *ENABLED
}

pub(super) fn transfers_enabled()->bool {
    static ENABLED:LazyLock<bool>=LazyLock::new(||std::env::var_os("NX_DEFER_TRANSFER_WAITS").is_some());
    *ENABLED
}

fn overlaps(a:u64,n:u64,b:u64,m:u64)->bool {
    n!=0 && m!=0 && a<b.saturating_add(m) && b<a.saturating_add(n)
}

fn write_aliases(gpu:u64,cpu:u64,len:u64,pools:&[(u64,u64,u64)])->Vec<(u64,u64)> {
    let mut out=vec![(gpu,len),(cpu,len)];
    for &(base,guest,size) in pools {
        let start=cpu.max(base);let end=cpu.saturating_add(len).min(base.saturating_add(size));
        if start<end {out.push((guest+(start-base),end-start));}
    }
    out.sort_unstable();out.dedup();out
}

fn whole_pool_ranges(state:&super::super::objects::State,pool:u64)->Vec<(u64,u64)> {
    let Some(p)=state.pools.get(&pool) else {return Vec::new();};
    if let Some(maps)=state.virtual_maps.get(&pool) {
        maps.iter().flat_map(|&(offset,len,_,_)|state.pool_cpu_ranges(pool,offset,len).unwrap_or_else(||vec![(0,u64::MAX)])).collect()
    } else if p.cpu!=0 {vec![(p.cpu,p.size)]} else {vec![(0,u64::MAX)]}
}

fn texture_cpu_ranges(state:&super::super::objects::State,b:&super::TexBuilder)->Vec<(u64,u64)> {
    let u=|at|u32::from_le_bytes(b[at..at+4].try_into().unwrap());
    let pool=u64::from_le_bytes(b[8..16].try_into().unwrap());
    let offset=u64::from_le_bytes(b[16..24].try_into().unwrap());
    // Stale/unselected descriptors are also present in a bindless pool. Keep
    // the conservative whole-pool guard when their layout cannot be bounded.
    let bounded=(0..=10).contains(&u(0x1c)) && (1..=16384).contains(&u(0x20))
        && (1..=16384).contains(&u(0x24)) && (1..=16384).contains(&u(0x28))
        && u(0x2c)<=15 && u(0x30)<=156 && matches!(u(0x34),0|1|2|4|8|16);
    if !bounded {return whole_pool_ranges(state,pool);}
    let (len,_)=super::super::texture_builder_storage(b.as_ptr() as u64);
    let Some(mut ranges)=state.pool_cpu_ranges(pool,offset,len).filter(|_|len!=0) else {return whole_pool_ranges(state,pool);};
    // Texture uploads resolve every physical subrange. The existing hash path
    // can additionally read a contiguous CPU span from the first mapping;
    // include it even for discontiguous virtual storage.
    if let Some(first)=state.pool_offset_cpu(pool,offset) {ranges.push((first,len));}
    ranges
}

impl CpuEvents {
    pub(super) fn clear(&mut self) {self.writes.clear();self.aliases=None;}
    pub(super) fn begin_submit(&mut self) {self.aliases=None;self.pool_aliases=None;self.texture_ranges=None;}
    fn pending(&self)->bool {!self.writes.is_empty()}
}

impl Engine {
    pub(super) fn wait_cpu_events(&mut self) {
        if !self.cpu_events.pending() {return;}
        if let Some(cb)=&self.last_submit {super::gpu_wait::complete(&cb.0);}
        self.cpu_events.clear();
    }
}

impl Replay<'_> {
    fn ensure_pool_aliases(&mut self) {
        if self.e.cpu_events.pool_aliases.is_none() {
            self.e.cpu_events.pool_aliases=Some(self.pools.iter().flat_map(|p|&p.chunks).map(|chunk|
                (chunk.buffer.0.contents().as_ptr() as u64+chunk.offset as u64,chunk.base,chunk.size)).collect());
        }
    }

    pub(super) fn aliases_for_write(&mut self,gpu:u64,cpu:u64,len:u64)->Vec<(u64,u64)> {
        self.ensure_pool_aliases();
        write_aliases(gpu,cpu,len,self.e.cpu_events.pool_aliases.as_ref().unwrap())
    }

    pub(super) fn record_cpu_event(&mut self,gpu:u64,cpu:u64) {
        if !enabled() {self.e.gpu_buffer_dirty=true;return;}
        self.record_known_write(gpu,cpu,4);
    }

    pub(super) fn record_transfer_write(&mut self,gpu:u64,cpu:u64,len:u64) {
        if !transfers_enabled() {self.e.gpu_buffer_dirty=true;return;}
        self.record_known_write(gpu,cpu,len);
    }

    pub(super) fn record_known_write(&mut self,gpu:u64,cpu:u64,len:u64) {
        if len==0 {return;}
        // Existing uniform overlays are snapshots. Preserve their old ordering
        // path rather than deferring an event that can alias such a snapshot.
        let overlay=!self.e.overlays.is_empty() && {
            let aliases=self.aliases_for_write(gpu,cpu,len);
            self.e.overlays.iter().any(|(&a,&(n,_,_))|aliases.iter().any(|&(p,m)|overlaps(a,n,p,m)))
        };
        if overlay {if super::submit_profile::active(){eprintln!("[cpu-write-window] fallback overlay {gpu:#x}+{len:#x}");}self.e.gpu_buffer_dirty=true;return;}
        if !self.e.cpu_events.writes.contains(&(gpu,cpu,len)) {
            if self.e.cpu_events.writes.len()>=256 {self.e.gpu_buffer_dirty=true;return;}
            self.e.cpu_events.writes.push((gpu,cpu,len));self.e.cpu_events.aliases=None;
        }
        if TRACE_DRAWS.load(Relaxed) {eprintln!("[cpu-event] deferred {gpu:#x}+{len:#x} shared{cpu:#x}");}
    }

    fn ensure_event_addresses(&mut self) {
        if self.e.cpu_events.aliases.is_some() {return;}
        self.ensure_pool_aliases();
        let mut ranges=Vec::new();
        for &(gpu,cpu,len) in &self.e.cpu_events.writes {
            ranges.extend(write_aliases(gpu,cpu,len,self.e.cpu_events.pool_aliases.as_ref().unwrap()));
        }
        ranges.sort_unstable();ranges.dedup();self.e.cpu_events.aliases=Some(ranges);
    }

    pub(super) fn prepare_cpu_read(&mut self,addr:u64,len:u64) {
        if addr==0 || len==0 || !self.e.cpu_events.pending() {return;}
        self.ensure_event_addresses();
        if self.e.cpu_events.aliases.as_ref().unwrap().iter().any(|&(p,n)|overlaps(addr,len,p,n)) {
            if TRACE_DRAWS.load(Relaxed) || super::submit_profile::active() {eprintln!("[cpu-event] host read wait {addr:#x}+{len:#x}");}
            self.commit_boundary(true);
        }
    }

    pub(super) fn pending_cpu_write(&mut self,addr:u64,len:u64)->bool {
        if !self.e.cpu_events.pending() {return false;}
        self.ensure_event_addresses();
        self.e.cpu_events.aliases.as_ref().unwrap().iter().any(|&(p,n)|overlaps(addr,len,p,n))
    }

    pub(super) fn prepare_stage_cpu_reads(&mut self,st:&GState,_compute:bool) {
        if !self.e.cpu_events.pending() {return;}
        // Validate control/first-code cache lines before using their lengths.
        for &(code,control) in st.stages() {
            self.prepare_cpu_read(control,0x760);self.prepare_cpu_read(code,0x90);
        }
        // Binding a GPU buffer does not read its payload on the CPU. Direct
        // GPU bindings retain normal command ordering. Actual CPU snapshots
        // (uniform updates, unmapped gpu_range copies and diagnostics) guard
        // their own reads at the point of use.
        if !self.e.cpu_events.pending() {return;}
        for &(code,control) in st.stages() {
            // The translator caps code at1MiB and embedded C1 at64KiB.
            let len=unsafe {(control as *const u8).add(0x700).cast::<u32>().read_unaligned()} as u64;
            let end=unsafe {(control as *const u8).add(0x704).cast::<u32>().read_unaligned()} as u64;
            self.prepare_cpu_read(code,len.max(end).clamp(0x90,0x11_0000));
        }
    }

    pub(super) fn prepare_texture_access(&mut self,key:Option<super::textures::TexKey>,b:&super::TexBuilder,render_target:bool) {
        if super::submit_profile::active() && key.is_some_and(|k|k.format==80 && k.width==1024 && k.height==512) {self.commit_boundary(true);}
        // A cached private render texture has no CPU backing to hash/upload.
        // Diagnostics that explicitly inspect backing still keep a guard.
        if self.e.textures.cpu_reads_for_get(key,render_target) || TRACE_DRAWS.load(Relaxed) {
            self.prepare_texture_cpu_read(b);
        }
    }

    pub(super) fn texture_keyed(&mut self,key:Option<super::textures::TexKey>,b:&super::TexBuilder,render_target:bool)->Option<(super::textures::TexKey,crate::gfx::Shared<crate::gfx::Texture>)> {
        self.prepare_texture_access(key,b,render_target);
        self.e.textures.get_keyed(key,b,render_target)
    }

    pub(super) fn texture(&mut self,b:&super::TexBuilder,render_target:bool)->Option<(super::textures::TexKey,crate::gfx::Shared<crate::gfx::Texture>)> {
        self.texture_keyed(super::textures::TexKey::of(b),b,render_target)
    }

    pub(super) fn prepare_texture_cpu_read(&mut self,b:&super::TexBuilder) {
        if !self.e.cpu_events.pending() {return;}
        self.ensure_event_addresses();
        let epoch=super::super::objects::VIRTUAL_MAPPING_EPOCH.load(Relaxed);
        let cache=self.e.cpu_events.texture_ranges.get_or_insert_with(||(epoch,Default::default()));
        if cache.0!=epoch || cache.1.len()>=4096 {cache.0=epoch;cache.1.clear();}
        let ranges=cache.1.entry(*b).or_insert_with(||texture_cpu_ranges(&STATE.lock().unwrap(),b));
        let hit=self.e.cpu_events.aliases.as_ref().unwrap().iter().any(|&(p,n)|ranges.iter().any(|&(a,m)|overlaps(a,m,p,n)));
        if hit {
            if TRACE_DRAWS.load(Relaxed) || super::submit_profile::active() {
                let pool=u64::from_le_bytes(b[8..16].try_into().unwrap());let offset=u64::from_le_bytes(b[16..24].try_into().unwrap());
                eprintln!("[cpu-event] selected texture host read pool{pool:#x} offset{offset:#x}");
            }
            self.commit_boundary(true);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn transfer_ranges_include_every_partial_alias_and_exact_endpoints() {
        let aliases=write_aliases(0x9000,0x1080,0x100,&[(0x1000,0x2000,0xc0),(0x1100,0x3000,0x100),(0x1180,0x4000,0x100)]);
        assert_eq!(aliases,[(0x1080,0x100),(0x2080,0x40),(0x3000,0x80),(0x9000,0x100)]);
        assert!(aliases.iter().any(|&(a,n)|overlaps(a,n,0x307f,1)));
        assert!(!aliases.iter().any(|&(a,n)|overlaps(a,n,0x3080,1)));
    }
}
