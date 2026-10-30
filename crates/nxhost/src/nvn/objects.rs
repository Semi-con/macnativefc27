// Mush8 - objects.rs
// Constantin Clerc (c22dev) - 0.1
use super::fx::{FastMap,FastSet};
use crate::stubs::Args;
use std::collections::BTreeMap;
use std::sync::{LazyLock, Mutex};

fn rd<T: Copy>(addr: u64) -> T {
    unsafe { (addr as *const T).read_unaligned() }
}

fn invalidate_mapped_range() {
    // SDK55.15's pool and buffer entry points end in 4be5c0 ->
    // nn::os::FlushDataCache (1436a0), without waiting for any GPU queue.
    // All host pool buffers use coherent StorageModeShared. Preserve CPU
    // ordering here; completion belongs to the game's fences/events and to
    // explicit host readback/upload barriers.
    std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
}

#[derive(Debug, Clone, Copy)]
pub struct MemoryPool {
    pub cpu: u64,
    pub size: u64,
    pub flags: u32,
}

#[derive(Debug, Clone, Copy)]
pub struct Buffer {
    pub pool: u64,
    pub offset: u64,
    pub size: u64,
}

#[derive(Clone)]
pub struct Texture {
    pub builder: [u8; 0x68],
}

impl Texture {
    fn u32(&self, off: usize) -> u32 {
        u32::from_le_bytes(self.builder[off..off + 4].try_into().unwrap())
    }
    pub fn target(&self) -> u32 {
        self.u32(0x1c)
    }
    pub fn width(&self) -> u32 {
        self.u32(0x20)
    }
    pub fn height(&self) -> u32 {
        self.u32(0x24)
    }
    pub fn depth(&self) -> u32 {
        self.u32(0x28)
    }
    pub fn levels(&self) -> u32 {
        self.u32(0x2c)
    }
    pub fn format(&self) -> u32 {
        self.u32(0x30)
    }
}

#[derive(Default)]
pub struct State {
    pub queues: FastSet<u64>,
    pub pools: BTreeMap<u64, MemoryPool>,
    pub buffers: FastMap<u64, Buffer>,
    pub registered_textures: FastMap<(u64, u32), [u8; 0x68]>,
    pub registered_samplers: FastMap<(u64, u32), [u8; 0x60]>,
    pub programs: FastMap<u64, Vec<(u64, u64)>>,
    pub virtual_maps: FastMap<u64, Vec<(u64, u64, u64, u64)>>,
    pub virtual_bases: FastMap<u64,u64>,
    domain_arena:super::gpu_domains::Arena,
    canonical_gpu_pools:FastSet<u64>,
    gpu_base_index:BTreeMap<u64,u64>,
    pub windows: FastMap<u64, (Vec<u64>, u32)>,
    pub window_layers:FastMap<u64,u64>,
    pub debug_labels: FastMap<u64, String>,
}

const TEXTURE_MAGIC: u64 = 0x5845_5458_4E56_4E58;
const BUFFER_MAGIC: u64 = 0x5246_4642_4E56_4E58;

fn buffer(s: &State, obj: u64) -> Option<Buffer> {
    if let Some(b) = s.buffers.get(&obj) {
        return Some(*b);
    }
    if obj == 0 || rd::<u64>(obj) != BUFFER_MAGIC {
        return None;
    }
    Some(Buffer { pool: rd(obj + 8), offset: rd(obj + 0x10), size: rd(obj + 0x18) })
}

pub fn texture(obj: u64) -> Option<Texture> {
    if obj == 0 || rd::<u64>(obj) != TEXTURE_MAGIC {
        return None;
    }
    let mut builder = [0u8; 0x68];
    unsafe { std::ptr::copy_nonoverlapping((obj + 8) as *const u8, builder.as_mut_ptr(), 0x68) };
    Some(Texture { builder })
}

pub fn apply_view(b: &mut [u8; 0x68], view: u64, keep_target: bool) {
    if view == 0 {
        return;
    }
    // Read the public SDK 22 view API. Its packed 16-byte object is unrelated
    // to the old SDK's byte flags/32-bit swizzle layout.
    let proc = |name| *super::DRIVER.get().unwrap().procs.get(name).expect("verified SDK view getter");
    let get = |name| -> Option<u32> {
        let f: extern "C" fn(u64, *mut u32) -> u32 = unsafe { std::mem::transmute(proc(name) as usize) };
        let mut value = 0;
        (f(view, &mut value) != 0).then_some(value)
    };
    if let Some(to) = get("nvnTextureViewGetFormat") {
        let from = u32::from_le_bytes(b[0x30..0x34].try_into().unwrap());
        b[0x30..0x34].copy_from_slice(&to.to_le_bytes());
        let ((fw, fh, _), (tw, th, _)) = (super::format_info(from), super::format_info(to));
        if (fw, fh) != (1, 1) && (tw, th) == (1, 1) {
            for (o, bd) in [(0x20, fw), (0x24, fh)] {
                let v = u32::from_le_bytes(b[o..o+4].try_into().unwrap());
                b[o..o+4].copy_from_slice(&v.div_ceil(bd).to_le_bytes());
            }
        }
    }
    let swizzle: extern "C" fn(u64, *mut u32, *mut u32, *mut u32, *mut u32) -> u32 = unsafe {
        std::mem::transmute(proc("nvnTextureViewGetSwizzle") as usize)
    };
    let (mut r, mut g, mut bl, mut a) = (0,0,0,0);
    if swizzle(view,&mut r,&mut g,&mut bl,&mut a) != 0 {
        for (i,v) in [r,g,bl,a].into_iter().enumerate() { b[0x38+4*i..0x3c+4*i].copy_from_slice(&v.to_le_bytes()); }
    }
    if !keep_target {
        if let Some(to) = get("nvnTextureViewGetTarget") { b[0x1c..0x20].copy_from_slice(&to.to_le_bytes()); }
    }
    if let Some(mode) = get("nvnTextureViewGetDepthStencilMode") { b[0x48..0x4c].copy_from_slice(&mode.to_le_bytes()); }
    let range = |name| -> Option<(u32,u32)> {
        let f: extern "C" fn(u64, *mut u32, *mut u32) -> u32 = unsafe { std::mem::transmute(proc(name) as usize) };
        let (mut first, mut count) = (0,0);
        (f(view,&mut first,&mut count) != 0).then_some((first,count))
    };
    let (layer, level) = (range("nvnTextureViewGetLayers"), range("nvnTextureViewGetLevels"));
    if layer.is_some() || level.is_some() {
        set_view_range(b, layer, level);
    }
}

const SUBRESOURCE_TAG: u8 = 0xB7;

// Host-only metadata in the canonical builder's reserved tail. A zero count
// means the remaining subresources; public guest builders never see this tail.
pub(super) fn set_view_range(b: &mut [u8;0x68], layer:Option<(u32,u32)>, level:Option<(u32,u32)>) {
    let (lf,lc)=layer.unwrap_or((0,0)); let (mf,mc)=level.unwrap_or((0,0));
    assert!(lf<=u16::MAX as u32 && mf<=u8::MAX as u32,"NVN view origin exceeds host texture limits");
    b[0x60..0x62].copy_from_slice(&(if lc==u32::MAX {0} else {u16::try_from(lc).expect("NVN view layer count")} ).to_le_bytes());
    b[0x62]=if mc==u32::MAX {0} else {u8::try_from(mc).expect("NVN view mip count")};
    b[0x64..0x66].copy_from_slice(&(lf as u16).to_le_bytes());b[0x66]=mf as u8;b[0x67]=SUBRESOURCE_TAG;
}

pub fn view_range(b:&[u8;0x68])->[u32;4] {
    let (layer,level)=view_subresource(b);
    if b[0x67]==SUBRESOURCE_TAG {[layer,u16::from_le_bytes([b[0x60],b[0x61]]) as u32,level,b[0x62] as u32]} else {[0;4]}
}

pub fn view_subresource(b: &[u8; 0x68]) -> (u32, u32) {
    if b[0x67] == SUBRESOURCE_TAG { (u16::from_le_bytes([b[0x64], b[0x65]]) as u32, b[0x66] as u32) } else { (0, 0) }
}

const SAMPLER_MAGIC: u64 = 0x504D_4153_4E56_4E58;

pub fn sampler(obj: u64) -> Option<[u8; 0x60]> {
    if obj == 0 || rd::<u64>(obj) != SAMPLER_MAGIC {
        return None;
    }
    Some(super::canonical_sampler_builder(obj + 8))
}

unsafe extern "C" {
    fn nx_pool_storage();
}
std::arch::global_asm!(".globl _nx_pool_storage", ".p2align 2", "_nx_pool_storage:", "ldr x0, [x0, #0xf8]", "ret");

static POOL_VTABLE: LazyLock<[u64; 8]> = LazyLock::new(|| {
    let f = nx_pool_storage as *const () as usize as u64;
    [0, 0, 0, 0, 0, f, 0, f]
});

fn init_pool_struct(obj: u64, p: &MemoryPool) {
    unsafe {
        (obj as *mut u64).write_unaligned(POOL_VTABLE.as_ptr() as u64);
        ((obj + 0x10) as *mut u32).write_unaligned(p.flags & 0x3fff_ffff);
        ((obj + 0x30) as *mut u64).write_unaligned(p.size);
        ((obj + 0xf8) as *mut u64).write_unaligned(p.cpu);
    }
}

const TEXTURE_FIELDS: &[std::ops::Range<usize>] = &[8..0x4c, 0x58..0x5a, 0x5c..0x60];

fn same_object(a: u64, b: u64, magic: u64, fields: &[std::ops::Range<usize>]) -> bool {
    if a == 0 || b == 0 || rd::<u64>(a) != magic || rd::<u64>(b) != magic {
        return false;
    }
    fields.iter().all(|r| {
        let len = r.end - r.start;
        let (x, y) = unsafe { (std::slice::from_raw_parts((a + 8 + r.start as u64) as *const u8, len), std::slice::from_raw_parts((b + 8 + r.start as u64) as *const u8, len)) };
        x == y
    })
}

pub static VIRTUAL_MAPPING_EPOCH:std::sync::atomic::AtomicU64=std::sync::atomic::AtomicU64::new(1);
pub static TEXTURE_DESCRIPTOR_EPOCH:std::sync::atomic::AtomicU64=std::sync::atomic::AtomicU64::new(1);

pub static STATE: LazyLock<Mutex<State>> = LazyLock::new(|| Mutex::new(State::default()));

fn state() -> std::sync::MutexGuard<'static, State> {
    STATE.lock().unwrap()
}

impl State {
    fn release_domain(&mut self,obj:u64) {
        self.canonical_gpu_pools.remove(&obj);
        if let Some(base)=self.virtual_bases.remove(&obj) {
            self.gpu_base_index.remove(&base);
            if super::gpu_domains::enabled() && (super::gpu_domains::START..super::gpu_domains::END).contains(&base) {if let Some(p)=self.pools.get(&obj) {self.domain_arena.release(base,p.size);}}
        }
    }
    fn register_pool(&mut self,obj:u64,p:MemoryPool) {
        VIRTUAL_MAPPING_EPOCH.fetch_add(1,std::sync::atomic::Ordering::Relaxed);
        self.virtual_maps.remove(&obj);self.release_domain(obj);
        if super::gpu_domains::enabled() && p.size!=0 && p.cpu==0 && p.flags&0x200!=0 {
            if let Some(base)=self.domain_arena.reserve(p.size) {self.virtual_bases.insert(obj,base);}
        }
        if p.cpu==0 && p.flags&0x200!=0 && !self.virtual_bases.contains_key(&obj) {
            static NEXT:std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0x1_0000_0000);
            let span = p.size.next_multiple_of(0x10000);
            let base = NEXT.fetch_add(span,std::sync::atomic::Ordering::Relaxed);
            assert!(base.checked_add(span).is_some_and(|end|end<0x40_0000_0000),"virtual GPU address arena exhausted");
            self.virtual_bases.insert(obj,base);
        }
        if let Some(&base)=self.virtual_bases.get(&obj) {self.gpu_base_index.insert(base,obj);}
        self.pools.insert(obj,p);
    }

    pub fn pool_gpu_address(&mut self,pool:u64,offset:u64) -> Option<u64> {
        let p = *self.pools.get(&pool)?;
        if offset>p.size {return None;}
        if super::gpu_domains::enabled() && p.cpu!=0 && p.size!=0 && !self.virtual_bases.contains_key(&pool) && !self.canonical_gpu_pools.contains(&pool) {
            // GPU address queries, rather than CPU-only textures, consume
            // the bounded40-bit arena. Exhaustion retains the canonical alias.
            if let Some(base)=self.domain_arena.reserve(p.size) {self.virtual_bases.insert(pool,base);self.gpu_base_index.insert(base,pool);VIRTUAL_MAPPING_EPOCH.fetch_add(1,std::sync::atomic::Ordering::Relaxed);eprintln!("[gpu-domain] pool{pool:#x} gpu{base:#x} cpu{:#x} size{:#x}",p.cpu,p.size);} else {self.canonical_gpu_pools.insert(pool);}
        }
        self.existing_pool_gpu_address(pool,offset)
    }
    pub fn existing_pool_gpu_address(&self,pool:u64,offset:u64)->Option<u64> {
        let p=self.pools.get(&pool)?;if offset>p.size {return None;}
        let base=self.virtual_bases.get(&pool).copied().or((p.cpu!=0).then_some(p.cpu))?;base.checked_add(offset)
    }
    pub fn gpu_cpu_span(&self,address:u64,size:u64)->Option<(u64,u64,u64)> {
        let (&base,&pool)=self.gpu_base_index.range(..=address).next_back()?;let p=self.pools.get(&pool)?;
        let offset=address.checked_sub(base)?;if offset>=p.size || size>p.size-offset {return None;}
        if p.cpu!=0 {return Some((base,p.cpu,p.size));}
        let parts=self.pool_cpu_ranges(pool,offset,size)?;let first=parts.first()?.0;let mut next=first;
        for(cpu,len)in parts{if cpu!=next{return None;}next=next.checked_add(len)?;}Some((address,first,size))
    }
    pub fn gpu_cpu_address(&self,address:u64,size:u64)->Option<u64> {let(base,cpu,_)=self.gpu_cpu_span(address,size)?;cpu.checked_add(address-base)}

    fn map_virtual(&mut self,pool:u64,requests:&[(u64,u64,u64,u64)]) -> bool {
        let Some(v) = self.pools.get(&pool).filter(|p|p.cpu==0 && p.flags&0x200!=0) else {return false;};
        for &(phys,poff,voff,size) in requests {
            if size==0 || (poff|voff|size)&0xfff!=0 || voff.checked_add(size).is_none_or(|end|end>v.size) {return false;}
            if phys!=0 && self.pools.get(&phys).is_none_or(|p|p.cpu==0 || poff.checked_add(size).is_none_or(|end|end>p.size)) {return false;}
        }
        let maps = self.virtual_maps.entry(pool).or_default();
        for &(phys,poff,voff,size) in requests {
            let end = voff+size; let mut next = Vec::new();
            for &(start,len,p,off) in maps.iter() {
                if start>=end || start+len<=voff {next.push((start,len,p,off));continue;}
                if start<voff {next.push((start,voff-start,p,off));}
                if start+len>end {next.push((end,start+len-end,p,off+end-start));}
            }
            if phys!=0 {next.push((voff,size,phys,poff));}
            next.sort_by_key(|m|m.0); *maps=next;
        }
        VIRTUAL_MAPPING_EPOCH.fetch_add(1,std::sync::atomic::Ordering::Relaxed);
        true
    }

    pub fn gpu_aliases(&self) -> Vec<(u64,u64,u64)> {
        let mut out = Vec::new();
        for (&pool,&base) in &self.virtual_bases {if let Some(p)=self.pools.get(&pool).filter(|p|p.cpu!=0) {out.push((base,p.cpu,p.size));}}
        for (&pool,maps) in &self.virtual_maps {
            let base = self.virtual_bases[&pool];
            for &(voff,size,phys,poff) in maps {
                if let Some(p) = self.pools.get(&phys) {out.push((base+voff,p.cpu+poff,size));}
            }
        }
        out.sort_by_key(|m|m.0); out
    }

    // Resolve a complete range, not just its first byte. Virtual pools can
    // join unrelated physical allocations; a hole rejects the entire request.
    pub fn pool_cpu_ranges(&self,pool:u64,offset:u64,size:u64)->Option<Vec<(u64,u64)>> {
        let p=self.pools.get(&pool)?;let end=offset.checked_add(size)?;
        if end>p.size {return None;}
        if size==0 {return Some(Vec::new());}
        if p.cpu!=0 {return Some(vec![(p.cpu.checked_add(offset)?,size)]);}
        let maps=self.virtual_maps.get(&pool)?;let mut at=offset;let mut out=Vec::new();
        while at<end {
            let &(start,len,phys,poff)=super::virtual_lookup::find(maps,at)?;
            let pp=self.pools.get(&phys)?;let off=poff.checked_add(at-start)?;
            let take=(start+len-at).min(end-at);
            if pp.cpu==0 || off.checked_add(take)?>pp.size {return None;}
            out.push((pp.cpu.checked_add(off)?,take));at+=take;
        }
        Some(out)
    }

    pub fn pool_offset_cpu(&self, pool: u64, offset: u64) -> Option<u64> {
        let p = self.pools.get(&pool)?;
        if offset>=p.size {return None;}
        if p.cpu != 0 {
            return Some(p.cpu + offset);
        }
        let (voff, _, phys, poff) = *super::virtual_lookup::find(self.virtual_maps.get(&pool)?,offset)?;
        let pp = self.pools.get(&phys)?;
        (pp.cpu != 0).then(|| pp.cpu + poff + (offset - voff))
    }

    pub fn pool_at(&self, addr: u64) -> Option<(u64, MemoryPool)> {
        self.pools.iter().find(|(_, p)| p.cpu != 0 && addr >= p.cpu && addr - p.cpu < p.size).map(|(&k, &p)| (k, p))
    }
}

#[cfg(test)]
mod virtual_pool_tests {
    use super::*;
    #[test]
    fn remap_transactions_preserve_sorted_disjoint_virtual_lookup() {
        let mut state = State::default();
        state.register_pool(1, MemoryPool { cpu: 0, size: 0x10000, flags: 0x221 });
        for pool in [2, 3] { state.register_pool(pool, MemoryPool { cpu: 0x7d00000000 + pool * 0x10000, size: 0x10000, flags: 0x10a }); }
        for transaction in [vec![(2, 0, 0, 0x10000)], vec![(3, 0x4000, 0x2000, 0x6000)], vec![(0, 0, 0x3000, 0x2000)], vec![(2, 0x1000, 0x4000, 0x3000)]] {
            assert!(state.map_virtual(1, &transaction));
            let maps = &state.virtual_maps[&1];
            assert!(maps.windows(2).all(|p| p[0].0 + p[0].1 <= p[1].0));
            for offset in (0..0x11000).step_by(127) {
                let reference = maps.iter().find(|m| offset >= m.0 && offset - m.0 < m.1);
                assert_eq!(super::super::virtual_lookup::find(maps, offset), reference);
            }
        }
    }
    #[test]
    fn exhausted_physical_domains_keep_a_stable_canonical_address_after_another_pool_finalizes() {
        if !super::super::gpu_domains::enabled() {return;}
        let mut s=State::default();
        for n in 0..65 {s.register_pool(n+1,MemoryPool{cpu:0x7d10000000+n*65536,size:65536,flags:0x12});let _=s.pool_gpu_address(n+1,0).unwrap();}
        let address=s.pool_gpu_address(65,0).unwrap();assert_eq!(address,0x7d10400000);
        s.release_domain(1);s.pools.remove(&1);assert_eq!(s.pool_gpu_address(65,0),Some(address));
        s.register_pool(66,MemoryPool{cpu:0x7d10500000,size:65536,flags:0x12});assert_ne!(s.pool_gpu_address(66,0),Some(0x7d10500000));
    }
    #[test]
    fn reservations_remap_subranges_preserve_neighbors_and_reject_invalid_transactions() {
        let mut s=State::default();
        s.register_pool(1,MemoryPool{cpu:0,size:0x10000,flags:0x221});
        s.register_pool(2,MemoryPool{cpu:0x7d10000000,size:0x10000,flags:0x10a});
        s.register_pool(3,MemoryPool{cpu:0x7d20000000,size:0x10000,flags:0x10a});
        let va=s.pool_gpu_address(1,0).unwrap(); assert_ne!(va,0);
        assert!(s.pool_offset_cpu(1,0).is_none());
        assert!(s.map_virtual(1,&[(2,0,0,0x4000)]));
        assert!(s.map_virtual(1,&[(3,0x5000,0x1000,0x1000)]));
        assert_eq!(s.pool_offset_cpu(1,0x800),Some(0x7d10000800));
        assert_eq!(s.pool_offset_cpu(1,0x1800),Some(0x7d20005800));
        assert_eq!(s.pool_offset_cpu(1,0x2800),Some(0x7d10002800));
        assert_eq!(s.pool_gpu_address(1,0),Some(va),"mapping must not change reserved GPU addresses");
        assert!(s.map_virtual(1,&[(0,0,0x2000,0x1000)]));
        assert!(s.pool_offset_cpu(1,0x2800).is_none());
        let before=s.gpu_aliases();
        assert!(!s.map_virtual(1,&[(2,0,0x4000,0x1000),(2,0x10000,0x8000,0x1000)]));
        assert_eq!(s.gpu_aliases(),before,"invalid later request partially changed mappings");
        assert!(!s.map_virtual(1,&[(2,1,0,0x1000)]));
        assert!(!s.map_virtual(2,&[(3,0,0,0x1000)]));
        assert_eq!(s.gpu_aliases().iter().filter(|&&(base,_,_)|base>=va && base<va+0x10000).count(),3);
        if super::super::gpu_domains::enabled() {let physical=s.pool_gpu_address(2,0).unwrap();assert!(s.gpu_aliases().contains(&(physical,0x7d10000000,0x10000)));assert_eq!(s.gpu_cpu_address(physical+7,32),Some(0x7d10000007));assert_eq!(s.gpu_cpu_address(physical+0xfff0,32),None);assert_eq!(s.gpu_cpu_address(va+0x1ff0,32),None);}
        assert_eq!(s.pool_cpu_ranges(1,0x800,0x1800),Some(vec![(0x7d10000800,0x800),(0x7d20005000,0x1000)]));
        assert!(s.pool_cpu_ranges(1,0x800,0x2800).is_none(),"a virtual hole must reject the complete request");
        assert!(s.pool_cpu_ranges(2,0xf000,0x2000).is_none(),"physical-pool end must be enforced");
    }
    #[test]
    fn canonical_view_metadata_keeps_counts_and_origins_independent_of_builder_fields() {
        let mut b=[0u8;0x68];b[..0x60].fill(0x5a);let prefix=b[..0x60].to_vec();
        set_view_range(&mut b,Some((6,12)),Some((2,3)));
        assert_eq!(view_range(&b),[6,12,2,3]);assert_eq!(view_subresource(&b),(6,2));assert_eq!(&b[..0x60],prefix);
        set_view_range(&mut b,None,Some((1,u32::MAX)));assert_eq!(view_range(&b),[0,0,1,0]);
        assert_eq!(view_range(&[0;0x68]),[0;4]);
    }
}

pub type HostFn = fn(&Args) -> u64;

pub(super) fn timestamp_ns() -> u64 {
    (crate::svc::system_tick() as u128 * 1_000_000_000 / nxabi::GUEST_TICK_HZ as u128) as u64
}

pub fn host_impl(name: &str) -> Option<HostFn> {
    Some(match name {
        "nvnDeviceSetDebugLabel" | "nvnQueueSetDebugLabel" | "nvnProgramSetDebugLabel" |
        "nvnMemoryPoolSetDebugLabel" | "nvnBufferSetDebugLabel" | "nvnTextureSetDebugLabel" |
        "nvnSamplerSetDebugLabel" | "nvnTexturePoolSetDebugLabel" | "nvnSamplerPoolSetDebugLabel" |
        "nvnCommandBufferSetDebugLabel" | "nvnWindowSetDebugLabel" | "nvnSyncSetDebugLabel" |
        "nvnEventSetDebugLabel" => |a| {
            let mut bytes = Vec::new();
            if a.0[1] != 0 {
                for i in 0..4096 {
                    let v: u8 = rd(a.0[1] + i);
                    if v == 0 { break; }
                    bytes.push(v);
                }
            }
            let label = String::from_utf8_lossy(&bytes).into_owned();
            state().debug_labels.insert(a.0[0], label);
            0
        },
        "nvnMemoryPoolInitialize" => |a| {
            let b = a.0[1];
            let p = MemoryPool { cpu: rd(b + 0x10), size: rd(b + 0x18), flags: rd(b + 4) };
            init_pool_struct(a.0[0], &p);
            state().register_pool(a.0[0], p);
            1
        },
        "nvnMemoryPoolMap" => |a| state().pools.get(&a.0[0]).filter(|p|p.flags&1==0).map_or(0, |p| p.cpu),
        "nvnMemoryPoolGetBufferAddress" => |a| {
            let address=state().pool_gpu_address(a.0[0],0).unwrap_or(0);
            if address==0 && a.0[0]!=0 && std::env::var_os("NX_TRACE_NULL_GPU").is_some() {
                static COUNT:std::sync::atomic::AtomicU32=std::sync::atomic::AtomicU32::new(0);
                if COUNT.fetch_add(1,std::sync::atomic::Ordering::Relaxed)<8 {
                    eprintln!("[gpu-origin] unknown pool header{:x?} tail{:x?}",crate::diag::read_words::<8>(a.0[0]),crate::diag::read_words::<2>(a.0[0]+0xf0));
                    crate::diag::trace_stub_context("zero MemoryPoolGetBufferAddress",a);
                }
            }
            address
        },
        "nvnMemoryPoolGetSize" => |a| state().pools.get(&a.0[0]).map_or(0, |p| p.size),
        "nvnMemoryPoolGetFlags" => |a| state().pools.get(&a.0[0]).map_or(0, |p| (p.flags & 0x01ff_ffff) as u64),
        "nvnMemoryPoolFinalize" => |a| {
            VIRTUAL_MAPPING_EPOCH.fetch_add(1,std::sync::atomic::Ordering::Relaxed);
            let removed={
                let mut s=state();
                s.virtual_maps.remove(&a.0[0]);s.release_domain(a.0[0]);
                for maps in s.virtual_maps.values_mut() {maps.retain(|m|m.2!=a.0[0]);}
                let removed=s.pools.remove(&a.0[0]);
                // Match submit's STATE -> pool-cache lock order. Evicting the
                // cache before taking STATE can let a concurrent snapshot
                // recreate a cache entry after finalization has forgotten it.
                crate::gfx::forget_pool(a.0[0]);
                removed
            };
            if let Some(p) = removed {
                super::draw::forget_memory(p.cpu, p.size);
                super::draw::forget_memory(a.0[0] | 1 << 63, p.size);
            }
            0
        },
        "nvnMemoryPoolFlushMappedRange" => |a| {
            let p = state().pools.get(&a.0[0]).copied();
            if let Some(p) = p { super::draw::flush_memory(p.cpu + a.0[1], a.0[2]); }
            0
        },
        "nvnMemoryPoolInvalidateMappedRange" => |a| {
            if std::path::Path::new("work/probe-cpu-invalidate").exists() {
                static COUNT:std::sync::atomic::AtomicU32=std::sync::atomic::AtomicU32::new(0);
                if COUNT.fetch_add(1,std::sync::atomic::Ordering::Relaxed)<512 {
                    if let Some(p)=state().pools.get(&a.0[0]).copied() {eprintln!("[cpu-invalidate] pool={:#x} cpu={:#x} offset={:#x} size={:#x}",a.0[0],p.cpu,a.0[1],a.0[2]);}
                }
            }
            invalidate_mapped_range(); 0
        },
        "nvnMemoryPoolMapVirtual" => |a| {
            if a.0[1]>0x10000 {return 0;}
            let mut requests = Vec::new();
            for i in 0..a.0[1] {
                let r = a.0[2] + 0x28 * i;
                let (phys, poff, voff, size): (u64, u64, u64, u64) = (rd(r), rd(r + 8), rd(r + 0x10), rd(r + 0x18));
                requests.push((phys,poff,voff,size));
            }
            state().map_virtual(a.0[0],&requests) as u64
        },

        "nvnBufferInitialize" => |a| {
            let b = a.0[1];
            let buf = Buffer { pool: rd(b + 0x10), offset: rd(b + 0x18), size: rd(b + 0x20) };
            unsafe {
                let o = a.0[0] as *mut u64;
                o.write_unaligned(BUFFER_MAGIC);
                o.add(1).write_unaligned(buf.pool);
                o.add(2).write_unaligned(buf.offset);
                o.add(3).write_unaligned(buf.size);
            }
            state().buffers.insert(a.0[0], buf);
            1
        },
        "nvnBufferFinalize" => |a| {
            state().buffers.remove(&a.0[0]);
            if a.0[0] != 0 {
                unsafe { (a.0[0] as *mut u64).write_unaligned(0) };
            }
            0
        },
        "nvnBufferMap" => |a| {
            let s=state();buffer(&s,a.0[0]).and_then(|b|s.pools.get(&b.pool).filter(|p|p.cpu!=0 && p.flags&1==0).map(|p|p.cpu+b.offset)).unwrap_or(0)
        },
        "nvnBufferGetAddress" => |a| {
            let mut s = state();
            let addr = buffer(&s, a.0[0]).and_then(|b| s.pool_gpu_address(b.pool,b.offset));
            if addr.is_none() {
                static WARNED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
                if !WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                    let lr = unsafe { *(a as *const _ as *const u64).sub(1) };
                    eprintln!("[nvn] buffer {:#x} has no address (asked at {})", a.0[0], crate::diag::symbolize(lr));
                }
            }
            addr.unwrap_or(0)
        },
        "nvnBufferGetSize" => |a| buffer(&state(), a.0[0]).map_or(0, |b| b.size),
        "nvnBufferGetMemoryPool" => |a| buffer(&state(),a.0[0]).map_or(0,|b|b.pool),
        "nvnBufferGetMemoryOffset" => |a| buffer(&state(),a.0[0]).map_or(0,|b|b.offset),
        "nvnBufferFlushMappedRange" => |a| {
            let s = state();
            let address = buffer(&s, a.0[0]).and_then(|b| s.pool_offset_cpu(b.pool, b.offset + a.0[1]));
            drop(s);
            if let Some(address) = address { super::draw::flush_memory(address, a.0[2]); }
            0
        },
        "nvnBufferInvalidateMappedRange" => |_| { invalidate_mapped_range(); 0 },

        "nvnTextureInitialize" => |a| {
            let builder = super::canonical_texture_builder(a.0[1]);
            unsafe {
                (a.0[0] as *mut u64).write_unaligned(TEXTURE_MAGIC);
                std::ptr::copy_nonoverlapping(builder.as_ptr(), (a.0[0] + 8) as *mut u8, 0x68);
            }
            1
        },
        "nvnTextureFinalize" => |a| {
            if a.0[0] != 0 { unsafe { (a.0[0] as *mut u64).write_unaligned(0); } }
            state().debug_labels.remove(&a.0[0]);
            0
        },
        "nvnTextureGetWidth" => |a| texture(a.0[0]).map_or(0, |t| t.width() as u64),
        "nvnTextureGetHeight" => |a| texture(a.0[0]).map_or(0, |t| t.height() as u64),
        "nvnTextureGetDepth" => |a| texture(a.0[0]).map_or(0, |t| t.depth() as u64),
        "nvnTextureGetTarget" => |a| texture(a.0[0]).map_or(0, |t| t.target() as u64),
        "nvnTextureGetFormat" => |a| texture(a.0[0]).map_or(0, |t| t.format() as u64),
        "nvnTextureGetLevels" => |a| texture(a.0[0]).map_or(0, |t| t.levels() as u64),
        "nvnTextureGetFlags" => |a| texture(a.0[0]).map_or(0, |t| t.u32(0x18) as u64),
        "nvnTextureGetSamples" => |a| texture(a.0[0]).map_or(0, |t| t.u32(0x34) as u64),
        "nvnTextureGetStride" => |a| texture(a.0[0]).map_or(0, |t| t.u32(0x5c) as u64),
        "nvnTextureGetDepthStencilMode" => |a| texture(a.0[0]).map_or(0, |t| t.u32(0x48) as u64),
        "nvnTextureGetRawStorageClass" => |a| {
            let t=texture(a.0[0]).unwrap_or_else(||crate::diag::report_and_exit("raw storage class queried for an invalid NVN texture"));
            // SDK55.15 4c8e00: public raw class, independent of Tegra page
            // kinds or the host's backing allocation (unlike StorageClass).
            if t.u32(0x18)&0x50!=0 {1} else {match t.format() {0=>0,49|50=>3,51|53=>4,54=>5,_=>2}}
        },
        "nvnTextureGetStorageClass" => |a| {
            let t=texture(a.0[0]).unwrap_or_else(||crate::diag::report_and_exit("storage class queried for an invalid NVN texture"));
            super::texture_storage_class(&t.builder)
        },
        "nvnTextureGetMemoryPool" => |a| texture(a.0[0]).map_or(0, |t| u64::from_le_bytes(t.builder[8..16].try_into().unwrap())),
        "nvnTextureGetMemoryOffset" => |a| texture(a.0[0]).map_or(0, |t| u64::from_le_bytes(t.builder[16..24].try_into().unwrap())),
        "nvnTextureGetTextureAddress" => |a| {
            let Some(t) = texture(a.0[0]) else { return 0; };
            let pool = u64::from_le_bytes(t.builder[8..16].try_into().unwrap());
            let offset = u64::from_le_bytes(t.builder[16..24].try_into().unwrap());
            let address = state().pool_gpu_address(pool, offset);
            address.unwrap_or_else(|| crate::diag::report_and_exit("NVN texture address has no GPU address reservation"))
        },
        "nvnTextureGetSwizzle" => |a| {
            if let Some(t) = texture(a.0[0]) {
                for i in 0..4 { unsafe { (a.0[i + 1] as *mut u32).write_unaligned(t.u32(0x38 + i * 4)); } }
            }
            0
        },
        "nvnTextureGetViewOffset" => |a| {
            let Some(t) = texture(a.0[0]) else { return 0; };
            let mut view = t.builder;
            apply_view(&mut view, a.0[1], true);
            let (layer, level) = view_subresource(&view);
            if layer == 0 && level == 0 { return 0; }
            let (shape, layout) = super::texture_layout(t.builder.as_ptr() as u64).unwrap_or_else(|| crate::diag::report_and_exit("NVN view offset requested for unsupported linear subresource"));
            if layer >= shape.layers || level >= shape.levels { crate::diag::report_and_exit("NVN view subresource outside texture"); }
            layout.layer_size * layer as u64 + layout.level_offsets[level as usize]
        },
        "nvnTextureGetZCullStorageSize" => |_| 0,
        "nvnTextureGetDebugID" => |_| 0,
        "nvnTextureGetStorageSize" => |a| {
            let b = texture(a.0[0]).map(|t| t.builder);
            b.map_or(0, |b| super::texture_builder_storage(b.as_ptr() as u64).0)
        },
        "nvnTextureWriteTexels" => |a| {
            if let Some(t) = texture(a.0[0]) {
                super::draw::write_texels(t.builder, a.0[1], rd(a.0[2]), a.0[3] as *const u8, None);
            }
            0
        },
        "nvnTextureWriteTexelsStrided" => |a| {
            if let Some(t) = texture(a.0[0]) {
                super::draw::write_texels(t.builder, a.0[1], rd(a.0[2]), a.0[3] as *const u8, Some((a.0[4] as usize, a.0[5] as usize)));
            }
            0
        },
        "nvnTextureReadTexels" => |a| {
            if let Some(t) = texture(a.0[0]) {
                super::draw::read_texels(t.builder, a.0[1], rd(a.0[2]), a.0[3] as *mut u8, None);
            }
            0
        },
        "nvnTextureReadTexelsStrided" => |a| {
            if let Some(t) = texture(a.0[0]) {
                super::draw::read_texels(t.builder, a.0[1], rd(a.0[2]), a.0[3] as *mut u8, Some((a.0[4] as usize, a.0[5] as usize)));
            }
            0
        },
        "nvnTextureFlushTexels" | "nvnTextureInvalidateTexels" => |_| 0,
        "nvnTextureCompare" => |a| same_object(a.0[0], a.0[1], TEXTURE_MAGIC, TEXTURE_FIELDS) as u64,
        "nvnSamplerInitialize" => |a| {
            unsafe {
                // SDK 55.15 sampler is exactly 64 bytes: header, the original
                // 48-byte packed builder, then reserved bytes. Canonical host
                // sampling state lives separately and must never spill past it.
                std::ptr::write_bytes(a.0[0] as *mut u8, 0, 0x40);
                (a.0[0] as *mut u64).write_unaligned(SAMPLER_MAGIC);
                std::ptr::copy_nonoverlapping(a.0[1] as *const u8, (a.0[0] + 8) as *mut u8, 0x30);
            }
            1
        },
        "nvnSamplerFinalize" => |a| {
            if a.0[0] != 0 { unsafe { (a.0[0] as *mut u64).write_unaligned(0); } }
            state().debug_labels.remove(&a.0[0]);
            0
        },
        "nvnTexturePoolInitialize" | "nvnSamplerPoolInitialize" => |a| {
            // SDK 55.15 public descriptor-pool layout, including reserved tail.
            unsafe {
                let p = a.0[0] as *mut u64;
                p.write_unaligned(a.0[1]);
                p.add(1).write_unaligned(a.0[2]);
                p.add(2).write_unaligned(a.0[3] as u32 as u64);
                p.add(3).write_unaligned(0);
            }
            1
        },
        "nvnTexturePoolGetMemoryPool" | "nvnSamplerPoolGetMemoryPool" => |a| rd(a.0[0]),
        "nvnTexturePoolGetMemoryOffset" | "nvnSamplerPoolGetMemoryOffset" => |a| rd(a.0[0] + 8),
        "nvnTexturePoolGetSize" | "nvnSamplerPoolGetSize" => |a| rd::<u32>(a.0[0] + 0x10) as u64,
        "nvnTexturePoolFinalize" => |a| {
            let mut s=state();s.registered_textures.retain(|&(pool, _), _| pool != a.0[0]);
            TEXTURE_DESCRIPTOR_EPOCH.fetch_add(1,std::sync::atomic::Ordering::Relaxed);
            0
        },
        "nvnSamplerPoolFinalize" => |a| {
            state().registered_samplers.retain(|&(pool, _), _| pool != a.0[0]);
            0
        },
        "nvnTexturePoolRegisterTexture" => |a| {
            let mut s = state();
            if let Some(mut b) = texture(a.0[2]).map(|t| t.builder) {
                apply_view(&mut b, a.0[3], false);
                s.registered_textures.insert((a.0[0], a.0[1] as u32), b);
                TEXTURE_DESCRIPTOR_EPOCH.fetch_add(1,std::sync::atomic::Ordering::Relaxed);
            }
            0
        },
        "nvnSamplerPoolRegisterSampler" => |a| {
            let mut s = state();
            if let Some(b) = sampler(a.0[2]) {
                s.registered_samplers.insert((a.0[0], a.0[1] as u32), b);
            }
            0
        },
        "nvnSamplerPoolRegisterSamplerBuilder" => |a| {
            let b = super::canonical_sampler_builder(a.0[2]);
            state().registered_samplers.insert((a.0[0], a.0[1] as u32), b);
            0
        },
        "nvnDeviceGetTextureHandle" => |a| ((a.0[1] as u32 | (a.0[2] as u32) << 20) as i32 as i64 as u64) | 1 << 32,
        "nvnDeviceGetImageHandle" => |a| a.0[1] as i32 as i64 as u64,

        "nvnSyncInitialize" => |a| super::draw::initialize_sync(a.0[0],a.0[1]),
        "nvnSyncFinalize" => |a| {super::draw::finalize_sync(a.0[0]);0},
        "nvnQueueFenceSync" => super::cmd::fence_sync,
        "nvnSyncWait" => super::cmd::sync_wait,
        "nvnQueueWaitSync" => |a| {require_queue(a.0[0]);u64::from(super::draw::sync_wait(a.0[1],u64::MAX)!=3)},

        "nvnQueueInitialize" => |a| {
            if a.0[0]==0 || a.0[1]==0 {return 0;}
            state().queues.insert(a.0[0]);1
        },
        "nvnQueueFinalize" => |a| {require_queue(a.0[0]);super::draw::finish();state().queues.remove(&a.0[0]);0},
        // Submit commits Metal command buffers immediately, so Flush has no
        // outstanding host-side batch. Errors from those actual submissions
        // terminate with their Metal diagnostic instead of being hidden here.
        "nvnQueueFlush" => |a| {require_queue(a.0[0]);0},
        "nvnQueueGetError" => |a| {
            require_queue(a.0[0]);
            // SDK22 returns NO_ERROR=0 without touching errorInfo on its
            // no-error branch (4c3628); the structure's size is irrelevant.
            0
        },
        "nvnQueueFinish" => |a| {
            require_queue(a.0[0]);
            super::draw::finish();
            0
        },
        "nvnQueueSubmitCommands" => super::cmd::submit,
        "nvnQueuePresentTexture" => super::cmd::present,
        "nvnWindowInitialize" => |a| {
            let b = a.0[1];
            let native=rd::<u64>(b+0x18);
            let layer=crate::services::vi::native_layer(native).unwrap_or_else(||crate::diag::report_and_exit(&format!("NVN native window {native:#x} has no verified VI layer binding")));
            let (count, list): (u32, u64) = (rd(b + 0x10), rd(b + 0x28));
            let textures = (0..count as u64).map(|i| rd::<u64>(list + 8 * i)).collect();
            state().windows.insert(a.0[0], (textures, 0));
            state().window_layers.insert(a.0[0],layer);
            1
        },
        "nvnWindowFinalize" => |a| {let mut s=state();s.windows.remove(&a.0[0]);s.window_layers.remove(&a.0[0]);0},
        "nvnWindowSetCrop" => |a| {
            crate::gfx::set_crop(a.0[1] as i32, a.0[2] as i32, a.0[3] as i32, a.0[4] as i32);
            0
        },
        "nvnWindowSetPresentInterval" => |a| {
            crate::gfx::set_present_interval(a.0[1] as u32);
            0
        },
        "nvnQueueAcquireTexture" => |a| {require_queue(a.0[0]);acquire(a.0[1], a.0[2],0)},
        "nvnWindowAcquireTexture" => |a| acquire(a.0[0], a.0[2],a.0[1]),

        "nvnCommandBufferInitialize" => |a| {
            unsafe { std::ptr::write_bytes(a.0[0] as *mut u8, 0, 0xa0); }
            super::cmd::initialize(a.0[0]);
            1
        },
        "nvnCommandBufferFinalize" => |a| { super::cmd::finalize(a.0[0]); 0 },
        "nvnCommandBufferBeginRecording" => |a| {
            super::cmd::begin(a.0[0]);
            0
        },
        "nvnCommandBufferEndRecording" => |a| super::cmd::end(a.0[0]),
        "nvnCommandBufferAddControlMemory" => |a| {
            super::cmd::add_control_memory(a.0[0], a.0[1], a.0[2]);
            0
        },
        "nvnCommandBufferSetRenderTargets" => super::cmd::set_render_targets,
        "nvnCommandBufferClearColor" => super::cmd::clear_color,
        "nvnCommandBufferClearDepthStencil" => super::cmd::clear_depth_stencil,
        "nvnCommandBufferCallCommands" => super::cmd::call_commands,
        "nvnCommandBufferBindUniformBuffer" => super::cmd::bind_uniform_buffer,
        "nvnCommandBufferBindTexture" => super::cmd::bind_texture,
        name if super::cmd::host_impl(name).is_some() => return super::cmd::host_impl(name),

        "nvnProgramInitialize" => |a| {
            state().programs.insert(a.0[0], Vec::new());
            1
        },
        "nvnProgramFinalize" => |a| {
            state().programs.remove(&a.0[0]);
            0
        },
        "nvnProgramSetShaders" => |a| {
            let stages: Vec<(u64, u64)> = (0..a.0[1]).map(|i| (rd::<u64>(a.0[2] + 16 * i), rd::<u64>(a.0[2] + 16 * i + 8))).collect();
            if std::env::var_os("NX_DUMP_SHADERS").is_some() {
                dump_shaders(a.0[0], &stages);
            }
            state().programs.insert(a.0[0], stages);
            1
        },

        "nvnDeviceSetWindowOriginMode" => |a| {
            eprintln!("[nvn] window origin mode {}", a.0[1]);
            super::draw::set_origin_upper_left(a.0[1] as u32 == 1);
            0
        },
        "nvnDeviceSetDepthMode" => |a| {
            eprintln!("[nvn] depth mode {}", a.0[1]);
            super::draw::set_depth_zero_to_one(a.0[1] as u32 == 1);
            0
        },
        "nvnDeviceInstallDebugCallback" | "nvnDeviceApplyDeferredFinalizes" => |_| 0,
        "nvnDeviceGetCurrentTimestampInNanoseconds" => |_| timestamp_ns(),
        // Host counter reports use nanoseconds in the same monotonic domain.
        // SDK 55.15 reads the counterData timestamp at byte 8, not counter value.
        "nvnDeviceGetTimestampInNanoseconds" => |a| rd(a.0[1] + 8),
        _ => return None,
    })
}

pub fn require_queue(queue:u64) {
    if !state().queues.contains(&queue) {crate::diag::report_and_exit(&format!("uninitialized NVN queue {queue:#x}"));}
}

fn acquire(window: u64, index_out: u64,sync:u64) -> u64 {
    let mut s = state();
    let Some((textures, next)) = s.windows.get_mut(&window) else { return 1 };
    if textures.is_empty() {return 1;}
    let n = textures.len() as u32;
    let texture=textures[*next as usize];
    unsafe { *(index_out as *mut i32) = *next as i32 };
    *next = (*next + 1) % n;
    drop(s);
    if sync!=0 {super::draw::acquire_sync(sync,texture);}
    0
}

fn dump_shaders(program: u64, stages: &[(u64, u64)]) {
    static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    static BYTES:std::sync::atomic::AtomicU64=std::sync::atomic::AtomicU64::new(0);
    let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    if n >= 512 {
        return;
    }
    let _ = std::fs::create_dir_all("work/shaders");
    for (i, &(code, control)) in stages.iter().enumerate() {
        let code=state().gpu_cpu_address(code,1).unwrap_or(code);
        let ctl = unsafe { std::slice::from_raw_parts(control as *const u8, 0x1000) };
        let _ = std::fs::write(format!("work/shaders/p{n:03}_s{i}.control"), ctl);
        let s = state();
        let requested=u32::from_le_bytes(ctl[0x700..0x704].try_into().unwrap()) as u64;
        let len = s.pool_at(code).map_or(0, |(_, p)| (p.cpu + p.size - code).min(requested.clamp(0x100,0x100000)));
        drop(s);
        if BYTES.fetch_update(std::sync::atomic::Ordering::Relaxed,std::sync::atomic::Ordering::Relaxed,|used|used.checked_add(len+0x1000).filter(|n|*n<=64*1024*1024)).is_err() {return;}
        if len > 0 {
            let c = unsafe { std::slice::from_raw_parts(code as *const u8, len as usize) };
            let _ = std::fs::write(format!("work/shaders/p{n:03}_s{i}.code"), c);
        }
        eprintln!("[shader] program {program:#x} #{n} stage {i}: code {code:#x} control {control:#x}");
    }
}

#[cfg(test)]
mod buffer_tests {
    use super::*;

    fn call(name: &str, obj: u64, builder: u64) -> u64 {
        host_impl(name).unwrap()(&Args([obj, builder, 0, 0, 0, 0, 0, 0], [0; 8]))
    }

    struct Fixture {
        storage: Box<[u8; 64]>,
        pool: Box<[u64; 32]>,
        object: Box<[u64; 16]>,
    }

    impl Fixture {
        fn new() -> Self {
            Self::with_flags(0)
        }
        fn with_flags(flags:u32) -> Self {
            let mut f = Self { storage: Box::new([0; 64]), pool: Box::new([0; 32]), object: Box::new([0; 16]) };
            let pool_builder = [(flags as u64)<<32, 0, f.storage.as_ptr() as u64, 64, 0];
            assert_eq!(call("nvnMemoryPoolInitialize", f.pool.as_mut_ptr() as u64, pool_builder.as_ptr() as u64), 1);
            let builder = [0, 0, f.pool.as_ptr() as u64, 8, 24];
            assert_eq!(call("nvnBufferInitialize", f.object.as_mut_ptr() as u64, builder.as_ptr() as u64), 1);
            f
        }

        fn address(&self) -> u64 {self.storage.as_ptr() as u64 + 8}
        fn gpu_address(&self)->u64 {state().pool_gpu_address(self.pool.as_ptr() as u64,8).unwrap()}
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let mut s = STATE.lock().unwrap();
            s.buffers.remove(&(self.object.as_ptr() as u64));
            let pool=self.pool.as_ptr() as u64;s.release_domain(pool);s.pools.remove(&pool);
        }
    }

    #[test]
    fn copied_buffer_preserves_address_mapping_and_size() {
        let f = Fixture::new();
        let copy = *f.object;
        for name in ["nvnBufferGetAddress", "nvnBufferMap"] {
            let expected=if name=="nvnBufferGetAddress" {f.gpu_address()} else {f.address()};
            assert_eq!(call(name, f.object.as_ptr() as u64, 0), expected);
            assert_eq!(call(name, copy.as_ptr() as u64, 0), expected, "{name} lost the copied buffer's storage");
        }
        assert_eq!(call("nvnBufferGetSize", copy.as_ptr() as u64, 0), 24);
    }
    #[test]
    fn cpu_no_access_does_not_hide_the_gpu_address_or_expose_a_cpu_mapping() {
        for flags in [0x21u32,0xa1] {
            let f=Fixture::with_flags(flags);let object=f.object.as_ptr() as u64;let pool=f.pool.as_ptr() as u64;
            assert_eq!(call("nvnMemoryPoolMap",pool,0),0);assert_eq!(call("nvnBufferMap",object,0),0);
            assert_eq!(call("nvnMemoryPoolGetBufferAddress",pool,0),f.gpu_address()-8);
            assert_eq!(call("nvnBufferGetAddress",object,0),f.gpu_address());
        }
        for flags in [0x12u32,0x24] {
            let f=Fixture::with_flags(flags);assert_eq!(call("nvnMemoryPoolMap",f.pool.as_ptr() as u64,0),f.storage.as_ptr() as u64);assert_eq!(call("nvnBufferMap",f.object.as_ptr() as u64,0),f.address());
        }
    }

    #[test]
    fn finalized_buffer_does_not_regain_its_address() {
        let mut f = Fixture::new();
        let obj = f.object.as_mut_ptr() as u64;
        call("nvnBufferFinalize", obj, 0);
        for name in ["nvnBufferGetAddress", "nvnBufferMap", "nvnBufferGetSize"] {
            assert_eq!(call(name, obj, 0), 0, "{name} resurrected a finalized buffer");
        }
    }

    #[test]
    fn finalizing_a_copy_does_not_finalize_the_original() {
        let f = Fixture::new();
        let mut copy = *f.object;
        let copy_addr = copy.as_mut_ptr() as u64;
        assert_eq!(call("nvnBufferGetAddress", copy_addr, 0), f.gpu_address());
        call("nvnBufferFinalize", copy_addr, 0);
        assert_eq!(call("nvnBufferGetAddress", copy_addr, 0), 0);
        assert_eq!(call("nvnBufferGetSize", copy_addr, 0), 0);
        assert_eq!(call("nvnBufferGetAddress", f.object.as_ptr() as u64, 0), f.gpu_address());
    }

    #[test]
    fn null_and_uninitialized_buffer_objects_return_zero() {
        let unknown = [0u64; 16];
        for obj in [0, unknown.as_ptr() as u64] {
            for name in ["nvnBufferGetAddress", "nvnBufferMap", "nvnBufferGetSize"] {
                assert_eq!(call(name, obj, 0), 0);
            }
        }
    }

    #[test]
    fn sdk_pool_flags_and_descriptor_pool_layout() {
        let mut object = [0u64; 32];
        let builder = [0x21u64 << 32, 0, 0x12340000, 0x20000, 0xdeadbeef];
        let obj = object.as_mut_ptr() as u64;
        assert_eq!(call("nvnMemoryPoolInitialize", obj, builder.as_ptr() as u64), 1);
        assert_eq!(call("nvnMemoryPoolGetFlags", obj, 0), 0x21);
        assert_eq!(object[0x30 / 8], 0x20000);
        assert_eq!(object[0xf8 / 8], 0x12340000);
        let s = state();
        assert_eq!(s.pool_at(0x12340010).unwrap().0, obj);
        assert!(s.pool_at(0x1233ffff).is_none());
        assert!(s.pool_at(0x12360000).is_none());
        drop(s);
        for class in ["TexturePool", "SamplerPool"] {
            let mut p = [u64::MAX; 4];
            let ptr = p.as_mut_ptr() as u64;
            let args = Args([ptr, obj, 0x1000, 4096, 0, 0, 0, 0], [0; 8]);
            assert_eq!(host_impl(&format!("nvn{class}Initialize")).unwrap()(&args), 1);
            assert_eq!(p, [obj, 0x1000, 4096, 0]);
            assert_eq!(call(&format!("nvn{class}GetMemoryPool"), ptr, 0), obj);
            assert_eq!(call(&format!("nvn{class}GetMemoryOffset"), ptr, 0), 0x1000);
            assert_eq!(call(&format!("nvn{class}GetSize"), ptr, 0), 4096);
        }
        call("nvnMemoryPoolFinalize", obj, 0);
    }

    #[test]
    fn debug_label_is_copied_and_null_clears_it() {
        let object = 0x8badf00d;
        let label = b"FC27 test program\0";
        call("nvnProgramSetDebugLabel", object, label.as_ptr() as u64);
        assert_eq!(state().debug_labels[&object], "FC27 test program");
        call("nvnProgramSetDebugLabel", object, 0);
        assert_eq!(state().debug_labels.remove(&object).unwrap(), "");
        assert!(host_impl("nvnUnknownObjectSetDebugLabel").is_none());
    }

    #[test]
    fn timestamps_use_monotonic_time_and_counter_timestamp_field() {
        let first = call("nvnDeviceGetCurrentTimestampInNanoseconds", 0, 0);
        assert!(first > 0);
        let data = [123u64, first];
        assert_eq!(call("nvnDeviceGetTimestampInNanoseconds", 0, data.as_ptr() as u64), first);
        assert!(call("nvnDeviceGetCurrentTimestampInNanoseconds", 0, 0) >= first);
    }

    #[test]
    fn sampler_sdk22_object_keeps_neighbor_freelist_intact() {
        let mut adjacent = [0xa5u8; 0x90];
        let mut builder = [0u8; 0x30];
        builder[0x1c..0x2c].copy_from_slice(&[0x3f800000u32.to_le_bytes(); 4].concat());
        builder[0x2c..0x30].copy_from_slice(&0x08110111u32.to_le_bytes());
        let obj = unsafe { adjacent.as_mut_ptr().add(16) } as u64;
        assert_eq!(call("nvnSamplerInitialize", obj, builder.as_ptr() as u64), 1);
        assert!(adjacent[..16].iter().chain(&adjacent[16 + 0x40..]).all(|v| *v == 0xa5));
        assert_eq!(&adjacent[24..24 + 0x30], &builder);
        let copy = adjacent[16..16 + 0x40].to_vec();
        assert_eq!(sampler(copy.as_ptr() as u64), Some(super::super::canonical_sampler_builder(builder.as_ptr() as u64)));
        call("nvnSamplerFinalize", obj, 0);
        assert!(sampler(obj).is_none());
    }

    #[test]
    fn texture_queries_preserve_pool_offset_and_opaque_object_bounds() {
        let f = Fixture::new();
        let mut builder = [0u8; 0x68];
        builder[0x10..0x18].copy_from_slice(&(f.pool.as_ptr() as u64).to_le_bytes());
        builder[0x18..0x20].copy_from_slice(&8u64.to_le_bytes());
        for (off, value) in [(0x20, 0x10u32), (0x24, 1), (0x28, 16), (0x2c, 4), (0x30, 1), (0x34, 1), (0x38, 37), (0x3c, 1), (0x58, 64)] {
            builder[off..off + 4].copy_from_slice(&value.to_le_bytes());
        }
        let mut object = [0xa5u8; 0xa0];
        let ptr = unsafe { object.as_mut_ptr().add(16) } as u64;
        assert_eq!(call("nvnTextureInitialize", ptr, builder.as_ptr() as u64), 1);
        assert!(object[..16].iter().chain(&object[16 + 0x80..]).all(|v| *v == 0xa5));
        let copy = object[16..16 + 0x80].to_vec();
        for obj in [ptr, copy.as_ptr() as u64] {
            assert_eq!(call("nvnTextureGetMemoryPool", obj, 0), f.pool.as_ptr() as u64);
            assert_eq!(call("nvnTextureGetMemoryOffset", obj, 0), 8);
            assert_eq!(call("nvnTextureGetTextureAddress", obj, 0), f.gpu_address());
            assert_eq!(call("nvnTextureGetStride", obj, 0), 64);
            assert_eq!(call("nvnTextureGetWidth", obj, 0), 16);
            assert_eq!(call("nvnTextureGetSamples", obj, 0), 1);
        }
        call("nvnTextureFinalize", ptr, 0);
        assert!(texture(ptr).is_none());
        assert!(texture(copy.as_ptr() as u64).is_some());
    }
}
