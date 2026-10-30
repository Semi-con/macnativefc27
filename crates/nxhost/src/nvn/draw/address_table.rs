// Immutable two-level index for complete 4 KiB pages. Partial pages, holes,
// overlapping mappings and bounded-out tables retain the original span search.
use super::state::Res;
use crate::gfx::{self, Shared};
use objc2_metal::*;
use std::collections::BTreeMap;
use std::sync::{Arc, LazyLock};

const ROOT_SHIFT:u32=24;
const PAGE_SHIFT:u32=12;
const LEAF_WORDS:usize=1<<(ROOT_SHIFT-PAGE_SHIFT);
const MAX_ROOT_WORDS:u64=1<<18;
const MAX_LEAVES:usize=1024; // at most 32 MiB, otherwise keep span search

pub(super) fn enabled()->bool {
    static ON:LazyLock<bool>=LazyLock::new(||std::env::var_os("NX_PAGED_GMEM").is_some());
    *ON
}

fn spans(pools:&[Arc<gfx::PoolMemory>])->Vec<[u64;3]> {
    let mut out:Vec<[u64;3]>=Vec::new();
    for c in pools.iter().flat_map(|p|&p.chunks) {
        let gpu=c.gpu+c.offset as u64;
        if let Some(last)=out.last_mut().filter(|last|last[0].checked_add(last[1])==Some(c.base) && last[2].checked_add(last[1])==Some(gpu)) {
            last[1]+=c.size;
        } else {out.push([c.base,c.size,gpu]);}
    }
    out
}

struct Image {bytes:Vec<u8>,root_base:u64,root_count:u64,root_offset:usize,leaf_offset:usize}

fn image(spans:&[[u64;3]])->Image {
    let mut out=Image{bytes:unsafe{std::slice::from_raw_parts(spans.as_ptr().cast::<u8>(),spans.len()*24)}.to_vec(),root_base:0,root_count:0,root_offset:0,leaf_offset:0};
    let mut leaves:BTreeMap<u64,Vec<u64>>=BTreeMap::new();
    let mut previous_end=0;
    for &[base,len,gpu] in spans {
        let Some(end)=base.checked_add(len) else {return out;};
        if base<previous_end {return out;}
        previous_end=end;
        let Some(first)=base.checked_add(4095).map(|v|v&!4095) else {return out;};
        let end=end&!4095;
        if first>=end {continue;}
        let first_root=first>>ROOT_SHIFT;let last_root=(end-1)>>ROOT_SHIFT;
        if last_root-first_root>=MAX_LEAVES as u64 {return out;}
        for root in first_root..=last_root {
            if !leaves.contains_key(&root) && leaves.len()==MAX_LEAVES {return out;}
            let page_start=first.max(root<<ROOT_SHIFT);
            let page_end=end.min((root+1)<<ROOT_SHIFT);
            let leaf=leaves.entry(root).or_insert_with(||vec![0;LEAF_WORDS]);
            for page in (page_start..page_end).step_by(4096) {
                let Some(pointer)=gpu.checked_add(page-base) else {return out;};
                if pointer!=0 {leaf[((page>>PAGE_SHIFT)&(LEAF_WORDS as u64-1)) as usize]=pointer;}
            }
        }
    }
    let Some((&first,_))=leaves.first_key_value() else {return out;};
    let last=*leaves.last_key_value().unwrap().0;
    let count=last-first+1;
    if count>MAX_ROOT_WORDS {return out;}
    let mut roots=vec![0u32;count as usize];
    let mut pages=Vec::with_capacity(leaves.len()*LEAF_WORDS);
    for (i,(root,leaf)) in leaves.into_iter().enumerate() {
        roots[(root-first) as usize]=(i+1) as u32;pages.extend(leaf);
    }
    out.root_base=first;out.root_count=count;out.root_offset=out.bytes.len();
    out.bytes.extend_from_slice(unsafe{std::slice::from_raw_parts(roots.as_ptr().cast::<u8>(),roots.len()*4)});
    out.bytes.resize(out.bytes.len().next_multiple_of(8),0);out.leaf_offset=out.bytes.len();
    out.bytes.extend_from_slice(unsafe{std::slice::from_raw_parts(pages.as_ptr().cast::<u8>(),pages.len()*8)});
    out
}

// Optional single-load index for complete64KiB pages in the existing40-bit
// GPU domain. The original4KiB index and span fallback remain behind it.
const FLAT_TAG:u64=1<<63;
const FLAT_START:u64=1<<39;
const FLAT_END:u64=1<<40;
const FLAT_SHIFT:u32=16;
const FLAT_WORDS:usize=((FLAT_END-FLAT_START)>>FLAT_SHIFT) as usize; //64MiB cap
pub(super) fn flat_enabled()->bool {static ON:LazyLock<bool>=LazyLock::new(||std::env::var_os("NX_FLAT_GMEM").is_some());enabled()&&*ON}
fn image_flat(spans:&[[u64;3]])->Image {
    let mut out=image(spans);if out.root_count==0{return out;}
    let ranges=spans.iter().filter_map(|&[base,len,gpu]| {
        let end=base.checked_add(len)?.min(FLAT_END);let first=base.max(FLAT_START).checked_add(65535)?&!65535;let end=end&!65535;
        (first<end).then_some((base,gpu,first,end))
    }).collect::<Vec<_>>();if ranges.is_empty(){return out;}
    let old_pages=out.bytes.split_off(out.leaf_offset);out.bytes.resize(out.leaf_offset+FLAT_WORDS*8,0);
    for (base,gpu,first,end) in ranges {for page in (first..end).step_by(65536) {
        let Some(pointer)=gpu.checked_add(page-base).filter(|p|*p!=0 && p.checked_add(65535).is_some()) else {continue;};
        let at=out.leaf_offset+(((page-FLAT_START)>>FLAT_SHIFT) as usize)*8;out.bytes[at..at+8].copy_from_slice(&pointer.to_le_bytes());
    }}
    // Ordinary leaf indices address the unchanged pages after the flat prefix.
    for at in (out.root_offset..out.root_offset+out.root_count as usize*4).step_by(4) {
        let word=u32::from_le_bytes(out.bytes[at..at+4].try_into().unwrap());if word!=0 {out.bytes[at..at+4].copy_from_slice(&(word+(FLAT_WORDS/LEAF_WORDS) as u32).to_le_bytes());}
    }
    out.bytes.extend_from_slice(&old_pages);out.root_base|=FLAT_TAG;out
}

#[derive(Default)]
pub(super) struct Cache {
    signature:Vec<[u64;3]>,
    pub(super) buffer:Option<Shared<gfx::Buffer>>,
    bindings:[u64;6],
}

impl Cache {
    pub(super) fn bind(&mut self,pools:&[Arc<gfx::PoolMemory>])->(Shared<gfx::Buffer>,[u64;6]) {
        let signature=spans(pools);
        if self.buffer.is_none() || self.signature!=signature {
            let image=if flat_enabled(){image_flat(&signature)}else{image(&signature)};let dev=&gfx::gpu().device.0;
            let buffer=dev.newBufferWithLength_options(image.bytes.len().max(8),MTLResourceOptions::StorageModeShared).expect("shader address page table");
            unsafe {std::ptr::copy_nonoverlapping(image.bytes.as_ptr(),buffer.contents().as_ptr().cast::<u8>(),image.bytes.len());}
            let gpu=buffer.gpuAddress();
            self.bindings=[gpu,signature.len() as u64,image.root_base,image.root_count,gpu+image.root_offset as u64,gpu+image.leaf_offset as u64];
            if super::TRACE_DRAWS.load(std::sync::atomic::Ordering::Relaxed) {eprintln!("[frame] indexed address table: {} spans, {} root entries, {} bytes",signature.len(),image.root_count,image.bytes.len());}
            self.signature=signature;self.buffer=Some(Shared(buffer));
        }
        (self.buffer.as_ref().unwrap().clone(),self.bindings)
    }
}

impl Res {
    pub(super) fn pool_bindings(&self)->[u64;6] {[self.pool_spans,self.pool_span_count,self.pool_root_base,self.pool_root_count,self.pool_roots,self.pool_pages]}
    pub(super) fn apply_pool_bindings(&mut self,b:[u64;6]) {
        [self.pool_spans,self.pool_span_count,self.pool_root_base,self.pool_root_count,self.pool_roots,self.pool_pages]=b;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn lookup(image:&Image,address:u64)->u64 {
        let root=(address>>ROOT_SHIFT).wrapping_sub(image.root_base&!FLAT_TAG);
        if image.root_base&FLAT_TAG!=0 && (FLAT_START..FLAT_END).contains(&address) {let at=image.leaf_offset+(((address-FLAT_START)>>FLAT_SHIFT) as usize)*8;let page=u64::from_le_bytes(image.bytes[at..at+8].try_into().unwrap());if page!=0{return page+(address&65535);}}
        if root>=image.root_count {return 0;}
        let at=image.root_offset+root as usize*4;
        let leaf=u32::from_le_bytes(image.bytes[at..at+4].try_into().unwrap());
        if leaf==0 {return 0;}
        let at=image.leaf_offset+(((leaf-1) as usize)*LEAF_WORDS+((address>>PAGE_SHIFT)&4095) as usize)*8;
        let page=u64::from_le_bytes(image.bytes[at..at+8].try_into().unwrap());
        if page==0 {0} else {page+(address&4095)}
    }
    #[test]
    fn complete_pages_preserve_edges_holes_aliases_and_bounded_fallback() {
        let spans=[[0x1003,0x5000,0x90000123],[0x100000000,8192,0x90000123],[0x100003000,4096,0xa0000000]];
        let built=image(&spans);assert_ne!(built.root_count,0);
        for address in (0..0x8000).chain(0x100000000..0x100005000) {
            let page=address&!4095;
            let expected=spans.iter().find(|s|page>=s[0] && page+4096<=s[0]+s[1]).map_or(0,|s|s[2]+address-s[0]);
            assert_eq!(lookup(&built,address),expected,"address{address:#x}");
        }
        assert_eq!(image(&[[0,8192,0x10000],[4096,8192,0x20000]]).root_count,0);
        assert_eq!(image(&[[0,4096,0x10000],[u64::MAX-4095,4096,0x20000]]).root_count,0);
        assert_eq!(image(&[[0,(MAX_LEAVES as u64+1)<<ROOT_SHIFT,0x10000]]).root_count,0);
        assert_eq!(image(&[[0,4096,0x10000],[(MAX_ROOT_WORDS+1)<<ROOT_SHIFT,4096,0x20000]]).root_count,0);
    }

    #[test]
    #[ignore="requires a real Metal GPU"]
    fn gpu_page_index_preserves_aliases_partial_pages_remapping_and_cache_lifetime() {
        use objc2::{runtime::ProtocolObject,rc::Retained};
        use objc2_foundation::NSString;
        objc2::rc::autoreleasepool(|_|unsafe {
            let page=crate::mem::PAGE as usize;
            let allocation=libc::mmap(std::ptr::null_mut(),page*2,libc::PROT_READ|libc::PROT_WRITE,libc::MAP_PRIVATE|libc::MAP_ANON,-1,0);
            assert_ne!(allocation,libc::MAP_FAILED);let cpu=allocation as u64;
            for i in 0..page*2/4 {(allocation as *mut u32).add(i).write(i as u32+101);}
            let mut pools=gfx::pool_memory(&[(0xdada,cpu,(page*2) as u64)]).unwrap();
            let base=0x28_0000_0000;
            gfx::add_pool_aliases(&mut pools,&[(base,cpu,4096),(base+8192,cpu+4096,4096),(base+0x2000000+4,cpu+4,8188)]).unwrap();
            let mut cache=Cache::default();let (table,bindings)=cache.bind(&pools);assert_ne!(bindings[3],0);
            let (same,same_bindings)=cache.bind(&pools);assert_eq!(Retained::as_ptr(&same.0),Retained::as_ptr(&table.0));assert_eq!(bindings,same_bindings);
            let mut res:Res=std::mem::zeroed();res.zero_gpu=gfx::gpu().zeros.0.gpuAddress();res.apply_pool_bindings(bindings);
            let addresses=[base,base+4092,base+4096,base+8192,base+12284,base+12288,base+0x2000000+4,base+0x2000000+4096,base+0x2000000+8188,base+0x2000000+8192];
            let list=addresses.iter().map(|a|format!("{a}ul")).collect::<Vec<_>>().join(",");
            let prelude=super::super::shaders::cbuf_source(nxshader::prelude()).replace("#define NX_PAGED_GMEM 0","#define NX_PAGED_GMEM 1");
            let source=format!("{prelude}\nkernel void paged(constant Res& r [[buffer(0)]],device ulong* out [[buffer(1)]]) {{ulong addresses[10]={{{list}}};GmemCache cache={{0ul,0ul,0ul}};for(uint i=0;i<10;i++) {{ulong address=addresses[i];out[i*5]=gmem_page(r,address);out[i*5+1]=gmem_gpu(r,address);out[i*5+2]=*gmem(r,address);out[i*5+3]=gmem_gpu_cached(cache,r,address);out[i*5+4]=*gmem_cached(cache,r,address);}} *gmem(r,{base}ul+8192ul)=0x12345678u;}}");
            let dev=&gfx::gpu().device.0;let library=dev.newLibraryWithSource_options_error(&NSString::from_str(&source),None).unwrap();let pipeline=dev.newComputePipelineStateWithFunction_error(&library.newFunctionWithName(&NSString::from_str("paged")).unwrap()).unwrap();let output=dev.newBufferWithLength_options(50*8,MTLResourceOptions::StorageModeShared).unwrap();
            let cb=gfx::gpu().queue.0.commandBuffer().unwrap();let enc=cb.computeCommandEncoder().unwrap();enc.setComputePipelineState(&pipeline);enc.useResource_usage(ProtocolObject::from_ref(&*table.0),MTLResourceUsage::Read);
            for c in pools.iter().flat_map(|p|&p.chunks) {enc.useResource_usage(ProtocolObject::from_ref(&*c.buffer.0),MTLResourceUsage::Read|MTLResourceUsage::Write);}
            enc.useResource_usage(ProtocolObject::from_ref(&*gfx::gpu().zeros.0),MTLResourceUsage::Read);enc.setBytes_length_atIndex(std::ptr::NonNull::from(&res).cast(),size_of::<Res>(),0);enc.setBuffer_offset_atIndex(Some(&output),0,1);enc.dispatchThreadgroups_threadsPerThreadgroup(MTLSize{width:1,height:1,depth:1},MTLSize{width:1,height:1,depth:1});enc.endEncoding();
            // Rebuild/remap before the old command executes: the old buffer is
            // immutable and independently retained by its actual Metal encoder.
            let mut remapped=gfx::pool_memory(&[(0xdada,cpu,(page*2) as u64)]).unwrap();gfx::add_pool_aliases(&mut remapped,&[(base,cpu+4096,4096)]).unwrap();let (replacement,replacement_bindings)=cache.bind(&remapped);assert_ne!(bindings,replacement_bindings);assert_ne!(Retained::as_ptr(&table.0),Retained::as_ptr(&replacement.0));drop(cache);drop(same);drop(table);
            super::super::commit(&cb,&pools);cb.waitUntilCompleted();assert_eq!(cb.status(),MTLCommandBufferStatus::Completed);
            let words=std::slice::from_raw_parts(output.contents().as_ptr().cast::<u64>(),50);
            let offsets=[Some(0),Some(4092),None,Some(4096),Some(8188),None,Some(4),Some(4096),Some(8188),None];
            for (i,offset) in offsets.into_iter().enumerate() {
                let expected=offset.map_or(0,|offset|offset/4+101) as u64;assert_eq!(words[i*5+2],expected,"value{i}");assert_eq!(words[i*5+1]!=0,offset.is_some(),"mapping{i}");
                assert_eq!(words[i*5+3],words[i*5+1],"cached mapping{i}");assert_eq!(words[i*5+4],expected,"cached value{i}");let indexed=matches!(i,0|1|3|4|7|8);assert_eq!(words[i*5]!=0,indexed,"complete page{i}");if indexed {assert_eq!(words[i*5],words[i*5+1]);}
            }
            assert_eq!((allocation as *const u32).add(1024).read(),0x12345678);
            gfx::forget_pool(0xdada);
        });
    }
}

#[cfg(test)]
#[path="flat_page_tests.rs"]
mod flat_page_tests;
