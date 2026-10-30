use super::*;

fn lookup(image:&Image,address:u64)->u64 {
    if image.root_base&FLAT_TAG!=0 && (FLAT_START..FLAT_END).contains(&address) {
        let at=image.leaf_offset+(((address-FLAT_START)>>FLAT_SHIFT) as usize)*8;
        let page=u64::from_le_bytes(image.bytes[at..at+8].try_into().unwrap());if page!=0{return page+(address&65535);}
    }
    let root=(address>>ROOT_SHIFT).wrapping_sub(image.root_base&!FLAT_TAG);if root>=image.root_count{return 0;}
    let at=image.root_offset+root as usize*4;let leaf=u32::from_le_bytes(image.bytes[at..at+4].try_into().unwrap());if leaf==0{return 0;}
    let at=image.leaf_offset+(((leaf-1) as usize)*LEAF_WORDS+((address>>PAGE_SHIFT)&4095) as usize)*8;
    let page=u64::from_le_bytes(image.bytes[at..at+8].try_into().unwrap());if page==0{0}else{page+(address&4095)}
}

#[test]
fn flat_index_matches_every_original_page_and_preserves_partials_holes_and_aliases(){
    let base=FLAT_START+0x10000000;
    let ranges=[[base+4,0x30000-8,0x90000004],[base+0x40000,0x20000,0x90000004],[FLAT_END-65536,65536,0xa0000000]];
    let old=image(&ranges);let flat=image_flat(&ranges);assert_ne!(flat.root_base&FLAT_TAG,0);assert_eq!(flat.root_count,old.root_count);
    assert_eq!(flat.bytes.len()-old.bytes.len(),FLAT_WORDS*8);
    for address in (base-4096..base+0x70000).chain(FLAT_END-65540..FLAT_END+4096){assert_eq!(lookup(&flat,address),lookup(&old,address),"{address:#x}");}
    for bad in [vec![[base,65536,0x10000],[base+4096,65536,0x20000]],vec![[u64::MAX-4095,4096,0x10000]]] {let flat=image_flat(&bad);assert_eq!(flat.root_count,0);}
    let outside=[[0x1000,65536,0x9000]];let old=image(&outside);let flat=image_flat(&outside);assert_eq!(old.bytes,flat.bytes);assert_eq!(old.root_count,flat.root_count);
}

#[test]
#[ignore="requires a real Metal GPU; pool fixtures must run serially"]
fn gpu_flat_index_preserves_reads_and_pointer_chains_and_reports_cost(){
    use objc2::runtime::ProtocolObject;
    use objc2_foundation::NSString;
    use super::super::{shaders,state};
    objc2::rc::autoreleasepool(|_|unsafe{
        const BYTES:usize=4*65536;const THREADS:usize=16384;const STEPS:u32=64;
        let allocation=libc::mmap(std::ptr::null_mut(),BYTES,libc::PROT_READ|libc::PROT_WRITE,libc::MAP_PRIVATE|libc::MAP_ANON,-1,0);assert_ne!(allocation,libc::MAP_FAILED);
        let base=FLAT_START+0x10000000;
        let words=std::slice::from_raw_parts_mut(allocation.cast::<u64>(),BYTES/8);
        for page in 0..32 {for node in 0..512 {words[page*512+node]=base+(page*4096+((node+37)%512)*8) as u64;}}
        let mut pools=gfx::pool_memory(&[(0xf174,allocation as u64,BYTES as u64)]).unwrap();
        gfx::add_pool_aliases(&mut pools,&[(base,allocation as u64,2*65536),(base+3*65536+4,allocation as u64+4,8192)]).unwrap();
        let signature=spans(&pools);let images=[image(&signature),image_flat(&signature)];assert_ne!(images[1].root_base&FLAT_TAG,0);
        let dev=&gfx::gpu().device.0;
        let tables=images.each_ref().map(|i| {let b=dev.newBufferWithLength_options(i.bytes.len().max(8),MTLResourceOptions::StorageModeShared).unwrap();std::ptr::copy_nonoverlapping(i.bytes.as_ptr(),b.contents().as_ptr().cast::<u8>(),i.bytes.len());b});
        let resources=std::array::from_fn::<_,2,_>(|mode| {let i=&images[mode];let b=&tables[mode];let mut r:state::Res=std::mem::zeroed();let gpu=b.gpuAddress();r.zero_gpu=gfx::gpu().zeros.0.gpuAddress();r.apply_pool_bindings([gpu,signature.len() as u64,i.root_base,i.root_count,gpu+i.root_offset as u64,gpu+i.leaf_offset as u64]);r});
        let addresses=[base-4,base,base+4092,base+65532,base+65536,base+131068,base+131072,base+196612,base+200704,base+204800,base+204804];
        let list=addresses.iter().map(|v|format!("{v}ul")).collect::<Vec<_>>().join(",");
        let prelude=nxshader::prelude().replace("#define NX_PAGED_GMEM 0","#define NX_PAGED_GMEM 1");
        let body=format!(r#"
kernel void edges(constant Res& r [[buffer(0)]],device ulong* out [[buffer(1)]]){{ulong a[11]={{{list}}};for(uint i=0;i<11;i++){{out[i*3]=gmem_page(r,a[i]);out[i*3+1]=gmem_gpu(r,a[i]);out[i*3+2]=*gmem(r,a[i]);}}}}
kernel void chase(constant Res& r [[buffer(0)]],device uint* out [[buffer(1)]],uint tid [[thread_position_in_grid]]){{ulong address={base}ul+ulong(tid%32u)*4096ul+ulong(tid%512u)*8ul;GmemCache cache={{0ul,0ul,0ul}};uint sum=tid;for(uint step=0;step<{STEPS}u;step++){{sum=rotate(sum,5u)^uint(address)^step;uint lo=*gmem_cached(cache,r,address);uint hi=*gmem_cached(cache,r,address+4ul);address=ulong(lo)|(ulong(hi)<<32ul);}}out[tid]=sum;}}
"#);
        let libraries=[prelude.clone(),shaders::flat_source(&prelude).into_owned()].map(|p|dev.newLibraryWithSource_options_error(&NSString::from_str(&(p+&body)),None).unwrap());
        let pipelines=libraries.each_ref().map(|l|dev.newComputePipelineStateWithFunction_error(&l.newFunctionWithName(&NSString::from_str("chase")).unwrap()).unwrap());
        let edge_pipelines=libraries.each_ref().map(|l|dev.newComputePipelineStateWithFunction_error(&l.newFunctionWithName(&NSString::from_str("edges")).unwrap()).unwrap());
        let output=dev.newBufferWithLength_options(THREADS*4,MTLResourceOptions::StorageModeShared).unwrap();
        for mode in 0..2 {
            let cb=gfx::gpu().queue.0.commandBuffer().unwrap();let e=cb.computeCommandEncoder().unwrap();e.setComputePipelineState(&edge_pipelines[mode]);e.useResource_usage(ProtocolObject::from_ref(&*tables[mode]),MTLResourceUsage::Read);
            for c in pools.iter().flat_map(|p|&p.chunks){e.useResource_usage(ProtocolObject::from_ref(&*c.buffer.0),MTLResourceUsage::Read);}
            e.useResource_usage(ProtocolObject::from_ref(&*gfx::gpu().zeros.0),MTLResourceUsage::Read);e.setBytes_length_atIndex(std::ptr::NonNull::from(&resources[mode]).cast(),size_of::<state::Res>(),0);e.setBuffer_offset_atIndex(Some(&output),0,1);e.dispatchThreadgroups_threadsPerThreadgroup(MTLSize{width:1,height:1,depth:1},MTLSize{width:1,height:1,depth:1});e.endEncoding();cb.commit();cb.waitUntilCompleted();assert_eq!(cb.status(),MTLCommandBufferStatus::Completed);
            let got=std::slice::from_raw_parts(output.contents().as_ptr().cast::<u64>(),addresses.len()*3);
            for (i,&a) in addresses.iter().enumerate(){let span=signature.iter().find(|s|a>=s[0] && a-s[0]<s[1]);let pointer=span.map_or(0,|s|s[2]+a-s[0]);let value=if a>=base&&a<base+131072 {std::ptr::read_unaligned((allocation as u64+a-base) as *const u32)}else if a>=base+196612&&a<base+204804 {std::ptr::read_unaligned((allocation as u64+4+a-(base+196612)) as *const u32)}else{0};assert_eq!(&got[i*3..i*3+3],&[lookup(&images[mode],a),pointer,value as u64],"mode{mode} address{a:#x}");}
        }
        let expected=(0..THREADS).map(|tid| {let mut a=base+((tid%32)*4096+(tid%512)*8) as u64;let mut sum=tid as u32;for step in 0..STEPS{sum=sum.rotate_left(5)^a as u32^step;a=words[((a-base)/8) as usize];}sum}).collect::<Vec<_>>();
        let mut times=[Vec::new(),Vec::new()];
        for round in 0..10 {let mode=round%2;let cb=gfx::gpu().queue.0.commandBuffer().unwrap();let e=cb.computeCommandEncoder().unwrap();e.setComputePipelineState(&pipelines[mode]);e.useResource_usage(ProtocolObject::from_ref(&*tables[mode]),MTLResourceUsage::Read);for c in pools.iter().flat_map(|p|&p.chunks){e.useResource_usage(ProtocolObject::from_ref(&*c.buffer.0),MTLResourceUsage::Read);}e.useResource_usage(ProtocolObject::from_ref(&*gfx::gpu().zeros.0),MTLResourceUsage::Read);e.setBytes_length_atIndex(std::ptr::NonNull::from(&resources[mode]).cast(),size_of::<state::Res>(),0);e.setBuffer_offset_atIndex(Some(&output),0,1);e.dispatchThreadgroups_threadsPerThreadgroup(MTLSize{width:THREADS/128,height:1,depth:1},MTLSize{width:128,height:1,depth:1});e.endEncoding();cb.commit();cb.waitUntilCompleted();assert_eq!(cb.status(),MTLCommandBufferStatus::Completed);assert_eq!(std::slice::from_raw_parts(output.contents().as_ptr().cast::<u32>(),THREADS),&expected);if round>=2 {times[mode].push((cb.GPUEndTime()-cb.GPUStartTime())*1000.0);}}
        eprintln!("[flat-index-benchmark] legacy_gpu_ms={:?} flat_gpu_ms={:?} exact_threads={THREADS} steps={STEPS}; syntheticonly,actualgamegainunproven",times[0],times[1]);
        drop(resources);drop(tables);gfx::forget_pool(0xf174);drop(pools);assert_eq!(libc::munmap(allocation,BYTES),0);
    });
}
