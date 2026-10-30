use super::*;
use objc2_foundation::NSString;

// A dependent pointer chain prevents the compiler from proving that successive
// guest addresses share a page. Compare the two real Metal helpers, validate
// every result, and report GPU time without a timing-based pass/fail threshold.
#[test]
#[ignore="requires a real Metal GPU; shared pool fixtures must run serially"]
fn gpu_complete_page_translation_cache_pointer_chain_matches_legacy_and_reports_cost() {
    objc2::rc::autoreleasepool(|_|unsafe {
        const BYTES:usize=32*4096;
        const THREADS:usize=16384;
        const STEPS:u32=64;
        let allocation=libc::mmap(std::ptr::null_mut(),BYTES,libc::PROT_READ|libc::PROT_WRITE,libc::MAP_PRIVATE|libc::MAP_ANON,-1,0);
        assert_ne!(allocation,libc::MAP_FAILED);
        let base=0x2c_0000_0000u64;
        let words=std::slice::from_raw_parts_mut(allocation.cast::<u64>(),BYTES/8);
        for page in 0..32usize {for node in 0..512usize {
            words[page*512+node]=base+(page*4096+((node+37)%512)*8) as u64;
        }}
        let mut pools=gfx::pool_memory(&[(0xca67,allocation as u64,BYTES as u64)]).unwrap();
        gfx::add_pool_aliases(&mut pools,&[(base,allocation as u64,BYTES as u64)]).unwrap();
        let mut table_cache=address_table::Cache::default();
        let (table,bindings)=table_cache.bind(&pools);assert_ne!(bindings[3],0);
        let mut res:state::Res=std::mem::zeroed();res.zero_gpu=gfx::gpu().zeros.0.gpuAddress();res.apply_pool_bindings(bindings);
        let device=&gfx::gpu().device.0;
        let output=device.newBufferWithLength_options(THREADS*4,MTLResourceOptions::StorageModeShared).unwrap();
        let expected=(0..THREADS).map(|tid| {
            let mut address=base+((tid%32)*4096+(tid%512)*8) as u64;let mut sum=tid as u32;
            for step in 0..STEPS {sum=sum.rotate_left(5) ^ address as u32 ^ step;address=words[((address-base)/8) as usize];}sum
        }).collect::<Vec<_>>();
        let source=nxshader::prelude().replace("#define NX_PAGED_GMEM 0","#define NX_PAGED_GMEM 1");
        let body=format!(r#"
kernel void chase(constant Res& r [[buffer(0)]],device uint* out [[buffer(1)]],uint tid [[thread_position_in_grid]]) {{
    ulong address={base}ul+ulong(tid%32u)*4096ul+ulong(tid%512u)*8ul;
    GmemCache global_memory_cache = {{0ul, 0ul, 0ul}};
    uint sum=tid;
    for(uint step=0u;step<{STEPS}u;step++) {{
        sum=rotate(sum,5u)^uint(address)^step;
        uint lo=*gmem_cached(global_memory_cache,res,address);uint hi=*gmem_cached(global_memory_cache,res,address+4ul);
        address=ulong(lo)|(ulong(hi)<<32ul);
    }}out[tid]=sum;
}}"#);
        let body=body.replace("constant Res& r [[buffer(0)]]","constant Res& res [[buffer(0)]]");
        let pipelines=[false,true].map(|enabled| {
            let translated=source.clone()+&body;
            let compiled=shaders::graphics_memory_source(&translated,false,false,true,enabled).into_owned();
            let compiled=if enabled {shaders::page_cached_source(&compiled).into_owned()}else{compiled};
            assert_eq!(compiled.contains("    GmemCache global_memory_cache = {0ul, 0ul, 0ul};"),enabled);
            assert_eq!(compiled.contains("gmem_cached(global_memory_cache,res,"),enabled);
            let lib=device.newLibraryWithSource_options_error(&NSString::from_str(&compiled),None).unwrap();
            device.newComputePipelineStateWithFunction_error(&lib.newFunctionWithName(&NSString::from_str("chase")).unwrap()).unwrap()
        });
        let mut timings=[Vec::new(),Vec::new()];
        // Warm both compiled forms, then alternate to reduce thermal/order bias.
        for round in 0..10 {let mode=round%2;
            let cb=gfx::gpu().queue.0.commandBuffer().unwrap();let enc=cb.computeCommandEncoder().unwrap();enc.setComputePipelineState(&pipelines[mode]);
            enc.useResource_usage(ProtocolObject::from_ref(&*table.0),MTLResourceUsage::Read);
            for chunk in pools.iter().flat_map(|p|&p.chunks) {enc.useResource_usage(ProtocolObject::from_ref(&*chunk.buffer.0),MTLResourceUsage::Read);}
            enc.useResource_usage(ProtocolObject::from_ref(&*gfx::gpu().zeros.0),MTLResourceUsage::Read);
            enc.setBytes_length_atIndex(std::ptr::NonNull::from(&res).cast(),size_of::<state::Res>(),0);enc.setBuffer_offset_atIndex(Some(&output),0,1);
            enc.dispatchThreadgroups_threadsPerThreadgroup(MTLSize{width:THREADS/128,height:1,depth:1},MTLSize{width:128,height:1,depth:1});enc.endEncoding();cb.commit();cb.waitUntilCompleted();
            assert_eq!(cb.status(),MTLCommandBufferStatus::Completed);
            assert_eq!(std::slice::from_raw_parts(output.contents().as_ptr().cast::<u32>(),THREADS),&expected,"mode{mode} round{round}");
            let (start,end)=(cb.GPUStartTime(),cb.GPUEndTime());assert!(start>0.0&&end>start);
            if round>=2 {timings[mode].push((end-start)*1000.0);}
        }
        eprintln!("[page-cache-benchmark] production_uncached_gpu_ms={:?} production_cached_gpu_ms={:?} validated_threads={THREADS} steps={STEPS}; synthetic only, actual-game gain unproven",timings[0],timings[1]);
        drop(table_cache);drop(table);gfx::forget_pool(0xca67);drop(pools);
        assert_eq!(libc::munmap(allocation,BYTES),0);
    });
}
