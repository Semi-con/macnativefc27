use super::*;
use objc2_foundation::NSString;

#[test]
fn buffer_read_specialization_preserves_store_source_and_has_bounded_page_guard(){
    let old=nxshader::prelude();let new=shaders::buffer_read_source(old);
    assert_ne!(new,old);assert!(new.contains("(address&4095ul)+ulong(bytes)<=4096ul"));
    let store="inline void buffer_write_raw(";
    assert_eq!(&old[old.find(store).unwrap()..],&new[new.find(store).unwrap()..]);
    assert_eq!(shaders::buffer_read_source("no buffer function"),"no buffer function");
}

#[test]
#[ignore="requires a real Metal GPU; pool fixtures must run serially"]
fn gpu_buffer_read_batch_preserves_all_formats_holes_aliases_and_pixel_edges(){
    objc2::rc::autoreleasepool(|_|unsafe{
        const BYTES:usize=65536;const THREADS:usize=16384;
        let allocation=libc::mmap(std::ptr::null_mut(),BYTES,libc::PROT_READ|libc::PROT_WRITE,libc::MAP_PRIVATE|libc::MAP_ANON,-1,0);assert_ne!(allocation,libc::MAP_FAILED);let cpu=allocation as u64;
        for i in 0..BYTES {(allocation as *mut u8).add(i).write(((i*37+i/13)%251) as u8);}
        let base=0x28_0000_0000u64;
        let aliases=[(base,cpu,8192),(base+0x10000,cpu+12288,4096),(base+0x11000,cpu+24576,4096),(base+0x20004,cpu+4,28),(base+0x30000,cpu+32769,4096)];
        let mut pools=gfx::pool_memory(&[(0xf175,cpu,BYTES as u64)]).unwrap();gfx::add_pool_aliases(&mut pools,&aliases).unwrap();let mut cache=address_table::Cache::default();let (table,bindings)=cache.bind(&pools);
        let dev=&gfx::gpu().device.0;let fault=dev.newBufferWithLength_options(32,MTLResourceOptions::StorageModeShared).unwrap();let output=dev.newBufferWithLength_options(THREADS*32,MTLResourceOptions::StorageModeShared).unwrap();
        let mut res:state::Res=std::mem::zeroed();res.zero_gpu=gfx::gpu().zeros.0.gpuAddress();res.fault_gpu=fault.gpuAddress();res.apply_pool_bindings(bindings);
        // Heap fast mappings take precedence over even a complete pool page.
        let physical=&pools.iter().find(|p|p.base==cpu).unwrap().chunks[0];
        res.heap_base=base+4096;res.heap_size=4096;res.heap_chunk_size=4096;res.heap_gpu[0]=physical.gpu+physical.offset as u64+49152;
        let heap=gfx::HeapBuffer{base:base+4096,size:4096,chunk_size:4096,chunks:vec![gfx::HeapChunk{buffer:physical.buffer.clone(),base:base+4096,size:4096,gpu:res.heap_gpu[0],offset:0}]};
        let swizzle=2u32<<8|3<<11|4<<14|5<<17;
        let mut cases:Vec<[u32;8]>=Vec::new();let mut expected=Vec::new();
        for f in 1u32..=48 {
            let components=1+(f-1)/12;let n=(f-1)%12+1;let bits=if n<=4{8}else if n<=9{16}else{32};let bytes=bits*components/8;
            for offset in [0u64,1,3,4080,4092,4095,8188,8192,0x10ffc,0x20004,0x20018,0x20020,0x30000,0x30001,0x30ff0] {
                for (x,lod) in [(0u32,0),(1,0),(4,0),(0,1)] {
                    let address=base+offset;let at=address+x as u64*bytes as u64;
                    cases.push([address as u32,(address>>32) as u32,4,f|swizzle,x,lod,0,0]);
                    let mut raw=[0u32;4];let mut valid=true;let mut typed_alignment_valid=true;
                    if x<4 {
                        let word_path=at&3==0&&bytes&3==0;let step=if word_path{4}else{1};
                        for i in (0..bytes).step_by(step as usize) {
                            let a=at+i as u64;let p=if a>=base+4096 && a<base+8192 {Some(cpu+49152+a-base-4096)}else{aliases.iter().find(|&&(b,_,l)|a>=b&&a-b<l).map(|&(b,p,_)|p+a-b)};
                            if let Some(p)=p {if word_path {typed_alignment_valid &= p&3==0;raw[(i/4) as usize]=std::ptr::read_unaligned(p as *const u32);}else{raw[(i/4) as usize]|=((p as *const u8).read() as u32)<<((i%4)*8);}}else{valid=false;break;}
                        }
                    }
                    if !valid{raw=[0;4];}expected.push(typed_alignment_valid.then_some(raw));
                }
            }
        }
        let direct_cases=cases.iter().map(|c|{let mut c=*c;let f=c[3]&255;let n=(f-1)%12+1;let bits=if n<=4{8}else if n<=9{16}else{32};let format=[bits,1+(f-1)/12,0,1];if let Some(m)=super::buffer_direct::metadata(&pools,Some(&heap),c[..4].try_into().unwrap(),format){c[..4].copy_from_slice(&m);}c}).collect::<Vec<_>>();
        assert!(direct_cases.iter().filter(|c|c[3]>>31!=0).count()>500);
        let inputs=[&cases,&direct_cases].map(|cases|{let b=dev.newBufferWithLength_options(cases.len()*32,MTLResourceOptions::StorageModeShared).unwrap();std::ptr::copy_nonoverlapping(cases.as_ptr().cast::<u8>(),b.contents().as_ptr().cast::<u8>(),cases.len()*32);b});
        let bench_meta=[0,40,256,46|swizzle];let direct_bench_meta=super::buffer_direct::metadata(&pools,Some(&heap),bench_meta,[32,4,2,1]).unwrap();
        // A descriptor spanning the heap priority boundary must fall back.
        assert!(super::buffer_direct::metadata(&pools,Some(&heap),[0,40,512,46|swizzle],[32,4,2,1]).is_none());
        let original=nxshader::prelude().replace("#define NX_PAGED_GMEM 0","#define NX_PAGED_GMEM 1");let optimized=shaders::buffer_read_source(&original).into_owned();assert_ne!(original,optimized);
        let body=r#"
kernel void edges(constant Res& r [[buffer(0)]],const device uint4* cases [[buffer(1)]],device uint4* out [[buffer(2)]],uint tid [[thread_position_in_grid]]){uint4 m=cases[tid*2u],c=cases[tid*2u+1u];out[tid*2u]=buffer_read_raw(r,m,c.x,0u);out[tid*2u+1u]=buffer_fetch(r,m,c.x,c.y,0u);}
kernel void fetches(constant Res& r [[buffer(0)]],device uint4* out [[buffer(2)]],constant uint4& m [[buffer(3)]],uint tid [[thread_position_in_grid]]){uint4 sum=uint4(tid);for(uint i=0u;i<32u;i++){uint x=(tid*13u+sum.x+i*17u)%256u;sum^=buffer_read_raw(r,m,x,0u);sum=rotate(sum,uint4(3u));}out[tid]=sum;}
"#;
        let direct=shaders::direct_buffer_source(&optimized).into_owned();assert_ne!(direct,optimized);
        let specialized_source=vertex_formats::buffer_source(&direct).into_owned()+r#"
kernel void metadata_edges(constant Res& r [[buffer(0)]],const device uint4* cases [[buffer(1)]],device uint4* out [[buffer(2)]],constant uint4* metadata [[buffer(3)]],uint tid [[thread_position_in_grid]]){uint4 m=nx_buffer_meta(metadata+tid*32u,tid&31u),c=cases[tid*2u+1u];out[tid*2u]=buffer_read_raw(r,m,c.x,0u);out[tid*2u+1u]=buffer_fetch(r,m,c.x,c.y,0u);}
"#;
        let specialized_library=dev.newLibraryWithSource_options_error(&NSString::from_str(&specialized_source),None).unwrap();
        let libraries=[original,optimized,direct].map(|s|dev.newLibraryWithSource_options_error(&NSString::from_str(&(s+body)),None).unwrap());
        let edges=libraries.each_ref().map(|l|dev.newComputePipelineStateWithFunction_error(&l.newFunctionWithName(&NSString::from_str("edges")).unwrap()).unwrap());
        let fetches=libraries.each_ref().map(|l|dev.newComputePipelineStateWithFunction_error(&l.newFunctionWithName(&NSString::from_str("fetches")).unwrap()).unwrap());
        let mut reference=None;let mut reference_fault=None;let mut times=[Vec::new(),Vec::new(),Vec::new()];
        for mode in 0..3 {
            std::ptr::write_bytes(fault.contents().as_ptr(),0,32);let cb=gfx::gpu().queue.0.commandBuffer().unwrap();let e=cb.computeCommandEncoder().unwrap();e.setComputePipelineState(&edges[mode]);
            for c in pools.iter().flat_map(|p|&p.chunks){e.useResource_usage(ProtocolObject::from_ref(&*c.buffer.0),MTLResourceUsage::Read);}
            for b in [&table.0,&fault,&gfx::gpu().zeros.0]{e.useResource_usage(ProtocolObject::from_ref(&**b),MTLResourceUsage::Read|MTLResourceUsage::Write);}
            e.setBytes_length_atIndex(std::ptr::NonNull::from(&res).cast(),size_of_val(&res),0);e.setBuffer_offset_atIndex(Some(&inputs[usize::from(mode==2)]),0,1);e.setBuffer_offset_atIndex(Some(&output),0,2);e.dispatchThreads_threadsPerThreadgroup(MTLSize{width:cases.len(),height:1,depth:1},MTLSize{width:64,height:1,depth:1});e.endEncoding();cb.commit();cb.waitUntilCompleted();assert_eq!(cb.status(),MTLCommandBufferStatus::Completed);
            let got=std::slice::from_raw_parts(output.contents().as_ptr().cast::<[u32;4]>(),cases.len()*2).to_vec();
            for (i,expected) in expected.iter().enumerate(){if let Some(expected)=expected{assert_eq!(got[i*2],*expected,"mode{mode} case{i} {:?}",cases[i]);}}
            let fault_code=*fault.contents().as_ptr().cast::<u32>();
            if mode==0{reference=Some(got);reference_fault=Some(fault_code);}else{assert_eq!(Some(&got),reference.as_ref());assert_eq!(Some(fault_code),reference_fault);}
        }
        // Group only by the exact metadata format word. Address/length remain
        // per-case data, including direct-tag changes, aliases, holes and edges.
        let mut groups=std::collections::BTreeMap::<u32,Vec<usize>>::new();
        for (i,c) in direct_cases.iter().enumerate(){groups.entry(c[3]).or_default().push(i);}
        let reference=reference.unwrap();
        for (word,indices) in groups {
            let indices=indices.iter().copied().cycle().take(indices.len().max(32)).collect::<Vec<_>>();
            let selected=indices.iter().map(|&i|direct_cases[i]).collect::<Vec<_>>();
            let metadata=selected.iter().flat_map(|c|std::iter::repeat_n(<[u32;4]>::try_from(&c[..4]).unwrap(),32)).collect::<Vec<_>>();
            let make=|bytes:&[u8]|{let b=dev.newBufferWithLength_options(bytes.len(),MTLResourceOptions::StorageModeShared).unwrap();std::ptr::copy_nonoverlapping(bytes.as_ptr(),b.contents().as_ptr().cast::<u8>(),bytes.len());b};
            let selected_buffer=make(std::slice::from_raw_parts(selected.as_ptr().cast::<u8>(),selected.len()*32));
            let metadata_buffer=make(std::slice::from_raw_parts(metadata.as_ptr().cast::<u8>(),metadata.len()*16));
            let generic_function=specialized_library.newFunctionWithName_constantValues_error(&NSString::from_str("metadata_edges"),&MTLFunctionConstantValues::new()).unwrap();
            let specialized_function=vertex_formats::buffer_function(&specialized_library,"metadata_edges",&[word;32]).unwrap();
            let pipelines=[&*generic_function,&*specialized_function.0].map(|f|dev.newComputePipelineStateWithFunction_error(f).unwrap());
            let mut generic_fault=None;
            for (mode,pipeline) in pipelines.iter().enumerate(){
                std::ptr::write_bytes(fault.contents().as_ptr(),0,32);let cb=gfx::gpu().queue.0.commandBuffer().unwrap();let e=cb.computeCommandEncoder().unwrap();e.setComputePipelineState(pipeline);
                for c in pools.iter().flat_map(|p|&p.chunks){e.useResource_usage(ProtocolObject::from_ref(&*c.buffer.0),MTLResourceUsage::Read);}
                for b in [&table.0,&fault,&gfx::gpu().zeros.0]{e.useResource_usage(ProtocolObject::from_ref(&**b),MTLResourceUsage::Read|MTLResourceUsage::Write);}
                e.setBytes_length_atIndex(std::ptr::NonNull::from(&res).cast(),size_of_val(&res),0);e.setBuffer_offset_atIndex(Some(&selected_buffer),0,1);e.setBuffer_offset_atIndex(Some(&output),0,2);e.setBuffer_offset_atIndex(Some(&metadata_buffer),0,3);
                e.dispatchThreads_threadsPerThreadgroup(MTLSize{width:selected.len(),height:1,depth:1},MTLSize{width:32,height:1,depth:1});e.endEncoding();cb.commit();cb.waitUntilCompleted();assert_eq!(cb.status(),MTLCommandBufferStatus::Completed);
                let got=std::slice::from_raw_parts(output.contents().as_ptr().cast::<[u32;4]>(),selected.len()*2);
                for (at,&original_index)in indices.iter().enumerate(){assert_eq!(got[at*2..at*2+2],reference[original_index*2..original_index*2+2],"buffer metadata mode{mode} word{word:#x} case{original_index}");}
                let code=*fault.contents().as_ptr().cast::<u32>();if mode==0{generic_fault=Some(code);}else{assert_eq!(Some(code),generic_fault,"metadata fault word{word:#x}");}
            }
        }
        eprintln!("[verify] all32 sampled metadata slots preserve2880 format/direct/alias/hole/edge cases and exact fault codes");
        // An exact-output dependent gather prevents the compiler from folding
        // repeated reads. Timings are observations, never pass criteria.
        let mut reference=None;
        for round in 0..15 {let mode=round%3;let cb=gfx::gpu().queue.0.commandBuffer().unwrap();let e=cb.computeCommandEncoder().unwrap();e.setComputePipelineState(&fetches[mode]);for c in pools.iter().flat_map(|p|&p.chunks){e.useResource_usage(ProtocolObject::from_ref(&*c.buffer.0),MTLResourceUsage::Read);}for b in [&table.0,&fault,&gfx::gpu().zeros.0]{e.useResource_usage(ProtocolObject::from_ref(&**b),MTLResourceUsage::Read|MTLResourceUsage::Write);}e.setBytes_length_atIndex(std::ptr::NonNull::from(&res).cast(),size_of_val(&res),0);e.setBuffer_offset_atIndex(Some(&output),0,2);let m=if mode==2{&direct_bench_meta}else{&bench_meta};e.setBytes_length_atIndex(std::ptr::NonNull::from(m).cast(),16,3);e.dispatchThreads_threadsPerThreadgroup(MTLSize{width:THREADS,height:1,depth:1},MTLSize{width:128,height:1,depth:1});e.endEncoding();cb.commit();cb.waitUntilCompleted();assert_eq!(cb.status(),MTLCommandBufferStatus::Completed);let got=std::slice::from_raw_parts(output.contents().as_ptr().cast::<u32>(),THREADS*4).to_vec();if let Some(ref reference)=reference{assert_eq!(&got,reference);}else{reference=Some(got);}if round>=3{times[mode].push((cb.GPUEndTime()-cb.GPUStartTime())*1000.0);}}
        eprintln!("[buffer-read-benchmark] original_gpu_ms={:?} batch_gpu_ms={:?} direct_gpu_ms={:?} formats=48 edge_cases={} exact_threads={THREADS}; synthetic only",times[0],times[1],times[2],cases.len());
        drop(cache);drop(table);gfx::forget_pool(0xf175);drop(pools);assert_eq!(libc::munmap(allocation,BYTES),0);
    });
}
