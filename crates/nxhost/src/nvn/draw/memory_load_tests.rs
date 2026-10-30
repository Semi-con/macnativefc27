use super::*;
use objc2_foundation::NSString;
use std::ptr::NonNull;

fn program(instructions:&[u64])->Vec<u8> {
    let mut instructions=instructions.to_vec();instructions.extend([0xe30000000007000fu64,0xe2400fffff87000f]);
    let mut words=Vec::new();for group in instructions.chunks(3) {words.push(0u64);words.extend_from_slice(group);}
    let mut bytes=words.iter().flat_map(|w|w.to_le_bytes()).collect::<Vec<_>>();bytes.resize(bytes.len().max(0x100),0);bytes
}
fn movc(dest:u64,bank:u64,word:u64)->u64 {0x4c98078000070000|(bank<<34)|(word<<20)|dest}
fn stg(words:u64,dest:u64,address:u64)->u64 {0xeed8000000070000|(words<<48)|(1<<45)|(address<<8)|dest}

#[test]
#[ignore="requires a real Metal GPU"]
fn gpu_original_multiword_loads_preserve_holes_aliases_heap_boundaries_and_overlapping_registers() {
    objc2::rc::autoreleasepool(|_|unsafe {
        let page=crate::mem::PAGE as usize;let length=page*5;
        let allocation=libc::mmap(std::ptr::null_mut(),length,libc::PROT_READ|libc::PROT_WRITE,libc::MAP_PRIVATE|libc::MAP_ANON,-1,0);assert_ne!(allocation,libc::MAP_FAILED);let cpu=allocation as u64;
        for i in 0..page*4/4 {(allocation as *mut u32).add(i).write(0x12340000+i as u32);}
        let base=0x28_0000_0000;let heap=0x18_0000_0000;let output_address=0x38_0000_0000;
        let aliases=[(base,cpu,8192),(base+0x10000,cpu+8192,4096),(base+0x11000,cpu+16384,4096),(base+0x20004,cpu+4,12),(base+0x30000,cpu+32,4),(heap,cpu+40960,12288),(output_address,cpu+(page*4) as u64,page as u64)];
        let mut pools=gfx::pool_memory(&[(0xda40,cpu,length as u64)]).unwrap();gfx::add_pool_aliases(&mut pools,&aliases).unwrap();
        let mut cache=address_table::Cache::default();let (table,bindings)=cache.bind(&pools);let device=&gfx::gpu().device.0;
        let uniform=device.newBufferWithLength_options(4096,MTLResourceOptions::StorageModeShared).unwrap();let fault=device.newBufferWithLength_options(32,MTLResourceOptions::StorageModeShared).unwrap();
        let mut res:state::Res=std::mem::zeroed();res.cb.fill(gfx::gpu().zeros.0.gpuAddress());res.cb[3]=uniform.gpuAddress();res.zero_gpu=gfx::gpu().zeros.0.gpuAddress();res.fault_gpu=fault.gpuAddress();res.apply_pool_bindings(bindings);
        let physical=&pools.iter().find(|p|p.base==cpu).unwrap().chunks[0];res.heap_base=heap;res.heap_size=8192;res.heap_chunk_size=4096;res.heap_gpu[0]=physical.gpu+physical.offset as u64+24576;res.heap_gpu[1]=physical.gpu+physical.offset as u64+32768;
        let addresses=[base,base+4,base+4084,base+4092,base+8188,base+8192,base+0x10ffc,base+0x20004,base+0x20008,base+0x30000,heap,heap+4092,heap+8188,heap+8192,u64::MAX-3];
        for size in [5u64,6] {
            let ldg=(0xeed4200000070000u64&!(7<<48))|(size<<48);
            assert_eq!(nxshader::table::decode(ldg).1,nxshader::table::Op::Ldg);assert_eq!(nxshader::table::decode(stg(size,0,4)).1,nxshader::table::Op::Stg);
            // LDG overwrites R0/R1, which also hold its original address.
            let code=program(&[movc(0,3,0),movc(1,3,1),movc(4,3,2),movc(5,3,3),ldg,stg(size,0,4)]);
            let reference=nxshader::translate_compute_with_memory(&code,0,0).unwrap();let candidate=nxshader::translate_configured(&code,0,0,0,None,nxshader::TranslationConfig{compute_memory:Some((0,0)),vector_global_loads:true,..Default::default()}).unwrap();assert!(candidate.msl.contains("nx_gmem_ld"));
            let mut pipelines=Vec::new();for translation in [&reference,&candidate] {
                let source=shaders::cbuf_source(&translation.msl);let options=MTLCompileOptions::new();options.setMathMode(MTLMathMode::Safe);
                let library=device.newLibraryWithSource_options_error(&NSString::from_str(&source),Some(&options)).unwrap();pipelines.push(device.newComputePipelineStateWithFunction_error(&library.newFunctionWithName(&NSString::from_str(&translation.entry)).unwrap()).unwrap());
            }
            let words=if size==5 {2}else{4};
            for address in addresses {
                let expected=(0..words).map(|k| {
                    let at=address.wrapping_add(k as u64*4);let offset=if at>=heap && at<heap+8192 {Some(if at-heap<4096 {24576+at-heap}else{32768+at-heap-4096})}
                        else {aliases.iter().find(|&&(base,_,len)|at>=base && at-base<len).map(|&(base,physical,_)|physical-cpu+at-base)};
                    offset.map_or(0,|offset|0x12340000+offset as u32/4)
                }).collect::<Vec<_>>();
                for pipeline in &pipelines {
                    let output=(cpu+(page*4) as u64) as *mut u32;for i in 0..4 {output.add(i).write(0xdeadbeef);}std::ptr::write_bytes(fault.contents().as_ptr(),0,32);
                    let values=[address as u32,(address>>32) as u32,output_address as u32,(output_address>>32) as u32];std::ptr::copy_nonoverlapping(values.as_ptr(),uniform.contents().as_ptr().cast::<u32>(),4);
                    let cb=gfx::gpu().queue.0.commandBuffer().unwrap();let encoder=cb.computeCommandEncoder().unwrap();encoder.setComputePipelineState(pipeline);
                    for chunk in pools.iter().flat_map(|p|&p.chunks) {encoder.useResource_usage(ProtocolObject::from_ref(&*chunk.buffer.0),MTLResourceUsage::Read|MTLResourceUsage::Write);}
                    for buffer in [&uniform,&fault,&table.0,&gfx::gpu().zeros.0] {encoder.useResource_usage(ProtocolObject::from_ref(&**buffer),MTLResourceUsage::Read|MTLResourceUsage::Write);}
                    encoder.setBytes_length_atIndex(NonNull::from(&res).cast(),size_of_val(&res),0);encoder.dispatchThreadgroups_threadsPerThreadgroup(MTLSize{width:1,height:1,depth:1},MTLSize{width:1,height:1,depth:1});encoder.endEncoding();cb.commit();cb.waitUntilCompleted();assert_eq!(cb.status(),MTLCommandBufferStatus::Completed);
                    assert_eq!(std::slice::from_raw_parts(output,words),expected,"original LDG size{size} address{address:#x}");if words==2 {assert_eq!(output.add(2).read(),0xdeadbeef);}
                    assert_eq!(*fault.contents().as_ptr().cast::<u32>(),0);
                }
            }
        }
        gfx::forget_pool(0xda40);
    });
}

#[test]
#[ignore="requires a real Metal GPU"]
fn gpu_branchless_embedded_constants_refresh_without_freezing_live_uniforms() {
    objc2::rc::autoreleasepool(|_|unsafe {
        let code=program(&[movc(0,1,0),movc(1,1,1),movc(2,3,0),movc(3,3,1),movc(4,3,2),movc(5,3,3),stg(6,0,4)]);
        let length=0x100+code.len();let mut container=vec![0u8;length+8];container[..4].copy_from_slice(&0x12345679u32.to_le_bytes());container[0x100..length].copy_from_slice(&code);
        let mut control=vec![0u8;0x880];for (offset,value) in [(0x700,length as u32),(0x704,(length+8) as u32),(0x708,0x100),(0x748,1),(0x74c,1),(0x750,1)] {control[offset..offset+4].copy_from_slice(&value.to_le_bytes());}
        let object=0xcf270050;STATE.lock().unwrap().pools.insert(object,crate::nvn::objects::MemoryPool{cpu:container.as_ptr() as u64,size:container.len() as u64,flags:0x22});
        let device=&gfx::gpu().device.0;let make=|length|device.newBufferWithLength_options(length,MTLResourceOptions::StorageModeShared).unwrap();let constants=make(4096);let uniforms=make(4096);let output=make(4096);let table=make(4096);let fault=make(32);let address=0x3840000000u64;
        std::ptr::copy_nonoverlapping([address,16,output.gpuAddress()].as_ptr(),table.contents().as_ptr().cast::<u64>(),3);
        let mut res:state::Res=std::mem::zeroed();res.cb.fill(gfx::gpu().zeros.0.gpuAddress());res.cb[1]=constants.gpuAddress();res.cb[3]=uniforms.gpuAddress();res.pool_spans=table.gpuAddress();res.pool_span_count=1;res.zero_gpu=gfx::gpu().zeros.0.gpuAddress();res.fault_gpu=fault.gpuAddress();
        let mut shaders=shaders::Shaders::default();let mut hashes=Vec::new();
        for (constant,live) in [(7u32,11u32),(19,23),(19,31),(7,41)] {
            let words=[constant,constant+1];for (i,word) in words.iter().enumerate() {container[length+i*4..length+i*4+4].copy_from_slice(&word.to_le_bytes());}std::ptr::copy_nonoverlapping(words.as_ptr(),constants.contents().as_ptr().cast::<u32>(),2);
            let values=[live,live+1,address as u32,(address>>32) as u32];std::ptr::copy_nonoverlapping(values.as_ptr(),uniforms.contents().as_ptr().cast::<u32>(),4);std::ptr::write_bytes(fault.contents().as_ptr(),0,32);
            let stage=shaders.stage(container.as_ptr() as u64,control.as_ptr() as u64).unwrap();assert_eq!(stage.constants_len,if shaders::embedded_all_enabled(){8}else{0});hashes.push(stage.hash);
            let shaders::Lookup::Ready(compiled)=shaders.function(&stage,0,0,0,true) else {panic!("branchless shader must compile")};let pipeline=device.newComputePipelineStateWithFunction_error(&compiled.func.0).unwrap();
            let cb=gfx::gpu().queue.0.commandBuffer().unwrap();let encoder=cb.computeCommandEncoder().unwrap();encoder.setComputePipelineState(&pipeline);
            for buffer in [&constants,&uniforms,&output,&table,&fault,&gfx::gpu().zeros.0] {encoder.useResource_usage(ProtocolObject::from_ref(&**buffer),MTLResourceUsage::Read|MTLResourceUsage::Write);}
            encoder.setBytes_length_atIndex(NonNull::from(&res).cast(),size_of_val(&res),0);encoder.dispatchThreadgroups_threadsPerThreadgroup(MTLSize{width:1,height:1,depth:1},MTLSize{width:1,height:1,depth:1});encoder.endEncoding();cb.commit();cb.waitUntilCompleted();assert_eq!(cb.status(),MTLCommandBufferStatus::Completed);
            assert_eq!(std::slice::from_raw_parts(output.contents().as_ptr().cast::<u32>(),4),[constant,constant+1,live,live+1]);assert_eq!(*fault.contents().as_ptr().cast::<u32>(),0);
        }
        assert_eq!(hashes[0]!=hashes[1],shaders::embedded_all_enabled());assert_eq!(hashes[1],hashes[2]);assert_eq!(hashes[0],hashes[3]);STATE.lock().unwrap().pools.remove(&object);
    });
}
