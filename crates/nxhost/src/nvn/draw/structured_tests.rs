// Replay the original integration shader that exposed a native watchdog hang.
// The small render and diagnostic instruction fuel keep this regression bounded.
use super::*;
use objc2_foundation::NSString;

#[test]
#[ignore="requires a real Metal GPU and NX_ORIGINAL_SHARED_COMPUTE_BLOB"]
fn gpu_original_shared_min_max_or_kernel_builds_with_original_memory_metadata(){
    let code=std::fs::read(std::env::var("NX_ORIGINAL_SHARED_COMPUTE_BLOB").expect("original compute capture path")).unwrap();
    let translated=nxshader::translate_compute_with_memory(&code,1536,536).unwrap();
    for function in ["atomic_fetch_min_explicit","atomic_fetch_max_explicit","atomic_fetch_or_explicit"]{assert!(translated.msl.contains(function));}
    objc2::rc::autoreleasepool(|_|unsafe{
        let options=MTLCompileOptions::new();options.setMathMode(MTLMathMode::Safe);options.setPreserveInvariance(true);
        let source=shaders::cbuf_source(&translated.msl);let device=&gfx::gpu().device.0;
        let library=device.newLibraryWithSource_options_error(&NSString::from_str(&source),Some(&options)).unwrap();
        let function=library.newFunctionWithName(&NSString::from_str(&translated.entry)).unwrap();let pipeline=device.newComputePipelineStateWithFunction_error(&function).unwrap();assert!(pipeline.maxTotalThreadsPerThreadgroup()>=64);
        eprintln!("[verify] original{}byte compute kernel/1536local/536shared: actualMetal pipeline built with MIN/MAX/OR. Original dispatch contents are not validated by compilation.",code.len());
    });
}

#[test]
#[ignore="requires a real Metal GPU and NX_ORIGINAL_MIXED_COMPUTE_BLOB"]
fn gpu_original_mixed_surface_compute_builds_with_one_shared_atomic_backing(){
    let code=std::fs::read(std::env::var("NX_ORIGINAL_MIXED_COMPUTE_BLOB").expect("original compute capture path")).unwrap();
    let translated=nxshader::translate_compute_with_memory(&code,0,0).unwrap();
    assert_eq!(translated.interface.atomic_images,1);assert!(translated.msl.contains("atomic_store_explicit(pointer"));
    objc2::rc::autoreleasepool(|_|unsafe{
        let options=MTLCompileOptions::new();options.setMathMode(MTLMathMode::Safe);options.setPreserveInvariance(true);
        let source=shaders::cbuf_source(&translated.msl);let device=&gfx::gpu().device.0;
        let library=device.newLibraryWithSource_options_error(&NSString::from_str(&source),Some(&options)).unwrap();
        let function=library.newFunctionWithName(&NSString::from_str(&translated.entry)).unwrap();let pipeline=device.newComputePipelineStateWithFunction_error(&function).unwrap();assert!(pipeline.maxTotalThreadsPerThreadgroup()>=64);
        eprintln!("[verify] original{}byte compute source translated and actualMetal pipeline built; one atomic binding, ordinary stores share its staging. Original dispatch image contents still need game verification.",code.len());
    });
}

#[test]
#[ignore="requires a real Metal GPU; run serially"]
fn gpu_clamped_c1_jump_table_matches_all_cases_unsigned_extremes_and_switch(){
    let address=|n:usize|((n/3*4+1+n%3)*8)as u32;
    let mov=|r:u64,v:f32|0x010000000007f000|(u64::from(v.to_bits())<<20)|r;
    let mut words=vec![0x4c98078c00070033u64,0x3820038000473320,0x384800000027205e,0xef94001000075e5e,0xe250000000075e0f];
    words[4]|=((-(address(4)as i64+8)as u64)&0xffffff)<<20;
    let mut constants=Vec::new();
    for n in 0..5 {constants.extend(address(words.len()).to_le_bytes());words.extend([mov(0,n as f32/8.0),mov(1,(n+1)as f32/8.0),mov(2,0.5),mov(3,1.0),0xe30000000007000f]);}
    constants.resize(256,0);words.push(0xe2400fffff87000f);
    let mut code=vec![0u8;128];code[0x30..0x34].copy_from_slice(&(5u32<<10).to_le_bytes());code[0x78..0x7c].copy_from_slice(&15u32.to_le_bytes());
    code.extend(words.chunks(3).flat_map(|g|std::iter::once(0u64).chain(g.iter().copied())).flat_map(u64::to_le_bytes));
    let reference=nxshader::translate_with_constants(&code,0,0,0,&constants).unwrap();
    let candidate=nxshader::translate_graphics_structured(&code,0,0,0,Some(&constants)).unwrap();
    assert!(candidate.msl.contains("// structured graphics:"));
    objc2::rc::autoreleasepool(|_|unsafe{
        let dev=&gfx::gpu().device.0;
        let vertex="\nvertex float4 jump_vertex(uint id [[vertex_id]]){float2 p=float2((id<<1)&2,id&2);return float4(p*2.0f-1.0f,0.5f,1.0f);}\n";
        let mut pipelines=Vec::new();
        for translated in [&reference,&candidate]{
            let source=translated.msl.replace("    uint pc = 8u;","    uint pc = 8u; uint control_steps=0u;").replace("    for (;;) {","    for (;;) {if(++control_steps>128u){shader_fault(res,32768u,pc);return epilogue();}")+vertex;
            let options=MTLCompileOptions::new();options.setMathMode(MTLMathMode::Safe);options.setPreserveInvariance(true);
            let lib=dev.newLibraryWithSource_options_error(&NSString::from_str(&source),Some(&options)).unwrap();
            let descriptor=MTLRenderPipelineDescriptor::new();descriptor.setVertexFunction(Some(&lib.newFunctionWithName(&NSString::from_str("jump_vertex")).unwrap()));descriptor.setFragmentFunction(Some(&lib.newFunctionWithName(&NSString::from_str(&translated.entry)).unwrap()));descriptor.colorAttachments().objectAtIndexedSubscript(0).setPixelFormat(MTLPixelFormat::RGBA32Float);pipelines.push(dev.newRenderPipelineStateWithDescriptor_error(&descriptor).unwrap());
        }
        let uniform=dev.newBufferWithLength_options(4096,MTLResourceOptions::StorageModeShared).unwrap();std::ptr::write_bytes(uniform.contents().as_ptr(),0,4096);
        let fault=dev.newBufferWithLength_options(32,MTLResourceOptions::StorageModeShared).unwrap();
        let descriptor=MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(MTLPixelFormat::RGBA32Float,4,4,false);descriptor.setStorageMode(MTLStorageMode::Shared);descriptor.setUsage(MTLTextureUsage::RenderTarget);let target=dev.newTextureWithDescriptor(&descriptor).unwrap();
        let mut res:state::Res=std::mem::zeroed();res.cb.fill(gfx::gpu().zeros.0.gpuAddress());res.cb[3]=uniform.gpuAddress();res.fault_gpu=fault.gpuAddress();
        for index in [0u32,1,2,3,4,5,7,31,256,0x7fffffff,0x80000000,u32::MAX]{
            uniform.contents().as_ptr().cast::<u32>().write(index);
            let n=index.min(4)as f32;let expected=[n/8.0,(n+1.0)/8.0,0.5,1.0];
            for (variant,pipeline)in pipelines.iter().enumerate(){
                std::ptr::write_bytes(fault.contents().as_ptr(),0,32);let pass=MTLRenderPassDescriptor::new();let color=pass.colorAttachments().objectAtIndexedSubscript(0);color.setTexture(Some(&target));color.setLoadAction(MTLLoadAction::Clear);color.setStoreAction(MTLStoreAction::Store);
                let cb=gfx::gpu().queue.0.commandBuffer().unwrap();let encoder=cb.renderCommandEncoderWithDescriptor(&pass).unwrap();encoder.setRenderPipelineState(pipeline);
                for b in [&uniform,&fault,&gfx::gpu().zeros.0]{encoder.useResource_usage_stages(ProtocolObject::from_ref(&**b),MTLResourceUsage::Read|MTLResourceUsage::Write,MTLRenderStages::Fragment);}
                encoder.setFragmentBytes_length_atIndex(std::ptr::NonNull::from(&res).cast(),size_of_val(&res),0);encoder.drawPrimitives_vertexStart_vertexCount(MTLPrimitiveType::Triangle,0,3);encoder.endEncoding();cb.commit();cb.waitUntilCompleted();assert_eq!(cb.status(),MTLCommandBufferStatus::Completed);assert_eq!(*fault.contents().as_ptr().cast::<u32>(),0,"index{index} variant{variant}");
                let mut pixels=[0f32;64];target.getBytes_bytesPerRow_fromRegion_mipmapLevel(std::ptr::NonNull::from(&mut pixels).cast(),64,MTLRegion{origin:MTLOrigin{x:0,y:0,z:0},size:MTLSize{width:4,height:4,depth:1}},0);
                for pixel in pixels.chunks_exact(4){assert_eq!(pixel,expected,"index{index} variant{variant}");}
            }
        }
        eprintln!("[verify] all5 real C1 cases,7 clamp/extreme inputs,16 pixels each: original switch and bounded structured dispatch match independent expected colors exactly");
    });
}

#[test]
#[ignore="requires a real Metal GPU and NX_BRDF_GRAPHICS_CAPTURE"]
fn gpu_original_brdf_nested_loop_matches_original_dispatch() {
    let root=std::path::PathBuf::from(std::env::var("NX_BRDF_GRAPHICS_CAPTURE").expect("set original BRDF capture"));
    let code=std::fs::read(root.join("work/msl_rt/028e09c5dd0ed5bc.bin")).unwrap();
    let reference=nxshader::translate_with(&code,nxshader::FRAGMENT_EARLY_TESTS,0,0).unwrap();
    let candidate=nxshader::translate_graphics_structured(&code,nxshader::FRAGMENT_EARLY_TESTS,0,0,None).unwrap();
    assert!(candidate.msl.contains("// structured graphics:"));
    assert!(reference.interface.textures.is_empty() && reference.interface.images.is_empty());
    objc2::rc::autoreleasepool(|_|unsafe {
        let dev=&gfx::gpu().device.0;
        let vertex="\nstruct BrdfVertex {float4 pos [[position]];float4 uv [[user(locn0)]];};vertex BrdfVertex brdf_vertex(uint id [[vertex_id]]) {float2 uv=float2((id<<1)&2,id&2);BrdfVertex v;v.pos=float4(uv*2.0f-1.0f,0.5f,1.0f);v.uv=float4(uv,0.0f,0.0f);return v;}\n";
        let mut pipelines=Vec::new();
        for (variant,translated) in [&reference,&candidate].into_iter().enumerate() {
            let needle="    uint pc = 8u;";assert_eq!(translated.msl.matches(needle).count(),1);
            let msl=if variant==1 {shaders::cbuf_source(&translated.msl).into_owned()} else {translated.msl.clone()};
            let source=msl.replace(needle,"    uint pc = 8u; uint control_steps=0u;")
                .replace("    for (;;) {","    for (;;) { if (++control_steps > 16384u) {shader_fault(res,32768u,pc);return epilogue();}")+vertex;
            let options=MTLCompileOptions::new();options.setMathMode(MTLMathMode::Safe);options.setPreserveInvariance(true);
            let library=dev.newLibraryWithSource_options_error(&NSString::from_str(&source),Some(&options)).unwrap();
            let descriptor=MTLRenderPipelineDescriptor::new();descriptor.setVertexFunction(Some(&library.newFunctionWithName(&NSString::from_str("brdf_vertex")).unwrap()));descriptor.setFragmentFunction(Some(&library.newFunctionWithName(&NSString::from_str(&translated.entry)).unwrap()));descriptor.colorAttachments().objectAtIndexedSubscript(0).setPixelFormat(MTLPixelFormat::RGBA32Float);
            pipelines.push(dev.newRenderPipelineStateWithDescriptor_error(&descriptor).unwrap());
        }
        let bank=dev.newBufferWithLength_options(4096,MTLResourceOptions::StorageModeShared).unwrap();
        let constants=std::slice::from_raw_parts_mut(bank.contents().as_ptr().cast::<f32>(),1024);constants.fill(1.0);constants[1]=-1.0;constants[2]=0.045;constants[7]=std::f32::consts::FRAC_1_SQRT_2;
        let uniform=dev.newBufferWithLength_options(4096,MTLResourceOptions::StorageModeShared).unwrap();std::ptr::write_bytes(uniform.contents().as_ptr(),0,4096);
        let fault=dev.newBufferWithLength_options(32,MTLResourceOptions::StorageModeShared).unwrap();
        let descriptor=MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(MTLPixelFormat::RGBA32Float,16,16,false);descriptor.setStorageMode(MTLStorageMode::Shared);descriptor.setUsage(MTLTextureUsage::RenderTarget);let target=dev.newTextureWithDescriptor(&descriptor).unwrap();
        let mut res:state::Res=std::mem::zeroed();res.cb.fill(gfx::gpu().zeros.0.gpuAddress());res.cb[1]=bank.gpuAddress();res.cb[3]=uniform.gpuAddress();res.fault_gpu=fault.gpuAddress();
        let mut first=None;let mut changed=false;
        for samples in [0u32,1,2,8,32,128] {
            uniform.contents().as_ptr().cast::<u32>().write(samples);let mut outputs=Vec::new();let mut timings=Vec::new();
            for pipeline in &pipelines {
                std::ptr::write_bytes(fault.contents().as_ptr(),0,32);
                let pass=MTLRenderPassDescriptor::new();let color=pass.colorAttachments().objectAtIndexedSubscript(0);color.setTexture(Some(&target));color.setLoadAction(MTLLoadAction::Clear);color.setStoreAction(MTLStoreAction::Store);
                let cb=gfx::gpu().queue.0.commandBuffer().unwrap();let encoder=cb.renderCommandEncoderWithDescriptor(&pass).unwrap();encoder.setRenderPipelineState(pipeline);
                for buffer in [&bank,&uniform,&fault,&gfx::gpu().zeros.0] {encoder.useResource_usage_stages(ProtocolObject::from_ref(&**buffer),MTLResourceUsage::Read|MTLResourceUsage::Write,MTLRenderStages::Fragment);}
                encoder.setFragmentBytes_length_atIndex(std::ptr::NonNull::from(&res).cast(),size_of_val(&res),0);encoder.drawPrimitives_vertexStart_vertexCount(MTLPrimitiveType::Triangle,0,3);encoder.endEncoding();cb.commit();cb.waitUntilCompleted();assert_eq!(cb.status(),MTLCommandBufferStatus::Completed);
                assert_eq!(*fault.contents().as_ptr().cast::<u32>(),0,"original BRDF control fuel exhausted at sample count{samples}");
                let mut pixels=vec![0f32;16*16*4];target.getBytes_bytesPerRow_fromRegion_mipmapLevel(std::ptr::NonNull::new(pixels.as_mut_ptr()).unwrap().cast(),16*16,MTLRegion{origin:MTLOrigin{x:0,y:0,z:0},size:MTLSize{width:16,height:16,depth:1}},0);outputs.push(pixels);timings.push((cb.GPUEndTime()-cb.GPUStartTime())*1000.0);
            }
            for (i,(&a,&b)) in outputs[0].iter().zip(&outputs[1]).enumerate() {assert!(a.to_bits()==b.to_bits() || (a.is_nan()&&b.is_nan()) || (a-b).abs()<=1e-5f32.max(a.abs().max(b.abs())*1e-4),"original BRDF sample count{samples} pixel word{i}: reference{a:?} candidate{b:?}");}
            let bits=outputs[0].iter().map(|v|v.to_bits()).collect::<Vec<_>>();if let Some(first)=&first {changed|=first!=&bits;} else {first=Some(bits);}
            assert!(outputs[0].iter().any(|v|v.is_finite()&&*v!=0.0),"BRDF output must contain actual finite data");
            eprintln!("[verify] original BRDF samples{samples} reference_gpu_ms{:.4} structured_gpu_ms{:.4}",timings[0],timings[1]);
        }
        assert!(changed,"sample count must affect actual original BRDF output");
    });
}
