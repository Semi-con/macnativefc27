use super::*;
use std::ptr::NonNull;

fn fixture(changed:bool)->nxshader::Translated {
    let extra=if changed {"float4(1,0,0,0)"} else {"float4(0)"};
    let body=format!("struct VOut {{float4 pos [[position,invariant]];float4 v0 [[user(locn0)]];float psize [[point_size]];}};\nstruct StaticTextures {{texture2d<float> t0 [[id(0)]];sampler s0 [[id(1)]];}};\nvertex VOut vmain(uint vid [[vertex_id]], uint iid [[instance_id]], constant uint4* colors [[buffer(7)]], constant StaticTextures& statictex [[buffer(12)]]) {{float2 p[3]={{float2(-1,-1),float2(3,-1),float2(-1,3)}};VOut o;o.pos=float4(p[vid],0,1);o.v0=as_type<float4>(colors[iid])+statictex.t0.sample(statictex.s0,float2(0.5f),level(0))+{extra};o.psize=1;return o;}}\n");
    nxshader::Translated{stage:nxshader::sph::Stage::Vertex,msl:format!("{}{body}",nxshader::prelude()),entry:"vmain".into(),interface:Default::default(),grid_write_footprint:None,descriptor_write_footprint:None,write_window_footprint:None}
}

fn normal(t:&nxshader::Translated,key:u64)->String {
    let (s,_)=t.named(&format!("{key:016x}"));format!("{}{}",super::super::shaders::cbuf_source(nxshader::prelude()),super::super::shaders::cbuf_source(&s))
}

fn render_and_compare_fallback(product:&Product)->[f32;4] {unsafe {
    let device=&gfx::gpu().device.0;
    let fallback=product.state.fallback_function().unwrap();
    let dyn_encoder=product.func.0.newArgumentEncoderWithBufferIndex(12);
    let ordinary_encoder=fallback.0.newArgumentEncoderWithBufferIndex(12);
    assert_eq!(dyn_encoder.encodedLength(),ordinary_encoder.encodedLength(),"fallback argument-buffer layout changed");
    let input_desc=MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(MTLPixelFormat::RGBA32Float,1,1,false);input_desc.setStorageMode(MTLStorageMode::Shared);input_desc.setUsage(MTLTextureUsage::ShaderRead);
    let texture=device.newTextureWithDescriptor(&input_desc).unwrap();let input=[0.25f32,0.5,0.75,1.0];
    texture.replaceRegion_mipmapLevel_withBytes_bytesPerRow(MTLRegion{origin:MTLOrigin{x:0,y:0,z:0},size:MTLSize{width:1,height:1,depth:1}},0,NonNull::new(input.as_ptr().cast_mut()).unwrap().cast(),16);
    let sampler_desc=MTLSamplerDescriptor::new();sampler_desc.setSupportArgumentBuffers(true);let sampler=device.newSamplerStateWithDescriptor(&sampler_desc).unwrap();
    let mut argument_buffers=Vec::new();
    for encoder in [&dyn_encoder,&ordinary_encoder] {
        let buffer=device.newBufferWithLength_options(encoder.encodedLength(),MTLResourceOptions::StorageModeShared).unwrap();std::ptr::write_bytes(buffer.contents().as_ptr(),0,buffer.length());
        encoder.setArgumentBuffer_offset(Some(&buffer),0);encoder.setTexture_atIndex(Some(&texture),0);encoder.setSamplerState_atIndex(Some(&sampler),1);argument_buffers.push(buffer);
    }
    let a=std::slice::from_raw_parts(argument_buffers[0].contents().as_ptr().cast::<u8>(),argument_buffers[0].length());let b=std::slice::from_raw_parts(argument_buffers[1].contents().as_ptr().cast::<u8>(),argument_buffers[1].length());assert_eq!(a,b,"fallback uses different encoded texture/sampler bytes");
    let colors=device.newBufferWithLength_options(32,MTLResourceOptions::StorageModeShared).unwrap();let color_data=[99f32,99.,99.,99.,0.,0.,0.,0.];std::ptr::copy_nonoverlapping(color_data.as_ptr(),colors.contents().as_ptr().cast(),8);
    let fragment_source="#include <metal_stdlib>\nusing namespace metal;\nstruct FIn {float4 pos [[position]];float4 v0 [[user(locn0)]];};fragment float4 output_color(FIn in [[stage_in]]) {return in.v0;}";
    let fragment_library=device.newLibraryWithSource_options_error(&NSString::from_str(fragment_source),None).unwrap();let fragment=fragment_library.newFunctionWithName(&NSString::from_str("output_color")).unwrap();
    let mut results=Vec::new();
    // Both pipelines consume the very same argument-buffer encoding. The draw
    // uses baseInstance1; a lost iid or vid changes the pixels or coverage.
    for (function,dynamic) in [(&product.func,true),(&fallback,false)] {
        let desc=MTLRenderPipelineDescriptor::new();desc.setVertexFunction(Some(&function.0));desc.setFragmentFunction(Some(&fragment));desc.colorAttachments().objectAtIndexedSubscript(0).setPixelFormat(MTLPixelFormat::RGBA32Float);
        if dynamic {desc.setVertexPreloadedLibraries(&NSArray::from_retained_slice(&[product.state.library.0.clone()]));}
        let pipeline=device.newRenderPipelineStateWithDescriptor_error(&desc).unwrap();
        let target_desc=MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(MTLPixelFormat::RGBA32Float,1,1,false);target_desc.setStorageMode(MTLStorageMode::Shared);target_desc.setUsage(MTLTextureUsage::RenderTarget);let target=device.newTextureWithDescriptor(&target_desc).unwrap();
        let pass=MTLRenderPassDescriptor::new();let attachment=pass.colorAttachments().objectAtIndexedSubscript(0);attachment.setTexture(Some(&target));attachment.setLoadAction(MTLLoadAction::Clear);attachment.setStoreAction(MTLStoreAction::Store);attachment.setClearColor(MTLClearColor{red:0.,green:0.,blue:0.,alpha:0.});
        let cb=gfx::gpu().queue.0.commandBuffer().unwrap();let enc=cb.renderCommandEncoderWithDescriptor(&pass).unwrap();enc.setRenderPipelineState(&pipeline);enc.setVertexBuffer_offset_atIndex(Some(&colors),0,7);enc.setVertexBuffer_offset_atIndex(Some(&argument_buffers[0]),0,12);enc.useResource_usage_stages(ProtocolObject::from_ref(&*texture),MTLResourceUsage::Read,MTLRenderStages::Vertex);enc.drawPrimitives_vertexStart_vertexCount_instanceCount_baseInstance(MTLPrimitiveType::Triangle,0,3,1,1);enc.endEncoding();cb.commit();cb.waitUntilCompleted();assert_eq!(cb.status(),MTLCommandBufferStatus::Completed);
        let mut guarded=[-99f32;8];target.getBytes_bytesPerRow_fromRegion_mipmapLevel(NonNull::new(guarded.as_mut_ptr().add(2)).unwrap().cast(),16,MTLRegion{origin:MTLOrigin{x:0,y:0,z:0},size:MTLSize{width:1,height:1,depth:1}},0);assert_eq!(&guarded[..2],&[-99f32;2]);assert_eq!(&guarded[6..],&[-99f32;2]);results.push(<[f32;4]>::try_from(&guarded[2..6]).unwrap());
    }
    assert_eq!(results[0],results[1],"dynamic body and ordinary entry render differently");results[0]
}}

#[test]
#[ignore="requires real Metal GPU; isolated dynamic-library cache and exact pixel/ABI readbacks"]
fn gpu_dynamic_vertex_disk_reload_source_identity_corruption_and_relocation() {
    objc2::rc::autoreleasepool(|_| {
        crate::kernel::KERNEL.aspace.get_or_init(||crate::mem::AddressSpace::reserve().unwrap());
        static NEXT:AtomicU64=AtomicU64::new(0);let root=std::env::temp_dir().join(format!("fc27-vertex-cache-{}-{}",std::process::id(),NEXT.fetch_add(1,Relaxed)));std::fs::create_dir_all(&root).unwrap();
        let key=0x152001;let t=fixture(false);let source=normal(&t,key);
        let first=compile_at(&t,key,&source,&root).expect("dynamic vertex failed to compile");assert!(!first.disk_hit);assert_eq!(render_and_compare_fallback(&first),[0.25,0.5,0.75,1.]);
        let fresh=compile_at(&t,key,&source,&root).unwrap();assert!(fresh.disk_hit);assert_eq!(render_and_compare_fallback(&fresh),[0.25,0.5,0.75,1.]);
        let changed=fixture(true);let other=compile_at(&changed,key,&normal(&changed,key),&root).unwrap();assert!(!other.disk_hit);assert_ne!(first.path,other.path);assert_eq!(render_and_compare_fallback(&other),[1.25,0.5,0.75,1.]);
        let corrupt=root.join("corrupt");std::fs::create_dir(&corrupt).unwrap();std::fs::write(corrupt.join(first.path.file_name().unwrap()),b"invalid cached Metal library").unwrap();
        let repaired=compile_at(&t,key,&source,&corrupt).unwrap();assert!(!repaired.disk_hit);assert_eq!(render_and_compare_fallback(&repaired),[0.25,0.5,0.75,1.]);assert!(compile_at(&t,key,&source,&corrupt).unwrap().disk_hit);
        let relocated=root.join("relocated");std::fs::create_dir(&relocated).unwrap();std::fs::copy(&first.path,relocated.join(first.path.file_name().unwrap())).unwrap();
        let moved=compile_at(&t,key,&source,&relocated).unwrap();assert!(!moved.disk_hit,"relocated absolute installName was accepted");assert_eq!(render_and_compare_fallback(&moved),[0.25,0.5,0.75,1.]);
        let bytes=std::fs::read(&first.path).unwrap();let blocked=root.join("blocked.metallib");std::fs::create_dir(&blocked).unwrap();assert!(!store(&blocked,&first.state.library.0));assert!(blocked.is_dir());assert_eq!(std::fs::read(&first.path).unwrap(),bytes);
        drop((first,fresh,other,repaired,moved));std::fs::remove_dir_all(root).unwrap();
        eprintln!("[verify] dynamic vertex disk reload, source identity, corruption recovery, relocation invalidation, failed-write preservation and ordinary fallback texture/sampler ABI pixels+guards passed");
    });
}
