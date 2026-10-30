// Runtime NVN descriptor handles use sparse 20-bit texture and 12-bit sampler
// IDs. Tier-2 Metal argument buffers preserve per-lane selection without a
// guessed constant-buffer binding or a dummy texture for unknown handles.
use super::{Replay,shaders,textures,STATE};
use crate::gfx::{self,Shared};
use nxshader::TexKind;
use objc2_metal::*;
use nxshader::msl::{DYNAMIC_TEXTURE_CAPACITY,DYNAMIC_SAMPLER_ARGUMENT,DYNAMIC_LOOKUP_ARGUMENT,DYNAMIC_COUNTS_ARGUMENT};

// Reflection describes a function's immutable argument layout. Encoding into
// each new allocation still happens under this lock, so reusing the encoder
// cannot race another user or change argument bytes already submitted to GPU.
#[derive(Default)]
pub(super) struct EncoderCache(std::sync::Mutex<std::collections::HashMap<usize,Shared<objc2::runtime::ProtocolObject<dyn MTLArgumentEncoder>>>>);

impl EncoderCache {
    pub(super) fn with<T>(&self,function:&shaders::Function,index:usize,encode:impl FnOnce(&objc2::runtime::ProtocolObject<dyn MTLArgumentEncoder>)->T)->T {
        static ENABLED:std::sync::LazyLock<bool>=std::sync::LazyLock::new(||std::env::var_os("NX_ARGUMENT_ENCODER_CACHE").is_some());
        if !*ENABLED {let encoder=unsafe {function.newArgumentEncoderWithBufferIndex(index)};return encode(&encoder);}
        let mut cache=self.0.lock().unwrap();
        let encoder=cache.entry(index).or_insert_with(||Shared(unsafe {function.newArgumentEncoderWithBufferIndex(index)}));
        encode(&encoder.0)
    }
}

pub(super) struct Bindings {
    pub allocation:super::state::Alloc,
    pub textures:Vec<Shared<gfx::Texture>>,
    samplers:Vec<Shared<textures::Sampler>>,
}

impl Bindings {
    pub fn retain(&self,cb:&gfx::CommandBuffer) {
        let textures=self.textures.clone();let samplers=self.samplers.clone();
        let handler=block2::RcBlock::new(move |_:std::ptr::NonNull<gfx::CommandBuffer>| {let _=(&textures,&samplers);});
        unsafe {cb.addCompletedHandler(block2::RcBlock::as_ptr(&handler));}
    }
}

impl Replay<'_> {
    pub(super) fn stage_static_arguments(&mut self,c:&shaders::Compiled,textures:&[(u32,Shared<gfx::Texture>,u32,Shared<textures::Sampler>)])->Result<Option<Bindings>,&'static str> {
        if !c.iface.argument_textures {return Ok(None);}
        let dev=&gfx::gpu().device.0;
        if dev.argumentBuffersSupport()!=MTLArgumentBuffersTier::Tier2 {return Err("large static sampler bank needs Tier-2 Metal argument buffers");}
        if textures.len()>256 || textures.len()>dev.maxArgumentBufferSamplerCount() {return Err("static sampler bank exceeds Metal capacity");}
        c.argument_encoders.with(&c.func.0,12,|encoder| {
        let allocation=self.e.arena.alloc(encoder.encodedLength());
        if allocation.offset%encoder.alignment()!=0 {return Err("static argument buffer alignment mismatch");}
        unsafe {std::ptr::write_bytes(allocation.ptr,0,encoder.encodedLength());encoder.setArgumentBuffer_offset(Some(&allocation.buffer.0),allocation.offset);}
        let mut bound=Bindings{allocation,textures:Vec::new(),samplers:Vec::new()};
        for (slot,texture,sampler_slot,sampler) in textures {
            if *slot!=*sampler_slot || *slot as usize>=textures.len() {return Err("static sampler/texture argument layout mismatch");}
            unsafe {encoder.setTexture_atIndex(Some(&texture.0),*slot as usize*2);encoder.setSamplerState_atIndex(Some(&sampler.0),*slot as usize*2+1);}
            bound.textures.push(texture.clone());bound.samplers.push(sampler.clone());
        }
        Ok(Some(bound))
        })
    }

    pub(super) fn stage_bindless(&mut self,st:&super::GState,c:&shaders::Compiled)->Result<Option<Bindings>,&'static str> {
        if c.iface.dynamic_textures==0 {return Ok(None);}
        let dev=&gfx::gpu().device.0;
        if dev.argumentBuffersSupport()!=MTLArgumentBuffersTier::Tier2 {return Err("runtime texture selection needs Tier-2 Metal argument buffers");}
        let (mut descriptors,mut samplers)={
            let s=STATE.lock().unwrap();
            let textures=s.registered_textures.iter().filter(|((pool,_),_)|*pool==st.texture_pool).map(|((_,id),b)|(*id,*b,textures::TexKey::of_in(b,&s))).collect::<Vec<_>>();
            let samplers=s.registered_samplers.iter().filter(|((pool,_),_)|*pool==st.sampler_pool).map(|((_,id),b)|(*id,*b)).collect::<Vec<_>>();
            (textures,samplers)
        };
        descriptors.sort_by_key(|(id,_,_)|*id);samplers.sort_by_key(|(id,_)|*id);
        if samplers.len()>256 || samplers.len()>dev.maxArgumentBufferSamplerCount() {return Err("dynamic sampler bank exceeds Metal capacity");}
        c.argument_encoders.with(&c.func.0,2,|encoder| {
        let allocation=self.e.arena.alloc(encoder.encodedLength());
        if allocation.offset%encoder.alignment()!=0 {return Err("dynamic argument buffer alignment mismatch");}
        unsafe {std::ptr::write_bytes(allocation.ptr,0,encoder.encodedLength());encoder.setArgumentBuffer_offset(Some(&allocation.buffer.0),allocation.offset);}
        let mut banks:[Vec<[u32;2]>;3]=std::array::from_fn(|_|Vec::new());
        let mut views:[std::collections::HashMap<usize,u32>;2]=std::array::from_fn(|_|Default::default());
        let mut bound=Bindings{allocation,textures:Vec::new(),samplers:Vec::new()};
        for (id,b,key) in descriptors {
            let Some(key)=key else {continue;};
            // Other dimensionalities and integer samplers cannot be selected
            // through a float/depth 2D instruction. A selection faults on GPU.
            if key.target!=1 {continue;}
            if key.format==49 {continue;}
            let depth=matches!(key.format,50..=54);
            let bank=usize::from(depth);
            if c.iface.dynamic_textures&(1<<bank)==0 {continue;}
            if !self.e.textures.bindless_available(key,&b) {continue;}
            let Some((key,tex))=self.texture_keyed(Some(key),&b,false) else {continue;};
            if !depth && shaders::integer_kind(tex.0.pixelFormat())!=0 {continue;}
            let kind=if depth {TexKind::Depth2D} else {TexKind::T2D};
            let Some(tex)=self.e.textures.view(key,&b,&tex,kind) else {continue;};
            if tex.0.textureType()!=MTLTextureType::Type2D {continue;}
            // Distinct sparse NVN IDs may name the exact same Metal view.
            // Preserve every ID in the sorted lookup, sharing only identical
            // retained view objects; format, subresource and swizzle views
            // with different identities remain independent entries.
            let identity=objc2::rc::Retained::as_ptr(&tex.0)as *const()as usize;
            let index=if let Some(&index)=views[bank].get(&identity){index}else{
                if views[bank].len()>=DYNAMIC_TEXTURE_CAPACITY{return Err("dynamic texture bank exceeds Metal capacity");}
                let index=views[bank].len()as u32;views[bank].insert(identity,index);
                unsafe{encoder.setTexture_atIndex(Some(&tex.0),bank*DYNAMIC_TEXTURE_CAPACITY+index as usize);}
                bound.textures.push(tex);index
            };
            banks[bank].push([id,index]);
        }
        for (id,b) in samplers {
            let index=banks[2].len();let sampler=self.e.textures.sampler(Some(&b));
            banks[2].push([id,index as u32]);unsafe {encoder.setSamplerState_atIndex(Some(&sampler.0),DYNAMIC_SAMPLER_ARGUMENT+index);}
            bound.samplers.push(sampler);
        }
        for (bank,items) in banks.iter().enumerate() {
            let bytes=unsafe {std::slice::from_raw_parts(items.as_ptr().cast::<u8>(),items.len()*8)};
            let table=self.e.arena.alloc_bytes(bytes);
            unsafe {encoder.setBuffer_offset_atIndex(Some(&table.buffer.0),table.offset,DYNAMIC_LOOKUP_ARGUMENT+bank);}
        }
        unsafe {encoder.constantDataAtIndex(DYNAMIC_COUNTS_ARGUMENT).cast::<[u32;4]>().as_ptr().write([banks[0].len() as u32,banks[1].len() as u32,banks[2].len() as u32,0]);}
        Ok(Some(bound))
        })
    }
}

#[cfg(all(test,target_os="macos"))]
mod tests {
    use super::*;
    use objc2::{runtime::ProtocolObject};
    use objc2_foundation::{NSString,NSRange};
    use std::ptr::NonNull;

    #[test]
    #[ignore="requires a real Metal GPU; run serially"]
    fn gpu_wide_dynamic_bank_preserves_last_slot_sparse_ids_and_depth_layout(){
        objc2::rc::autoreleasepool(|_|unsafe{
            let dev=&gfx::gpu().device.0;assert_eq!(dev.argumentBuffersSupport(),MTLArgumentBuffersTier::Tier2);
            let source=format!("{}{}\nkernel void wide_probe(device float4* output [[buffer(0)]],constant DynamicTexturesWide& bank [[buffer(2)]]){{uint first=bindless_find(bank.color_lookup,bank.counts.x,0x123u);uint last=bindless_find(bank.color_lookup,bank.counts.x,0xfffffu);uint filter=bindless_find(bank.sampler_lookup,bank.counts.z,0xabcu);output[0]=bank.color[first].sample(bank.samplers[filter],float2(0.5f));output[1]=bank.color[last].sample(bank.samplers[filter],float2(0.5f));uint d=bindless_find(bank.depth_lookup,bank.counts.y,0x789u);output[2]=float4(bank.depth[d].sample(bank.samplers[filter],float2(0.5f)));}}",nxshader::prelude(),nxshader::msl::dynamic_texture_layout());
            let lib=dev.newLibraryWithSource_options_error(&NSString::from_str(&source),None).unwrap();let function=lib.newFunctionWithName(&NSString::from_str("wide_probe")).unwrap();let pipeline=dev.newComputePipelineStateWithFunction_error(&function).unwrap();let arguments=function.newArgumentEncoderWithBufferIndex(2);
            let buffer=dev.newBufferWithLength_options(arguments.encodedLength(),MTLResourceOptions::StorageModeShared).unwrap();std::ptr::write_bytes(buffer.contents().as_ptr(),0,buffer.length());arguments.setArgumentBuffer_offset(Some(&buffer),0);
            let descriptor=MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(MTLPixelFormat::RGBA32Float,1,1,false);descriptor.setStorageMode(MTLStorageMode::Shared);descriptor.setUsage(MTLTextureUsage::ShaderRead);
            let mut textures=Vec::new();
            for index in 0..DYNAMIC_TEXTURE_CAPACITY{
                let tex=dev.newTextureWithDescriptor(&descriptor).unwrap();let color=[index as f32,2.0,3.0,1.0];tex.replaceRegion_mipmapLevel_withBytes_bytesPerRow(MTLRegion{origin:MTLOrigin{x:0,y:0,z:0},size:MTLSize{width:1,height:1,depth:1}},0,NonNull::from(&color).cast(),16);arguments.setTexture_atIndex(Some(&tex),index);textures.push(tex);
            }
            let depth_descriptor=MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(MTLPixelFormat::Depth32Float,1,1,false);depth_descriptor.setStorageMode(MTLStorageMode::Private);depth_descriptor.setUsage(MTLTextureUsage::ShaderRead|MTLTextureUsage::RenderTarget);let depth=dev.newTextureWithDescriptor(&depth_descriptor).unwrap();arguments.setTexture_atIndex(Some(&depth),DYNAMIC_TEXTURE_CAPACITY*2-1);
            let sampler_descriptor=MTLSamplerDescriptor::new();sampler_descriptor.setSupportArgumentBuffers(true);let sampler=dev.newSamplerStateWithDescriptor(&sampler_descriptor).unwrap();arguments.setSamplerState_atIndex(Some(&sampler),DYNAMIC_SAMPLER_ARGUMENT);
            let mut tables=Vec::new();
            for (bank,items)in [vec![[0x123u32,0],[0xfffff,(DYNAMIC_TEXTURE_CAPACITY-1)as u32]],vec![[0x789,(DYNAMIC_TEXTURE_CAPACITY-1)as u32]],vec![[0xabc,0]]].iter().enumerate(){
                let table=dev.newBufferWithLength_options(items.len()*8,MTLResourceOptions::StorageModeShared).unwrap();std::ptr::copy_nonoverlapping(items.as_ptr().cast::<u8>(),table.contents().as_ptr().cast::<u8>(),items.len()*8);arguments.setBuffer_offset_atIndex(Some(&table),0,DYNAMIC_LOOKUP_ARGUMENT+bank);tables.push(table);
            }
            arguments.constantDataAtIndex(DYNAMIC_COUNTS_ARGUMENT).cast::<[u32;4]>().as_ptr().write([2,1,1,0]);
            let output=dev.newBufferWithLength_options(48,MTLResourceOptions::StorageModeShared).unwrap();let command=gfx::gpu().queue.0.commandBuffer().unwrap();
            let pass=MTLRenderPassDescriptor::new();let attachment=pass.depthAttachment();attachment.setTexture(Some(&depth));attachment.setLoadAction(MTLLoadAction::Clear);attachment.setStoreAction(MTLStoreAction::Store);attachment.setClearDepth(0.25);command.renderCommandEncoderWithDescriptor(&pass).unwrap().endEncoding();
            let encoder=command.computeCommandEncoder().unwrap();encoder.setComputePipelineState(&pipeline);encoder.setBuffer_offset_atIndex(Some(&output),0,0);encoder.setBuffer_offset_atIndex(Some(&buffer),0,2);
            for texture in textures.iter().chain(std::iter::once(&depth)){encoder.useResource_usage(ProtocolObject::from_ref(&**texture),MTLResourceUsage::Read);}
            for table in &tables{encoder.useResource_usage(ProtocolObject::from_ref(&**table),MTLResourceUsage::Read);}
            encoder.dispatchThreadgroups_threadsPerThreadgroup(MTLSize{width:1,height:1,depth:1},MTLSize{width:1,height:1,depth:1});encoder.endEncoding();command.commit();command.waitUntilCompleted();assert_eq!(command.status(),MTLCommandBufferStatus::Completed);
            let actual=std::slice::from_raw_parts(output.contents().as_ptr().cast::<f32>(),12);assert_eq!(&actual[..4],&[0.0,2.0,3.0,1.0]);assert_eq!(&actual[4..8],&[(DYNAMIC_TEXTURE_CAPACITY-1)as f32,2.0,3.0,1.0]);assert_eq!(&actual[8..],&[0.25;4]);
            eprintln!("[verify]{} distinct live color views, first+last sparse20bitIDs, lastdepthslot, sampler andlookup/countABI read correctly",textures.len());
        });
    }

    #[test]
    #[ignore="requires a real Metal GPU"]
    fn gpu_reused_argument_encoders_preserve_inflight_texture_sampler_tables_and_compute_counters() {
        objc2::rc::autoreleasepool(|_|unsafe {
            let device=&gfx::gpu().device.0;
            let source="#include <metal_stdlib>\nusing namespace metal;struct Table {texture2d<float> image [[id(0)]];sampler filter [[id(1)]];};kernel void cache_probe(device float4* output [[buffer(0)]],constant Table& a [[buffer(2)]],constant Table& b [[buffer(12)]]) {output[0]=a.image.sample(a.filter,float2(0.5f));output[1]=b.image.sample(b.filter,float2(0.5f));}";
            let library=device.newLibraryWithSource_options_error(&NSString::from_str(source),None).unwrap();
            let function=library.newFunctionWithName(&NSString::from_str("cache_probe")).unwrap();
            let pipeline=device.newComputePipelineStateWithFunction_error(&function).unwrap();
            let descriptor=MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(MTLPixelFormat::RGBA32Float,1,1,false);descriptor.setStorageMode(MTLStorageMode::Shared);descriptor.setUsage(MTLTextureUsage::ShaderRead);
            let colors=[[1f32,2.0,3.0,4.0],[5f32,6.0,7.0,8.0]];
            let textures=colors.iter().map(|color| {
                let texture=device.newTextureWithDescriptor(&descriptor).unwrap();
                texture.replaceRegion_mipmapLevel_withBytes_bytesPerRow(MTLRegion{origin:MTLOrigin{x:0,y:0,z:0},size:MTLSize{width:1,height:1,depth:1}},0,NonNull::new(color.as_ptr().cast_mut()).unwrap().cast(),16);texture
            }).collect::<Vec<_>>();
            let sampler_descriptor=MTLSamplerDescriptor::new();sampler_descriptor.setSupportArgumentBuffers(true);
            let sampler=device.newSamplerStateWithDescriptor(&sampler_descriptor).unwrap();
            let event=device.newSharedEvent().unwrap();
            let cache=EncoderCache::default();let mut commands=Vec::new();let mut outputs=Vec::new();let mut keep=Vec::new();let mut counters=Vec::new();
            for round in 0..2 {
                let output=device.newBufferWithLength_options(32,MTLResourceOptions::StorageModeShared).unwrap();std::ptr::write_bytes(output.contents().as_ptr(),0,32);
                let command=gfx::gpu().queue.0.commandBuffer().unwrap();command.encodeWaitForEvent_value(ProtocolObject::from_ref(&*event),1);
                let counter=gfx::timestamp_samples(2);
                let encoder=gfx::begin_compute_sampled(&command,counter.as_ref().map(|c|&*c.0)).unwrap();encoder.setComputePipelineState(&pipeline);encoder.setBuffer_offset_atIndex(Some(&output),0,0);
                for (slot,index) in [2usize,12].into_iter().enumerate() {
                    let argument=cache.with(&function,index,|arguments| {
                        let buffer=device.newBufferWithLength_options(arguments.encodedLength().max(16),MTLResourceOptions::StorageModeShared).unwrap();std::ptr::write_bytes(buffer.contents().as_ptr(),0,buffer.length());
                        arguments.setArgumentBuffer_offset(Some(&buffer),0);arguments.setTexture_atIndex(Some(&textures[(round+slot)%2]),0);arguments.setSamplerState_atIndex(Some(&sampler),1);buffer
                    });
                    encoder.setBuffer_offset_atIndex(Some(&argument),0,index);keep.push(argument);
                }
                for texture in &textures {encoder.useResource_usage(ProtocolObject::from_ref(&**texture),MTLResourceUsage::Read);}
                encoder.dispatchThreadgroups_threadsPerThreadgroup(MTLSize{width:1,height:1,depth:1},MTLSize{width:1,height:1,depth:1});encoder.endEncoding();command.commit();
                commands.push(command);outputs.push(output);counters.push(counter);
            }
            // Both tables have been encoded before either GPU command can run.
            event.setSignaledValue(1);
            for (round,command) in commands.iter().enumerate() {
                command.waitUntilCompleted();assert_eq!(command.status(),MTLCommandBufferStatus::Completed);
                let actual=std::slice::from_raw_parts(outputs[round].contents().as_ptr().cast::<f32>(),8);
                assert_eq!(&actual[..4],&colors[round]);assert_eq!(&actual[4..],&colors[1-round]);
                if let Some(counter)=&counters[round] {
                    let data=counter.0.resolveCounterRange(NSRange{location:0,length:2}).unwrap();assert_eq!(data.length(),16);
                    let mut values=[0u64;2];data.getBytes_length(NonNull::from(&mut values).cast(),16);assert!(values[0]>0 && values[0]!=u64::MAX && values[1]>=values[0] && values[1]!=u64::MAX);
                }
            }
            std::hint::black_box(keep);
        });
    }
}
