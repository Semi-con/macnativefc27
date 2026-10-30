// Mush8 - state.rs
// Constantin Clerc (c22dev) - 0.1
use super::shaders::Compiled;
use super::textures::compare_fn;
use crate::gfx::{self, Shared};
use objc2_metal::*;
use objc2::rc::Retained;
use objc2_foundation::{NSArray, NSString, NSURL};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, LazyLock, Mutex};

pub type Pipeline = objc2::runtime::ProtocolObject<dyn MTLRenderPipelineState>;

use super::super::fx::{FastMap, FastSet};
pub type DepthState = objc2::runtime::ProtocolObject<dyn MTLDepthStencilState>;

#[derive(Clone, Copy)]
pub struct GState {
    pub program: [(u64, u64); 8],
    pub program_len: u8,
    pub attribs: [u32; 16],
    pub streams: [(u32, u32); 16],
    pub vbufs: [(u64, u64); 16],
    pub ubos: [[(u64, u64); 15]; 3],
    pub textures: [[u64; 32]; 3],
    pub images: [[u64;8];3],
    pub separate_textures: [[u64;128];3],
    pub separate_samplers: [[u64;128];3],
    pub storage: [[(u64, u64); 16]; 3],
    pub color_state: u32,
    pub channel_mask: u32,
    pub blend: [u64; 8],
    pub depth_stencil: u64,
    pub polygon: u32,
    pub viewport: Option<[f32; 4]>,
    pub scissor: Option<[i32; 4]>,
    pub blend_color: [f32; 4],
    pub alpha_ref: f32,
    pub multisample: u32,
    pub sample_mask: u32,
    pub scratch_memory: (u64,u64,u64),
    pub stencil_ref: [u32; 2],
    pub stencil_read: [u32; 2],
    pub stencil_write: [u32; 2],
    pub texture_pool: u64,
    pub sampler_pool: u64,
    pub depth_clamp: bool,
    pub rasterizer_discard: bool,
    pub depth_range: (f32, f32),
    pub polygon_offset: [f32; 3],
    pub render_enable: bool,
}

impl Default for GState {
    fn default() -> Self {
        Self {
            program: [(0, 0); 8],
            program_len: 0,
            attribs: [0; 16],
            streams: [(0, 0); 16],
            vbufs: [(0, 0); 16],
            ubos: [[(0, 0); 15]; 3],
            textures: [[0; 32]; 3],
            images: [[0;8];3],
            separate_textures: [[0;128];3],
            separate_samplers: [[0;128];3],
            storage: [[(0, 0); 16]; 3],
            color_state: 0,
            channel_mask: 0xFFFF_FFFF,
            blend: std::array::from_fn(|i| i as u64),
            depth_stencil: 0,
            polygon: 2 << 3 | 1 << 2,
            viewport: None,
            scissor: None,
            blend_color: [0.0; 4],
            alpha_ref: 0.0,
            multisample: 5,
            sample_mask: u32::MAX,
            scratch_memory: (0,0,0),
            stencil_ref: [0; 2],
            stencil_read: [0xFF; 2],
            stencil_write: [0xFF; 2],
            texture_pool: 0,
            sampler_pool: 0,
            depth_clamp: false,
            rasterizer_discard: false,
            depth_range: (0.0, 1.0),
            polygon_offset: [0.0; 3],
            render_enable: true,
        }
    }
}

impl GState {
    pub fn fragment_options(&self,targets:&Targets)->Result<u32,&'static str> {
        let alpha=self.color_state>>24&15;
        let mut options=if alpha!=0 && alpha!=8 {nxshader::FRAGMENT_ALPHA_TEST} else {0};
        for (i,target) in targets.colors.iter().enumerate() {
            if target.is_none() || self.channel_mask>>(4*i)&15==0 || self.color_state>>(8+i)&1==0 {continue;}
            let factors=(self.blend[i]>>32) as u32;
            if (0..4).any(|c|(16..=19).contains(&(factors>>(8*c)&127))) {
                if i!=0 {return Err("secondary blend sources on nonzero render targets are unsupported");}
                options|=nxshader::FRAGMENT_DUAL_SOURCE;
            }
        }
        Ok(options)
    }
    pub fn validate_single_sample(&self) -> Result<(),&'static str> {
        let w=self.multisample;
        // SDK55 defaults to word 5: rasterization enabled, dither enabled,
        // alpha-to-coverage disabled and zero (single) sample counts.
        if (w>>8&31)>1 || (w>>15&31)>1 {return Err("multisample render targets require MSAA texture/pipeline/resolve support");}
        if w&2!=0 || w&8!=0 || w&(3<<13)!=0 || w&(1<<20)!=0 {return Err("alpha-to-coverage or custom coverage/sample-location state is unsupported");}
        Ok(())
    }
    pub fn stages(&self) -> &[(u64, u64)] {
        &self.program[..self.program_len as usize]
    }

    pub fn set_program(&mut self, stages: &[(u64, u64)]) {
        let n = stages.len().min(self.program.len());
        self.program[..n].copy_from_slice(&stages[..n]);
        self.program_len = n as u8;
    }

    pub fn bind_program(&mut self,stages:&[(u64,u64)],mask:u32)->Result<(),&'static str> {
        if mask&!0x3f!=0 {return Err("invalid NVN shader-stage mask");}
        let mut next=[(0,0);8];let mut count=0;let mut selected=0;
        for &(code,control) in self.stages() {
            let bit=super::shaders::stage_bit(code).ok_or("bound program has an unknown SPH stage")?;
            if bit&mask==0 {next[count]=(code,control);count+=1;}
        }
        for &(code,control) in stages {
            let bit=super::shaders::stage_bit(code).ok_or("program has an unknown SPH stage")?;
            if bit&mask!=0 {
                if bit&selected!=0 {return Err("multiple shaders for one NVN stage are unsupported");}
                if count>=next.len() {return Err("too many bound NVN shader stages");}
                selected|=bit;next[count]=(code,control);count+=1;
            }
        }
        self.set_program(&next[..count]);Ok(())
    }
}

#[cfg(test)]
mod coverage_tests {
    use super::*;
    #[test]
    fn sdk_default_is_single_sample_and_optional_coverage_is_rejected() {
        let mut s=GState::default();assert!(s.validate_single_sample().is_ok());
        for value in [0,5,5|(1<<8)|(1<<15)] {s.multisample=value;assert!(s.validate_single_sample().is_ok());}
        for value in [5|2,5|8,5|(2<<8),5|(4<<15),5|(1<<13),5|(1<<20)] {s.multisample=value;assert!(s.validate_single_sample().is_err());}
    }
    #[test]
    fn partial_program_bindings_preserve_graphics_across_compute_and_unbind_selected_stages() {
        let shader=|stage:u32| {let mut b=vec![0u8;128];b[..4].copy_from_slice(&0x12345678u32.to_le_bytes());b[0x30..0x34].copy_from_slice(&(stage<<10).to_le_bytes());b};
        let vs=shader(1);let fs=shader(5);let replacement=shader(5);let compute=vec![0u8;128];let mut state=GState::default();
        let v=(vs.as_ptr() as u64,1);let f=(fs.as_ptr() as u64,2);let f2=(replacement.as_ptr() as u64,3);let c=(compute.as_ptr() as u64,4);
        state.bind_program(&[v,f],3).unwrap();state.bind_program(&[c],0x20).unwrap();assert_eq!(state.stages(),&[v,f,c]);
        state.bind_program(&[f2],2).unwrap();assert_eq!(state.stages(),&[v,c,f2]);state.bind_program(&[],0x20).unwrap();assert_eq!(state.stages(),&[v,f2]);
        state.bind_program(&[],0).unwrap();assert_eq!(state.stages(),&[v,f2]);state.bind_program(&[],2).unwrap();assert_eq!(state.stages(),&[v]);
        assert!(state.bind_program(&[f,f2],2).is_err());assert!(state.bind_program(&[],0x40).is_err());
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Res {
    pub cb: [u64; 18],
    pub vb: [u64; 16],
    pub vb_stride: [u32; 16],
    pub vb_divisor: [u32; 16],
    pub attr_format: [u32; 16],
    pub attr_stream: [u32; 16],
    pub attr_offset: [u32; 16],
    pub base_vertex: i32,
    pub base_instance: u32,
    pub flags: u32,
    pub alpha_ref: f32,
    pub alpha_func: u32,
    pub heap_base: u64,
    pub heap_gpu: [u64; gfx::MAX_HEAP_CHUNKS],
    pub heap_size: u64,
    pub heap_chunk_size: u64,
    pub zero_gpu: u64,
    pub pool_spans: u64,
    pub pool_span_count: u64,
    pub fault_gpu:u64,
    pub shader_hash:u64,
    pub pool_root_base:u64,
    pub pool_root_count:u64,
    pub pool_roots:u64,
    pub pool_pages:u64,
}

const _: () = assert!(size_of::<Res>() <= 4096);

impl Res {
    pub fn set_pools(&mut self, arena:&mut Arena, pools: &[Arc<gfx::PoolMemory>]) -> Alloc {
        let mut spans:Vec<[u64;3]>=Vec::new();
        for c in pools.iter().flat_map(|p| &p.chunks) {
            let gpu=c.gpu+c.offset as u64;
            if let Some(last)=spans.last_mut().filter(|last|last[0].checked_add(last[1])==Some(c.base) && last[2].checked_add(last[1])==Some(gpu)) {
                last[1]+=c.size;
            } else {spans.push([c.base,c.size,gpu]);}
        }
        if super::TRACE_DRAWS.load(std::sync::atomic::Ordering::Relaxed) {eprintln!("[frame] shader address table: {} spans",spans.len());}
        self.pool_span_count=spans.len() as u64;
        let bytes=unsafe {std::slice::from_raw_parts(spans.as_ptr().cast::<u8>(),spans.len()*24)};
        let allocation=arena.alloc_bytes(bytes);self.pool_spans=allocation.gpu;allocation
    }
}

#[cfg(all(test, target_os = "macos"))]
mod segmented_heap_tests {
    use super::*;
    use objc2::runtime::ProtocolObject;

    #[test]
    #[ignore="requires a real Metal GPU"]
    fn gpu_dynamic_pool_table_resolves_more_than_128_aliases_and_preserves_holes() {
        objc2::rc::autoreleasepool(|_| unsafe {
            let page=crate::mem::PAGE as usize;
            let memory=libc::mmap(std::ptr::null_mut(),page,libc::PROT_READ|libc::PROT_WRITE,libc::MAP_PRIVATE|libc::MAP_ANON,-1,0);
            assert_ne!(memory,libc::MAP_FAILED);let cpu=memory as u64;(memory as *mut u32).write(42);
            let mut pools=gfx::pool_memory(&[(0xa500,cpu,page as u64)]).unwrap();
            let base=0x20_0000_0000u64;let merged=base+200*8192;(memory as *mut u32).add(1024).write(99);let aliases=(0..160).map(|i|(base+i*8192,cpu,4096)).chain([(merged,cpu,4096),(merged+4096,cpu+4096,4096)]).collect::<Vec<_>>();gfx::add_pool_aliases(&mut pools,&aliases).unwrap();
            let mut arena=Arena::default();let mut res:Res=std::mem::zeroed();res.zero_gpu=gfx::gpu().zeros.0.gpuAddress();let table=res.set_pools(&mut arena,&pools);assert_eq!(res.pool_span_count,162);assert!(size_of::<Res>()<=4096);
            let source=format!("{}\nkernel void large_table(constant Res& r [[buffer(0)]],device uint* out [[buffer(1)]]) {{out[0]=*gmem(r,{base}ul);out[1]=*gmem(r,{}ul);out[2]=*gmem(r,{}ul);out[3]=uint(gmem_gpu(r,{}ul)!=0ul);*gmem(r,{}ul)=77u;GmemCache cache={{0ul,0ul,0ul}};out[4]=*gmem_cached(cache,r,{base}ul);out[5]=*gmem_cached(cache,r,{base}ul+4ul);out[6]=uint(gmem_gpu_cached(cache,r,{base}ul+159ul*8192ul+4096ul)!=0ul);out[7]=*gmem_cached(cache,r,{base}ul+159ul*8192ul+4ul);uint sum=0u;for(uint i=0u;i<160u;i++) sum+=*gmem_cached(cache,r,{base}ul+ulong(i)*8192ul);out[8]=sum;out[9]=*gmem_cached(cache,r,{base}ul+159ul*8192ul+4096ul);out[10]=*gmem_cached(cache,r,{merged}ul);out[11]=*gmem_cached(cache,r,{merged}ul+4096ul);}}",nxshader::prelude(),base+79*8192,base+159*8192,base+159*8192+4096,base+159*8192+4);
            let dev=&gfx::gpu().device.0;let lib=dev.newLibraryWithSource_options_error(&NSString::from_str(&source),None).unwrap();let pipeline=dev.newComputePipelineStateWithFunction_error(&lib.newFunctionWithName(&NSString::from_str("large_table")).unwrap()).unwrap();let output=dev.newBufferWithLength_options(48,MTLResourceOptions::StorageModeShared).unwrap();
            let cb=gfx::gpu().queue.0.commandBuffer().unwrap();let enc=cb.computeCommandEncoder().unwrap();enc.setComputePipelineState(&pipeline);enc.useResource_usage(ProtocolObject::from_ref(&*table.buffer.0),MTLResourceUsage::Read);for chunk in pools.iter().flat_map(|p|&p.chunks) {enc.useResource_usage(ProtocolObject::from_ref(&*chunk.buffer.0),MTLResourceUsage::Read|MTLResourceUsage::Write);}enc.useResource_usage(ProtocolObject::from_ref(&*gfx::gpu().zeros.0),MTLResourceUsage::Read);enc.setBytes_length_atIndex(std::ptr::NonNull::from(&res).cast(),size_of_val(&res),0);enc.setBuffer_offset_atIndex(Some(&output),0,1);enc.dispatchThreadgroups_threadsPerThreadgroup(MTLSize{width:1,height:1,depth:1},MTLSize{width:1,height:1,depth:1});enc.endEncoding();gfx::forget_pool(0xa500);super::super::commit(&cb,&pools);drop(pools);cb.waitUntilCompleted();assert_eq!(cb.status(),MTLCommandBufferStatus::Completed);assert_eq!(std::slice::from_raw_parts(output.contents().as_ptr().cast::<u32>(),12),[42,42,42,0,42,77,0,77,6720,0,42,99]);assert_eq!((memory as *const u32).add(1).read(),77);
        });
    }

    #[test]
    #[ignore = "requires a real Metal GPU"]
    fn gpu_reads_and_writes_discontiguous_native_pools() {
        objc2::rc::autoreleasepool(|_| unsafe {
            let page = crate::mem::PAGE as usize;
            let allocation = libc::mmap(std::ptr::null_mut(), page * 8, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_PRIVATE | libc::MAP_ANON, -1, 0);
            assert_ne!(allocation, libc::MAP_FAILED);
            let base = allocation as u64;
            // Include 4 KiB NVN alignment inside a 16 KiB host page.
            let ranges = [(base + 4096, page as u64 - 4096), (base + page as u64 * 3, page as u64), (base + page as u64 * 6 + 4096, page as u64 - 4096)];
            let input: Vec<_> = ranges.iter().enumerate().rev().map(|(i, &(b, n))| (0x7100 + i as u64, b, n)).collect();
            let pools = gfx::pool_memory(&input).unwrap();
            assert_eq!(pools.iter().map(|p| p.base).collect::<Vec<_>>(), ranges.map(|p| p.0));
            let mut addresses = Vec::new();
            for (i, &(b, n)) in ranges.iter().enumerate() {
                assert!(gfx::pool_buffer_range(&pools, b, n).is_some());
                assert!(gfx::pool_buffer_range(&pools, b + n - 4, 8).is_none());
                for (j, addr) in [b, b + n - 4].into_iter().enumerate() {
                    (addr as *mut u32).write((i * 2 + j + 1) as u32);
                    addresses.push(addr);
                }
            }
            let mut res: Res = std::mem::zeroed();
            res.zero_gpu = gfx::gpu().zeros.0.gpuAddress();
            let mut arena=Arena::default();let table=res.set_pools(&mut arena,&pools);
            assert_eq!(res.pool_span_count, 3);
            let address_constants = addresses.iter().map(|a| format!("{a}ul")).collect::<Vec<_>>().join(",");
            let invalid = ranges[0].0 + ranges[0].1;
            let source = format!("{}\nkernel void pool_probe(constant Res& r [[buffer(0)]], device uint* out [[buffer(1)]]) {{ ulong addresses[6] = {{{address_constants}}}; for(uint i = 0; i < 6; ++i) {{ out[i] = *gmem(r, addresses[i]); if (gmem_gpu(r, addresses[i]) != 0ul) *gmem(r, addresses[i]) = out[i] + 100u; }} out[6] = *gmem(r, {invalid}ul); out[7] = uint(gmem_gpu(r, {invalid}ul) != 0ul); }}", nxshader::prelude());
            let dev = &gfx::gpu().device.0;
            let lib = dev.newLibraryWithSource_options_error(&NSString::from_str(&source), None).unwrap();
            let func = lib.newFunctionWithName(&NSString::from_str("pool_probe")).unwrap();
            let pipeline = dev.newComputePipelineStateWithFunction_error(&func).unwrap();
            let out = dev.newBufferWithLength_options(32, MTLResourceOptions::StorageModeShared).unwrap();
            let cb = gfx::gpu().queue.0.commandBuffer().unwrap();
            let enc = cb.computeCommandEncoder().unwrap();
            enc.setComputePipelineState(&pipeline);
            enc.useResource_usage(ProtocolObject::from_ref(&*table.buffer.0), MTLResourceUsage::Read);
            for c in pools.iter().flat_map(|p| &p.chunks) {
                enc.useResource_usage(ProtocolObject::from_ref(&*c.buffer.0), MTLResourceUsage::Read | MTLResourceUsage::Write);
            }
            enc.useResource_usage(ProtocolObject::from_ref(&*gfx::gpu().zeros.0), MTLResourceUsage::Read|MTLResourceUsage::Write);
            enc.setBytes_length_atIndex(std::ptr::NonNull::from(&res).cast(), size_of::<Res>(), 0);
            enc.setBuffer_offset_atIndex(Some(&out), 0, 1);
            enc.dispatchThreadgroups_threadsPerThreadgroup(MTLSize { width: 1, height: 1, depth: 1 }, MTLSize { width: 1, height: 1, depth: 1 });
            enc.endEncoding();
            for &(id, _, _) in &input { gfx::forget_pool(id); }
            super::super::commit(&cb, &pools);
            drop(pools);
            cb.waitUntilCompleted();
            assert_eq!(cb.status(), MTLCommandBufferStatus::Completed);
            assert_eq!(std::slice::from_raw_parts(out.contents().as_ptr() as *const u32, 8), &[1, 2, 3, 4, 5, 6, 0, 0]);
            for (i, addr) in addresses.into_iter().enumerate() { assert_eq!((addr as *const u32).read(), i as u32 + 101); }
        });
    }

    #[test]
    #[ignore = "requires a real Metal GPU"]
    fn gpu_reads_and_writes_across_heap_chunks() {
        objc2::rc::autoreleasepool(|_| unsafe {
            let page = crate::mem::PAGE as usize;
            let size = page * 5;
            let chunk = page * 2;
            let mem = libc::mmap(std::ptr::null_mut(), size, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_PRIVATE | libc::MAP_ANON, -1, 0);
            assert_ne!(mem, libc::MAP_FAILED);
            std::env::set_var("NX_GPU_HEAP_CHUNK_BYTES", chunk.to_string());
            let mapped = gfx::wrap_heap(mem as u64, size as u64);
            std::env::remove_var("NX_GPU_HEAP_CHUNK_BYTES");
            assert!(mapped);
            let h = gfx::heap().unwrap();
            assert_eq!(h.chunks.iter().map(|c| c.size).collect::<Vec<_>>(), vec![chunk as u64, chunk as u64, page as u64]);
            assert!(h.buffer_range(h.base + chunk as u64 - 4, 8).is_none());
            assert!(h.buffer_range(h.base + (size - 4) as u64, 4).is_some());
            assert!(h.buffer_range(h.base + (size - 4) as u64, 8).is_none());
            let offsets = [0, chunk - 4, chunk, chunk * 2 - 4, chunk * 2, size - 4];
            let values = [17u32, 41, 82, 123, 164, 205];
            let words = std::slice::from_raw_parts_mut(mem as *mut u32, size / 4);
            for (&off, &value) in offsets.iter().zip(&values) {
                words[off / 4] = value;
            }
            let dev = &gfx::gpu().device.0;
            let offsets = offsets.map(|v| format!("{v}ul")).join(", ");
            let source = format!("{}\nkernel void heap_probe(constant Res& r [[buffer(0)]], device uint* out [[buffer(1)]]) {{ ulong offsets[6] = {{{offsets}}}; for (uint i = 0; i < 6; ++i) {{ out[i] = *gmem(r, r.heap_base + offsets[i]); *gmem(r, r.heap_base + offsets[i]) = out[i] + 1000u; }} out[6] = *gmem(r, r.heap_base + r.heap_size); out[7] = *gmem(r, r.heap_base - 4ul); }}", nxshader::prelude());
            let lib = dev.newLibraryWithSource_options_error(&NSString::from_str(&source), None).unwrap();
            let func = lib.newFunctionWithName(&NSString::from_str("heap_probe")).unwrap();
            let pipeline = dev.newComputePipelineStateWithFunction_error(&func).unwrap();
            let out = dev.newBufferWithLength_options(32, MTLResourceOptions::StorageModeShared).unwrap();
            let mut res: Res = std::mem::zeroed();
            res.heap_base = h.base;
            res.heap_size = h.size;
            res.heap_chunk_size = h.chunk_size;
            res.heap_gpu = h.gpu_addresses();
            res.zero_gpu = gfx::gpu().zeros.0.gpuAddress();
            *(gfx::gpu().zeros.0.contents().as_ptr() as *mut u32) = 0;
            let cb = gfx::gpu().queue.0.commandBuffer().unwrap();
            let enc = cb.computeCommandEncoder().unwrap();
            enc.setComputePipelineState(&pipeline);
            for chunk in &h.chunks {
                enc.useResource_usage(ProtocolObject::from_ref(&*chunk.buffer.0), MTLResourceUsage::Read | MTLResourceUsage::Write);
            }
            enc.useResource_usage(ProtocolObject::from_ref(&*gfx::gpu().zeros.0), MTLResourceUsage::Read|MTLResourceUsage::Write);
            enc.setBytes_length_atIndex(std::ptr::NonNull::from(&res).cast(), size_of::<Res>(), 0);
            enc.setBuffer_offset_atIndex(Some(&out), 0, 1);
            enc.dispatchThreadgroups_threadsPerThreadgroup(MTLSize { width: 1, height: 1, depth: 1 }, MTLSize { width: 1, height: 1, depth: 1 });
            enc.endEncoding();
            cb.commit();
            cb.waitUntilCompleted();
            assert_eq!(cb.status(), MTLCommandBufferStatus::Completed);
            assert_eq!(std::slice::from_raw_parts(out.contents().as_ptr() as *const u32, 8), &[17, 41, 82, 123, 164, 205, 0, 0]);
            for (off, value) in [0, chunk - 4, chunk, chunk * 2 - 4, chunk * 2, size - 4].into_iter().zip(values) {
                assert_eq!(words[off / 4], value + 1000);
            }
        });
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Targets {
    pub colors: Vec<Option<MTLPixelFormat>>,
    pub depth: Option<MTLPixelFormat>,
    pub stencil: bool,
}

fn blend_factor(f: u32) -> MTLBlendFactor {
    use MTLBlendFactor as B;
    match f {
        1 => B::Zero,
        2 => B::One,
        3 => B::SourceColor,
        4 => B::OneMinusSourceColor,
        5 => B::SourceAlpha,
        6 => B::OneMinusSourceAlpha,
        7 => B::DestinationAlpha,
        8 => B::OneMinusDestinationAlpha,
        9 => B::DestinationColor,
        10 => B::OneMinusDestinationColor,
        11 => B::SourceAlphaSaturated,
        16 => B::Source1Color,
        17 => B::OneMinusSource1Color,
        18 => B::Source1Alpha,
        19 => B::OneMinusSource1Alpha,
        0x61 => B::BlendColor,
        0x62 => B::OneMinusBlendColor,
        0x63 => B::BlendAlpha,
        0x64 => B::OneMinusBlendAlpha,
        _ => B::One,
    }
}

fn blend_op(e: u32) -> MTLBlendOperation {
    match e {
        2 => MTLBlendOperation::Subtract,
        3 => MTLBlendOperation::ReverseSubtract,
        4 => MTLBlendOperation::Min,
        5 => MTLBlendOperation::Max,
        _ => MTLBlendOperation::Add,
    }
}

fn stencil_op(o: u32) -> MTLStencilOperation {
    use MTLStencilOperation as S;
    match o {
        2 => S::Zero,
        3 => S::Replace,
        4 => S::IncrementClamp,
        5 => S::DecrementClamp,
        6 => S::Invert,
        7 => S::IncrementWrap,
        8 => S::DecrementWrap,
        _ => S::Keep,
    }
}

#[derive(Clone, Copy)]
struct PipeKey {
    words: [u64; 64],
    len: u8,
}

impl PipeKey {
    fn new() -> Self {
        Self { words: [0; 64], len: 0 }
    }

    fn push(&mut self, w: u64) {
        assert!((self.len as usize)<self.words.len(),"pipeline key capacity");
        if (self.len as usize) < self.words.len() {
            self.words[self.len as usize] = w;
            self.len += 1;
        }
    }

    fn buffer_formats(&mut self,formats:&[u32;32]) {
        for pair in formats.chunks_exact(2) {self.push(pair[0]as u64 | (pair[1]as u64)<<32);}
    }

    fn vertex_formats(&mut self,formats:&[u32;16]) {
        for pair in formats.chunks_exact(2) {self.push(pair[0] as u64 | (pair[1] as u64)<<32);}
    }
}

#[cfg(test)]
mod vertex_format_key_tests {
    use super::*;
    #[test]
    fn maximum_render_state_key_retains_every_vertex_input_format() {
        let make=|formats:&[u32;16]|{let mut k=PipeKey::new();for n in 0..23{k.push(n);}k.vertex_formats(formats);k};
        let formats=std::array::from_fn(|i|i as u32*61);let original=make(&formats);assert_eq!(original.len,31);
        for i in 0..16 {let mut changed=formats;changed[i]^=1;assert!(original!=make(&changed),"attribute{i} alias");}
        assert!(original==make(&formats));
    }
    #[test]
    fn buffer_formats_preserve_all32_words_and_presence_in_full_pipeline_key() {
        let make=|formats:Option<&[u32;32]>|{let mut k=PipeKey::new();for n in 0..23{k.push(n);}k.vertex_formats(&[46;16]);k.push(formats.is_some()as u64);if let Some(f)=formats{k.buffer_formats(f);}k};
        let formats=std::array::from_fn(|i|0x80000000u32 | (i as u32*0x10203));let original=make(Some(&formats));assert_eq!(original.len,48);
        assert!(make(None)!=make(Some(&[0;32])));
        for i in 0..32 {let mut changed=formats;changed[i]^=0x80000000;assert!(original!=make(Some(&changed)),"slot{i} direct-tag alias");changed[i]^=0x80000001;assert!(original!=make(Some(&changed)),"slot{i} format alias");}
    }
}

impl PartialEq for PipeKey {
    fn eq(&self, other: &Self) -> bool {
        self.words[..self.len as usize] == other.words[..other.len as usize]
    }
}
impl Eq for PipeKey {}
impl Hash for PipeKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.words[..self.len as usize].hash(state);
    }
}

#[derive(Default)]
pub struct States {
    pipelines: FastMap<PipeKey, Option<Shared<Pipeline>>>,
    pending: FastSet<PipeKey>,
    depth: FastMap<(u64, [u32; 4], bool, bool), Shared<DepthState>>,
}

static DONE: LazyLock<Mutex<Vec<(PipeKey, Option<Shared<Pipeline>>)>>> = LazyLock::new(Default::default);
static PREWARMED: LazyLock<Mutex<HashMap<Recorded, Shared<Pipeline>>>> = LazyLock::new(Default::default);

impl States {
    pub fn pipeline(&mut self, vs: &Arc<Compiled>, vs_key: (u64, u32), fs: Option<&Arc<Compiled>>, fs_key: Option<(u64,u32)>, t: &Targets, st: &GState, wait: bool) -> Result<Shared<Pipeline>, &'static str> {
        self.pipeline_with_buffer_formats(vs,vs_key,fs,fs_key,t,st,wait,None)
    }

    pub(super) fn pipeline_with_buffer_formats(&mut self,vs:&Arc<Compiled>,vs_key:(u64,u32),fs:Option<&Arc<Compiled>>,fs_key:Option<(u64,u32)>,t:&Targets,st:&GState,wait:bool,buffer_formats:Option<[u32;32]>)->Result<Shared<Pipeline>,&'static str> {
        let mut key = PipeKey::new();
        for w in [vs_key.0, vs_key.1 as u64, fs_key.map_or(0,|k|k.0),fs_key.map_or(0,|k|k.1 as u64), fs.is_some() as u64, t.depth.map_or(0, |f| f.0 as u64), t.stencil as u64] {
            key.push(w);
        }
        for (i, c) in t.colors.iter().enumerate() {
            let enabled = st.color_state >> (8 + i) & 1 != 0;
            let mask = st.channel_mask >> (4 * i) & 0xF;
            key.push(c.map_or(0, |f| f.0 as u64) | (enabled as u64) << 32 | (mask as u64) << 33);
            if enabled {
                key.push(st.blend[i]);
            }
        }
        let vertex_formats=vs.vertex_formats.as_ref().filter(|_|super::vertex_formats::enabled()).map(|_|super::vertex_formats::formats(st));
        if let Some(formats)=vertex_formats {key.vertex_formats(&formats);}
        key.push(buffer_formats.is_some()as u64);
        if let Some(formats)=buffer_formats {key.buffer_formats(&formats);}
        loop {
            if !self.pending.is_empty() {
                for (k, p) in std::mem::take(&mut *DONE.lock().unwrap()) {
                    self.pending.remove(&k);
                    self.pipelines.insert(k, p);
                }
            }
            if let Some(p) = self.pipelines.get(&key) {
                return p.clone().ok_or("pipeline failed");
            }
            if !self.pending.contains(&key) {
                break;
            }
            if !wait {
                return Err("pipeline compiling");
            }
            crate::kernel::blocking(|| std::thread::sleep(std::time::Duration::from_millis(1)));
        }
        let (cs, cm, blend) = (st.color_state, st.channel_mask, st.blend);
        if let Some(r) = fs.filter(|_|vertex_formats.is_none() && buffer_formats.is_none()).and_then(|fs|Recorded::of(vs, fs, t, cs, cm, &blend)) {
            if let Some(p) = PREWARMED.lock().unwrap().remove(&r) {
                self.pipelines.insert(key, Some(p.clone()));
                return Ok(p);
            }
            record(r);
        }
        let (jvs, jfs, jt) = (vs.clone(), fs.cloned(), t.clone());
        let name = format!("{:016x}/{}", vs_key.0, fs_key.map_or_else(||"depth-only".into(),|(hash,options)|format!("{hash:016x}.{options:x}")));
        if wait || super::shaders::sync_mode() {
            let t0 = std::time::Instant::now();
            let p = crate::kernel::blocking(|| build_pipeline(&jvs, jfs.as_deref(), &jt, cs, cm, &blend, &name,vertex_formats,buffer_formats));
            if wait && !super::shaders::sync_mode() && t0.elapsed().as_millis() >= 5 {
                eprintln!("[gfx] pipeline {name}: built while the render thread waited (new render target), {} ms", t0.elapsed().as_millis());
            }
            self.pipelines.insert(key, p.clone());
            return p.ok_or("pipeline failed");
        }
        self.pending.insert(key);
        super::shaders::spawn_job(Box::new(move || {
            let p = build_pipeline(&jvs, jfs.as_deref(), &jt, cs, cm, &blend, &name,vertex_formats,buffer_formats);
            DONE.lock().unwrap().push((key, p));
        }));
        Err("pipeline compiling")
    }

    pub fn depth_stencil(&mut self, st: &GState, has_depth: bool, has_stencil: bool) -> Shared<DepthState> {
        let masks = [st.stencil_read[0], st.stencil_read[1], st.stencil_write[0], st.stencil_write[1]];
        let key = (st.depth_stencil, masks, has_depth, has_stencil);
        self.depth
            .entry(key)
            .or_insert_with(|| {
                let (w0, w1) = (st.depth_stencil as u32, (st.depth_stencil >> 32) as u32);
                let d = MTLDepthStencilDescriptor::new();
                if has_depth && w0 & 1 != 0 {
                    d.setDepthCompareFunction(compare_fn(w0 >> 4 & 0xF));
                    d.setDepthWriteEnabled(w0 & 2 != 0);
                }
                if has_stencil && w0 & 4 != 0 {
                    for face in 0..2 {
                        let f = w1 >> (16 * face);
                        let s = MTLStencilDescriptor::new();
                        s.setStencilCompareFunction(compare_fn(f & 0xF));
                        s.setStencilFailureOperation(stencil_op(f >> 4 & 0xF));
                        s.setDepthStencilPassOperation(stencil_op(f >> 8 & 0xF));
                        s.setDepthFailureOperation(stencil_op(f >> 12 & 0xF));
                        s.setReadMask(masks[face as usize]);
                        s.setWriteMask(masks[2 + face as usize]);
                        if face == 0 {
                            d.setFrontFaceStencil(Some(&s));
                        } else {
                            d.setBackFaceStencil(Some(&s));
                        }
                    }
                }
                Shared(gfx::gpu().device.0.newDepthStencilStateWithDescriptor(&d).expect("depth stencil state"))
            })
            .clone()
    }
}

fn build_pipeline(vs: &Compiled, fs: Option<&Compiled>, t: &Targets, color_state: u32, channel_mask: u32, blend: &[u64; 8], name: &str,vertex_formats:Option<[u32;16]>,buffer_formats:Option<[u32;32]>) -> Option<Shared<Pipeline>> {
    let t0 = std::time::Instant::now();
    let r = build_pipeline_inner(vs, fs, t, color_state, channel_mask, blend, name,vertex_formats,buffer_formats);
    let ms = t0.elapsed().as_millis();
    if ms > 500 {
        eprintln!("[gfx] pipeline {name}: building took {ms} ms");
    }
    r
}

fn build_pipeline_inner(vs: &Compiled, fs: Option<&Compiled>, t: &Targets, color_state: u32, channel_mask: u32, blend: &[u64; 8], name: &str,vertex_formats:Option<[u32;16]>,buffer_formats:Option<[u32;32]>) -> Option<Shared<Pipeline>> {
    let specialized=vs.vertex_formats.as_ref().filter(|_|vertex_formats.is_some() || buffer_formats.is_some()).and_then(|library|library.function(vertex_formats.as_ref(),buffer_formats.as_ref()));
    let function=specialized.as_ref().unwrap_or(&vs.func);
    let d = descriptor(&function.0, fs.map(|f|&*f.func.0), t, color_state, channel_mask, blend);
    if let Some(vertex)=&vs.dynamic_vertex {d.setVertexPreloadedLibraries(&NSArray::from_retained_slice(&[vertex.library.0.clone()]));}
    let dev = &gfx::gpu().device.0;
    if let Some(a) = &*ARCHIVES {
        use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
        static HITS: AtomicUsize = AtomicUsize::new(0);
        static MISSES: AtomicUsize = AtomicUsize::new(0);
        d.setBinaryArchives(Some(&a.0));
        if let Ok(p) = dev.newRenderPipelineStateWithDescriptor_options_reflection_error(&d, MTLPipelineOption::FailOnBinaryArchiveMiss, None) {
            HITS.fetch_add(1, Relaxed);
            return Some(Shared(p));
        }
        let misses = MISSES.fetch_add(1, Relaxed) + 1;
        eprintln!("[gfx] pipeline {name}: not in the archive, compiling ({} archive hits, {misses} misses)", HITS.load(Relaxed));
    }
    match dev.newRenderPipelineStateWithDescriptor_error(&d) {
        Ok(p) => Some(Shared(p)),
        Err(e) => {
            if let Some(vertex)=&vs.dynamic_vertex {
                if let Some(function)=vertex.fallback_function() {
                    d.setVertexFunction(Some(&function.0));d.setVertexPreloadedLibraries(&NSArray::new());d.setBinaryArchives(None);
                    if let Ok(p)=dev.newRenderPipelineStateWithDescriptor_error(&d) {
                        eprintln!("[vertex-cache] pipeline {name}: used ordinary vertex fallback after dynamic link error: {}",e.localizedDescription());
                        return Some(Shared(p));
                    }
                }
            }
            eprintln!("[gfx] pipeline {name} failed: {}", e.localizedDescription());
            None
        }
    }
}

pub type Buffer = gfx::Buffer;
const CHUNK: usize = 16 << 20;

#[derive(Default)]
pub struct Arena {
    free: Vec<Shared<Buffer>>,
    pub used: Vec<Shared<Buffer>>,
    offset: usize,
    in_flight: Vec<(Shared<gfx::CommandBuffer>, Vec<Shared<Buffer>>)>,
}

pub struct Alloc {
    pub buffer: Shared<Buffer>,
    pub offset: usize,
    pub ptr: *mut u8,
    pub gpu: u64,
    pub new_chunk: bool,
}

impl Arena {
    pub fn alloc(&mut self, size: usize) -> Alloc {
        let size = size.max(1).next_multiple_of(256);
        let mut new_chunk = false;
        if self.used.last().is_none_or(|b| self.offset + size > b.0.length()) {
            let b = self.free.iter().position(|b| b.0.length() >= size).map(|i| self.free.swap_remove(i)).unwrap_or_else(|| {
                let dev = &gfx::gpu().device.0;
                Shared(dev.newBufferWithLength_options(size.max(CHUNK), MTLResourceOptions::StorageModeShared).expect("scratch buffer"))
            });
            self.used.push(b);
            self.offset = 0;
            new_chunk = true;
        }
        let b = self.used.last().unwrap().clone();
        let offset = self.offset;
        self.offset += size;
        let ptr = unsafe { (b.0.contents().as_ptr() as *mut u8).add(offset) };
        let gpu = b.0.gpuAddress() + offset as u64;
        Alloc { buffer: b, offset, ptr, gpu, new_chunk }
    }

    pub fn alloc_bytes(&mut self, data: &[u8]) -> Alloc {
        let a = self.alloc(data.len());
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), a.ptr, data.len()) };
        a
    }

    pub fn retire(&mut self, cb: Shared<gfx::CommandBuffer>) {
        let used = std::mem::take(&mut self.used);
        self.offset = 0;
        if !used.is_empty() {
            self.in_flight.push((cb, used));
        }
    }

    pub fn recycle(&mut self) {
        let mut i = 0;
        while i < self.in_flight.len() {
            if self.in_flight[i].0.0.status() == MTLCommandBufferStatus::Completed {
                let (_, bufs) = self.in_flight.swap_remove(i);
                self.free.extend(bufs);
            } else {
                i += 1;
            }
        }
    }
}

type Archive = objc2::runtime::ProtocolObject<dyn MTLBinaryArchive>;
type Function = objc2::runtime::ProtocolObject<dyn MTLFunction>;

const PIPELINE_LIST: &str = "data/pipelines.tsv";
const USER_LIST: &str = "work/pipelines.tsv";
const ARCHIVE_FILE: &str = "pipelines.metalar";

static ARCHIVES: LazyLock<Option<Shared<NSArray<Archive>>>> = LazyLock::new(|| {
    if std::env::var_os("NX_PIPELINE_ARCHIVE").is_none() {
        return None;
    }
    let dir = super::shaders::aot_dir();
    let mut files: Vec<String> = std::fs::read_dir(dir)
        .ok()?
        .flatten()
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.starts_with("pipelines") && n.ends_with(".metalar"))
        .collect();
    files.sort();
    let archives: Vec<Retained<Archive>> = files
        .iter()
        .filter_map(|f| {
            let d = MTLBinaryArchiveDescriptor::new();
            d.setUrl(Some(&NSURL::fileURLWithPath(&NSString::from_str(&format!("{dir}/{f}")))));
            gfx::gpu().device.0.newBinaryArchiveWithDescriptor_error(&d).ok()
        })
        .collect();
    eprintln!("[gfx] {} pipeline archive(s) loaded", archives.len());
    (!archives.is_empty()).then(|| Shared(NSArray::from_retained_slice(&archives)))
});

#[derive(Clone, PartialEq, Eq, Hash)]
struct Recorded {
    vs: (u64, [u32; 3]),
    fs: (u64, [u32; 3]),
    colors: Vec<u64>,
    depth: u64,
    stencil: bool,
    color_state: u32,
    channel_mask: u32,
    blend: [u64; 8],
}

impl Recorded {
    fn of(vs: &Compiled, fs: &Compiled, t: &Targets, color_state: u32, channel_mask: u32, blend: &[u64; 8]) -> Option<Self> {
        Some(Self {
            vs: vs.source?,
            fs: fs.source?,
            colors: t.colors.iter().map(|c| c.map_or(0, |f| f.0 as u64)).collect(),
            depth: t.depth.map_or(0, |f| f.0 as u64),
            stencil: t.stencil,
            color_state,
            channel_mask,
            blend: *blend,
        })
    }

    fn line(&self) -> String {
        let v = |(k, [a, b, c]): (u64, [u32; 3])| format!("{k:016x}:{a:x}:{b:x}:{c:x}");
        let list = |x: &[u64]| x.iter().map(|n| format!("{n:x}")).collect::<Vec<_>>().join(",");
        format!(
            "{}\t{}\t{}\t{:x}\t{}\t{:x}\t{:x}\t{}",
            v(self.vs),
            v(self.fs),
            list(&self.colors),
            self.depth,
            self.stencil as u8,
            self.color_state,
            self.channel_mask,
            list(&self.blend)
        )
    }

    fn parse(l: &str) -> Option<Self> {
        let f: Vec<&str> = l.split('\t').collect();
        let hex = |s: &str| u64::from_str_radix(s, 16).ok();
        let list = |s: &str| if s.is_empty() { Some(Vec::new()) } else { s.split(',').map(hex).collect::<Option<Vec<u64>>>() };
        let v = |s: &str| {
            let p: Vec<u64> = s.split(':').map(hex).collect::<Option<_>>()?;
            (p.len() == 4).then(|| (p[0], [p[1] as u32, p[2] as u32, p[3] as u32]))
        };
        let blend: Vec<u64> = list(f.get(7)?)?;
        Some(Self {
            vs: v(f.first()?)?,
            fs: v(f.get(1)?)?,
            colors: list(f.get(2)?)?,
            depth: hex(f.get(3)?)?,
            stencil: *f.get(4)? == "1",
            color_state: hex(f.get(5)?)? as u32,
            channel_mask: hex(f.get(6)?)? as u32,
            blend: blend.try_into().ok()?,
        })
    }

    fn targets(&self) -> Targets {
        let fmt = |n: u64| (n != 0).then_some(MTLPixelFormat(n as usize));
        Targets { colors: self.colors.iter().map(|&n| fmt(n)).collect(), depth: fmt(self.depth), stencil: self.stencil }
    }
}

fn read_list() -> Vec<Recorded> {
    let shipped = std::env::var("NX_PIPELINE_LIST").unwrap_or_else(|_| PIPELINE_LIST.into());
    let mut seen = std::collections::HashSet::new();
    [shipped.as_str(), USER_LIST]
        .iter()
        .flat_map(|p| std::fs::read_to_string(p).unwrap_or_default().lines().filter_map(Recorded::parse).collect::<Vec<_>>())
        .filter(|r| seen.insert(r.clone()))
        .collect()
}

pub fn prewarm() {
    std::thread::Builder::new()
        .name("prewarm".into())
        .spawn(|| {
            use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
            let t0 = std::time::Instant::now();
            let list = read_list();
            let (next, built) = (AtomicUsize::new(0), AtomicUsize::new(0));
            std::thread::scope(|s| {
                for _ in 0..2 {
                    s.spawn(|| {
                        unsafe { libc::pthread_set_qos_class_self_np(libc::qos_class_t::QOS_CLASS_UTILITY, 0) };
                        while let Some(r) = list.get(next.fetch_add(1, Relaxed)) {
                            objc2::rc::autoreleasepool(|_| {
                                let (Some(vs), Some(fs)) = (super::shaders::aot_compiled(r.vs), super::shaders::aot_compiled(r.fs)) else { return };
                                let d = descriptor(&vs.func.0, Some(&fs.func.0), &r.targets(), r.color_state, r.channel_mask, &r.blend);
                                if let Some(a) = &*ARCHIVES {
                                    d.setBinaryArchives(Some(&a.0));
                                }
                                if let Ok(p) = gfx::gpu().device.0.newRenderPipelineStateWithDescriptor_error(&d) {
                                    PREWARMED.lock().unwrap().insert(r.clone(), Shared(p));
                                    built.fetch_add(1, Relaxed);
                                }
                            });
                        }
                    });
                }
            });
            eprintln!("[gfx] prewarmed {}/{} recorded pipelines in {:.1} s", built.into_inner(), list.len(), t0.elapsed().as_secs_f32());
        })
        .unwrap();
}

fn record(r: Recorded) {
    static TARGET: LazyLock<&str> = LazyLock::new(|| if std::env::var_os("NX_RECORD_PIPELINES").is_some() { PIPELINE_LIST } else { USER_LIST });
    static KNOWN: LazyLock<Mutex<std::collections::HashSet<Recorded>>> = LazyLock::new(|| Mutex::new(read_list().into_iter().collect()));
    if KNOWN.lock().unwrap().insert(r.clone()) {
        use std::io::Write;
        if let Some(dir) = std::path::Path::new(*TARGET).parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(*TARGET) {
            let _ = writeln!(f, "{}", r.line());
        }
    }
}

fn add(archive: &Shared<Archive>, d: &MTLRenderPipelineDescriptor) -> bool {
    archive.0.addRenderPipelineFunctionsWithDescriptor_error(d).is_ok()
}

pub fn precompile_pipelines(programs: &std::collections::BTreeMap<u64, Vec<u8>>) -> anyhow::Result<()> {
    use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
    let t0 = std::time::Instant::now();
    let list = read_list();
    if list.is_empty() {
        eprintln!("[gfx] no recorded pipelines ({PIPELINE_LIST})");
        return Ok(());
    }
    let archive = gfx::gpu().device.0.newBinaryArchiveWithDescriptor_error(&MTLBinaryArchiveDescriptor::new()).map_err(|e| anyhow::anyhow!("{}", e.localizedDescription()))?;
    let archive = Shared(archive);
    let function = |(key, [extra, uints, sints]): (u64, [u32; 3])| {
        let blob = programs.get(&key)?;
        super::shaders::compile(blob, false, extra, uints, sints, key)
    };
    let (next, built) = (AtomicUsize::new(0), AtomicUsize::new(0));
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| loop {
                let Some(r) = list.get(next.fetch_add(1, Relaxed)) else { return };
                objc2::rc::autoreleasepool(|_| {
                    let (Some(vs), Some(fs)) = (function(r.vs), function(r.fs)) else { return };
                    let d = descriptor(&vs.func.0, Some(&fs.func.0), &r.targets(), r.color_state, r.channel_mask, &r.blend);
                    if add(&archive, &d) {
                        built.fetch_add(1, Relaxed);
                    }
                });
            });
        }
    });
    let path = format!("{}/{ARCHIVE_FILE}", super::shaders::aot_dir());
    archive.0.serializeToURL_error(&NSURL::fileURLWithPath(&NSString::from_str(&path))).map_err(|e| anyhow::anyhow!("{}", e.localizedDescription()))?;
    eprintln!("[gfx] precompiled {}/{} recorded pipelines in {:.0} s", built.into_inner(), list.len(), t0.elapsed().as_secs_f32());
    Ok(())
}

pub fn recorded_variants() -> Vec<(u64, [u32; 3])> {
    let mut v: Vec<(u64, [u32; 3])> = read_list().iter().flat_map(|r| [r.vs, r.fs]).filter(|(_, p)| *p != [0; 3]).collect();
    v.sort();
    v.dedup();
    v
}

fn descriptor(vs: &Function, fs: Option<&Function>, t: &Targets, color_state: u32, channel_mask: u32, blend: &[u64; 8]) -> Retained<MTLRenderPipelineDescriptor> {
    let d = MTLRenderPipelineDescriptor::new();
    d.setVertexFunction(Some(vs));
    d.setFragmentFunction(fs);
    for (i, c) in t.colors.iter().enumerate() {
        let Some(fmt) = c else { continue };
        let a = unsafe { d.colorAttachments().objectAtIndexedSubscript(i) };
        a.setPixelFormat(*fmt);
        let m = if fs.is_some() {channel_mask >> (4 * i) & 0xF} else {0};
        let mut wm = MTLColorWriteMask::empty();
        for (bit, w) in [(1, MTLColorWriteMask::Red), (2, MTLColorWriteMask::Green), (4, MTLColorWriteMask::Blue), (8, MTLColorWriteMask::Alpha)] {
            if m & bit != 0 {
                wm |= w;
            }
        }
        a.setWriteMask(wm);
        if m != 0 && color_state >> (8 + i) & 1 != 0 {
            let (w0, w1) = (blend[i] as u32, (blend[i] >> 32) as u32);
            a.setBlendingEnabled(true);
            a.setRgbBlendOperation(blend_op(w0 >> 24 & 0xF));
            a.setAlphaBlendOperation(blend_op(w0 >> 28 & 0xF));
            a.setSourceRGBBlendFactor(blend_factor(w1 & 0x7F));
            a.setDestinationRGBBlendFactor(blend_factor(w1 >> 8 & 0x7F));
            a.setSourceAlphaBlendFactor(blend_factor(w1 >> 16 & 0x7F));
            a.setDestinationAlphaBlendFactor(blend_factor(w1 >> 24 & 0x7F));
        }
    }
    if let Some(f) = t.depth {
        d.setDepthAttachmentPixelFormat(f);
        if t.stencil {
            d.setStencilAttachmentPixelFormat(f);
        }
    }
    d
}

#[cfg(all(test,target_os="macos"))]
mod depth_only_tests {
    use super::*;
    #[test]
    #[ignore="requires a real Metal GPU"]
    fn gpu_vertex_only_pipeline_writes_depth_preserves_color_and_has_distinct_cache_entry() {
        objc2::rc::autoreleasepool(|_| unsafe {
            let dev=&gfx::gpu().device.0;
            let source="#include <metal_stdlib>\nusing namespace metal;vertex float4 depth_triangle(uint id [[vertex_id]]) {float2 p=float2((id<<1)&2,id&2);return float4(p*2.0f-1.0f,0.25f,1.0f);}fragment float4 color_fragment() {return float4(0.75f);}";
            let lib=dev.newLibraryWithSource_options_error(&NSString::from_str(source),None).unwrap();
            let compiled=|name|Arc::new(Compiled{write_window_footprint:None,descriptor_write_footprint:None,grid_write_footprint:None,func:Shared(lib.newFunctionWithName(&NSString::from_str(name)).unwrap()),iface:Default::default(),source:None,argument_encoders:Default::default(),vertex_formats:None,dynamic_vertex:None});
            let vs=compiled("depth_triangle");let fs=compiled("color_fragment");
            let make=|format| {let td=MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(format,4,4,false);td.setStorageMode(MTLStorageMode::Private);td.setUsage(MTLTextureUsage::RenderTarget|MTLTextureUsage::ShaderRead);dev.newTextureWithDescriptor(&td).unwrap()};
            let depth=make(MTLPixelFormat::Depth32Float);let color=make(MTLPixelFormat::RGBA32Float);
            let targets=Targets{colors:vec![Some(MTLPixelFormat::RGBA32Float)],depth:Some(MTLPixelFormat::Depth32Float),stencil:false};
            let mut state=GState::default();state.depth_stencil=3|(2<<4);let mut states=States::default();
            let ds=states.depth_stencil(&state,true,false);
            let depth_pipeline=states.pipeline(&vs,(1,0),None,None,&targets,&state,true).unwrap();
            let color_pipeline=states.pipeline(&vs,(1,0),Some(&fs),Some((0,0)),&targets,&state,true).unwrap();
            assert_ne!(Retained::as_ptr(&depth_pipeline.0),Retained::as_ptr(&color_pipeline.0));
            let output=dev.newBufferWithLength_options(320,MTLResourceOptions::StorageModeShared).unwrap();
            for (pipeline,expected_color) in [(&depth_pipeline,[0.125f32;4]),(&color_pipeline,[0.75f32;4])] {
                let pass=MTLRenderPassDescriptor::new();let da=pass.depthAttachment();da.setTexture(Some(&depth));da.setLoadAction(MTLLoadAction::Clear);da.setStoreAction(MTLStoreAction::Store);da.setClearDepth(1.0);
                let ca=pass.colorAttachments().objectAtIndexedSubscript(0);ca.setTexture(Some(&color));ca.setLoadAction(MTLLoadAction::Clear);ca.setStoreAction(MTLStoreAction::Store);ca.setClearColor(MTLClearColor{red:0.125,green:0.125,blue:0.125,alpha:0.125});
                let cb=gfx::gpu().queue.0.commandBuffer().unwrap();let enc=cb.renderCommandEncoderWithDescriptor(&pass).unwrap();enc.setRenderPipelineState(&pipeline.0);enc.setDepthStencilState(Some(&ds.0));enc.drawPrimitives_vertexStart_vertexCount(MTLPrimitiveType::Triangle,0,3);enc.endEncoding();
                super::super::copy::copy_into(&cb,&depth,(0,0),[0,0,0,4,4,1],52,&output,0).unwrap();super::super::copy::copy_into(&cb,&color,(0,0),[0,0,0,4,4,1],47,&output,64).unwrap();cb.commit();cb.waitUntilCompleted();assert_eq!(cb.status(),MTLCommandBufferStatus::Completed);
                let values=std::slice::from_raw_parts(output.contents().as_ptr().cast::<f32>(),80);assert!(values[..16].iter().all(|v|*v==0.25));for pixel in values[16..].chunks_exact(4) {assert_eq!(pixel,expected_color);}
            }
        });
    }
}
