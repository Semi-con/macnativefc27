// Mush8 - exec.rs
// Constantin Clerc (c22dev) - 0.1
use super::state::Res;
use super::{Replay, UPPER_LEFT, ZERO_TO_ONE};
use crate::gfx;
use objc2::runtime::ProtocolObject;
use objc2_metal::*;
use std::ptr::NonNull;
use std::sync::atomic::Ordering::Relaxed;

fn rd<T: Copy>(addr: u64) -> T {
    unsafe { (addr as *const T).read_unaligned() }
}

fn descriptor_ids(st:&super::GState,stage:usize,binding:u32)->(u32,u32) {
    let (texture,sampler)=if binding>=0x1000 {
        let pair=binding-0x1000;
        (st.separate_textures[stage].get((pair/128) as usize).copied().unwrap_or(0),st.separate_samplers[stage].get((pair%128) as usize).copied().unwrap_or(0))
    } else if (32..40).contains(&binding) {
        // C2[0x120..0x160] contains image handles. TXQ reads the same
        // descriptor table and can query these without a sampler binding.
        (st.images[stage][(binding-32) as usize],0)
    } else {
        let combined=st.textures[stage].get(binding as usize).copied().unwrap_or(0);(combined,combined)
    };
    ((texture&0xfffff) as u32,((sampler>>20)&0xfff) as u32)
}

struct AtomicImage {
    binding:u32,
    texture:gfx::Shared<gfx::Texture>,
    allocation:super::state::Alloc,
    layout:[u32;4],
}

impl AtomicImage {
    fn copy(&self,cb:&gfx::CommandBuffer,to_texture:bool)->Result<(),&'static str> {
        if self.layout[3]!=0 {return Ok(());}
        let enc=cb.blitCommandEncoder().ok_or("no image-atomic transfer encoder")?;
        let [width,height,row_words,_]=self.layout;let row=row_words as usize*4;
        let size=MTLSize{width:width as usize,height:height as usize,depth:1};let zero=MTLOrigin{x:0,y:0,z:0};let a=&self.allocation;
        unsafe {
            if to_texture {enc.copyFromBuffer_sourceOffset_sourceBytesPerRow_sourceBytesPerImage_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin(&a.buffer.0,a.offset,row,row*height as usize,size,&self.texture.0,0,0,zero);}
            else {enc.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toBuffer_destinationOffset_destinationBytesPerRow_destinationBytesPerImage(&self.texture.0,0,0,zero,size,&a.buffer.0,a.offset,row,row*height as usize);}
        }
        enc.endEncoding();Ok(())
    }
}

impl Replay<'_> {
    fn prepare_atomic_images(&mut self,images:&[(u32,gfx::Shared<gfx::Texture>,[u32;4])],mask:u32)->Result<Vec<AtomicImage>,&'static str> {
        // Multiple aliases would need a shared scratch allocation. Reject that
        // case rather than allowing separate snapshots to overwrite each other.
        if mask.count_ones()>1 {return Err("multiple atomic-image bindings require alias-aware shared staging");}
        let mut out=Vec::new();
        for (binding,texture,meta) in images {
            if mask&(1<<binding)==0 {continue;}
            if meta[0]!=32 || meta[1]!=1 || !matches!(meta[2],0|1) || texture.0.textureType()!=MTLTextureType::Type2D {return Err("surface atomic requires a 1D/2D R32 integer image");}
            if meta[3]!=0 {
                out.push(AtomicImage{binding:*binding,texture:texture.clone(),allocation:self.e.arena.alloc(4),layout:[0,0,0,1]});continue;
            }
            let (width,height)=(texture.0.width(),texture.0.height());let row=(width*4).next_multiple_of(256);
            let allocation=self.e.arena.alloc(row*height);
            let image=AtomicImage{binding:*binding,texture:texture.clone(),allocation,layout:[width as u32,height as u32,(row/4) as u32,0]};
            image.copy(&self.cb,false)?;out.push(image);
        }
        Ok(out)
    }
    pub(super) fn draw(&mut self, d: &super::Draw) -> Result<(), &'static str> {
        self.draw_with(d,None)
    }

    pub(super) fn draw_indirect(&mut self,d:&super::Draw,address:u64)->Result<(),&'static str> {
        if address%4!=0 {return Err("indirect draw argument alignment");}
        if !matches!(d.mode,0|1|3|4|5) {return Err("indirect primitive conversion is unsupported");}
        if d.index.is_some_and(|(ty,_)|!matches!(ty,1|2)) {return Err("indirect indices require uint16 or uint32");}
        self.draw_with(d,Some(address))
    }

    fn draw_with(&mut self,d:&super::Draw,indirect:Option<u64>)->Result<(),&'static str> {
        self.e.st.validate_single_sample().inspect_err(|_|eprintln!("[gfx] NVN multisample state {:#x}",self.e.st.multisample))?;
        if d.count == 0 || d.instances == 0 || self.e.st.rasterizer_discard || !self.e.st.render_enable || self.e.st.sample_mask&1==0 {
            if super::submit_profile::active() {eprintln!("[draw-reason] count={} instances={} rasterizer_discard={} render_enable={} sample_mask={:#x} stages={:x?}",d.count,d.instances,self.e.st.rasterizer_discard,self.e.st.render_enable,self.e.st.sample_mask,self.e.st.stages());}
            return Ok(());
        }
        let st = self.e.st;
        self.prepare_stage_cpu_reads(&st,false);
        let mut vs = None;
        let mut fs = None;
        for &(code, ctl) in st.stages() {
            // Compute remains bound across graphics draws. Do not rehash its
            // full code body while selecting this draw's graphics stages.
            if super::shaders::stage_bit(code)==Some(0x20) {continue;}
            let Some(s) = self.e.shaders.stage(code, ctl) else { continue };
            if s.fragment {
                fs = Some(s)
            } else if s.local_size.is_none() {
                vs = Some(s)
            }
        }
        let Some(vs) = vs else {
            eprintln!("[gfx] incomplete graphics program stages {:?}",st.stages().iter().map(|&(code,ctl)|(code,ctl,super::shaders::stage_bit(code))).collect::<Vec<_>>());
            return Err("draw without a vertex stage");
        };
        self.ensure_pass()?;
        let targets = self.pass.as_ref().unwrap().targets.clone();
        let (mut uints, mut sints) = (0, 0);
        for (i, f) in targets.colors.iter().enumerate() {
            match f.map_or(0, super::shaders::integer_kind) {
                1 => uints |= 1 << i,
                2 => sints |= 1 << i,
                _ => {}
            }
        }
        use super::shaders::Lookup;
        let wait = self.pass.as_ref().unwrap().fresh;
        let fragment_options=if fs.is_some() {st.fragment_options(&targets)?} else {0};
        let fragment_options=fragment_options | if fs.is_some_and(|f|f.early_fragment_tests) && fragment_options&nxshader::FRAGMENT_ALPHA_TEST==0 {nxshader::FRAGMENT_EARLY_TESTS} else {0};
        let fsc = match fs.map(|fs|self.e.shaders.function(&fs, fragment_options, uints, sints, wait)) {
            Some(Lookup::Ready(f)) => Some(f),
            Some(Lookup::Pending) => {if super::submit_profile::active() {eprintln!("[draw-reason] fragment compiling count={} hash={:016x}",d.count,fs.unwrap().hash);}return Err("shader compiling");},
            Some(Lookup::Failed) => return Err("fragment shader failed"),
            None => None,
        };
        let extra = fsc.as_ref().map_or(0,|f|f.iface.attributes);
        let vsc = match self.e.shaders.function(&vs, extra, 0, 0, wait) {
            Lookup::Ready(f) => f,
            Lookup::Pending => {if super::submit_profile::active() {eprintln!("[draw-reason] vertex compiling count={} hash={:016x}",d.count,vs.hash);}return Err("shader compiling");},
            Lookup::Failed => return Err("vertex shader failed"),
        };
        let (w, h) = self.pass.as_ref().unwrap().size;

        let ds = self.e.states.depth_stencil(&st, targets.depth.is_some(), targets.stencil);

        let (sx, sy, sw, sh) = match st.scissor {
            Some([x, y, sw, sh]) => {
                let (x0, y0) = (x.max(0) as u32, y.max(0) as u32);
                let x1 = ((x as i64 + sw as i64).clamp(0, w as i64)) as u32;
                let y1 = ((y as i64 + sh as i64).clamp(0, h as i64)) as u32;
                if x1 <= x0 || y1 <= y0 {
                    return Ok(());
                }
                (x0, y0, x1 - x0, y1 - y0)
            }
            None => (0, 0, w, h),
        };
        let upper_left = UPPER_LEFT.load(Relaxed);

        self.heap = gfx::heap();
        let mut bound = Vec::new();
        let mut static_arguments:[Option<super::bindless::Bindings>;2]=std::array::from_fn(|_|None);
        for (stage, c) in std::iter::once((0usize,&vsc)).chain(fsc.as_ref().map(|f|(1,f))) {
            let textures=self.stage_textures(&st,stage,c);
            static_arguments[stage]=self.stage_static_arguments(c,&textures)?;
            for (b, tex, slot, sampler) in textures {
                bound.push((stage == 1, b, tex, slot, sampler));
            }
        }
        let vertex_images=self.stage_images(&st,0,&vsc)?;
        let fragment_images=if let Some(f)=&fsc {self.stage_images(&st,1,f)?} else {Vec::new()};
        let vertex_buffer_metadata=self.stage_buffer_metadata(&st,0,&vsc)?;
        let fragment_buffer_metadata=if let Some(f)=&fsc {self.stage_buffer_metadata(&st,1,f)?} else {None};
        let buffer_formats=if super::vertex_formats::buffer_enabled() && crate::input::test_control::buffer_formats_active() {vertex_buffer_metadata.as_ref().map(super::vertex_formats::buffer_formats)}else{None};
        let pipeline = self.e.states.pipeline_with_buffer_formats(&vsc, (vs.hash, extra), fsc.as_ref(), fs.map(|fs|(fs.hash ^ (uints as u64) << 48 ^ (sints as u64) << 56,fragment_options)), &targets, &st, wait,buffer_formats)?;
        let vertex_bindless=self.stage_bindless(&st,&vsc)?;
        let fragment_bindless=if let Some(f)=&fsc {self.stage_bindless(&st,f)?} else {None};

        let draw_probe=if super::TRACE_DRAWS.load(Relaxed) {
            let requested=fs.is_some_and(|fs|std::fs::read_to_string("work/probe-draw-hashes").ok().is_some_and(|value|value.split(',').filter_map(|v|u64::from_str_radix(v.trim(),16).ok()).any(|hash|hash==fs.hash)));
            static SEEN:std::sync::LazyLock<std::sync::Mutex<std::collections::HashSet<u64>>>=std::sync::LazyLock::new(Default::default);
            if requested {
                let hash=fs.unwrap().hash;let mut seen=SEEN.lock().unwrap();
                if seen.len()<8 && seen.insert(hash) {Some(format!("work/draw-probes/{hash:016x}"))} else {None}
            } else {None}
        } else {None};
        let profile=super::TRACE_DRAWS.load(Relaxed) && std::path::Path::new("work/profile-gpu-stages").exists();
        if draw_probe.is_some() {self.commit_boundary(true);} else if profile {self.end_pass();}
        let flags = (!upper_left) as u32 | ((!ZERO_TO_ONE.load(Relaxed)) as u32) << 1;
        let alpha = st.color_state >> 24 & 0xF;
        let mut res = [self.res(&st, 0, vs.constants, flags), self.res(&st, 1, fs.map_or(0,|f|f.constants), flags)];
        res[0].shader_hash=vs.hash;res[1].shader_hash=fs.map_or(0,|f|f.hash);
        if alpha != 0 && alpha != 8 {
            res[1].flags |= 4;
            res[1].alpha_func = alpha - 1;
        }
        let r = &mut res[0];
        for i in 0..16 {
            let a = st.attribs[i];
            if a & 0x40 == 0 {
                continue;
            }
            let stream = (a & 0x1F) as usize % 16;
            r.attr_format[i] = (a >> 21) & 0x3FF;
            r.attr_offset[i] = (a >> 7) & 0x3FFF;
            r.attr_stream[i] = stream as u32;
        }
        for s in 0..16 {
            r.vb[s] = if st.vbufs[s].0 != 0 { self.gpu_range(st.vbufs[s].0, st.vbufs[s].1, false) } else { gfx::gpu().zeros.0.gpuAddress() };
            r.vb_stride[s] = st.streams[s].0;
            r.vb_divisor[s] = st.streams[s].1;
        }
        r.base_instance = d.base_instance;

        if let Some(directory)=&draw_probe {
            let _=std::fs::create_dir_all(directory);
            let mut metadata=format!("vertex={:016x}\nfragment={:016x}\nviewport={:?}\n",vs.hash,fs.unwrap().hash,st.viewport);
            for bank in 0..18 {
                let count=if bank<3 {4096} else {(st.ubos[1][bank-3].1 as usize).min(65536)};
                if count==0 {continue;}
                if let Some(bytes)=self.probe_gpu_bytes(res[1].cb[bank],count) {let _=std::fs::write(format!("{directory}/cb{bank}.bin"),bytes);}
                if bank>=3 {
                    let (address,size)=st.ubos[1][bank-3];metadata.push_str(&format!("cb{bank}={address:#x},{size:#x}\n"));
                    if address!=0 {if let Ok(bytes)=self.cpu_bytes(address,count) {let _=std::fs::write(format!("{directory}/cb{bank}.cpu"),bytes);}}
                }
            }
            for (fragment,binding,texture,_,_) in &bound {
                if !fragment {continue;}
                let logical=fsc.as_ref().unwrap().iface.texture_slots.iter().find_map(|(logical,slot)|(*slot==*binding).then_some(*logical));
                if let Some(logical)=logical {
                    let (id,sampler)=descriptor_ids(&st,1,logical);
                    let (builder,sampler)={let state=super::STATE.lock().unwrap();(state.registered_textures.get(&(st.texture_pool,id)).copied(),state.registered_samplers.get(&(st.sampler_pool,sampler)).copied())};
                    if let Some(sampler)=sampler {let _=std::fs::write(format!("{directory}/input{binding}.sampler"),sampler);}
                    if let Some(builder)=builder {
                        let _=std::fs::write(format!("{directory}/input{binding}.builder"),builder);
                        if let Some(key)=super::textures::TexKey::of(&builder) {
                            metadata.push_str(&format!("input{binding} logical{logical} id{id}: {key:?}\n"));
                            if matches!(texture.0.textureType(),MTLTextureType::Type2D|MTLTextureType::Type3D) {super::dump_volume(&texture.0,&key,&format!("{directory}/input{binding}"));}
                        }
                    }
                }
            }
            for (i,&(address,size)) in st.storage[1].iter().enumerate() {
                if address!=0 && size!=0 {
                    metadata.push_str(&format!("storage{i}={address:#x},{size:#x}\n"));
                    if let Ok(bytes)=self.cpu_bytes(address,size.min(65536) as usize) {let _=std::fs::write(format!("{directory}/storage{i}.bin"),bytes);}
                }
            }
            let _=std::fs::write(format!("{directory}/metadata.txt"),metadata);
        }
        // An actual texture upload may have waited for a preceding GPU
        // transfer and ended the old encoder. Reopen the same targets and
        // retain indirect resources on the command buffer that draws them.
        self.ensure_pass()?;
        let fault=self.fault_record();res[0].fault_gpu=fault;res[1].fault_gpu=fault;
        for bindings in static_arguments.iter().chain([&vertex_bindless,&fragment_bindless]).flatten() {bindings.retain(&self.cb);}
        let heap = self.heap;
        let pass = self.pass.as_mut().unwrap();
        if let Some(hb) = heap {
            if pass.heap != hb.chunks[0].gpu {
                for chunk in &hb.chunks {
                    pass.enc.useResource_usage_stages(ProtocolObject::from_ref(&*chunk.buffer.0), MTLResourceUsage::Read, MTLRenderStages::Vertex | MTLRenderStages::Fragment);
                }
                pass.heap = hb.chunks[0].gpu;
            }
        }

        let args=if let Some(address)=indirect {
            let bytes=if d.index.is_some() {20} else {16};
            if let Some((ty,address))=d.index {
                let size=if ty==1 {2} else {4};
                if address%size!=0 || self.buffer_range(address,size).is_none() {return Err("indirect index buffer is unaligned or unmapped");}
            }
            Some(self.buffer_range(address,bytes).map(|(chunk,offset)|(chunk.buffer.clone(),offset)).ok_or("indirect draw arguments are not mapped to Metal")?)
        } else {None};
        let prim = self.primitives(d)?;
        let vertex_global=if vs.global_writes {vsc.write_window_footprint.as_deref().and_then(|p|self.window_write_ranges(p,&st,0))}else{Some(Vec::new())};
        let vertex_writes=self.shared_write_ranges(&st,0,&vsc,vertex_global);
        let fragment_writes=if let Some(c)=&fsc {
            let global=if fs.is_some_and(|s|s.global_writes) {c.write_window_footprint.as_deref().and_then(|p|self.window_write_ranges(p,&st,1))}else{Some(Vec::new())};self.shared_write_ranges(&st,1,c,global)
        }else{Some(Vec::new())};
        let vertex_buffer_writes=vertex_images.iter().any(|(b,_,meta)|meta[3]!=0 && (!super::write_windows::enabled() || vsc.iface.image_writes&(1u32<<b)!=0));
        let fragment_buffer_writes=fragment_images.iter().any(|(b,_,meta)|meta[3]!=0 && (!super::write_windows::enabled() || fsc.as_ref().is_some_and(|c|c.iface.image_writes&(1u32<<b)!=0)));
        let mut buffer_writers=MTLRenderStages::empty();
        if (vs.global_writes || vertex_buffer_writes) && (vertex_writes.is_none() || !super::write_windows::enabled()) {buffer_writers|=MTLRenderStages::Vertex;}
        if (fs.is_some_and(|s|s.global_writes) || fragment_buffer_writes) && (fragment_writes.is_none() || !super::write_windows::enabled()) {buffer_writers|=MTLRenderStages::Fragment;}
        for (writes,stage) in [(&vertex_writes,MTLRenderStages::Vertex),(&fragment_writes,MTLRenderStages::Fragment)] {
            if let Some(writes)=writes {for buffer in self.proven_write_buffers(writes) {self.pass.as_ref().unwrap().enc.useResource_usage_stages(ProtocolObject::from_ref(&*buffer.0),MTLResourceUsage::Write,stage);}}
        }
        if !buffer_writers.is_empty() {
            let enc=&self.pass.as_ref().unwrap().enc;let resources=self.pool_resources();
            if !resources.is_empty() {unsafe {enc.useResources_count_usage_stages(NonNull::new(resources.as_ptr().cast_mut()).unwrap(),resources.len(),MTLResourceUsage::Write,buffer_writers);}}
            if let Some(heap)=self.heap {for chunk in &heap.chunks {enc.useResource_usage_stages(ProtocolObject::from_ref(&*chunk.buffer.0),MTLResourceUsage::Write,buffer_writers);}}
        }

        if let Some(pass)=self.pass.as_mut() {
            if pass.counter.is_some() {super::submit_profile::workload(self.passes,pass.size,vs.hash,fs.map_or(0,|s|s.hash),d.count,d.instances,indirect.is_some());}
            if pass.counter.is_some() && !pass.profile_context_reported {
                super::submit_profile::context(self.passes,pass.size,st.viewport,vs.hash,fs.map_or(0,|s|s.hash));
                pass.profile_context_reported=true;
            }
        }
        if super::TRACE_DRAWS.load(Relaxed) {
            let attrs: Vec<String> = (0..16).filter(|&i| st.attribs[i] & 0x40 != 0).map(|i| format!("{i}:{}", (st.attribs[i] >> 21) & 0x3FF)).collect();
            eprintln!("[frame]   attributes {attrs:?}");
            let rt: Vec<String> = self.e.colors.iter().map(|c| c.as_ref().and_then(|b| super::textures::TexKey::of(b)).map_or("-".into(), |k| format!("{:#x}/{}/{}x{}", k.addr, k.format, k.width, k.height))).collect();
            let tex: Vec<String> = bound.iter().map(|(f, b, t, _, _)| format!("{}{b}:{}x{}/{}", if *f { "f" } else { "v" }, t.0.width(), t.0.height(), t.0.pixelFormat().0)).collect();
            eprintln!(
                "[frame] draw rt {rt:?} vs {:016x} fs {:016x} poly {:#x} mode {} count {} inst {} vp {:?} sc {:?} ds {:#x} cs {:#x} mask {:#x} tex {tex:?}",
                vs.hash, fs.map_or(0,|f|f.hash), st.polygon, d.mode, d.count, d.instances, st.viewport, st.scissor, st.depth_stencil, st.color_state, st.channel_mask
            );
        }
        if st.polygon & 3 == 3 {
            return Ok(());
        }
        let prev = self.pass.as_ref().unwrap().fixed;
        let next = {
            let enc = &self.pass.as_ref().unwrap().enc;
            bind_fixed(enc, prev, &pipeline, &ds, &st, upper_left, (w, h), [sx, sy, sw, sh])
        };
        self.pass.as_mut().unwrap().fixed = Some(next);
        let enc = &self.pass.as_ref().unwrap().enc;
        unsafe {
            enc.setVertexBytes_length_atIndex(NonNull::from(&res[0]).cast(), size_of::<Res>(), 0);
            enc.setFragmentBytes_length_atIndex(NonNull::from(&res[1]).cast(), size_of::<Res>(), 0);
            for (fragment,index,bindings) in [(false,2,&vertex_bindless),(true,2,&fragment_bindless),(false,12,&static_arguments[0]),(true,12,&static_arguments[1])] {
                if let Some(bindings)=bindings {
                    let stages=if fragment {MTLRenderStages::Fragment} else {MTLRenderStages::Vertex};
                    for texture in &bindings.textures {enc.useResource_usage_stages(ProtocolObject::from_ref(&*texture.0),MTLResourceUsage::Read,stages);}
                    for buffer in &self.e.arena.used {enc.useResource_usage_stages(ProtocolObject::from_ref(&*buffer.0),MTLResourceUsage::Read,stages);}
                    let a=&bindings.allocation;
                    if fragment {enc.setFragmentBuffer_offset_atIndex(Some(&a.buffer.0),a.offset,index);}
                    else {enc.setVertexBuffer_offset_atIndex(Some(&a.buffer.0),a.offset,index);}
                }
            }
            for (fragment,metadata) in [(false,&vertex_buffer_metadata),(true,&fragment_buffer_metadata)] {
                if let Some(a)=metadata {
                    if fragment {enc.setFragmentBuffer_offset_atIndex(Some(&a.buffer.0),a.offset,13);}
                    else {enc.setVertexBuffer_offset_atIndex(Some(&a.buffer.0),a.offset,13);}
                }
            }
            for (fragment,images) in [(false,&vertex_images),(true,&fragment_images)] {
                if images.is_empty() {continue;}
                let mut meta=[[0u32;4];8];
                for (binding,texture,descriptor) in images {
                    meta[*binding as usize]=*descriptor;
                    if fragment {enc.setFragmentTexture_atIndex(Some(&texture.0),32+*binding as usize);}
                    else {enc.setVertexTexture_atIndex(Some(&texture.0),32+*binding as usize);}
                }
                if fragment {enc.setFragmentBytes_length_atIndex(NonNull::from(&meta).cast(),size_of_val(&meta),1);}
                else {enc.setVertexBytes_length_atIndex(NonNull::from(&meta).cast(),size_of_val(&meta),1);}
            }
            for (frag, b, tex, slot, sampler) in &bound {
                if static_arguments[usize::from(*frag)].is_some() {continue;}
                if *frag {
                    enc.setFragmentTexture_atIndex(Some(&tex.0), *b as usize);
                    enc.setFragmentSamplerState_atIndex(Some(&sampler.0), *slot as usize);
                } else {
                    enc.setVertexTexture_atIndex(Some(&tex.0), *b as usize);
                    enc.setVertexSamplerState_atIndex(Some(&sampler.0), *slot as usize);
                }
            }
            match prim {
                Prim::Direct(ty) => if let Some((buffer,offset))=&args {enc.drawPrimitives_indirectBuffer_indirectBufferOffset(ty,&buffer.0,*offset);} else {enc.drawPrimitives_vertexStart_vertexCount_instanceCount_baseInstance(ty, d.first as usize, d.count as usize, d.instances as usize, 0);},
                Prim::Indexed{ty,itype,buffer,offset,..} if args.is_some()=> {
                    let (args,at)=args.as_ref().unwrap();
                    enc.drawIndexedPrimitives_indexType_indexBuffer_indexBufferOffset_indirectBuffer_indirectBufferOffset(ty,itype,&buffer.0,offset,&args.0,*at);
                }
                Prim::Indexed { ty, itype, buffer, offset, count, base_vertex } => enc
                    .drawIndexedPrimitives_indexCount_indexType_indexBuffer_indexBufferOffset_instanceCount_baseVertex_baseInstance(
                        ty,
                        count,
                        itype,
                        &buffer.0,
                        offset,
                        d.instances as usize,
                        base_vertex as isize,
                        0,
                    ),
            }
        }
        for (writes,active,stage) in [(vertex_writes,vs.global_writes || vertex_buffer_writes,0),(fragment_writes,fs.is_some_and(|s|s.global_writes) || fragment_buffer_writes,1)] {
            if !active {continue;}
            if let Some(ranges)=writes {
                if super::write_windows::enabled() && !ranges.is_empty() {self.e.window_batch_writes=true;}
                if super::submit_profile::active() {eprintln!("[cpu-write] graphics proven stage{stage} ranges={}",ranges.len());}
                for (gpu,cpu,len) in ranges {self.record_known_write(gpu,cpu,len);}
            } else {self.e.gpu_buffer_dirty=true;if super::submit_profile::active(){eprintln!("[cpu-write] graphics unknown stage{stage} vs={:016x} fs={:016x}",vs.hash,fs.map_or(0,|s|s.hash));}}
        }
        if profile {
            if let Some(counter)=self.pass.as_ref().and_then(|p|p.counter.clone()) {
                let (vs,fs,count,instances)=(vs.hash,fs.map_or(0,|s|s.hash),d.count,d.instances);
                let handler=block2::RcBlock::new(move |_:NonNull<gfx::CommandBuffer>| {
                    if let Some(data)=unsafe {counter.0.resolveCounterRange(objc2_foundation::NSRange{location:0,length:4})} {
                        if data.length()==32 {
                            let mut values=[0u64;4];unsafe {data.getBytes_length(NonNull::from(&mut values).cast(),32);}
                            if values.iter().all(|v|*v!=0 && *v!=u64::MAX) && values[1]>=values[0] && values[3]>=values[2] {
                                eprintln!("[gpu-stages] vs={vs:016x} fs={fs:016x} count={count} instances={instances} target={w}x{h} vertex_ms={:.6} fragment_ms={:.6}",(values[1]-values[0]) as f64/1e6,(values[3]-values[2]) as f64/1e6);
                            }
                        }
                    }
                });
                unsafe {self.cb.addCompletedHandler(block2::RcBlock::as_ptr(&handler));}
            }
            self.end_pass();
        }
        if let Some(directory)=draw_probe {
            self.commit_boundary(true);
            for (i,builder) in self.e.colors.iter().enumerate() {
                if let Some(key)=builder.as_ref().and_then(|b|super::textures::TexKey::of(b)) {
                    if let Some((_,texture))=self.e.textures.cached(&key) {super::dump_volume(&texture.0,&key,&format!("{directory}/output{i}"));}
                }
            }
        }
        Ok(())
    }

    fn probe_gpu_bytes(&self,pointer:u64,count:usize)->Option<Vec<u8>> {
        let buffers=self.e.arena.used.iter().chain(self.pools.iter().flat_map(|p|p.chunks.iter().map(|c|&c.buffer))).chain(self.heap.into_iter().flat_map(|h|h.chunks.iter().map(|c|&c.buffer))).chain(std::iter::once(&gfx::gpu().zeros));
        for buffer in buffers {
            let base=buffer.0.gpuAddress();
            if pointer>=base && pointer-base+(count as u64)<=buffer.0.length() as u64 {
                return Some(unsafe {std::slice::from_raw_parts((buffer.0.contents().as_ptr() as *const u8).add((pointer-base) as usize),count)}.to_vec());
            }
        }
        None
    }

    fn stage_textures(&mut self, st: &super::GState, stage: usize, c: &super::shaders::Compiled) -> Vec<(u32, gfx::Shared<gfx::Texture>, u32, gfx::Shared<super::textures::Sampler>)> {
        let mut bound = Vec::with_capacity(c.iface.textures.len());
        let ids=|binding|descriptor_ids(st,stage,binding);
        let descriptors: Vec<_> = {
            let s = super::STATE.lock().unwrap();
            c.iface
                .textures
                .keys()
                .map(|&b| {
                    let (tid, sid) = ids(b);
                    let tb = s.registered_textures.get(&(st.texture_pool, tid)).copied();
                    let key = tb.as_ref().map(|tb| super::textures::TexKey::of_in(tb, &s));
                    (tb, key, s.registered_samplers.get(&(st.sampler_pool, sid)).copied())
                })
                .collect()
        };
        for ((&b, &kind), (tb, key, sb)) in c.iface.textures.iter().zip(descriptors) {
            let (tid, _) = ids(b);
            let got = tb.and_then(|tb| {
                if let Some(k)=key.flatten().filter(|k|k.target==10) {
                    if !c.iface.buffer_metadata || kind!=nxshader::TexKind::T2D {crate::diag::report_and_exit("buffer texture requires a supported one-dimensional fetch/query");}
                    super::textures::buffer_descriptor(&tb).unwrap_or_else(|why|crate::diag::report_and_exit(why));
                    Some((k,self.e.textures.dummy(nxshader::TexKind::T2D),tb))
                } else {self.texture_keyed(key.flatten(), &tb, false).map(|(k, t)| (k, t, tb))}
            });
            if super::TRACE_DRAWS.load(Relaxed) {
                if let Some((k, t, tb)) = got.as_ref().filter(|(_, t, _)| matches!(t.0.textureType(),MTLTextureType::Type2D|MTLTextureType::Type3D)) {
                    self.e.sampled.entry(*k).or_insert_with(|| t.clone());
                    if t.0.textureType()==MTLTextureType::Type3D {
                        // Read-only diagnostic evidence: preserve the builder
                        // and bounded CPU backing alongside the GPU volume.
                        let path=format!("work/volume-probes/{:x}_f{}",k.addr,k.format);
                        if !std::path::Path::new(&format!("{path}.builder")).exists() {
                            let _=std::fs::create_dir_all("work/volume-probes");
                            let _=std::fs::write(format!("{path}.builder"),tb);
                            if let Some((shape,layout))=super::super::texture_layout(tb.as_ptr() as u64) {
                                let _=std::fs::write(format!("{path}.layout"),format!("shape={shape:?}\nlayout={layout:?}\n"));
                                if layout.total_size<=16*1024*1024 && crate::kernel::aspace().readable(k.addr,layout.total_size) {
                                    let bytes=unsafe {std::slice::from_raw_parts(k.addr as *const u8,layout.total_size as usize)};
                                    let _=std::fs::write(format!("{path}.cpu"),bytes);
                                }
                            }
                        }
                    }
                }
                let swz = tb.map(|b| [0x38usize, 0x3c, 0x40, 0x44].map(|o| u32::from_le_bytes(b[o..o + 4].try_into().unwrap())));
                eprintln!("[frame]   tex {}{b}: id {tid} pool {:#x} -> {:?} swizzle {swz:?}", ["v", "f", "c"][stage], st.texture_pool, got.as_ref().map(|(k, _, _)| (k.addr, k.format, k.width, k.height)));
            }
            let tex = got
                .and_then(|(k, t, tb)| self.e.textures.view(k, &tb, &t, kind))
                .filter(|t| !kind.is_depth() || gfx_is_depth(t.0.pixelFormat()))
                .unwrap_or_else(|| self.e.textures.dummy(kind));
            let sampler = self.e.textures.sampler(sb.as_ref());
            let slot = c.iface.samplers.get(&b).copied().unwrap_or(0);
            bound.push((c.iface.texture_slots.get(&b).copied().unwrap_or(b), tex, slot, sampler));
        }
        bound
    }

    fn stage_buffer_metadata(&mut self,st:&super::GState,stage:usize,c:&super::shaders::Compiled)->Result<Option<super::state::Alloc>,&'static str> {
        if !c.iface.buffer_metadata {return Ok(None);}
        let descriptors={
            let state=super::STATE.lock().unwrap();let mut descriptors=Vec::new();
            for (&binding,&slot) in &c.iface.texture_slots {
                let (id,_)=descriptor_ids(st,stage,binding);
                if let Some(builder)=state.registered_textures.get(&(st.texture_pool,id)) {descriptors.push((slot as usize,*builder));}
            }
            for &binding in c.iface.images.keys() {
                let handle=st.images[stage][binding as usize];
                if let Some(builder)=state.registered_textures.get(&(st.texture_pool,(handle&0xfffff) as u32)) {descriptors.push((256+binding as usize,*builder));}
            }
            descriptors
        };
        let mut metadata=[[0u32;4];264];
        for (slot,builder) in descriptors {
            if u32::from_le_bytes(builder[0x1c..0x20].try_into().unwrap())==10 {
                let (format,mut meta)=super::textures::buffer_descriptor(&builder)?;
                if slot<256 && super::buffer_direct::enabled() {
                    if let Some(direct)=super::buffer_direct::metadata(&self.pools,self.heap,meta,format) {meta=direct;}
                }
                if super::TRACE_DRAWS.load(Relaxed) {eprintln!("[frame] buffer metadata stage={stage} slot={slot} width={} format={} direct={}",meta[2],meta[3]&255,meta[3]>>31);}
                metadata[slot]=meta;
            }
        }
        let bytes=unsafe {std::slice::from_raw_parts(metadata.as_ptr().cast::<u8>(),size_of_val(&metadata))};
        Ok(Some(self.e.arena.alloc_bytes(bytes)))
    }

    fn stage_images(&mut self,st:&super::GState,stage:usize,c:&super::shaders::Compiled) -> Result<Vec<(u32,gfx::Shared<gfx::Texture>,[u32;4])>,&'static str> {
        let descriptors={
            let s=super::STATE.lock().unwrap();
            c.iface.images.iter().map(|(&binding,&image)| {
                let handle=st.images[stage].get(binding as usize).copied().unwrap_or(0);
                let descriptor=s.registered_textures.get(&(st.texture_pool,(handle&0xfffff) as u32)).copied();
                if descriptor.is_none() {
                    for line in &self.e.resource_history {eprintln!("[binding-history] {line}");}
                    eprintln!("[binding-history] program{:x?} renderEnabled{} groups{:?}; UBO0tail@640{:x?}",st.stages(),st.render_enable,self.e.debug_groups,crate::diag::read_words::<2>(st.ubos[stage][0].0+640));
                    let mut ids=s.registered_textures.keys().filter(|(pool,_)|*pool==st.texture_pool).map(|(_,id)|*id).collect::<Vec<_>>();ids.sort_unstable();let count=ids.len();ids.truncate(24);
                    let header=crate::diag::read_words::<3>(st.texture_pool);
                    let mut aliases=Vec::new();
                    for (&(pool,id),_) in &s.registered_textures {
                        if id!=(handle&0xfffff) as u32 || pool==st.texture_pool {continue;}
                        if let Some(other)=crate::diag::read_words::<3>(pool) {
                            if Some(other)==header {aliases.push((pool,other));if aliases.len()==4 {break;}}
                        }
                    }
                    eprintln!("[gfx] missing image: function {} stage{stage} binding{binding} handle{handle:#x} pool{:#x} header{header:x?}; registered count{count} firstIDs{ids:x?}; same-storage registered pools{aliases:x?}; images{:x?}; expected{:?}; UBO0{:x?} CPUview{:x?}",c.func.0.name(),st.texture_pool,st.images[stage],c.iface.images,st.ubos[stage][0],crate::diag::read_words::<8>(st.ubos[stage][0].0));
                }
                (binding,image,descriptor)
            }).collect::<Vec<_>>()
        };
        let mut out=Vec::new();
        for (binding,image,builder) in descriptors {
            let builder=builder.ok_or("storage image descriptor is unbound or unregistered")?;
            let buffer=u32::from_le_bytes(builder[0x1c..0x20].try_into().unwrap())==10;
            if buffer && !c.iface.buffer_metadata {return Err("buffer image requires one-dimensional surface metadata");}
            if !buffer {self.prepare_texture_access(super::textures::TexKey::of(&builder),&builder,false);}
            let view=if buffer {self.e.textures.buffer_image_view(&builder,image)} else {self.e.textures.image_view(&builder,image)};
            let (texture,meta)=view.map_err(|why| {
                eprintln!("[gfx] storage image: stage {stage} binding {binding}, shader source {:?}, image {image:?}, texture {:?}: {why}",c.source,super::textures::TexKey::of(&builder));
                why
            })?;
            if !super::write_windows::enabled() {self.e.gpu_buffer_dirty |= buffer;}
            out.push((binding,texture,meta));
        }
        Ok(out)
    }

    pub(super) fn dispatch(&mut self, groups: [u32; 3]) -> Result<(), &'static str> {
        self.dispatch_with(groups,None)
    }

    pub(super) fn dispatch_indirect(&mut self,address:u64)->Result<(),&'static str> {
        if address%4!=0 {return Err("indirect compute argument alignment");}
        let (buffer,offset)=self.buffer_range(address,12).map(|(chunk,offset)|(chunk.buffer.clone(),offset)).ok_or("indirect compute arguments are not mapped to Metal")?;
        self.dispatch_with([1,1,1],Some((buffer,offset)))
    }

    fn dispatch_with(&mut self,groups:[u32;3],indirect:Option<(gfx::Shared<gfx::Buffer>,usize)>)->Result<(),&'static str> {
        if groups.contains(&0) {
            return Ok(());
        }
        self.heap = gfx::heap();
        let st = self.e.st;
        self.prepare_stage_cpu_reads(&st,true);
        let s = st.stages().iter().find_map(|&(code, ctl)| self.e.shaders.stage(code, ctl).filter(|s| s.local_size.is_some())).ok_or("dispatch without a compute stage")?;
        let c = match self.e.shaders.function(&s, 0, 0, 0, false) {
            super::shaders::Lookup::Ready(c) => c,
            super::shaders::Lookup::Pending => return Err("compute shader compiling"),
            super::shaders::Lookup::Failed => return Err("compute shader failed"),
        };
        // Bounded by profile-submit (at most four submissions). Observe only
        // immutable uniform-overlay bytes; never add a GPU wait or inspect an
        // unsynchronized live parameter merely for this provenance probe.
        if super::submit_profile::active() && s.global_writes {
            let (addr,len)=st.ubos[2][0];
            let row=addr.checked_add(64).filter(|_|len>=68);
            let pending=row.is_some_and(|p|self.pending_cpu_write(p,4));
            let overlay=row.and_then(|p|self.e.overlays.range(..=p).next_back().and_then(|(&base,&(size,_,ptr))|
                (p-base<=size && size-(p-base)>=4).then(||unsafe{((ptr as u64+p-base) as *const u32).read_unaligned()})));
            eprintln!("[grid-writer-input] hash={:016x} groups={groups:?} unknown_pending={} ubo0={addr:#x}+{len:#x} row_pending={pending} immutable_overlay_row={overlay:?}",s.hash,self.e.gpu_buffer_dirty);
        }
        let local=s.local_size.unwrap();
        let thread_count=local.into_iter().map(|n|n as u64).product::<u64>();
        if thread_count>1024 {
            eprintln!("[gfx] compute {:016x} invalid original threadgroup {local:?}, dispatch {groups:?}",s.hash);
            return Err("original compute threadgroup exceeds the device limit");
        }
        let pipeline = match self.e.kernels.get(&s.hash) {
            Some(p) => p.clone(),
            None => {
                // Supply the original block size when compiling. The default
                // pipeline may reserve registers for fewer threads (384 on
                // M1 Max), although this function can support 1024 when the
                // compiler knows the required occupancy. Dispatch dimensions,
                // shared memory and barriers remain those of the original.
                let descriptor=MTLComputePipelineDescriptor::new();
                descriptor.setComputeFunction(Some(&c.func.0));
                descriptor.setMaxTotalThreadsPerThreadgroup((thread_count as usize).next_multiple_of(32));
                let p = gfx::gpu().device.0.newComputePipelineStateWithDescriptor_options_reflection_error(&descriptor,MTLPipelineOption::empty(),None).map_err(|error| {
                    eprintln!("[gfx] compute {:016x}, local {local:?}, shared {}: {}",s.hash,s.shared_memory,error.localizedDescription());
                    "compute pipeline failed"
                })?;
                let p = gfx::Shared(p);
                self.e.kernels.insert(s.hash, p.clone());
                p
            }
        };
        if super::TRACE_DRAWS.load(Relaxed) {eprintln!("[frame] compute {:016x} groups{groups:?} local{:?}",s.hash,s.local_size);}
        let textures = self.stage_textures(&st, 2, &c);
        if pipeline.0.threadExecutionWidth()!=32 {return Err("translated Maxwell compute requires 32-lane SIMD execution");}
        if thread_count>pipeline.0.maxTotalThreadsPerThreadgroup() as u64 {
            eprintln!("[gfx] compute {:016x}: original {local:?}, Metal maximum {}, dispatch {groups:?}",s.hash,pipeline.0.maxTotalThreadsPerThreadgroup());
            return Err("compute threadgroup exceeds this Metal pipeline limit");
        }
        let images=self.stage_images(&st,2,&c)?;
        let buffer_metadata=self.stage_buffer_metadata(&st,2,&c)?;
        let atomic_images=self.prepare_atomic_images(&images,c.iface.atomic_images)?;
        let bindless=self.stage_bindless(&st,&c)?;
        let static_arguments=self.stage_static_arguments(&c,&textures)?;
        let requested=std::env::var("NX_PROBE_COMPUTE_HASH").ok().is_some_and(|value|value.split(',').filter_map(|v|u64::from_str_radix(v.trim(),16).ok()).any(|hash|hash==s.hash || hash==s.base_hash));
        let marked=std::fs::read_to_string("work/probe-compute-hashes").ok().is_some_and(|value|value.split(',').filter_map(|v|u64::from_str_radix(v.trim(),16).ok()).any(|hash|hash==s.hash || hash==s.base_hash));
        static MARKED_SEEN:std::sync::LazyLock<std::sync::Mutex<std::collections::HashSet<u64>>>=std::sync::LazyLock::new(Default::default);
        let requested=requested || (marked && MARKED_SEEN.lock().unwrap().insert(s.hash));
        // The explicit marker plus one-frame trace bounds whole-chain capture.
        // Ordinary startup screenshots never activate this diagnostic.
        let color_chain=super::TRACE_DRAWS.load(Relaxed) && std::path::Path::new("work/probe-color-chain").exists() && (s.global_writes || images.iter().any(|(binding,_,_)| {
            let id=st.images[2][*binding as usize] as u32&0xfffff;
            let b=super::super::objects::STATE.lock().unwrap().registered_textures.get(&(st.texture_pool,id)).copied();
            b.is_some_and(|b| {
                let u=|o|u32::from_le_bytes(b[o..o+4].try_into().unwrap());
                let (target,w,h,f)=(u(0x1c),u(0x20),u(0x24),u(0x30));
                target==1 && ((61..=63).contains(&f) || (matches!(f,10|17|41|46) && w as u64*h as u64<=4096))
            })
        }));
        static CHAIN_SEEN:std::sync::LazyLock<std::sync::Mutex<std::collections::HashSet<u64>>>=std::sync::LazyLock::new(Default::default);
        let requested=requested || (color_chain && CHAIN_SEEN.lock().unwrap().insert(s.hash));
        static COUNT:std::sync::atomic::AtomicU32=std::sync::atomic::AtomicU32::new(0);
        let probe_index=requested.then(||COUNT.fetch_add(1,Relaxed)).filter(|index|*index<if color_chain {32}else{8});
        let profile=super::TRACE_DRAWS.load(Relaxed) && std::path::Path::new("work/profile-gpu-steps").exists();
        if probe_index.is_some() || profile {self.commit_boundary(true);}
        let mut res = self.res(&st, 2, s.constants, 0);res.shader_hash=s.hash;
        let mut probe_output_directory=None;
        if let Some(capture_index)=probe_index {
            {
                let read=|ptr:u64,count:usize| {
                    let buffers=self.e.arena.used.iter().chain(self.pools.iter().flat_map(|p|p.chunks.iter().map(|c|&c.buffer))).chain(self.heap.into_iter().flat_map(|h|h.chunks.iter().map(|c|&c.buffer)));
                    for buffer in buffers {
                        let base=buffer.0.gpuAddress();
                        if ptr>=base && ptr-base+(count*4) as u64<=buffer.0.length() as u64 {
                            let words=unsafe {std::slice::from_raw_parts((buffer.0.contents().as_ptr() as *const u8).add((ptr-base) as usize).cast::<u32>(),count)};
                            return Some(words.to_vec());
                        }
                    }
                    None
                };
                eprintln!("[probe] compute {:016x} groups {groups:?} local {:?} shared {} localMemory {} UBO0 {:?} CB3ptr {:#x} words {:x?}",s.hash,s.local_size,s.shared_memory,s.local_memory,st.ubos[2][0],res.cb[3],read(res.cb[3],64));
                eprintln!("[probe] storage {:x?}; CB0 {:x?}",st.storage[2],read(res.cb[0],256));
                eprintln!("[probe] embedded constants guest {:#x} GPU {:#x} CB1 {:x?}",s.constants,res.cb[1],read(res.cb[1],64));
                let _=std::fs::create_dir_all("work/compute-probes");
                let directory=format!("work/compute-probes/{:016x}-{capture_index}",s.hash);
                let _=std::fs::create_dir_all(&directory);
                probe_output_directory=Some(directory.clone());
                let mut metadata=format!("shader={:016x}\ngroups={groups:?}\nlocal={:?}\nlocal_memory={}\nshared_memory={}\n",s.hash,s.local_size.unwrap(),s.local_memory,s.shared_memory);
                for (binding,texture,_,_) in textures.iter().take(8) {
                    let logical=c.iface.texture_slots.iter().find_map(|(logical,slot)|(*slot==*binding).then_some(*logical));
                    if let Some(logical)=logical {
                        let (id,sampler_id)=descriptor_ids(&st,2,logical);
                        let sampler=super::super::objects::STATE.lock().unwrap().registered_samplers.get(&(st.sampler_pool,sampler_id)).copied();
                        if let Some(sampler)=sampler {let _=std::fs::write(format!("{directory}/input{binding}.sampler"),sampler);}
                        let builder=super::super::objects::STATE.lock().unwrap().registered_textures.get(&(st.texture_pool,id)).copied();
                        if let Some(builder)=builder {
                            if let Some(key)=super::textures::TexKey::of(&builder) {
                                metadata.push_str(&format!("input{binding} logical{logical} id{id}: {key:?}\n"));
                                let _=std::fs::write(format!("{directory}/input{binding}.builder"),builder);
                                if texture.0.textureType()==MTLTextureType::Type3D {
                                    if let Some((shape,layout))=super::super::texture_layout(builder.as_ptr() as u64) {
                                        let _=std::fs::write(format!("{directory}/input{binding}.layout"),format!("shape{shape:?} layout{layout:?}"));
                                        if layout.total_size<=4*1024*1024 && crate::kernel::aspace().readable(key.addr,layout.total_size) {
                                            let data=unsafe{std::slice::from_raw_parts(key.addr as *const u8,layout.total_size as usize)};
                                            let _=std::fs::write(format!("{directory}/input{binding}.cpu"),data);
                                        }
                                    }
                                }
                                if matches!(texture.0.textureType(),MTLTextureType::Type2D|MTLTextureType::Type3D) {super::dump_volume(&texture.0,&key,&format!("{directory}/input{binding}"));}
                            }
                        }
                    }
                }
                for (binding,texture,conversion) in &images {
                    let id=st.images[2][*binding as usize] as u32&0xfffff;
                    let builder=super::super::objects::STATE.lock().unwrap().registered_textures.get(&(st.texture_pool,id)).copied();
                    metadata.push_str(&format!("output{binding}: {:?}, conversion{conversion:?}, size{}x{}x{}\n",builder.as_ref().and_then(super::textures::TexKey::of),texture.0.width(),texture.0.height(),texture.0.depth()));
                    if let Some(key)=builder.as_ref().and_then(super::textures::TexKey::of) {
                        if key.target!=10 && matches!(texture.0.textureType(),MTLTextureType::Type2D|MTLTextureType::Type3D) {super::dump_volume(&texture.0,&key,&format!("{directory}/before{binding}"));}
                    }
                    if let Some(builder)=builder {let _=std::fs::write(format!("{directory}/output{binding}.builder"),builder);}
                }
                for bank in [1usize,3] {
                    if let Some(words)=read(res.cb[bank],if bank==1 {1024} else {64}) {
                        let bytes:Vec<u8>=words.iter().flat_map(|v|v.to_le_bytes()).collect();
                        let _=std::fs::write(format!("work/compute-probes/{:016x}.cb{bank}.bin",s.hash),bytes);
                    }
                }
                for (bank,count) in [(0usize,512usize),(1,1024)] {
                    if let Some(words)=read(res.cb[bank],count).or_else(||read(res.cb[bank],64)) {
                        let bytes=words.iter().flat_map(|v|v.to_le_bytes()).collect::<Vec<_>>();
                        let _=std::fs::write(format!("{directory}/cb{bank}.bin"),bytes);
                    }
                }
                for (index,&(address,size)) in st.ubos[2].iter().enumerate() {
                    if address==0 || size==0 {continue;}
                    let count=(size.min(65536)/4) as usize;
                    metadata.push_str(&format!("cb{}={address:#x},{size:#x}\n",index+3));
                    if let Some(words)=read(res.cb[index+3],count) {
                        let bytes=words.iter().flat_map(|v|v.to_le_bytes()).collect::<Vec<_>>();
                        let _=std::fs::write(format!("{directory}/cb{}.bin",index+3),bytes);
                    }
                }
                for (i,&(address,size)) in st.storage[2].iter().enumerate() {
                    if address!=0 && size!=0 {
                        metadata.push_str(&format!("storage{i}={address:#x},{size:#x}\n"));
                        if let Ok(bytes)=self.cpu_bytes(address,size.min(65536) as usize) {
                            let _=std::fs::write(format!("work/compute-probes/{:016x}.storage{i}.bin",s.hash),&bytes);
                            let _=std::fs::write(format!("{directory}/storage{i}.bin"),bytes);
                        }
                    }
                }
                let _=std::fs::write(format!("{directory}/metadata.txt"),metadata);
            }
        }
        if super::submit_profile::active() && super::write_windows::enabled() {
            eprintln!("[cpu-write-window] candidate hash={:016x} proof={} image_bindings={}",s.hash,c.write_window_footprint.as_ref().map_or(0,|p|p.len()),c.iface.images.len());
        }
        let mut used_window=false;
        let known_global_writes=if !s.global_writes {Some(Vec::new())} else {
            let exact=if indirect.is_none() {
                c.grid_write_footprint.and_then(|proof|self.grid_write_ranges(proof,&st,&res,groups))
                    .or_else(||c.descriptor_write_footprint.as_deref().and_then(|proof|self.descriptor_write_ranges(proof,&st)))
            } else {None};
            // A high-word domain covers every possible low word/group, so it
            // also applies when the dispatch dimensions are generated by GPU.
            exact.or_else(|| {let ranges=c.write_window_footprint.as_deref().and_then(|proof|self.window_write_ranges(proof,&st,2));used_window=ranges.is_some();ranges})
        };
        let known_shader_writes=self.shared_write_ranges(&st,2,&c,known_global_writes);
        let shared_image_writes=images.iter().any(|(b,_,meta)|meta[3]!=0 && c.iface.image_writes&(1u32<<b)!=0);
        used_window|=super::write_windows::enabled() && shared_image_writes && known_shader_writes.is_some();
        res.fault_gpu=self.fault_record();
        let write_buffers=known_shader_writes.as_deref().map(|ranges|self.proven_write_buffers(ranges));
        let boundary_counter=if super::submit_profile::active() {gfx::timestamp_samples(2)} else {None};
        if let Some(bindings)=&bindless {bindings.retain(&self.cb);}
        if let Some(bindings)=&static_arguments {bindings.retain(&self.cb);}
        let Some(enc) = gfx::begin_compute_sampled(&self.cb,boundary_counter.as_ref().map(|c|&*c.0)) else { return Err("no compute encoder") };
        enc.setComputePipelineState(&pipeline.0);
        let buffer_usage=if ((s.global_writes || shared_image_writes) && write_buffers.is_none()) || (!super::write_windows::enabled() && images.iter().any(|(_,_,meta)|meta[3]!=0)) {MTLResourceUsage::Read|MTLResourceUsage::Write} else {MTLResourceUsage::Read};
        if let Some(h) = self.heap {
            for chunk in &h.chunks {
                enc.useResource_usage(ProtocolObject::from_ref(&*chunk.buffer.0),buffer_usage);
            }
        }
        let resources=self.pool_resources();
        if !resources.is_empty() {unsafe {enc.useResources_count_usage(NonNull::new(resources.as_ptr().cast_mut()).unwrap(),resources.len(),buffer_usage);}}
        if let Some(buffers)=write_buffers {for b in buffers {enc.useResource_usage(ProtocolObject::from_ref(&*b.0),MTLResourceUsage::Write);}}
        let isolated=super::fault_records::enabled();
        enc.useResource_usage(ProtocolObject::from_ref(&*gfx::gpu().zeros.0),if isolated {MTLResourceUsage::Read} else {MTLResourceUsage::Read|MTLResourceUsage::Write});
        let arena_usage=if isolated && atomic_images.is_empty() {MTLResourceUsage::Read} else {MTLResourceUsage::Read|MTLResourceUsage::Write};
        for b in &self.e.arena.used {enc.useResource_usage(ProtocolObject::from_ref(&*b.0),arena_usage);}
        if let Some(fault)=&self.fault {enc.useResource_usage(ProtocolObject::from_ref(&*fault.buffer.0),MTLResourceUsage::Read|MTLResourceUsage::Write);}
        if let Some(table)=&self.e.address_table.buffer {
            enc.useResource_usage(ProtocolObject::from_ref(&*table.0),MTLResourceUsage::Read);
        }
        unsafe {
            enc.setBytes_length_atIndex(NonNull::from(&res).cast(), size_of::<Res>(), 0);
            if let Some(a)=&buffer_metadata {enc.setBuffer_offset_atIndex(Some(&a.buffer.0),a.offset,13);}
            for (index,bindings) in [(2,&bindless),(12,&static_arguments)] {
                if let Some(bindings)=bindings {
                    for texture in &bindings.textures {enc.useResource_usage(ProtocolObject::from_ref(&*texture.0),MTLResourceUsage::Read);}
                    let a=&bindings.allocation;enc.setBuffer_offset_atIndex(Some(&a.buffer.0),a.offset,index);
                }
            }
            if static_arguments.is_none() {
                for (b, tex, slot, sampler) in &textures {
                    enc.setTexture_atIndex(Some(&tex.0), *b as usize);
                    enc.setSamplerState_atIndex(Some(&sampler.0), *slot as usize);
                }
            }
            if !images.is_empty() {
                let mut meta=[[0u32;4];8];
                for (binding,texture,descriptor) in &images {
                    meta[*binding as usize]=*descriptor;
                    enc.setTexture_atIndex(Some(&texture.0),32+*binding as usize);
                }
                enc.setBytes_length_atIndex(NonNull::from(&meta).cast(),size_of_val(&meta),1);
            }
            if !atomic_images.is_empty() {
                let mut layout=[[0u32;4];8];
                for image in &atomic_images {
                    layout[image.binding as usize]=image.layout;let a=&image.allocation;
                    enc.setBuffer_offset_atIndex(Some(&a.buffer.0),a.offset,3+image.binding as usize);
                }
                enc.setBytes_length_atIndex(NonNull::from(&layout).cast(),size_of_val(&layout),11);
            }
        }
        let [x, y, z] = s.local_size.unwrap().map(|n| n as usize);
        let [gx, gy, gz] = groups.map(|n| n as usize);
        let local=MTLSize{width:x,height:y,depth:z};
        if let Some((buffer,offset))=indirect {
            unsafe {enc.dispatchThreadgroupsWithIndirectBuffer_indirectBufferOffset_threadsPerThreadgroup(&buffer.0,offset,local);}
        } else {enc.dispatchThreadgroups_threadsPerThreadgroup(MTLSize { width: gx, height: gy, depth: gz },local);}
        enc.endEncoding();
        if let Some(counter)=boundary_counter {super::submit_profile::watch(&self.cb,counter,format!("compute={:016x} groups={groups:?} local={:?}",s.hash,s.local_size),2);}
        for image in &atomic_images {image.copy(&self.cb,true)?;}
        if s.global_writes || shared_image_writes {
            if let Some(ranges)=known_shader_writes {
                if super::submit_profile::active() {eprintln!("[cpu-write] compute proven hash={:016x} ranges={ranges:x?}",s.hash);}
                if used_window && !ranges.is_empty() {self.e.window_batch_writes=true;}
                if used_window && super::submit_profile::active() {eprintln!("[cpu-write-window] hash={:016x} mapped_ranges={}",s.hash,ranges.len());}
                for (gpu,cpu,len) in ranges {self.record_known_write(gpu,cpu,len);}
            } else {
                self.e.gpu_buffer_dirty=true;
                if super::submit_profile::active() {eprintln!("[cpu-write] compute unknown hash={:016x}",s.hash);}
            }
        }
        if probe_output_directory.is_some() || profile {
            let completed=self.commit_boundary(true);
            if profile {eprintln!("[gpu-step] compute hash={:016x} groups={groups:?} local={local:?} gpu_ms={:.6}",s.hash,(completed.GPUEndTime()-completed.GPUStartTime())*1000.0);}
        }
        if let Some(directory)=probe_output_directory {
            for (i,&(address,size)) in st.storage[2].iter().enumerate() {
                if address!=0 && size!=0 {if let Ok(bytes)=self.cpu_bytes(address,size.min(65536) as usize) {let _=std::fs::write(format!("{directory}/after-storage{i}.bin"),bytes);}}
            }
            for (binding,texture,_) in &images {
                if !matches!(texture.0.textureType(),MTLTextureType::Type2D|MTLTextureType::Type3D) {continue;}
                let id=st.images[2][*binding as usize] as u32&0xfffff;
                let builder=super::super::objects::STATE.lock().unwrap().registered_textures.get(&(st.texture_pool,id)).copied();
                if let Some(key)=builder.as_ref().and_then(super::textures::TexKey::of) {super::dump_volume(&texture.0,&key,&format!("{directory}/output{binding}"));}
            }
        }
        Ok(())
    }

    pub(super) fn res(&mut self, st: &super::GState, stage: usize, constants: u64, flags: u32) -> Res {
        let zeros = gfx::gpu().zeros.0.gpuAddress();
        let mut r = Res {
            cb: [zeros; 18],
            vb: [zeros; 16],
            vb_stride: [0; 16],
            vb_divisor: [0; 16],
            attr_format: [0; 16],
            attr_stream: [0; 16],
            attr_offset: [0; 16],
            base_vertex: 0,
            base_instance: 0,
            flags,
            alpha_ref: st.alpha_ref,
            alpha_func: 7,
            heap_base: 0,
            heap_gpu: [0; gfx::MAX_HEAP_CHUNKS],
            heap_size: 0,
            heap_chunk_size: 0,
            zero_gpu: zeros,
            pool_spans: 0,
            pool_span_count: 0,
            fault_gpu:self.fault_record(),
            shader_hash:0,
            pool_root_base:0,pool_root_count:0,pool_roots:0,pool_pages:0,
        };
        if let Some(bindings)=self.pool_table {r.apply_pool_bindings(bindings);}
        else if super::address_table::enabled() {
            let (table,bindings)=self.e.address_table.bind(&self.pools);
            r.apply_pool_bindings(bindings);self.pool_table=Some(bindings);self.use_chunk(&table);
            let keep=table;
            let handler=block2::RcBlock::new(move |_:std::ptr::NonNull<gfx::CommandBuffer>|{std::hint::black_box(&keep);});
            unsafe {self.cb.addCompletedHandler(block2::RcBlock::as_ptr(&handler));}
        }
        else {
            let table=r.set_pools(&mut self.e.arena,&self.pools);
            if table.new_chunk {self.use_chunk(&table.buffer);}
            self.pool_table=Some(r.pool_bindings());
        }
        if super::TRACE_DRAWS.load(Relaxed) {eprintln!("[frame] bound address table: {} spans, {} page roots, flat64k={}, buffer_fast={}",r.pool_span_count,r.pool_root_count,r.pool_root_base>>63,super::shaders::buffer_read_enabled());}
        if let Some(h) = self.heap {
            (r.heap_base, r.heap_gpu, r.heap_size, r.heap_chunk_size) = (h.base, h.gpu_addresses(), h.size, h.chunk_size);
        }
        if st.storage[stage].iter().any(|s| s.0 != 0) {
            let mut bank0 = [0u32; 0x200];
            // Pinned SDK 55.15 constant-bank ABI: vertex 0x110, fragment
            // 0x510, compute 0x310. Each SSBO descriptor is 16 bytes. FC27's
            // original programs read C0[196/200] and fragment C0[324].
            let base=[0x44,0x144,0xc4][stage];
            for (i, &(addr, size)) in st.storage[stage].iter().enumerate() {
                let w = base + 4 * i;
                bank0[w] = addr as u32;
                bank0[w + 1] = (addr >> 32) as u32;
                bank0[w + 2] = size as u32;
            }
            let a = self.e.arena.alloc_bytes(unsafe { std::slice::from_raw_parts(bank0.as_ptr() as *const u8, 4 * bank0.len()) });
            r.cb[0] = a.gpu;
            if a.new_chunk {
                self.use_chunk(&a.buffer);
            }
        }
        r.cb[1] = self.gpu_range(constants, 0x10000, false);
        let handles = st.textures[stage];
        if self.e.bank_ok[stage] && self.e.bank_tex[stage] == handles {
            r.cb[2] = self.e.bank_gpu[stage];
        } else {
            let mut bank2 = [0u32; 0x270];
            for (i, h) in handles.iter().enumerate() {
                bank2[8 + 2 * i] = *h as u32;
                bank2[9 + 2 * i] = (*h >> 32) as u32;
            }
            for (i,h) in st.images[stage].iter().enumerate() {bank2[0x120/4+2*i]=*h as u32;}
            for (i,h) in st.separate_textures[stage].iter().enumerate() {bank2[0x168/4+2*i]=*h as u32;}
            for (i,h) in st.separate_samplers[stage].iter().enumerate() {bank2[0x568/4+2*i]=*h as u32;}
            let a = self.e.arena.alloc_bytes(unsafe { std::slice::from_raw_parts(bank2.as_ptr() as *const u8, 4 * bank2.len()) });
            r.cb[2] = a.gpu;
            if a.new_chunk {
                self.use_chunk(&a.buffer);
            }
            self.e.bank_tex[stage] = handles;
            self.e.bank_gpu[stage] = a.gpu;
            self.e.bank_ok[stage] = true;
        }
        for (i, &(addr, size)) in st.ubos[stage].iter().enumerate() {
            if addr != 0 {
                r.cb[3 + i] = self.gpu_range(addr, size, true);
            }
        }
        r
    }

    fn primitives(&mut self, d: &super::Draw) -> Result<Prim, &'static str> {
        d.validate_indices()?;
        let (ty, convert): (MTLPrimitiveType, Option<fn(&[u32]) -> Vec<u32>>) = match d.mode {
            0 => (MTLPrimitiveType::Point, None),
            1 => (MTLPrimitiveType::Line, None),
            2 => (MTLPrimitiveType::Line, Some(line_loop)),
            3 => (MTLPrimitiveType::LineStrip, None),
            4 => (MTLPrimitiveType::Triangle, None),
            5 => (MTLPrimitiveType::TriangleStrip, None),
            6 | 9 => (MTLPrimitiveType::Triangle, Some(fan)),
            7 => (MTLPrimitiveType::Triangle, Some(quads)),
            8 => (MTLPrimitiveType::Triangle, Some(quad_strip)),
            _ => return Err("unsupported primitive"),
        };
        let index_size = |t: u32| match t {
            0 => 1u64,
            1 => 2,
            _ => 4,
        };
        let read_indices = |t: u32, addr: u64, n: u32| -> Vec<u32> {
            (0..n as u64)
                .map(|i| match t {
                    0 => rd::<u8>(addr + i) as u32,
                    1 => rd::<u16>(addr + 2 * i) as u32,
                    _ => rd::<u32>(addr + 4 * i),
                })
                .collect()
        };
        let upload32 = |r: &mut Self, v: &[u32]| {
            let a = r.e.arena.alloc_bytes(unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, 4 * v.len()) });
            if a.new_chunk {
                r.use_chunk(&a.buffer);
            }
            (a.buffer, a.offset)
        };
        match (d.index, convert) {
            (None, None) => Ok(Prim::Direct(ty)),
            (None, Some(f)) => {
                let src: Vec<u32> = (d.first..d.first + d.count).collect();
                let v = f(&src);
                let (buffer, offset) = upload32(self, &v);
                Ok(Prim::Indexed { ty, itype: MTLIndexType::UInt32, buffer, offset, count: v.len(), base_vertex: 0 })
            }
            (Some((t, addr)), conv) => {
                let size = index_size(t);
                let range = self.buffer_range(addr, size * d.count as u64);
                if let Some((chunk, off)) = range.filter(|_| conv.is_none() && t != 0 && addr % size == 0) {
                    let itype = if t == 1 { MTLIndexType::UInt16 } else { MTLIndexType::UInt32 };
                    return Ok(Prim::Indexed { ty, itype, buffer: chunk.buffer.clone(), offset: off, count: d.count as usize, base_vertex: d.base_vertex });
                }
                if !crate::kernel::aspace().readable(addr, size * d.count as u64) {
                    static WARNED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
                    if !WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                        eprintln!("[gfx] draw skipped: its {} indices at {addr:#x} are outside guest memory", d.count);
                    }
                    return Err("index buffer outside guest memory");
                }
                let src = read_indices(t, addr, d.count);
                let v = match conv {
                    Some(f) => f(&src),
                    None => src,
                };
                let (buffer, offset) = upload32(self, &v);
                Ok(Prim::Indexed { ty, itype: MTLIndexType::UInt32, buffer, offset, count: v.len(), base_vertex: d.base_vertex })
            }
        }
    }
}

fn fbits(x: f32) -> u32 {
    x.to_bits()
}

fn bind_fixed(
    enc: &crate::gfx::RenderEncoder,
    prev: Option<super::EncState>,
    pipeline: &gfx::Shared<super::state::Pipeline>,
    ds: &gfx::Shared<super::state::DepthState>,
    st: &super::GState,
    upper_left: bool,
    target: (u32, u32),
    scissor: [u32; 4],
) -> super::EncState {
    let p = st.polygon;
    let cull = p & 3;
    let ccw = (p >> 2 & 1 != 0) == upper_left;
    let lines = p >> 3 & 3 == 1;
    let bias = if p >> 15 & (1 << (p >> 3 & 3)) != 0 { st.polygon_offset } else { [0.0; 3] };
    let (w, h) = target;
    let [vx, vy, vw, vh] = st.viewport.unwrap_or([0.0, 0.0, w as f32, h as f32]);
    let next = super::EncState {
        pipeline: std::ptr::from_ref(&*pipeline.0) as usize,
        depth: std::ptr::from_ref(&*ds.0) as usize,
        cull,
        winding: u32::from(ccw),
        fill: u32::from(lines),
        bias: bias.map(fbits),
        depth_clamp: st.depth_clamp,
        origin: [fbits(vx), fbits(vy), fbits(vw), fbits(vh)],
        zrange: [fbits(st.depth_range.0), fbits(st.depth_range.1)],
        scissor,
        stencil: st.stencil_ref,
        blend: st.blend_color.map(fbits),
    };
    if prev.is_none_or(|e| e.pipeline != next.pipeline) {
        enc.setRenderPipelineState(&pipeline.0);
    }
    if prev.is_none_or(|e| e.depth != next.depth) {
        enc.setDepthStencilState(Some(&ds.0));
    }
    if prev.is_none_or(|e| e.cull != next.cull) {
        enc.setCullMode(match cull {
            1 => MTLCullMode::Front,
            2 => MTLCullMode::Back,
            _ => MTLCullMode::None,
        });
    }
    if prev.is_none_or(|e| e.winding != next.winding) {
        enc.setFrontFacingWinding(if ccw { MTLWinding::CounterClockwise } else { MTLWinding::Clockwise });
    }
    if prev.is_none_or(|e| e.fill != next.fill) {
        enc.setTriangleFillMode(if lines { MTLTriangleFillMode::Lines } else { MTLTriangleFillMode::Fill });
    }
    if prev.is_none_or(|e| e.bias != next.bias) {
        enc.setDepthBias_slopeScale_clamp(bias[1], bias[0], bias[2]);
    }
    if prev.is_none_or(|e| e.depth_clamp != next.depth_clamp) {
        enc.setDepthClipMode(if st.depth_clamp { MTLDepthClipMode::Clamp } else { MTLDepthClipMode::Clip });
    }
    if prev.is_none_or(|e| e.origin != next.origin || e.zrange != next.zrange) {
        enc.setViewport(MTLViewport {
            originX: vx as f64,
            originY: vy as f64,
            width: vw as f64,
            height: vh as f64,
            znear: st.depth_range.0 as f64,
            zfar: st.depth_range.1 as f64,
        });
    }
    if prev.is_none_or(|e| e.scissor != next.scissor) {
        let [x, y, sw, sh] = scissor;
        enc.setScissorRect(MTLScissorRect { x: x as usize, y: y as usize, width: sw as usize, height: sh as usize });
    }
    if prev.is_none_or(|e| e.stencil != next.stencil) {
        enc.setStencilFrontReferenceValue_backReferenceValue(st.stencil_ref[0], st.stencil_ref[1]);
    }
    if prev.is_none_or(|e| e.blend != next.blend) {
        let c = st.blend_color;
        enc.setBlendColorRed_green_blue_alpha(c[0], c[1], c[2], c[3]);
    }
    next
}

fn gfx_is_depth(f: MTLPixelFormat) -> bool {
    matches!(f, MTLPixelFormat::Depth16Unorm | MTLPixelFormat::Depth32Float | MTLPixelFormat::Depth32Float_Stencil8)
}

pub(super) enum Prim {
    Direct(MTLPrimitiveType),
    Indexed { ty: MTLPrimitiveType, itype: MTLIndexType, buffer: gfx::Shared<gfx::Buffer>, offset: usize, count: usize, base_vertex: i32 },
}

fn line_loop(v: &[u32]) -> Vec<u32> {
    let mut o = Vec::with_capacity(v.len() * 2);
    for i in 0..v.len() {
        o.extend([v[i], v[(i + 1) % v.len()]]);
    }
    o
}

fn fan(v: &[u32]) -> Vec<u32> {
    (1..v.len().saturating_sub(1)).flat_map(|i| [v[0], v[i], v[i + 1]]).collect()
}

fn quads(v: &[u32]) -> Vec<u32> {
    v.chunks_exact(4).flat_map(|q| [q[0], q[1], q[2], q[0], q[2], q[3]]).collect()
}

fn quad_strip(v: &[u32]) -> Vec<u32> {
    (0..v.len().saturating_sub(2) / 2).flat_map(|i| {
        let q = &v[2 * i..2 * i + 4];
        [q[0], q[1], q[3], q[0], q[3], q[2]]
    }).collect()
}
