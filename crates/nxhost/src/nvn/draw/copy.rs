// Mush8 - copy.rs
// Constantin Clerc (c22dev) - 0.1
use crate::gfx::{self, Attachments, Load, Shared, Texture};
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSRange, NSString};
use objc2_metal::*;
use std::collections::HashMap;

type Pipeline = ProtocolObject<dyn MTLRenderPipelineState>;
type Sampler = ProtocolObject<dyn MTLSamplerState>;

pub const COPY_LINEAR_FILTER: u32 = 1;
pub const COPY_MIRROR_X: u32 = 4;
pub const COPY_MIRROR_Y: u32 = 8;

const MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;
struct P { float4 src; float4 color; };
struct V { float4 pos [[position]]; float2 uv; };
vertex V vs(uint id [[vertex_id]], constant P& p [[buffer(0)]]) {
    float2 q = float2((id << 1) & 2, id & 2);
    V v;
    v.pos = float4(q * 2.0 - 1.0, 0.0, 1.0);
    v.uv = mix(p.src.xy, p.src.zw, float2(q.x, 1.0 - q.y));
    return v;
}
fragment float4 copy(V v [[stage_in]], texture2d<float> t [[texture(0)]], sampler s [[sampler(0)]]) {
    return t.sample(s, v.uv, level(0.0));
}
fragment float4 fill(constant P& p [[buffer(0)]]) {
    return p.color;
}
"#;

#[repr(C)]
#[derive(Clone, Copy)]
struct Params {

    src: [f32; 4],
    color: [f32; 4],
}

type Function = ProtocolObject<dyn MTLFunction>;

#[derive(Clone)]
struct Lib {
    vs: Shared<Function>,
    copy: Shared<Function>,
    fill: Shared<Function>,
    linear: Shared<Sampler>,
    nearest: Shared<Sampler>,
}

#[derive(Default)]
pub struct Copier {
    lib: Option<Lib>,

    pipelines: HashMap<(MTLPixelFormat, u8, bool), Shared<Pipeline>>,
}

pub type Region = [i32; 6];

fn drawable(t: &Texture) -> bool {
    let f = t.pixelFormat();
    super::shaders::integer_kind(f) == 0 && !is_depth(f) && !is_block_compressed(f) && t.textureType() != MTLTextureType::Type3D
}

fn is_depth(f: MTLPixelFormat) -> bool {
    matches!(f, MTLPixelFormat::Depth16Unorm | MTLPixelFormat::Depth32Float | MTLPixelFormat::Depth32Float_Stencil8 | MTLPixelFormat::Stencil8)
}

fn is_block_compressed(f: MTLPixelFormat) -> bool {

    (130..=234).contains(&f.0)
}

fn layered(t: &Texture) -> bool {
    matches!(t.textureType(), MTLTextureType::Type2DArray | MTLTextureType::TypeCube | MTLTextureType::TypeCubeArray)
}

fn level_size(t: &Texture, level: u32) -> (usize, usize) {
    ((t.width() >> level).max(1), (t.height() >> level).max(1))
}

impl Copier {
    fn lib(&mut self) -> Option<Lib> {
        if self.lib.is_none() {
            let dev = &gfx::gpu().device.0;
            let lib = dev.newLibraryWithSource_options_error(&NSString::from_str(MSL), None).ok()?;
            let f = |n: &str| lib.newFunctionWithName(&NSString::from_str(n)).map(Shared);
            let sampler = |filter| {
                let d = MTLSamplerDescriptor::new();
                d.setMinFilter(filter);
                d.setMagFilter(filter);
                d.setSAddressMode(MTLSamplerAddressMode::ClampToEdge);
                d.setTAddressMode(MTLSamplerAddressMode::ClampToEdge);
                dev.newSamplerStateWithDescriptor(&d).map(Shared)
            };
            self.lib = Some(Lib {
                vs: f("vs")?,
                copy: f("copy")?,
                fill: f("fill")?,
                linear: sampler(MTLSamplerMinMagFilter::Linear)?,
                nearest: sampler(MTLSamplerMinMagFilter::Nearest)?,
            });
        }
        self.lib.clone()
    }

    fn pipeline(&mut self, format: MTLPixelFormat, mask: u8, fill: bool) -> Option<Shared<Pipeline>> {
        if let Some(p) = self.pipelines.get(&(format, mask, fill)) {
            return Some(p.clone());
        }
        let lib = self.lib()?;
        let d = MTLRenderPipelineDescriptor::new();
        d.setVertexFunction(Some(&lib.vs.0));
        d.setFragmentFunction(Some(if fill { &lib.fill.0 } else { &lib.copy.0 }));
        let att = unsafe { d.colorAttachments().objectAtIndexedSubscript(0) };
        att.setPixelFormat(format);

        att.setWriteMask(MTLColorWriteMask((mask.reverse_bits() >> 4) as usize));
        let p = Shared(gfx::gpu().device.0.newRenderPipelineStateWithDescriptor_error(&d).ok()?);
        self.pipelines.insert((format, mask, fill), p.clone());
        Some(p)
    }

    fn quad(&mut self, cb: &gfx::CommandBuffer, dst: &Shared<Texture>, sub: (u32, u32), rect: [usize; 4], mask: u8, src: Option<(&Texture, &Sampler)>, p: Params) {
        let Some(pipe) = self.pipeline(dst.0.pixelFormat(), mask, src.is_none()) else { return };
        let a = Attachments { colors: vec![Some((dst.clone(), Load::Keep))], depth: None, subs: vec![sub], depth_sub: (0, 0) };
        let Some(enc) = gfx::begin_pass(cb, &a) else { return };
        let [x, y, w, h] = rect;
        enc.setRenderPipelineState(&pipe.0);
        enc.setViewport(MTLViewport { originX: x as f64, originY: y as f64, width: w as f64, height: h as f64, znear: 0.0, zfar: 1.0 });
        enc.setScissorRect(MTLScissorRect { x, y, width: w, height: h });
        unsafe {
            enc.setVertexBytes_length_atIndex(std::ptr::NonNull::from(&p).cast(), size_of::<Params>(), 0);
            enc.setFragmentBytes_length_atIndex(std::ptr::NonNull::from(&p).cast(), size_of::<Params>(), 0);
            if let Some((t, s)) = src {
                enc.setFragmentTexture_atIndex(Some(t), 0);
                enc.setFragmentSamplerState_atIndex(Some(s), 0);
            }
            enc.drawPrimitives_vertexStart_vertexCount(MTLPrimitiveType::Triangle, 0, 3);
        }
        enc.endEncoding();
    }

    #[allow(clippy::too_many_arguments)]
    pub fn copy(
        &mut self,
        cb: &gfx::CommandBuffer,
        src: &Shared<Texture>,
        (src_layer, src_level): (u32, u32),
        sr: Region,
        dst: &Shared<Texture>,
        (dst_layer, dst_level): (u32, u32),
        dr: Region,
        flags: u32,
    ) -> Result<(), &'static str> {
        let [sx, sy, sz, sw, sh, sd] = sr.map(|v| v.max(0) as usize);
        let [dx, dy, dz, dw, dh, dd] = dr.map(|v| v.max(0) as usize);
        if sw == 0 || sh == 0 || dw == 0 || dh == 0 {
            return Ok(());
        }
        let (slw, slh) = level_size(&src.0, src_level);
        let (dlw, dlh) = level_size(&dst.0, dst_level);
        let (sw, sh, dw, dh) = (sw.min(slw.saturating_sub(sx)), sh.min(slh.saturating_sub(sy)), dw.min(dlw.saturating_sub(dx)), dh.min(dlh.saturating_sub(dy)));
        if sw == 0 || sh == 0 || dw == 0 || dh == 0 {
            return Ok(());
        }
        let mirror = flags & (COPY_MIRROR_X | COPY_MIRROR_Y) != 0;

        let layers = sd.max(1).min(dd.max(1));
        // NVN permits a depth-plane copy from D32F/S8 to D32F. Metal's
        // texture-to-texture blit requires matching formats; extract the depth
        // plane through a private buffer, without sampling or float conversion.
        if (sw,sh)==(dw,dh) && !mirror && src.0.pixelFormat()==MTLPixelFormat::Depth32Float_Stencil8 && dst.0.pixelFormat()==MTLPixelFormat::Depth32Float && src.0.textureType()!=MTLTextureType::Type3D && dst.0.textureType()!=MTLTextureType::Type3D {
            let row=(sw*4).next_multiple_of(256);let image=row.checked_mul(sh).ok_or("depth copy span overflow")?;
            let len=image.checked_mul(layers).ok_or("depth copy layer overflow")?;
            let staging=gfx::gpu().device.0.newBufferWithLength_options(len,MTLResourceOptions::StorageModePrivate).ok_or("depth plane staging buffer")?;
            let enc=cb.blitCommandEncoder().ok_or("depth plane copy encoder")?;
            for i in 0..layers {
                let source=if layered(&src.0) {src_layer as usize+sz+i} else if i==0 {0} else {return Err("depth source does not have requested array layers");};
                let destination=if layered(&dst.0) {dst_layer as usize+dz+i} else if i==0 {0} else {return Err("depth destination does not have requested array layers");};
                let size=MTLSize{width:sw,height:sh,depth:1};
                unsafe {
                    enc.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toBuffer_destinationOffset_destinationBytesPerRow_destinationBytesPerImage_options(&src.0,source,src_level as usize,MTLOrigin{x:sx,y:sy,z:0},size,&staging,i*image,row,image,MTLBlitOption::DepthFromDepthStencil);
                    enc.copyFromBuffer_sourceOffset_sourceBytesPerRow_sourceBytesPerImage_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin(&staging,i*image,row,image,size,&dst.0,destination,dst_level as usize,MTLOrigin{x:dx,y:dy,z:0});
                }
            }
            enc.endEncoding();return Ok(());
        }
        if (sw, sh) == (dw, dh) && !mirror && src.0.pixelFormat() == dst.0.pixelFormat() {
            let Some(enc) = cb.blitCommandEncoder() else { return Err("no blit encoder") };
            let three_d = src.0.textureType() == MTLTextureType::Type3D;
            for i in 0..if three_d { 1 } else { layers } {
                let slice = |t: &Texture, base: u32, z: usize| if layered(t) { base as usize + z + i } else { 0 };
                let oz = |t: &Texture, z: usize| if t.textureType() == MTLTextureType::Type3D { z } else { 0 };
                unsafe {
                    enc.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin(
                        &src.0,
                        slice(&src.0, src_layer, sz),
                        src_level as usize,
                        MTLOrigin { x: sx, y: sy, z: oz(&src.0, sz) },
                        MTLSize { width: sw, height: sh, depth: if three_d { sd.max(1) } else { 1 } },
                        &dst.0,
                        slice(&dst.0, dst_layer, dz),
                        dst_level as usize,
                        MTLOrigin { x: dx, y: dy, z: oz(&dst.0, dz) },
                    )
                };
            }
            enc.endEncoding();
            return Ok(());
        }
        if !drawable(&src.0) || !drawable(&dst.0) {
            return Err("scaled or converting copy of an integer, depth, compressed or 3D texture");
        }
        let lib = self.lib().ok_or("copy shaders")?;
        let sampler = if flags & COPY_LINEAR_FILTER != 0 { &lib.linear.0 } else { &lib.nearest.0 };
        for i in 0..layers {
            let src_slice = if layered(&src.0) { src_layer as usize + sz + i } else { 0 };
            let dst_slice = if layered(&dst.0) { dst_layer as usize + dz + i } else { 0 };

            let Some(view) = (unsafe {
                src.0.newTextureViewWithPixelFormat_textureType_levels_slices(
                    src.0.pixelFormat(),
                    MTLTextureType::Type2D,
                    NSRange { location: src_level as usize, length: 1 },
                    NSRange { location: src_slice, length: 1 },
                )
            }) else {
                return Err("source view");
            };
            let (fw, fh) = (slw as f32, slh as f32);
            let (mut u0, mut v0, mut u1, mut v1) = (sx as f32 / fw, sy as f32 / fh, (sx + sw) as f32 / fw, (sy + sh) as f32 / fh);
            if flags & COPY_MIRROR_X != 0 {
                std::mem::swap(&mut u0, &mut u1);
            }
            if flags & COPY_MIRROR_Y != 0 {
                std::mem::swap(&mut v0, &mut v1);
            }
            let p = Params { src: [u0, v0, u1, v1], color: [0.0; 4] };
            self.quad(cb, dst, (dst_slice as u32, dst_level), [dx, dy, dw, dh], 0xF, Some((&view, sampler)), p);
        }
        Ok(())
    }

    pub fn clear(&mut self, cb: &gfx::CommandBuffer, dst: &Shared<Texture>, (layer, level): (u32, u32), r: Region, color: [f32; 4], mask: u32) -> Result<(), &'static str> {
        let [x, y, z, w, h, d] = r.map(|v| v.max(0) as usize);
        let (lw, lh) = level_size(&dst.0, level);
        let (w, h) = (w.min(lw.saturating_sub(x)), h.min(lh.saturating_sub(y)));
        let mask = (mask & 0xF) as u8;
        if w == 0 || h == 0 || mask == 0 {
            return Ok(());
        }
        let f = dst.0.pixelFormat();
        if is_depth(f) || is_block_compressed(f) {
            return Err("clear of a depth or compressed texture");
        }
        let whole = (x, y, w, h) == (0, 0, lw, lh) && mask == 0xF;
        if !whole && super::shaders::integer_kind(f) != 0 {
            return Err("partial clear of an integer texture");
        }
        let three_d = dst.0.textureType() == MTLTextureType::Type3D;
        for i in 0..d.max(1) {
            let slice = if layered(&dst.0) || three_d { layer as usize + z + i } else if i == 0 { 0 } else { break };
            if whole {
                let c = color.map(|v| v as f64);
                let a = Attachments { colors: vec![Some((dst.clone(), Load::ClearColor(c)))], depth: None, subs: vec![(slice as u32, level)], depth_sub: (0, 0) };
                gfx::run_pass(cb, &a);
            } else {
                self.quad(cb, dst, (slice as u32, level), [x, y, w, h], mask, None, Params { src: [0.0; 4], color });
            }
        }
        Ok(())
    }
}

pub fn buffer_layout(w: usize, h: usize, d: usize, format: u32, strides: (u32, u32)) -> Result<(usize, usize, usize), &'static str> {
    let (bw, bh, bpb) = super::super::format_info(format);
    let tight = w.div_ceil(bw as usize).checked_mul(bpb as usize).ok_or("copy row overflow")?;
    let rows = h.div_ceil(bh as usize);
    let row = if strides.0 == 0 { tight } else { strides.0 as usize };
    if row < tight { return Err("NVN copy row stride is smaller than a row"); }
    let image_min = row.checked_mul(rows).ok_or("copy image overflow")?;
    let image = if strides.1 == 0 { image_min } else { strides.1 as usize };
    if image < image_min { return Err("NVN copy image stride is smaller than an image"); }
    let len = image.checked_mul(d.max(1) - 1).and_then(|n| n.checked_add(row.checked_mul(rows.saturating_sub(1))?)).and_then(|n| n.checked_add(tight)).ok_or("copy span overflow")?;
    Ok((row, image, len))
}

pub fn copy_to_buffer(cb: &gfx::CommandBuffer, src: &Texture, (layer, level): (u32, u32), r: Region, nvn_format: u32, dst: u64, pools: &[std::sync::Arc<gfx::PoolMemory>], strides: (u32, u32)) -> Result<Vec<(u64,u64,u64)>, &'static str> {
    let [_, _, _, w, h, d] = r.map(|v| v.max(0) as usize);
    if w == 0 || h == 0 {
        return Ok(Vec::new());
    }
    let (_, _, len) = buffer_layout(w, h, d, nvn_format, strides)?;
    let len = len as u64;
    let heap = gfx::heap();
    let range = |addr, len| gfx::pool_buffer_range(pools, addr, len).or_else(|| heap.and_then(|h| h.buffer_range(addr, len)));
    dst.checked_add(len).ok_or("readback destination span overflow")?;
    // Validate every destination and retain its actual buffer before encoding
    // any part. The returned canonical ranges also cover split virtual aliases.
    let mut pieces=Vec::new();let mut copied=0;
    while copied<len {
        let (chunk,off)=range(dst+copied,1).ok_or("readback destination is not mapped to Metal")?;
        let n=(chunk.base+chunk.size-dst-copied).min(len-copied);
        pieces.push((chunk.buffer.clone(),off,n,dst+copied,chunk.buffer.0.contents().as_ptr() as u64+off as u64));copied+=n;
    }
    let writes=pieces.iter().map(|(_,_,len,gpu,cpu)|(*gpu,*cpu,*len)).collect();
    if pieces.len()==1 {
        let (buffer,off,_,_,_)=&pieces[0];
        copy_into_strided(cb,src,(layer,level),r,nvn_format,&buffer.0,*off,strides)?;
        return Ok(writes);
    }
    let temp = gfx::gpu().device.0.newBufferWithLength_options(len as usize, MTLResourceOptions::StorageModeShared).ok_or("texture readback staging buffer")?;
    copy_into_strided(cb, src, (layer, level), r, nvn_format, &temp, 0, strides)?;
    let enc = cb.blitCommandEncoder().ok_or("no readback split encoder")?;
    let mut copied = 0;
    for (buffer,off,n,_,_) in pieces {
        unsafe { enc.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(&temp, copied as usize, &buffer.0, off, n as usize) };
        copied += n;
    }
    enc.endEncoding();
    Ok(writes)
}

pub fn copy_into(cb: &gfx::CommandBuffer, src: &Texture, (layer, level): (u32, u32), r: Region, nvn_format: u32, buffer: &gfx::Buffer, offset: usize) -> Result<(), &'static str> {
    copy_into_strided(cb, src, (layer, level), r, nvn_format, buffer, offset, (0, 0))
}

pub fn copy_into_strided(cb: &gfx::CommandBuffer, src: &Texture, (layer, level): (u32, u32), r: Region, nvn_format: u32, buffer: &gfx::Buffer, offset: usize, strides: (u32, u32)) -> Result<(), &'static str> {
    if gfx::bc::soft(nvn_format).is_some() {return Err("copying software-decoded BC textures back into compressed blocks requires a BC encoder");}
    let [x, y, z, w, h, d] = r.map(|v| v.max(0) as usize);
    if w == 0 || h == 0 {
        return Ok(());
    }
    let (row, image, _) = buffer_layout(w, h, d, nvn_format, strides)?;
    let options = match src.pixelFormat() {
        MTLPixelFormat::Depth32Float_Stencil8 => MTLBlitOption::DepthFromDepthStencil,
        _ => MTLBlitOption::None,
    };
    let three_d = src.textureType() == MTLTextureType::Type3D;
    let Some(enc) = cb.blitCommandEncoder() else { return Err("no blit encoder") };
    for i in 0..if three_d { 1 } else { d.max(1) } {
        let slice = if layered(src) { layer as usize + z + i } else if i == 0 { 0 } else { break };
        unsafe {
            enc.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toBuffer_destinationOffset_destinationBytesPerRow_destinationBytesPerImage_options(
                src,
                slice,
                level as usize,
                MTLOrigin { x, y, z: if three_d { z } else { 0 } },
                MTLSize { width: w, height: h, depth: if three_d { d.max(1) } else { 1 } },
                buffer,
                offset + i * image,
                row,
                image,
                options,
            )
        };
    }
    enc.endEncoding();
    Ok(())
}
