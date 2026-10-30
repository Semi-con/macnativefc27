// Optional specialization of vertex input formats. The original conversion
// body and all per-draw addressing remain shared with the generic function.
use crate::gfx::Shared;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::*;
use std::sync::{LazyLock,atomic::{AtomicU64,Ordering::Relaxed}};

type MetalLibrary=ProtocolObject<dyn MTLLibrary>;
type Function=ProtocolObject<dyn MTLFunction>;
pub(super) struct Library { library:Shared<MetalLibrary>,entry:String }

pub(super) fn buffer_enabled()->bool {
    static ENABLED:LazyLock<bool>=LazyLock::new(||std::env::var_os("NX_BUFFER_FORMATS").is_some() && std::env::var_os("NX_DYNAMIC_VERTEX_CACHE").is_none());*ENABLED
}

pub(super) fn enabled()->bool {
    static ENABLED:LazyLock<bool>=LazyLock::new(||std::env::var_os("NX_VERTEX_FORMATS").is_some() && std::env::var_os("NX_DYNAMIC_VERTEX_CACHE").is_none());
    *ENABLED
}

pub(super) fn source(source:&str)->std::borrow::Cow<'_,str> {
    const OLD:&str="static inline uint4 fetch_attr(constant Res& res, uint i, uint vid, uint iid) {\n    uint fmt = res.attr_format[i];";
    if !source.contains(OLD) {return std::borrow::Cow::Borrowed(source);}
    let mut prefix=String::new();
    for i in 0..16 {prefix.push_str(&format!("constant uint nx_vertex_format_{i} [[function_constant({})]];\n",128+i));}
    prefix.push_str("static inline uint nx_vertex_format(constant Res& res, uint i) {\n    switch (i) {\n");
    for i in 0..16 {prefix.push_str(&format!("    case {i}u: if (is_function_constant_defined(nx_vertex_format_{i})) return nx_vertex_format_{i}; break;\n"));}
    prefix.push_str("    default: break;\n    }\n    return res.attr_format[i];\n}\nstatic inline uint4 fetch_attr(constant Res& res, uint i, uint vid, uint iid) {\n    uint fmt = nx_vertex_format(res, i);");
    std::borrow::Cow::Owned(source.replacen(OLD,&prefix,1))
}

pub(super) fn buffer_source(source:&str)->std::borrow::Cow<'_,str> {
    let marker="static inline uint4 fetch_attr(";
    let mut result=source.to_string();
    if result.contains(marker) {
        let mut prefix=String::new();
        for i in 0..32 {prefix.push_str(&format!("constant uint nx_buffer_format_{i} [[function_constant({})]];\n",512+i));}
        prefix.push_str("inline uint4 nx_buffer_meta(constant uint4* metadata,uint i) {\n    uint4 value=metadata[i];\n    switch(i) {\n");
        for i in 0..32 {prefix.push_str(&format!("    case {i}u: if(is_function_constant_defined(nx_buffer_format_{i})) value.w=nx_buffer_format_{i}; break;\n"));}
        prefix.push_str("    default: break;\n    }\n    return value;\n}\n");
        result=result.replacen(marker,&(prefix+marker),1);
    }
    for i in 0..32 {result=result.replace(&format!("buffer_meta[{i}]"),&format!("nx_buffer_meta(buffer_meta,{i}u)"));}
    if result==source {std::borrow::Cow::Borrowed(source)} else {std::borrow::Cow::Owned(result)}
}

pub(super) fn buffer_formats(allocation:&super::state::Alloc)->[u32;32] {
    // stage_buffer_metadata initializes264 uint4 entries in this CPU-owned
    // arena allocation. It remains immutable until GPU retirement.
    assert!(allocation.offset+264*16<=allocation.buffer.0.length());
    unsafe {std::array::from_fn(|i|allocation.ptr.cast::<[u32;4]>().add(i).read()[3])}
}

#[cfg(test)]
pub(super) fn buffer_function(library:&MetalLibrary,entry:&str,buffers:&[u32;32])->Option<Shared<Function>> {specialize(library,entry,None,Some(buffers))}

pub(super) fn formats(st:&super::state::GState)->[u32;16] {
    st.attribs.map(|a|if a&0x40!=0 {(a>>21)&0x3ff}else{0})
}

pub(super) fn retain(library:Shared<MetalLibrary>,entry:&str,func:&Function)->Option<Library> {
    ((enabled() || buffer_enabled()) && func.functionType()==MTLFunctionType::Vertex).then(||Library{library,entry:entry.into()})
}

// Even optional function constants require the specialization API before a
// Metal function may enter a pipeline. Empty values select the generic branch.
pub(super) fn generic(library:&MetalLibrary,entry:&str)->Option<Shared<Function>> {
    if !enabled() && !buffer_enabled() {return library.newFunctionWithName(&NSString::from_str(entry)).map(Shared);}
    library.newFunctionWithName_constantValues_error(&NSString::from_str(entry),&MTLFunctionConstantValues::new()).ok().map(Shared)
}

impl Library {
    pub(super) fn function(&self,formats:Option<&[u32;16]>,buffers:Option<&[u32;32]>)->Option<Shared<Function>> {
        specialize(&self.library.0,&self.entry,formats,buffers)
    }
}

fn specialize(library:&MetalLibrary,entry:&str,formats:Option<&[u32;16]>,buffers:Option<&[u32;32]>)->Option<Shared<Function>> {
    let constants=MTLFunctionConstantValues::new();
    for (i,value) in formats.into_iter().flatten().enumerate() {
        // The pointer remains valid throughout the synchronous copying call.
        unsafe {constants.setConstantValue_type_atIndex(std::ptr::NonNull::from(value).cast(),MTLDataType::UInt,128+i);}
    }
    for (i,value) in buffers.into_iter().flatten().enumerate() {
        unsafe {constants.setConstantValue_type_atIndex(std::ptr::NonNull::from(value).cast(),MTLDataType::UInt,512+i);}
    }
    match library.newFunctionWithName_constantValues_error(&NSString::from_str(entry),&constants) {
        Ok(function)=>{
            static BUILT:AtomicU64=AtomicU64::new(0);let n=BUILT.fetch_add(1,Relaxed)+1;
            if n<=8 || n.is_power_of_two(){eprintln!("[vertex-formats] specialized={n} formats={formats:?} buffers={buffers:?}");}
            Some(Shared(function))
        }
        Err(e)=>{eprintln!("[vertex-formats] original function retained after specialization error: {}",e.localizedDescription());None}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn format_key_matches_original_enable_and_ten_bit_extraction() {
        let mut st=super::super::state::GState::default();
        for i in 0..16 {st.attribs[i]=((i as u32*61+7)<<21)|0x40|((i as u32*7)<<7)|(15-i as u32);}
        let expected=st.attribs.map(|a|(a>>21)&0x3ff);assert_eq!(formats(&st),expected);
        st.attribs[8]&=!0x40;assert_eq!(formats(&st)[8],0);
        st.attribs[8]^=0x3fff<<7;assert_eq!(formats(&st)[8],0);
        let old=nxshader::prelude();let specialized=source(old);assert_ne!(specialized,old);
        let body="    if (fmt == 0u) return uint4";
        assert_eq!(&old[old.find(body).unwrap()..],&specialized[specialized.find(body).unwrap()..]);
        assert_eq!(source("unrelated"),"unrelated");
    }

    #[test]
    fn buffer_specialization_rewrites_only_first32_sampled_metadata_words() {
        let body="buffer_meta[0].w + buffer_meta[31].z + buffer_meta[32].w + buffer_meta[256].x + buffer_meta[index].w";
        assert_eq!(buffer_source(body),"nx_buffer_meta(buffer_meta,0u).w + nx_buffer_meta(buffer_meta,31u).z + buffer_meta[32].w + buffer_meta[256].x + buffer_meta[index].w");
        let old=nxshader::prelude();let changed=buffer_source(old);assert_ne!(changed,old);
        let store="inline void buffer_write_raw(";assert_eq!(&old[old.find(store).unwrap()..],&changed[changed.find(store).unwrap()..]);
    }

    #[cfg(target_os="macos")]
    #[test]
    #[ignore="requires actual Metal function-constant readback"]
    fn gpu_distinct_buffer_format_constants_keep_addresses_lengths_and_fallback_slot() {
        objc2::rc::autoreleasepool(|_|unsafe {
            let dev=&crate::gfx::gpu().device.0;
            let source=buffer_source(nxshader::prelude()).into_owned()+"\nkernel void meta_probe(constant uint4* metadata [[buffer(0)]],device uint4* out [[buffer(1)]],uint tid [[thread_position_in_grid]]){out[tid]=nx_buffer_meta(metadata,tid);}";
            let lib=dev.newLibraryWithSource_options_error(&NSString::from_str(&source),None).unwrap();
            let mut data:[ [u32;4];33]=std::array::from_fn(|i|[i as u32*17,0x11223344+i as u32,0x55667788-i as u32,0x80000000|(i as u32*0x10203)]);
            let output=dev.newBufferWithLength_options(size_of_val(&data),MTLResourceOptions::StorageModeShared).unwrap();
            for round in 0..3 {
                let mut words=std::array::from_fn(|i|data[i][3]);let mut expected=data;
                if round==2 {words[7]^=0x80000001;expected[7][3]=words[7];}
                let function=buffer_function(&lib,"meta_probe",&words).unwrap();let pipeline=dev.newComputePipelineStateWithFunction_error(&function.0).unwrap();
                let cb=crate::gfx::gpu().queue.0.commandBuffer().unwrap();let e=cb.computeCommandEncoder().unwrap();e.setComputePipelineState(&pipeline);e.setBytes_length_atIndex(std::ptr::NonNull::from(&data).cast(),size_of_val(&data),0);e.setBuffer_offset_atIndex(Some(&output),0,1);
                e.dispatchThreads_threadsPerThreadgroup(MTLSize{width:33,height:1,depth:1},MTLSize{width:32,height:1,depth:1});e.endEncoding();cb.commit();cb.waitUntilCompleted();assert_eq!(cb.status(),MTLCommandBufferStatus::Completed);
                assert_eq!(std::slice::from_raw_parts(output.contents().as_ptr().cast::<[u32;4]>(),33),expected,"round{round}");
                for row in &mut data {row[0]=row[0].wrapping_add(0x10000);row[1]^=0x65432100;row[2]=row[2].wrapping_add(19);}
            }
        });
    }

    #[cfg(target_os="macos")]
    #[test]
    #[ignore="requires a real Metal GPU; input readback is exact"]
    fn gpu_vertex_format_constants_preserve_formats_streams_divisors_and_disabled_inputs() {
        objc2::rc::autoreleasepool(|_|unsafe {
            let dev=&crate::gfx::gpu().device.0;
            let input=dev.newBufferWithLength_options(16*16384,MTLResourceOptions::StorageModeShared).unwrap();
            for i in 0..input.length(){input.contents().as_ptr().cast::<u8>().add(i).write(((i*37+i/13)%251)as u8);}
            let mut r:super::super::state::Res=std::mem::zeroed();
            for i in 0..16 {let stream=(15-i)as usize;r.attr_stream[i]=stream as u32;r.attr_offset[i]=(i*4)as u32;r.vb[stream]=input.gpuAddress()+stream as u64*16384;r.vb_stride[stream]=96;r.vb_divisor[stream]=[0,1,2,3,7][i%5];}
            r.base_instance=5;
            let mut body=String::from("kernel void probe(constant Res& r [[buffer(0)]],device uint4* out [[buffer(1)]],uint tid [[thread_position_in_grid]]) {\n");
            // Literal attribute indices mirror translated vertex entry points.
            for i in 0..16 {body.push_str(&format!("out[tid*16u+{i}u]=fetch_attr(r,{i}u,tid,tid%11u);\n"));}
            body.push_str("}\n");
            let options=MTLCompileOptions::new();options.setMathMode(MTLMathMode::Safe);options.setPreserveInvariance(true);
            let original=nxshader::prelude().to_string()+&body;let changed=source(nxshader::prelude()).into_owned()+&body;
            let libs=[original,changed].map(|s|dev.newLibraryWithSource_options_error(&NSString::from_str(&s),Some(&options)).unwrap());
            let generic=libs.each_ref().map(|l|{
                let function=l.newFunctionWithName_constantValues_error(&NSString::from_str("probe"),&MTLFunctionConstantValues::new()).unwrap();
                dev.newComputePipelineStateWithFunction_error(&function).unwrap()
            });
            let output=dev.newBufferWithLength_options(32*16*16,MTLResourceOptions::StorageModeShared).unwrap();
            let read=|pipeline:&ProtocolObject<dyn MTLComputePipelineState>,r:&super::super::state::Res| {
                let cb=crate::gfx::gpu().queue.0.commandBuffer().unwrap();let e=cb.computeCommandEncoder().unwrap();e.setComputePipelineState(pipeline);
                e.useResource_usage(ProtocolObject::from_ref(&*input),MTLResourceUsage::Read);
                e.setBytes_length_atIndex(std::ptr::NonNull::from(r).cast(),size_of_val(r),0);e.setBuffer_offset_atIndex(Some(&output),0,1);
                e.dispatchThreads_threadsPerThreadgroup(MTLSize{width:32,height:1,depth:1},MTLSize{width:32,height:1,depth:1});e.endEncoding();cb.commit();cb.waitUntilCompleted();assert_eq!(cb.status(),MTLCommandBufferStatus::Completed);
                std::slice::from_raw_parts(output.contents().as_ptr().cast::<u32>(),32*16*4).to_vec()
            };
            for round in 0..7 {
                r.attr_format=std::array::from_fn(|i|(round*16+i)as u32);
                let expected=read(&generic[0],&r);assert_eq!(read(&generic[1],&r),expected,"undefined constants round{round}");
                let f=specialize(&libs[1],"probe",Some(&r.attr_format),None).unwrap();let pipeline=dev.newComputePipelineStateWithFunction_error(&f.0).unwrap();
                assert_eq!(read(&pipeline,&r),expected,"specialized round{round}");
                // Same source and bound bytes, different format keys must yield
                // different functions rather than a stale prior specialization.
                r.attr_format=[0;16];let zero=specialize(&libs[1],"probe",Some(&r.attr_format),None).unwrap();let zero_pipeline=dev.newComputePipelineStateWithFunction_error(&zero.0).unwrap();
                assert_eq!(read(&zero_pipeline,&r),read(&generic[0],&r),"disabled round{round}");
            }
        });
    }
}
