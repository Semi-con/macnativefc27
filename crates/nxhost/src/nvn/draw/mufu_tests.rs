// GPU accuracy is compared with independent f64 CPU arithmetic and the original
// safe Metal path. Exceptional values stay on that original path.
use super::*;
use objc2_foundation::NSString;
use std::ptr::NonNull;

#[test]
#[ignore="requires a real Metal GPU"]
fn gpu_refined_mufu_reciprocal_and_rsqrt_preserve_precision_domains_and_special_values() {
    let mut inputs=vec![0,0x80000000,1,0x007fffff,0x00800000,0x7f7fffff,0x7f800000,0xff800000,0x7fc00001,0xffc00001,0x3f800000,0xbf800000];
    for boundary in [0x00800000u32,0x7e800000,0x21800000,0x5d800000,0x3f800000,0x40800000] {
        for bits in [boundary-1,boundary,boundary+1] {inputs.push(bits);inputs.push(bits|0x80000000);}
    }
    let mut random=0xfc270001u32;
    for _ in 0..131072 {random=random.wrapping_mul(1664525).wrapping_add(1013904223);inputs.push(random);}
    // Cover each exponent and multiple mantissas independently of the RNG.
    for exponent in 0..256u32 {for mantissa in [0,1,0x155555,0x400000,0x7ffffe,0x7fffff] {inputs.push(exponent<<23|mantissa);inputs.push(exponent<<23|mantissa|0x80000000);}}
    objc2::rc::autoreleasepool(|_|unsafe {
        let device=&gfx::gpu().device.0;
        let source=nxshader::prelude().replace("#define NX_APPROX_MUFU 0","#define NX_APPROX_MUFU 1")+"\nkernel void math_probe(device const uint* inputs [[buffer(0)]],device float4* outputs [[buffer(1)]],constant uint& count [[buffer(2)]],uint id [[thread_position_in_grid]]) {if(id>=count)return;float x=as_type<float>(inputs[id]);outputs[id]=float4(1.0f/x,nx_mufu_rcp(x),rsqrt(x),nx_mufu_rsqrt(x));}\n";
        let options=MTLCompileOptions::new();options.setMathMode(MTLMathMode::Safe);options.setPreserveInvariance(true);
        let library=device.newLibraryWithSource_options_error(&NSString::from_str(&source),Some(&options)).unwrap();let function=library.newFunctionWithName(&NSString::from_str("math_probe")).unwrap();let pipeline=device.newComputePipelineStateWithFunction_error(&function).unwrap();
        let input=device.newBufferWithBytes_length_options(NonNull::new(inputs.as_mut_ptr()).unwrap().cast(),size_of_val(inputs.as_slice()),MTLResourceOptions::StorageModeShared).unwrap();
        let output=device.newBufferWithLength_options(inputs.len()*16,MTLResourceOptions::StorageModeShared).unwrap();let count=inputs.len() as u32;
        let command=gfx::gpu().queue.0.commandBuffer().unwrap();let encoder=command.computeCommandEncoder().unwrap();encoder.setComputePipelineState(&pipeline);encoder.setBuffer_offset_atIndex(Some(&input),0,0);encoder.setBuffer_offset_atIndex(Some(&output),0,1);encoder.setBytes_length_atIndex(NonNull::from(&count).cast(),4,2);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(MTLSize{width:inputs.len().div_ceil(64),height:1,depth:1},MTLSize{width:64,height:1,depth:1});encoder.endEncoding();command.commit();command.waitUntilCompleted();assert_eq!(command.status(),MTLCommandBufferStatus::Completed);
        let actual=std::slice::from_raw_parts(output.contents().as_ptr().cast::<[f32;4]>(),inputs.len());let mut rcp_max=0;let mut rsqrt_max=0;let mut rcp_count=0;let mut rsqrt_count=0;
        for (&bits,values) in inputs.iter().zip(actual) {
            let x=f32::from_bits(bits);let magnitude=bits&0x7fffffff;
            if (0x00800000..=0x7e800000).contains(&magnitude) {
                let reference=(1.0f64/f64::from(x)) as f32;let error=values[1].to_bits().abs_diff(reference.to_bits());rcp_max=rcp_max.max(error);rcp_count+=1;
                assert!(error<=1,"reciprocal x={bits:08x} candidate={:08x} independent={:08x} ulp={error}",values[1].to_bits(),reference.to_bits());
            } else {assert!(values[0].to_bits()==values[1].to_bits() || (values[0].is_nan()&&values[1].is_nan()),"reciprocal fallback x={bits:08x}: {values:?}");}
            if (0x21800000..=0x5d800000).contains(&bits) {
                let reference=(1.0f64/f64::from(x).sqrt()) as f32;let error=values[3].to_bits().abs_diff(reference.to_bits());rsqrt_max=rsqrt_max.max(error);rsqrt_count+=1;
                assert!(error<=2,"rsqrt x={bits:08x} candidate={:08x} independent={:08x} ulp={error}",values[3].to_bits(),reference.to_bits());
                let exponent=(((bits>>23)&255) as i32-127)&!1;let normalized=f64::from(x)*2.0f64.powi(-exponent);let normalized_result=f64::from(values[3])*2.0f64.powi(exponent/2);
                assert!((normalized_result-1.0/normalized.sqrt()).abs()<=2.0f64.powf(-22.4),"normalized Maxwell rsqrt accuracy exceeded x={bits:08x}");
            } else {assert!(values[2].to_bits()==values[3].to_bits() || (values[2].is_nan()&&values[3].is_nan()),"rsqrt fallback x={bits:08x}: {values:?}");}
        }
        assert!(rcp_count>100000 && rsqrt_count>20000);eprintln!("[verify] refined MUFU inputs={} reciprocal_n={rcp_count} max_ulp={rcp_max} rsqrt_n={rsqrt_count} max_ulp={rsqrt_max}; exceptional fallbacks preserved",inputs.len());
    });
}
