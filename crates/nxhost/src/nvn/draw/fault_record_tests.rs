use super::*;
use objc2_foundation::NSString;
use std::ptr::NonNull;

#[test]
#[ignore="requires a real Metal GPU; shared NVN fixture must run serially"]
fn gpu_fault_records_stay_distinct_while_prior_command_buffers_are_in_flight() {
    objc2::rc::autoreleasepool(|_|unsafe {
        let device=&gfx::gpu().device.0;
        let source="#include <metal_stdlib>\nusing namespace metal;kernel void record_marker(constant ulong& address [[buffer(0)]],constant uint& value [[buffer(1)]]) {reinterpret_cast<device uint*>(address)[4]=value;}";
        let library=device.newLibraryWithSource_options_error(&NSString::from_str(source),None).unwrap();let function=library.newFunctionWithName(&NSString::from_str("record_marker")).unwrap();let pipeline=device.newComputePipelineStateWithFunction_error(&function).unwrap();
        let event=device.newSharedEvent().unwrap();let cb=gfx::gpu().queue.0.commandBuffer().unwrap();cb.encodeWaitForEvent_value(ProtocolObject::from_ref(&*event),1);
        let mut e=Engine::default();let mut r=Replay{e:&mut e,cb,pass:None,heap:None,pools:Vec::new(),passes:0,copies:HashMap::new(),fault:None,pool_table:None,resource_buffers:Default::default()};
        let mut records=Vec::new();let mut commands=Vec::new();
        for value in [0x12345678u32,0xabcdef01] {
            let gpu=r.fault_record();let a=r.fault.as_ref().unwrap();records.push((a.buffer.clone(),a.offset));
            if fault_records::enabled() {assert_eq!(a.buffer.0.length(),32);assert!(!r.e.arena.used.iter().any(|b|b.0.gpuAddress()==a.buffer.0.gpuAddress()),"diagnostic record still aliases constant arena");}
            let encoder=r.cb.computeCommandEncoder().unwrap();encoder.setComputePipelineState(&pipeline);encoder.useResource_usage(ProtocolObject::from_ref(&*a.buffer.0),MTLResourceUsage::Read|MTLResourceUsage::Write);
            encoder.setBytes_length_atIndex(NonNull::from(&gpu).cast(),8,0);encoder.setBytes_length_atIndex(NonNull::from(&value).cast(),4,1);encoder.dispatchThreads_threadsPerThreadgroup(MTLSize{width:1,height:1,depth:1},MTLSize{width:1,height:1,depth:1});encoder.endEncoding();commands.push(r.commit_boundary(false));
        }
        assert_ne!(records[0].0.0.gpuAddress()+records[0].1 as u64,records[1].0.0.gpuAddress()+records[1].1 as u64,"in-flight fault record was recycled early");
        event.setSignaledValue(1);
        for command in commands {gpu_wait::complete(&command);assert_eq!(command.status(),MTLCommandBufferStatus::Completed);}
        for ((buffer,offset),value) in records.into_iter().zip([0x12345678,0xabcdef01]) {let words=std::slice::from_raw_parts((buffer.0.contents().as_ptr() as *const u8).add(offset).cast::<u32>(),8);assert_eq!(words[0],0);assert_eq!(words[4],value,"the next command buffer overwrote the earlier record");}
        eprintln!("[verify] real GPU separate in-flight fault records and command-completion-only recycling passed");
    });
}
