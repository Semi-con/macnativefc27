// Ordered GPU timestamp reports. Counter sampling, resolving and publication
// stay on the GPU; CPU readers use the same known-write guards as transfers.
use super::{gfx,Replay};
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSString,NSRange};
use objc2_metal::*;
use std::{ptr::NonNull,sync::{LazyLock,atomic::{AtomicBool,Ordering::Relaxed}}};

static TEST_SYNCHRONOUS:AtomicBool=AtomicBool::new(false);

pub(super) fn enabled()->bool {
    static ENABLED:LazyLock<bool>=LazyLock::new(||std::env::var_os("NX_GPU_TIMESTAMP_REPORTS").is_some());*ENABLED && !TEST_SYNCHRONOUS.load(Relaxed)
}
// Explicitly enabled local test sessions can compare the same live scene.
// Consume at most one mode request per32submits; production never reads it.
pub(super) fn poll_test_mode(submit:u32)->bool {
    static LIVE:LazyLock<bool>=LazyLock::new(||std::env::var_os("NX_TEST_CONTROL").is_some());
    if !*LIVE || submit%32!=0 {return false;}
    let Ok(mode)=std::fs::read_to_string("work/gpu-counter-mode") else {return false;};
    let _=std::fs::remove_file("work/gpu-counter-mode");
    let value=match mode.trim() {"sync"=>true,"gpu"=>false,_=>{eprintln!("[counter-mode] rejected unsupported test request");return false;}};
    let changed=TEST_SYNCHRONOUS.swap(value,Relaxed)!=value;
    eprintln!("[counter-mode] submit={submit} synchronous={value} changed={changed}");changed
}

struct Kernels {marker:gfx::Shared<ProtocolObject<dyn MTLComputePipelineState>>,publish:gfx::Shared<ProtocolObject<dyn MTLComputePipelineState>>}
static KERNELS:LazyLock<Option<Kernels>>=LazyLock::new(|| {
    let source=r#"#include <metal_stdlib>
using namespace metal;
kernel void mark_time(device uint* scratch [[buffer(0)]]) {scratch[0]=0x1234u;}
kernel void publish_time(device const ulong* samples [[buffer(0)]],device ulong* output [[buffer(1)]],constant ulong3& parameters [[buffer(2)]],constant ulong& fault_address [[buffer(3)]]) {
    ulong raw=samples[parameters.z];
    if(raw==0ul || raw==~0ul) {
        device atomic_uint* flag=reinterpret_cast<device atomic_uint*>(fault_address);
        uint expected=0u;while(expected==0u && !atomic_compare_exchange_weak_explicit(flag,&expected,1024u,memory_order_relaxed,memory_order_relaxed)) {}
        return;
    }
    ulong value=raw>=parameters.x?parameters.y+(raw-parameters.x):parameters.y-(parameters.x-raw);
    output[0]=value;output[1]=value;
}"#;
    let device=&gfx::gpu().device.0;
    let library=device.newLibraryWithSource_options_error(&NSString::from_str(source),None).map_err(|e|eprintln!("[counter] timestamp kernels: {e}")).ok()?;
    let pipeline=|name| {let function=library.newFunctionWithName(&NSString::from_str(name))?;device.newComputePipelineStateWithFunction_error(&function).ok().map(gfx::Shared)};
    Some(Kernels{marker:pipeline("mark_time")?,publish:pipeline("publish_time")?})
});

impl Replay<'_> {
    pub(super) fn gpu_report_counter(&mut self,kind:u32,address:u64)->Result<bool,&'static str> {
        if !matches!(kind,0|16) {return Err("unsupported NVN GPU timestamp counter");}
        if !enabled() || address==0 {return Ok(false);}
        let device=&gfx::gpu().device.0;
        // Apple GPU counters have nanosecond units and share the CPU clock.
        // Other vendors need a clock-rate interpolation path, kept on fallback.
        if !device.supportsFamily(MTLGPUFamily::Apple7) {return Ok(false);}
        let Some(kernels)=KERNELS.as_ref() else {return Ok(false);};
        let Some(counter)=gfx::timestamp_samples(2) else {return Ok(false);};
        if address%8!=0 {return Err("unaligned NVN timestamp report");}
        let end=address.checked_add(16).ok_or("timestamp destination overflow")?;
        // Resolve the complete destination before changing the command stream.
        let mut parts=Vec::new();let mut at=address;
        while at<end {
            let (chunk,offset)=self.buffer_range(at,1).ok_or("timestamp destination mapping hole")?;
            let n=(chunk.base+chunk.size-at).min(end-at);
            let cpu=chunk.buffer.0.contents().as_ptr() as u64+offset as u64;
            parts.push((chunk.buffer.clone(),offset,n,at,cpu));at+=n;
        }
        let before=super::super::objects::timestamp_ns();let mut cpu=0u64;let mut gpu=0u64;
        unsafe {device.sampleTimestamps_gpuTimestamp(NonNull::from(&mut cpu),NonNull::from(&mut gpu));}
        let after=super::super::objects::timestamp_ns();
        if gpu==0 || gpu==u64::MAX || after<before {return Ok(false);}
        let parameters=[gpu,before+(after-before)/2,if kind==16 {0u64}else{1u64},0u64];
        self.end_pass();self.flush_clears();
        let event=if kind==0 {
            // A bottom timestamp follows all earlier stages. Enforce that on
            // the GPU across the command-buffer boundary, without a CPU wait.
            let event=device.newSharedEvent().ok_or("timestamp ordering event")?;
            self.cb.encodeSignalEvent_value(ProtocolObject::from_ref(&*event),1);
            self.commit_boundary(false);
            self.cb.encodeWaitForEvent_value(ProtocolObject::from_ref(&*event),1);Some(event)
        } else {None};
        let raw=self.e.arena.alloc(16);let output=self.e.arena.alloc(16);
        let encoder=gfx::begin_compute_sampled(&self.cb,Some(&counter.0)).ok_or("timestamp sampling encoder")?;
        encoder.setComputePipelineState(&kernels.marker.0);
        unsafe {encoder.setBuffer_offset_atIndex(Some(&raw.buffer.0),raw.offset,0);}
        encoder.dispatchThreadgroups_threadsPerThreadgroup(MTLSize{width:1,height:1,depth:1},MTLSize{width:1,height:1,depth:1});encoder.endEncoding();
        let encoder=self.cb.blitCommandEncoder().ok_or("timestamp resolve encoder")?;
        unsafe {encoder.resolveCounters_inRange_destinationBuffer_destinationOffset(&counter.0,NSRange{location:0,length:2},&raw.buffer.0,raw.offset);}
        encoder.endEncoding();
        let fault=self.fault_record();
        let encoder=self.cb.computeCommandEncoder().ok_or("timestamp publication encoder")?;encoder.setComputePipelineState(&kernels.publish.0);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&raw.buffer.0),raw.offset,0);encoder.setBuffer_offset_atIndex(Some(&output.buffer.0),output.offset,1);
            encoder.setBytes_length_atIndex(NonNull::from(&parameters).cast(),32,2);encoder.setBytes_length_atIndex(NonNull::from(&fault).cast(),8,3);
        }
        if let Some(record)=&self.fault {encoder.useResource_usage(ProtocolObject::from_ref(&*record.buffer.0),MTLResourceUsage::Read|MTLResourceUsage::Write);}
        encoder.dispatchThreadgroups_threadsPerThreadgroup(MTLSize{width:1,height:1,depth:1},MTLSize{width:1,height:1,depth:1});encoder.endEncoding();
        let encoder=self.cb.blitCommandEncoder().ok_or("timestamp output encoder")?;let mut source_offset=output.offset;
        for (buffer,offset,n,_,_) in &parts {unsafe {encoder.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(&output.buffer.0,source_offset,&buffer.0,*offset,*n as usize);}source_offset+=*n as usize;}
        encoder.endEncoding();
        for (_,_,n,gpu,cpu) in parts {self.record_known_write(gpu,cpu,n);}
        let keep=counter;let handler=block2::RcBlock::new(move |_:NonNull<gfx::CommandBuffer>|{std::hint::black_box((&keep,&event));});
        unsafe {self.cb.addCompletedHandler(block2::RcBlock::as_ptr(&handler));}
        if super::submit_profile::active() {eprintln!("[cpu-write] GPU timestamp type={kind} ->{address:#x}+0x10");}
        Ok(true)
    }
}
