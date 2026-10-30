// Avoid a logical-core handoff for already completed or very short GPU work.
// Every path still invokes Metal's original completion wait, including handlers.
use crate::gfx;
use objc2_metal::*;
use std::sync::LazyLock;

pub(super) fn enabled()->bool {
    static ENABLED:LazyLock<bool>=LazyLock::new(||std::env::var_os("NX_HOST_WAIT_FASTPATHS").is_some());
    *ENABLED
}

// True means the guest scheduling slot was released for a potentially long wait.
pub(super) fn complete(command:&gfx::CommandBuffer)->bool {
    if enabled() {
        let start=std::time::Instant::now();
        for _ in 0..128 {
            if matches!(command.status(),MTLCommandBufferStatus::Completed|MTLCommandBufferStatus::Error) {
                command.waitUntilCompleted();return false;
            }
            if start.elapsed()>=std::time::Duration::from_micros(64) {break;}
            std::hint::spin_loop();
        }
    }
    crate::kernel::blocking(||command.waitUntilCompleted());true
}

#[cfg(all(test,target_os="macos"))]
mod tests {
    use super::*;
    use objc2::runtime::ProtocolObject;

    #[test]
    #[ignore="requires a real Metal GPU"]
    fn gpu_completion_fastpath_waits_for_actual_fenced_writes_and_reuses_completed_work() {
        objc2::rc::autoreleasepool(|_|unsafe {
            let device=&gfx::gpu().device.0;let event=gfx::Shared(device.newSharedEvent().unwrap());
            let buffer=device.newBufferWithLength_options(16384,MTLResourceOptions::StorageModeShared).unwrap();std::ptr::write_bytes(buffer.contents().as_ptr(),0,16384);
            let command=gfx::gpu().queue.0.commandBuffer().unwrap();command.encodeWaitForEvent_value(ProtocolObject::from_ref(&*event.0),1);
            let encoder=command.blitCommandEncoder().unwrap();encoder.fillBuffer_range_value(&buffer,objc2_foundation::NSRange::new(0,16384),0x3c);encoder.endEncoding();command.commit();
            assert_ne!(command.status(),MTLCommandBufferStatus::Completed);assert_eq!(*buffer.contents().as_ptr().cast::<u32>(),0);
            let release=std::thread::spawn(move || {std::thread::sleep(std::time::Duration::from_millis(20));event.0.setSignaledValue(1);});
            complete(&command);release.join().unwrap();assert_eq!(command.status(),MTLCommandBufferStatus::Completed);
            assert!(std::slice::from_raw_parts(buffer.contents().as_ptr().cast::<u32>(),4096).iter().all(|&v|v==0x3c3c3c3c),"completion must not return before the real GPU write");
            let handoff=complete(&command);assert_eq!(handoff,!enabled(),"already completed work needs no logical-core handoff in the gated path");
        });
    }
}
