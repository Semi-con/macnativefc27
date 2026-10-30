// A completed command buffer's fault record is the only CPU consumer. Keep
// these records separate from read-only uniforms; recycle only after completion.
use super::{gfx,state};
use objc2_metal::*;
use std::sync::{LazyLock,Mutex};

pub(super) fn enabled()->bool {
    static ENABLED:LazyLock<bool>=LazyLock::new(||std::env::var_os("NX_ISOLATE_SHADER_FAULTS").is_some());
    *ENABLED
}
static FREE:LazyLock<Mutex<Vec<gfx::Shared<gfx::Buffer>>>>=LazyLock::new(Default::default);

pub(super) fn allocate()->state::Alloc {
    let buffer=FREE.lock().unwrap().pop().unwrap_or_else(||gfx::Shared(gfx::gpu().device.0.newBufferWithLength_options(32,MTLResourceOptions::StorageModeShared).expect("shader fault record")));
    let ptr=buffer.0.contents().as_ptr() as *mut u8;let gpu=buffer.0.gpuAddress();
    state::Alloc{buffer,offset:0,ptr,gpu,new_chunk:false}
}

pub(super) fn recycle(buffer:gfx::Shared<gfx::Buffer>) {
    let mut free=FREE.lock().unwrap();if free.len()<128 {free.push(buffer);}
}
