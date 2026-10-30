// Mush8 - guest.rs
// Constantin Clerc (c22dev) - 0.1
use nxabi::ctx;
use std::arch::global_asm;
use std::sync::atomic::{AtomicU64, Ordering};

#[unsafe(no_mangle)]
pub static NX_CTX_SLOT: AtomicU64 = AtomicU64::new(0);

static KEY: AtomicU64 = AtomicU64::new(u64::MAX);

#[repr(C)]
pub struct GuestThread {
    pub x18: u64,
    pub tpidr: u64,
    pub tpidrro: u64,
    pub spill: [u64; 3],
    pub gate_stack_top: u64,
    pub host_sp: u64,
    pub renderer_tls: u64,
    pub handle: u32,
    gate_stack: Vec<u8>,
    renderer_tls_storage: Vec<u64>,
}

const _: () = {
    assert!(std::mem::offset_of!(GuestThread, x18) == ctx::X18 as usize);
    assert!(std::mem::offset_of!(GuestThread, tpidr) == ctx::TPIDR as usize);
    assert!(std::mem::offset_of!(GuestThread, tpidrro) == ctx::TPIDRRO as usize);
    assert!(std::mem::offset_of!(GuestThread, spill) == ctx::SPILL0 as usize);
    assert!(std::mem::offset_of!(GuestThread, gate_stack_top) == ctx::GATE_STACK as usize);
    assert!(std::mem::offset_of!(GuestThread, host_sp) == 0x38);
    assert!(std::mem::offset_of!(GuestThread, renderer_tls) == ctx::RENDERER_TLS as usize);
};

const GATE_STACK_SIZE: usize = 512 * 1024;

pub fn init_tsd() {
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| {
        let mut key: libc::pthread_key_t = 0;
        assert_eq!(unsafe { libc::pthread_key_create(&mut key, None) }, 0);
        NX_CTX_SLOT.store(key as u64 * 8, Ordering::Relaxed);
        KEY.store(key as u64, Ordering::Release);
    });
}

impl GuestThread {
    pub(crate) fn owns_gate_stack_span(&self, address:u64, len:usize)->bool {
        let start=self.gate_stack.as_ptr() as u64;
        address>=start && address.checked_add(len as u64)
            .is_some_and(|end|end<=start+self.gate_stack.len() as u64)
    }

    pub fn new(handle: u32, tlr: u64) -> Box<Self> {
        let gate_stack = vec![0u8; GATE_STACK_SIZE];
        let top = (gate_stack.as_ptr() as u64 + GATE_STACK_SIZE as u64) & !15;
        // Matching renderer .nrs PT_TLS: filesz=0, memsz=0x208, align=8.
        // One frame-data pointer plus 32 independent 16-byte TLS slots.
        let renderer_tls_storage = vec![0u64; 0x208 / 8];
        let renderer_tls = renderer_tls_storage.as_ptr() as u64;
        Box::new(Self { x18: 0, tpidr: 0, tpidrro: tlr, spill: [0; 3], gate_stack_top: top, host_sp: 0,
            renderer_tls, handle, gate_stack, renderer_tls_storage })
    }

    pub fn install(self: &mut Box<Self>) {
        let base: u64;
        unsafe { std::arch::asm!("mrs {}, tpidrro_el0", out(reg) base) };
        assert_eq!(base & 7, 0, "Darwin TSD base has low bits set; rewritten code assumes it does not");
        let p: *mut GuestThread = &mut **self;
        assert_eq!(unsafe { libc::pthread_setspecific(KEY.load(Ordering::Relaxed) as _, p as *const _) }, 0);
    }
}

pub fn try_current() -> Option<&'static mut GuestThread> {
    let key = KEY.load(Ordering::Acquire);
    if key == u64::MAX { return None; }
    let p = unsafe { libc::pthread_getspecific(key as _) } as *mut GuestThread;
    (!p.is_null()).then(|| unsafe { &mut *p })
}

pub fn current() -> &'static mut GuestThread {
    try_current().expect("not a guest thread")
}

#[repr(C)]
pub struct Frame {
    pub x: [u64; 31],
    pub sp: u64,
    pub nzcv: u64,
    pub fpcr: u64,
    pub fpsr: u64,
    _pad: u64,
    pub q: [u128; 32],
}
const _: () = assert!(std::mem::size_of::<Frame>() == 0x320);

#[repr(C)]
pub struct Entry {
    pub pc: u64,
    pub sp: u64,
    pub x: [u64; 8],
}

#[allow(improper_ctypes)]
unsafe extern "C" {
    pub fn nx_renderer_tlsdesc();
    pub fn nx_enter_guest(entry: *const Entry, ctx: *mut GuestThread);
    pub fn nx_exit_to_host(ctx: *mut GuestThread) -> !;
    pub fn nx_svc_gate();
}

// TLSDESC has a stronger ABI than a regular function call: all registers
// except x0, including NZCV and SIMD state, are preserved. Return an offset
// from the guest's current TP, rather than reading Darwin's thread pointer.
global_asm!(
    ".globl _nx_renderer_tlsdesc", ".p2align 2", "_nx_renderer_tlsdesc:",
    "stp x1, x2, [sp, #-16]!", "mrs x1, tpidrro_el0",
    "adrp x2, {slot}@PAGE", "ldr x2, [x2, {slot}@PAGEOFF]",
    "ldr x1, [x1, x2]", "ldr x2, [x1, #0x40]", "ldr x1, [x1, #8]",
    "ldr x0, [x0, #8]", "add x0, x2, x0", "sub x0, x0, x1",
    "ldp x1, x2, [sp], #16", "ret", slot = sym NX_CTX_SLOT,
);

#[cfg(test)]
mod tls_tests {
    use super::*;
    #[test]
    fn tlsdesc_preserves_special_abi_and_isolates_native_threads() {
        static INIT: std::sync::Once = std::sync::Once::new();
        INIT.call_once(init_tsd);
        let barrier = std::sync::Arc::new(std::sync::Barrier::new(2));
        let threads: Vec<_> = [0x111u64,0x222].into_iter().map(|marker| {
            let barrier = barrier.clone();
            std::thread::spawn(move || {
                let mut ctx = GuestThread::new(marker as u32,0);
                ctx.install(); ctx.tpidr = 0x100000;
                ctx.renderer_tls_storage[0] = marker;
                ctx.renderer_tls_storage[1] = marker+1;
                barrier.wait();
                for arg in [0u64,8] {
                    let desc = [nx_renderer_tlsdesc as *const () as usize as u64,arg];
                    let (mut r0, mut r1, mut r2) = (desc.as_ptr() as u64,0x1234u64,0x5678u64);
                    let flags: u64;
                    unsafe { std::arch::asm!(
                        "mov x9, #0x60000000", "msr nzcv, x9",
                        "bl _nx_renderer_tlsdesc", "mrs x9, nzcv",
                        inout("x0") r0, inout("x1") r1, inout("x2") r2,
                        lateout("x9") flags, clobber_abi("C"),
                    ); }
                    assert_eq!((r1,r2,flags),(0x1234,0x5678,0x60000000));
                    let address = ctx.tpidr.wrapping_add(r0);
                    assert_eq!(address,ctx.renderer_tls+arg);
                    assert_eq!(unsafe { *(address as *const u64) },marker+arg/8);
                }
                unsafe { libc::pthread_setspecific(KEY.load(Ordering::Relaxed) as _,std::ptr::null()); }
            })
        }).collect();
        for t in threads { t.join().unwrap(); }
    }
}

global_asm!(
    ".globl _nx_enter_guest",
    ".p2align 2",
    "_nx_enter_guest:",
    "stp x19, x20, [sp, #-0xa0]!",
    "stp x21, x22, [sp, #0x10]",
    "stp x23, x24, [sp, #0x20]",
    "stp x25, x26, [sp, #0x30]",
    "stp x27, x28, [sp, #0x40]",
    "stp x29, x30, [sp, #0x50]",
    "stp d8, d9, [sp, #0x60]",
    "stp d10, d11, [sp, #0x70]",
    "stp d12, d13, [sp, #0x80]",
    "stp d14, d15, [sp, #0x90]",
    "mov x9, sp",
    "str x9, [x1, #0x38]",
    "ldp x9, x10, [x0]",
    "mov sp, x10",
    "ldp x4, x5, [x0, #0x30]",
    "ldp x6, x7, [x0, #0x40]",
    "ldp x2, x3, [x0, #0x20]",
    "ldp x0, x1, [x0, #0x10]",
    "adr x30, 1f",
    "br x9",
    "1:",
    "mrs x0, tpidrro_el0",
    "adrp x1, {slot}@PAGE",
    "ldr x1, [x1, {slot}@PAGEOFF]",
    "ldr x0, [x0, x1]",
    ".globl _nx_exit_to_host",
    "_nx_exit_to_host:",
    "ldr x9, [x0, #0x38]",
    "mov sp, x9",
    "ldp x21, x22, [sp, #0x10]",
    "ldp x23, x24, [sp, #0x20]",
    "ldp x25, x26, [sp, #0x30]",
    "ldp x27, x28, [sp, #0x40]",
    "ldp x29, x30, [sp, #0x50]",
    "ldp d8, d9, [sp, #0x60]",
    "ldp d10, d11, [sp, #0x70]",
    "ldp d12, d13, [sp, #0x80]",
    "ldp d14, d15, [sp, #0x90]",
    "ldp x19, x20, [sp], #0xa0",
    "ret",
    ".globl _nx_svc_gate",
    ".p2align 2",
    "_nx_svc_gate:",
    "stp x0, x1, [sp, #-32]!",
    "stp x2, x3, [sp, #16]",
    "mrs x0, tpidrro_el0",
    "adrp x1, {slot}@PAGE",
    "ldr x1, [x1, {slot}@PAGEOFF]",
    "ldr x0, [x0, x1]",
    "ldr x1, [x0, #0x30]",
    "add x2, sp, #32",
    "cmp x2, x1",
    "b.hi 2f",
    "sub x3, x1, #0x80, lsl #12",
    "cmp x2, x3",
    "b.ls 2f",
    "sub x1, x2, #32",
    "2:",
    "sub x1, x1, #0x320",
    "and x1, x1, #~15",
    "ldp x2, x3, [sp, #16]",
    "stp x2, x3, [x1, #0x10]",
    "stp x4, x5, [x1, #0x20]",
    "stp x6, x7, [x1, #0x30]",
    "stp x8, x9, [x1, #0x40]",
    "stp x10, x11, [x1, #0x50]",
    "stp x12, x13, [x1, #0x60]",
    "stp x14, x15, [x1, #0x70]",
    "stp x16, x17, [x1, #0x80]",
    "stp x18, x19, [x1, #0x90]",
    "stp x20, x21, [x1, #0xa0]",
    "stp x22, x23, [x1, #0xb0]",
    "stp x24, x25, [x1, #0xc0]",
    "stp x26, x27, [x1, #0xd0]",
    "stp x28, x29, [x1, #0xe0]",
    "str x30, [x1, #0xf0]",
    "ldp x2, x3, [sp], #32",
    "stp x2, x3, [x1]",
    "mov x2, sp",
    "str x2, [x1, #0xf8]",
    "mrs x2, nzcv",
    "str x2, [x1, #0x100]",
    "mrs x2, fpcr",
    "str x2, [x1, #0x108]",
    "mrs x2, fpsr",
    "str x2, [x1, #0x110]",
    "add x2, x1, #0x120",
    "stp q0, q1, [x2, #0x00]",
    "stp q2, q3, [x2, #0x20]",
    "stp q4, q5, [x2, #0x40]",
    "stp q6, q7, [x2, #0x60]",
    "stp q8, q9, [x2, #0x80]",
    "stp q10, q11, [x2, #0xa0]",
    "stp q12, q13, [x2, #0xc0]",
    "stp q14, q15, [x2, #0xe0]",
    "stp q16, q17, [x2, #0x100]",
    "stp q18, q19, [x2, #0x120]",
    "stp q20, q21, [x2, #0x140]",
    "stp q22, q23, [x2, #0x160]",
    "stp q24, q25, [x2, #0x180]",
    "stp q26, q27, [x2, #0x1a0]",
    "stp q28, q29, [x2, #0x1c0]",
    "stp q30, q31, [x2, #0x1e0]",
    "mov sp, x1",
    "mov x0, sp",
    "bl {dispatch}",
    "mov x1, sp",
    "add x2, x1, #0x120",
    "ldp q0, q1, [x2, #0x00]",
    "ldp q2, q3, [x2, #0x20]",
    "ldp q4, q5, [x2, #0x40]",
    "ldp q6, q7, [x2, #0x60]",
    "ldp q8, q9, [x2, #0x80]",
    "ldp q10, q11, [x2, #0xa0]",
    "ldp q12, q13, [x2, #0xc0]",
    "ldp q14, q15, [x2, #0xe0]",
    "ldp q16, q17, [x2, #0x100]",
    "ldp q18, q19, [x2, #0x120]",
    "ldp q20, q21, [x2, #0x140]",
    "ldp q22, q23, [x2, #0x160]",
    "ldp q24, q25, [x2, #0x180]",
    "ldp q26, q27, [x2, #0x1a0]",
    "ldp q28, q29, [x2, #0x1c0]",
    "ldp q30, q31, [x2, #0x1e0]",
    "ldr x2, [x1, #0x100]",
    "msr nzcv, x2",
    "ldr x2, [x1, #0x108]",
    "msr fpcr, x2",
    "ldr x2, [x1, #0x110]",
    "msr fpsr, x2",
    "ldr x2, [x1, #0xf8]",
    "mov sp, x2",
    "ldp x2, x3, [x1, #0x10]",
    "ldp x4, x5, [x1, #0x20]",
    "ldp x6, x7, [x1, #0x30]",
    "ldp x8, x9, [x1, #0x40]",
    "ldp x10, x11, [x1, #0x50]",
    "ldp x12, x13, [x1, #0x60]",
    "ldp x14, x15, [x1, #0x70]",
    "ldp x16, x17, [x1, #0x80]",
    "ldp x20, x21, [x1, #0xa0]",
    "ldp x22, x23, [x1, #0xb0]",
    "ldp x24, x25, [x1, #0xc0]",
    "ldp x26, x27, [x1, #0xd0]",
    "ldp x28, x29, [x1, #0xe0]",
    "ldr x19, [x1, #0x98]",
    "ldr x30, [x1, #0xf0]",
    "ldp x0, x1, [x1]",
    "ret",
    slot = sym NX_CTX_SLOT,
    dispatch = sym crate::svc::dispatch,
);
