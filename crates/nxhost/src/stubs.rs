// Mush8 - stubs.rs
// Constantin Clerc (c22dev) - 0.1
use std::sync::OnceLock;
use std::sync::atomic::{AtomicUsize, Ordering};

#[repr(C)]
pub struct Args(pub [u64; 8], pub [u64; 8]);

impl Args {
    pub fn f32(&self, i: usize) -> f32 {
        f32::from_bits(self.1[i] as u32)
    }
}

pub type Handler = fn(tag: u64, args: &Args) -> u64;

const STUB_SIZE: u64 = 8;
const MAX_STUBS: usize = 8192;

static HANDLERS: [OnceLock<(Handler, u64)>; MAX_STUBS] = [const { OnceLock::new() }; MAX_STUBS];
static NEXT: AtomicUsize = AtomicUsize::new(0);

#[unsafe(no_mangle)]
extern "C" fn nx_stub_dispatch(idx: u64, args: &Args) -> u64 {
    let (h, tag) = *HANDLERS[idx as usize].get().expect("stub called before it was made");
    h(tag, args)
}

unsafe extern "C" {
    fn nx_stubs();
}
std::arch::global_asm!(
    ".globl _nx_stubs",
    ".p2align 4",
    "_nx_stubs:",
    ".set nx_stub_i, 0",
    ".rept 8192",
    "movz x16, #nx_stub_i",
    "b _nx_stub_entry",
    ".set nx_stub_i, nx_stub_i + 1",
    ".endr",
    ".p2align 2",
    "_nx_stub_entry:",
    "stp x0, x1, [sp, #-32]!",
    "stp x2, x3, [sp, #16]",
    "add x2, sp, #32",
    "mrs x0, tpidrro_el0",
    "adrp x1, {slot}@PAGE",
    "ldr x1, [x1, {slot}@PAGEOFF]",
    "ldr x0, [x0, x1]",
    "sub x1, x2, #32",
    "cbz x0, 3f",
    "ldr x1, [x0, #0x30]",
    "cmp x2, x1",
    "b.hi 3f",
    "sub x3, x1, #0x80, lsl #12",
    "cmp x2, x3",
    "b.ls 3f",
    "sub x1, x2, #32",
    "3:",
    "and x1, x1, #~15",
    "str x2, [x1, #-16]!",
    "mov x17, x1",
    "ldp x2, x3, [sp, #16]",
    "ldp x0, x1, [sp], #32",
    "mov sp, x17",
    "stp x29, x30, [sp, #-0x90]!",
    "mov x29, sp",
    "stp x0, x1, [sp, #0x10]",
    "stp x2, x3, [sp, #0x20]",
    "stp x4, x5, [sp, #0x30]",
    "stp x6, x7, [sp, #0x40]",
    "stp d0, d1, [sp, #0x50]",
    "stp d2, d3, [sp, #0x60]",
    "stp d4, d5, [sp, #0x70]",
    "stp d6, d7, [sp, #0x80]",
    "mov x0, x16",
    "add x1, sp, #0x10",
    "bl {f}",
    "ldp x29, x30, [sp], #0x90",
    "ldr x17, [sp]",
    "mov sp, x17",
    "ret",
    f = sym nx_stub_dispatch,
    slot = sym crate::guest::NX_CTX_SLOT,
);

pub fn make(h: Handler, tag: u64) -> u64 {
    let i = NEXT.fetch_add(1, Ordering::Relaxed);
    assert!(i < MAX_STUBS, "out of stubs");
    let _ = HANDLERS[i].set((h, tag));
    nx_stubs as *const () as usize as u64 + i as u64 * STUB_SIZE
}
