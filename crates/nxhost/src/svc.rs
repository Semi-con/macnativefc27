// Mush8 - svc.rs
// Constantin Clerc (c22dev) - 0.1
use crate::guest::{self, Frame};
use crate::ipc;
use crate::kernel::{self, KERNEL, Obj, rc};
use crate::mem::{Block, perm, state};
use std::sync::atomic::Ordering::Relaxed;

pub const PROGRAM_ID: u64 = crate::game::TITLE_ID;
const MEMORY_SIZE: u64 = 0xCD50_0000;
fn parse_memory_budget(value:&str)->Option<u64> {
    let bytes=value.parse::<u64>().ok()?.checked_mul(1<<20)?;
    (MEMORY_SIZE..=8u64<<30).contains(&bytes).then_some(bytes)
}
fn memory_size()->u64 {
    static SIZE:std::sync::LazyLock<u64>=std::sync::LazyLock::new(|| {
        let size=std::env::var("NX_MEMORY_BUDGET_MIB").map_or(MEMORY_SIZE,|value|parse_memory_budget(&value).expect("NX_MEMORY_BUDGET_MIB must be 3285..8192"));
        eprintln!("[memory] process physical-memory budget {} MiB",size>>20);size
    });*SIZE
}
pub static SYSTEM_RESOURCE_SIZE: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
pub static THREAD_PERMISSIONS:std::sync::OnceLock<(u64,u64)> = std::sync::OnceLock::new();
pub static DEFAULT_CORE:std::sync::OnceLock<i32> = std::sync::OnceLock::new();

fn cstr(addr: u64, max: usize) -> String {
    let s = unsafe { std::slice::from_raw_parts(addr as *const u8, max) };
    String::from_utf8_lossy(&s[..s.iter().position(|&c| c == 0).unwrap_or(max)]).into_owned()
}

fn trace() -> bool {
    static T: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *T.get_or_init(|| std::env::var_os("NX_TRACE_SVC").is_some())
}

thread_local! {
    static GUEST_FP: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

pub fn current_guest_fp() -> u64 {
    GUEST_FP.with(|c| c.get())
}

#[unsafe(no_mangle)]
pub extern "C" fn dispatch(f: &mut Frame) {
    GUEST_FP.with(|c| c.set(f.x[29]));
    kernel::record_svc(f);
    kernel::sched_point();
    let wait_profile=crate::input::test_control::profile_wait(f);
    dispatch_inner(f);
    if let Some(profile)=wait_profile {profile.finish(f.x[0]);}
    kernel::sched_point();
}

fn dispatch_inner(f: &mut Frame) {
    let n = f.x[16] as u32;
    if trace() {
        eprintln!("[svc] {n:#04x} x0={:#x} x1={:#x} x2={:#x} x3={:#x}", f.x[0], f.x[1], f.x[2], f.x[3]);
    }
    let x = f.x;
    let r: u32 = match n {
        0x01 => set_heap_size(f, x[1]),
        0x02 => {
            let a = kernel::aspace();
            let (_, _, cur) = a.map.lock().unwrap().query(x[0]);
            match a.set(x[0], x[1], Block { perm: x[2] as u32, ..cur }) {
                Ok(()) => rc::OK,
                Err(_) => rc::INVALID_ADDRESS,
            }
        }
        0x03 => set_memory_attribute(x[0],x[1],x[2] as u32,x[3] as u32),
        0x04 => map_memory(x[0], x[1], x[2]),
        0x05 => unmap_memory(x[0], x[1], x[2]),
        0x06 => {
            let (base, size, b) = kernel::aspace().map.lock().unwrap().query(x[2]);
            let info = x[0] as *mut u64;
            unsafe {
                *info = base;
                *info.add(1) = size;
                *info.add(2) = b.state as u64 | (b.attr as u64) << 32;
                *info.add(3) = b.perm as u64;
                *info.add(4) = 0;
            }
            f.x[1] = 0;
            rc::OK
        }
        0x07 => {
            eprintln!("[svc] ExitProcess");
            std::process::exit(0);
        }
        0x08 => match kernel::create_thread(x[1], x[2], x[3], x[4] as i32, x[5] as i32) {
            Ok(h) => {
                f.x[1] = h as u64;
                rc::OK
            }
            Err(e) => {
                eprintln!("[svc] CreateThread: {e}");
                0xD201
            }
        },
        0x09 => match kernel::thread(x[0] as u32) {
            Some(t) => {
                let args = [t.arg, t.handle as u64, 0, 0];
                match kernel::start_thread(t, args) {
                    Ok(_) => rc::OK,
                    Err(_) => 0xD201,
                }
            }
            None => rc::INVALID_HANDLE,
        },
        0x0A => {
            let ctx = guest::current();
            unsafe { guest::nx_exit_to_host(ctx) }
        }
        0x0B => {
            kernel::sleep(x[0] as i64);
            rc::OK
        }
        0x0C => match kernel::thread(x[1] as u32) {
            Some(t) => {
                f.x[1] = t.prio.load(Relaxed) as u64;
                rc::OK
            }
            None => rc::INVALID_HANDLE,
        },
        0x0D => match kernel::thread(x[0] as u32) {
            Some(t) => {
                t.prio.store(x[1] as i32, Relaxed);
                kernel::notify_all();
                rc::OK
            }
            None => rc::INVALID_HANDLE,
        },
        0x0E => match kernel::thread(x[2] as u32) {
            Some(t) => {
                f.x[1] = t.core.load(Relaxed) as u64;
                f.x[2] = t.affinity.load(Relaxed);
                rc::OK
            }
            None => rc::INVALID_HANDLE,
        },
        0x0F => match kernel::thread(x[0] as u32) {
            Some(t) => kernel::set_core_mask(&t,x[1] as i32,x[2]),
            None => rc::INVALID_HANDLE,
        },
        0x10 => {
            f.x[0] = kernel::processor_number();
            return;
        }
        0x11 | 0x12 | 0x17 => {
            let obj = KERNEL.lock().get(x[0] as u32);
            match obj {
                Some(Obj::WritableEvent(e)) | Some(Obj::ReadableEvent(e)) => {
                    if n == 0x11 {
                        kernel::signal_event(&e);
                        rc::OK
                    } else {
                        let was = e.signaled.swap(false, Relaxed);
                        if n == 0x17 && !was { rc::INVALID_STATE } else { rc::OK }
                    }
                }
                _ => rc::INVALID_HANDLE,
            }
        }
        0x13 => map_shared_memory(x[0] as u32, x[1], x[2],x[3] as u32),
        0x14 => unmap_shared_memory(x[0] as u32,x[1],x[2]),
        0x15 => {
            if x[1]%0x1000!=0 || !kernel::aspace().readable(x[1],x[2]) {f.x[0]=rc::INVALID_ADDRESS as u64;return;}
            if x[2]==0 || x[2]%0x1000!=0 {f.x[0]=rc::INVALID_SIZE as u64;return;}
            if ![0,perm::R as u64,perm::RW as u64].contains(&x[3]) {f.x[0]=rc::INVALID_ENUM as u64;return;}
            let h = KERNEL.lock().add(Obj::TransferMemory {address:x[1],size:x[2],permission:x[3] as u32});
            f.x[1] = h as u64;
            rc::OK
        }
        0x16 => {
            let object = { KERNEL.lock().close(x[0] as u32) };
            let existed = object.is_some();
            // Closing a service may join its CoreAudio route thread; release
            // the kernel lock and the guest's scheduling slot before Drop.
            kernel::blocking(|| drop(object));
            if existed { rc::OK } else { rc::INVALID_HANDLE }
        }
        0x18 => {
            let handles = unsafe { std::slice::from_raw_parts(x[1] as *const u32, x[2] as usize) }.to_vec();
            let (r, idx) = kernel::wait_synchronization(&handles, x[3] as i64);
            f.x[1] = idx as u64;
            r
        }
        0x19 => match kernel::thread(x[0] as u32) {
            Some(t) => {
                kernel::cancel_synchronization(&t);
                rc::OK
            }
            None => rc::INVALID_HANDLE,
        },
        0x1A => kernel::arbitrate_lock(x[0] as u32, x[1], x[2] as u32),
        0x1B => kernel::arbitrate_unlock(x[0]),
        0x1C => kernel::wait_process_wide_key(x[0], x[1], x[2] as u32, x[3] as i64),
        0x1D => kernel::signal_process_wide_key(x[0], x[1] as i32),
        0x1E => {
            f.x[0] = system_tick();
            return;
        }
        0x1F => {
            let name = cstr(x[1], 12);
            if name == "sm:" {
                f.x[1] = ipc::new_session_handle(std::sync::Arc::new(ipc::ServiceManager)) as u64;
                rc::OK
            } else {
                eprintln!("[svc] ConnectToNamedPort {name:?}: not found");
                rc::NOT_FOUND
            }
        }
        0x21 => send_sync_request(x[0] as u32, guest::current().tpidrro, 0x100),
        0x22 => send_sync_request(x[2] as u32, x[0], x[1] as usize),
        0x24 => {
            f.x[1] = 0x51;
            rc::OK
        }
        0x25 => match kernel::thread(x[1] as u32) {
            Some(t) => {
                f.x[1] = t.id;
                rc::OK
            }
            None => rc::INVALID_HANDLE,
        },
        0x26 => {
            let caller = unsafe { *((f.sp + 8) as *const u64) };
            eprintln!("[svc] Break(reason={:#x}, info={:#x}, size={:#x}) from {}", x[0], x[1], x[2], crate::diag::symbolize(caller));
            if x[0]&0x8000_0000 != 0 {
                // First-chance C++ exceptions and module load notifications
                // explicitly request notification only, then resume the guest.
                rc::OK
            } else {
                crate::diag::backtrace(x[29]);
                crate::diag::report_and_exit("guest called svcBreak");
            }
        }
        0x27 => {
            let s = unsafe { std::slice::from_raw_parts(x[0] as *const u8, x[1] as usize) };
            eprint!("[guest] {}", String::from_utf8_lossy(s));
            rc::OK
        }
        0x29 => get_info(f, x[1] as u32, x[2] as u32, x[3]),
        0x2A | 0x2B => rc::OK,
        0x2C => physical_memory(x[0],x[1],true),
        0x2D => physical_memory(x[0],x[1],false),
        0x30 | 0x31 => {
            f.x[1] = match (n, x[2]) {
                (0x30, 0) => memory_size(),
                (0x30, 1) => 256,
                (0x30, 2) => 1000,
                (0x30, 3) => 200,
                (0x30, 4) => 900,
                (_, 0) => *HEAP_SIZE.lock().unwrap(),
                _ => 0,
            };
            rc::OK
        }
        0x32 => rc::OK,
        0x34 => kernel::wait_for_address(x[0], x[1] as u32, x[2] as i32, x[3] as i64),
        0x35 => kernel::signal_to_address(x[0], x[1] as u32, x[2] as i32, x[3] as i32),
        0x45 => {
            let (w, r) = kernel::create_event();
            f.x[1] = w as u64;
            f.x[2] = r as u64;
            rc::OK
        }
        _ => {
            let caller = unsafe { *((f.sp + 8) as *const u64) };
            eprintln!("[svc] UNIMPLEMENTED svc {n:#04x} x0={:#x} x1={:#x} x2={:#x} from {}", x[0], x[1], x[2], crate::diag::symbolize(caller));
            0xF001
        }
    };
    f.x[0] = r as u64;
}

pub fn system_tick() -> u64 {
    let (v, freq): (u64, u64);
    unsafe {
        std::arch::asm!("mrs {}, cntvct_el0", out(reg) v);
        std::arch::asm!("mrs {}, cntfrq_el0", out(reg) freq);
    }
    ((v as u128 * nxabi::GUEST_TICK_HZ as u128) / freq as u128) as u64
}

static HEAP_SIZE: std::sync::Mutex<u64> = std::sync::Mutex::new(0);

fn set_memory_attribute(addr: u64,size: u64,mask: u32,value: u32) -> u32 {
    if addr%0x1000 != 0 {return rc::INVALID_ADDRESS;}
    if size == 0 || size%0x1000 != 0 {return rc::INVALID_SIZE;}
    // SDK 22 ProtectRelro locks permissions after applying RELRO. Cache-mode
    // changes are unsupported until a caller with such a requirement is bound.
    if mask&!0x10 != 0 || value&!mask != 0 {return rc::INVALID_ENUM;}
    let a = kernel::aspace();
    let Some(end) = addr.checked_add(size) else {return rc::INVALID_ADDRESS};
    if addr < a.base || end > a.base+a.layout.total() {return rc::INVALID_ADDRESS;}
    let mut memory = a.map.lock().unwrap();
    let mut changes = Vec::new();
    let mut cursor = addr;
    while cursor < end {
        let (lo,len,b) = memory.query(cursor);
        if b.state == state::FREE {return rc::INVALID_STATE;}
        let next = end.min(lo+len);
        changes.push((cursor,next,Block {attr:(b.attr&!mask)|value,..b})); cursor = next;
    }
    for (start,end,b) in changes {memory.set(start,end,b);}
    rc::OK
}

fn physical_memory(addr: u64,size: u64,map: bool) -> u32 {
    if addr%0x1000 != 0 {return rc::INVALID_ADDRESS;}
    if size == 0 || size%0x1000 != 0 {return rc::INVALID_SIZE;}
    if SYSTEM_RESOURCE_SIZE.get().copied().unwrap_or(0) == 0 {return rc::INVALID_STATE;}
    let a = kernel::aspace();
    let Some(end) = addr.checked_add(size) else {return rc::INVALID_ADDRESS};
    if addr < a.alias() || end > a.alias()+a.layout.alias_size {return rc::INVALID_ADDRESS;}
    let mut cursor = addr;
    let mut changes = Vec::new();
    {
        let memory = a.map.lock().unwrap();
        while cursor < end {
            let (lo,len,b) = memory.query(cursor);
            if b.state != state::FREE && (b.state != state::NORMAL || b.perm != perm::RW || b.attr != 0) {return rc::INVALID_STATE;}
            let next = end.min(lo+len);
            if (map && b.state == state::FREE) || (!map && b.state == state::NORMAL) {changes.push((cursor,next-cursor));}
            cursor = next;
        }
        let extra: u64 = if map {changes.iter().map(|(_,len)|len).sum()} else {0};
        let used=memory.private_bytes();let limit=memory_size()-SYSTEM_RESOURCE_SIZE.get().copied().unwrap_or(0);
        if used+extra > limit {eprintln!("[memory] physical commit rejected address{addr:#x} size{size:#x} used{used:#x} extra{extra:#x} budget{limit:#x}");return 0xCE01;}
    }
    for (start,len) in changes {
        let result = if map {
            a.set(start,len,Block {state:state::NORMAL,perm:perm::RW,attr:0})
        } else {a.discard(start,len)};
        if result.is_err() {return rc::INVALID_ADDRESS;}
    }
    rc::OK
}

fn set_heap_size(f: &mut Frame, size: u64) -> u32 {
    let a = kernel::aspace();
    if size % 0x20_0000 != 0 || size > a.layout.heap_size {
        return rc::INVALID_SIZE;
    }
    let mut cur = HEAP_SIZE.lock().unwrap();
    if size > *cur && a.map.lock().unwrap().private_bytes()+size-*cur > memory_size()-SYSTEM_RESOURCE_SIZE.get().copied().unwrap_or(0) {
        eprintln!("[memory] heap expansion rejected old{:#x} requested{size:#x} budget{:#x}",*cur,memory_size());
        return 0xCE01;
    }
    let r = if size >= *cur {
        a.set(a.heap() + *cur, size - *cur, Block { state: state::NORMAL, perm: perm::RW, attr: 0 })
    } else {
        a.discard(a.heap() + size, *cur - size)
    };
    if let Err(e) = r {
        eprintln!("[svc] SetHeapSize({size:#x}): {e}");
        return 0xD001;
    }
    if size > *cur {
        crate::gfx::wrap_heap(a.heap(), size);
    }
    *cur = size;
    f.x[1] = a.heap();
    rc::OK
}

fn map_memory(dst: u64, src: u64, size: u64) -> u32 {
    let a = kernel::aspace();
    let st = if (a.stack()..a.stack() + a.layout.stack_size).contains(&dst) { state::STACK } else { state::ALIAS };
    if a.set(dst, size, Block { state: st, perm: perm::RW, attr: 0 }).is_err() {
        return rc::INVALID_ADDRESS;
    }
    unsafe { std::ptr::copy_nonoverlapping(src as *const u8, dst as *mut u8, size as usize) };
    a.map.lock().unwrap().set(src, src + size, Block { state: state::NORMAL, perm: perm::NONE, attr: 1 });
    rc::OK
}

fn unmap_memory(dst: u64, src: u64, size: u64) -> u32 {
    let a = kernel::aspace();
    a.map.lock().unwrap().set(src, src + size, Block { state: state::NORMAL, perm: perm::RW, attr: 0 });
    unsafe { std::ptr::copy_nonoverlapping(dst as *const u8, src as *mut u8, size as usize) };
    let _ = a.discard(dst, size);
    rc::OK
}

type SharedMap = std::collections::HashMap<u64,(std::sync::Arc<kernel::SharedMemory>,u64)>;
static SHARED_MAPS: std::sync::LazyLock<std::sync::Mutex<SharedMap>> = std::sync::LazyLock::new(|| std::sync::Mutex::new(SharedMap::new()));

fn map_shared_memory(h: u32, addr: u64, size: u64, permission: u32) -> u32 {
    let obj = KERNEL.lock().get(h);
    let Some(Obj::SharedMemory(sm)) = obj else { return rc::INVALID_HANDLE };
    if size != sm.size || size == 0 || size%0x1000 != 0 { return rc::INVALID_SIZE; }
    let a = kernel::aspace();
    if addr%0x1000 != 0 || addr < a.base || addr.checked_add(size).is_none_or(|e| e > a.base+a.layout.total()) { return rc::INVALID_ADDRESS; }
    if ![perm::R,perm::RW].contains(&permission) || (sm.snapshot && permission != perm::R) {return rc::INVALID_ENUM;}
    let (lo,len,b) = a.map.lock().unwrap().query(addr);
    if b.state != state::FREE || addr+size > lo+len {return rc::INVALID_STATE;}
    if sm.snapshot {
        if a.set(addr,size,Block {state:state::SHARED,perm:perm::RW,attr:0}).is_err() {return rc::INVALID_ADDRESS;}
        unsafe {std::ptr::copy_nonoverlapping(sm.host as *const u8,addr as *mut u8,size as usize);}
        if a.set(addr,size,Block {state:state::SHARED,perm:permission,attr:0}).is_err() {return rc::INVALID_ADDRESS;}
    } else {
        if addr%crate::mem::PAGE != 0 || size%crate::mem::PAGE != 0 {return rc::INVALID_ADDRESS;}
        let prot = libc::PROT_READ | if permission&perm::W != 0 {libc::PROT_WRITE} else {0};
        let r = unsafe {libc::mmap(addr as *mut _,size as usize,prot,libc::MAP_SHARED|libc::MAP_FIXED,sm.fd,0)};
        if r as u64 != addr {return rc::INVALID_ADDRESS;}
        a.map.lock().unwrap().set(addr,addr+size,Block {state:state::SHARED,perm:permission,attr:0});
    }
    SHARED_MAPS.lock().unwrap().insert(addr,(sm,size));
    rc::OK
}

fn unmap_shared_memory(h: u32,addr: u64,size: u64) -> u32 {
    let Some(Obj::SharedMemory(sm)) = KERNEL.lock().get(h) else {return rc::INVALID_HANDLE};
    let mut mappings = SHARED_MAPS.lock().unwrap();
    let Some((mapped,len)) = mappings.get(&addr) else {return rc::INVALID_ADDRESS};
    if !std::sync::Arc::ptr_eq(mapped,&sm) || *len != size {return rc::INVALID_SIZE;}
    if kernel::aspace().discard(addr,size).is_err() {return rc::INVALID_ADDRESS;}
    mappings.remove(&addr); rc::OK
}

#[cfg(test)]
mod shared_memory_tests {
    use super::*;
    use crate::mem;
    #[test]
    fn subpage_clock_snapshots_preserve_neighbors_and_live_aliases_share_writes() {
        KERNEL.aspace.get_or_init(|| mem::AddressSpace::reserve().unwrap());
        let first = kernel::aspace().alias()+0x1000;
        let make = |pattern: u8,snapshot: bool,size: u64| {
            let mut s = crate::shm::create(size).unwrap(); s.snapshot = snapshot;
            unsafe {std::ptr::write_bytes(s.host as *mut u8,pattern,size as usize);}
            let s = std::sync::Arc::new(s); let h = KERNEL.lock().add(Obj::SharedMemory(s.clone())); (h,s)
        };
        let (ha,a) = make(0xa5,true,0x1000); let (hb,b) = make(0x5a,true,0x1000);
        assert_eq!(map_shared_memory(ha,first,0x1000,perm::R),rc::OK);
        assert_eq!(map_shared_memory(hb,first+0x1000,0x1000,perm::R),rc::OK);
        unsafe {assert_eq!(*(first as *const u8),0xa5);assert_eq!(*((first+0x1000) as *const u8),0x5a);}
        assert_eq!(map_shared_memory(ha,first,0x1000,perm::R),rc::INVALID_STATE);
        assert_eq!(unmap_shared_memory(ha,first,0x1000),rc::OK);
        unsafe {assert_eq!(*((first+0x1000) as *const u8),0x5a);}
        assert_eq!(unmap_shared_memory(hb,first+0x1000,0x1000),rc::OK);
        let live = kernel::aspace().alias()+mem::PAGE*2;
        let (hc,c) = make(43,false,mem::PAGE);
        assert_eq!(map_shared_memory(hc,live,mem::PAGE,perm::RW),rc::OK);
        unsafe {
            assert_eq!(*(live as *const u8),43);
            (live as *mut u8).write(87); assert_eq!(*(c.host as *const u8),87);
            (c.host as *mut u8).write(99); assert_eq!(*(live as *const u8),99);
        }
        assert_eq!(unmap_shared_memory(hc,live,mem::PAGE),rc::OK);
        for h in [ha,hb,hc] {let removed = {KERNEL.lock().close(h)};drop(removed);}
        drop((a,b,c));
    }
}

#[cfg(test)]
mod physical_memory_tests {
    use super::*;
    #[test]
    fn native_memory_budget_is_bounded_and_uses_bytes_consistently() {
        assert_eq!(parse_memory_budget("3285"),Some(MEMORY_SIZE));
        assert_eq!(parse_memory_budget("4096"),Some(4u64<<30));
        assert_eq!(parse_memory_budget("8192"),Some(8u64<<30));
        for invalid in ["0","3284","8193","18446744073709551615","x"] {assert!(parse_memory_budget(invalid).is_none());}
    }
    #[test]
    fn native_partial_physical_maps_preserve_neighbors_and_account_pages() {
        let _ = SYSTEM_RESOURCE_SIZE.set(0x600000);
        KERNEL.aspace.get_or_init(|| crate::mem::AddressSpace::reserve().unwrap());
        let addr = kernel::aspace().alias()+0x08000000;
        let bytes = || kernel::aspace().map.lock().unwrap().private_bytes();
        let baseline = bytes();
        assert_eq!(physical_memory(addr,0x1000,true),rc::OK);
        assert_eq!(physical_memory(addr,0x3000,true),rc::OK);
        assert_eq!(bytes(),baseline+0x3000);
        unsafe {std::ptr::write_bytes(addr as *mut u8,0xa5,0x3000);}
        assert_eq!(physical_memory(addr+0x1000,0x1000,false),rc::OK);
        assert_eq!(bytes(),baseline+0x2000);
        unsafe {assert_eq!(*(addr as *const u8),0xa5);assert_eq!(*((addr+0x2000) as *const u8),0xa5);}
        assert_eq!(physical_memory(addr+0x1000,0x1000,true),rc::OK);
        unsafe {assert_eq!(*((addr+0x1000) as *const u8),0);}
        assert_eq!(set_memory_attribute(addr,0x1000,0x10,0x10),rc::OK);
        assert_eq!(physical_memory(addr,0x1000,false),rc::INVALID_STATE);
        assert_eq!(set_memory_attribute(addr,0x1000,0x10,0),rc::OK);
        assert_eq!(physical_memory(addr,0x3000,false),rc::OK);
        assert_eq!(bytes(),baseline);
        assert_eq!(physical_memory(kernel::aspace().heap(),0x1000,true),rc::INVALID_ADDRESS);
        assert_eq!(physical_memory(addr,0,true),rc::INVALID_SIZE);
        let space=kernel::aspace();
        assert!(space.layout.alias_size>=0x2_0000_0000);
        assert_eq!(space.layout.heap_size,0x1_0000_0000);
        let tail=space.alias()+space.layout.alias_size-0x1000;
        assert_eq!(physical_memory(tail,0x1000,true),rc::OK);
        unsafe {(tail as *mut u64).write(0x123456789abcdef0);assert_eq!((tail as *const u64).read(),0x123456789abcdef0);}
        assert_eq!(physical_memory(tail,0x1000,false),rc::OK);
        assert_eq!(bytes(),baseline);
    }
}

fn send_sync_request(h: u32, buf: u64, size: usize) -> u32 {
    let obj = KERNEL.lock().get(h);
    let Some(Obj::Session(s)) = obj else {
        eprintln!("[svc] SendSyncRequest on non-session handle {h:#x}");
        return rc::INVALID_HANDLE;
    };
    let b = unsafe { std::slice::from_raw_parts_mut(buf as *mut u8, size) };
    ipc::handle(&s, h, b);
    rc::OK
}

fn get_info(f: &mut Frame, id0: u32, handle: u32, id1: u64) -> u32 {
    let a = kernel::aspace();
    let v = match id0 {
        0 => THREAD_PERMISSIONS.get().map_or(0xf,|p|p.0),
        1 => THREAD_PERMISSIONS.get().map_or(u64::MAX,|p|p.1),
        2 => a.alias(),
        3 => a.layout.alias_size,
        4 => a.heap(),
        5 => a.layout.heap_size,
        6 => memory_size(),
        21 => memory_size()-SYSTEM_RESOURCE_SIZE.get().copied().unwrap_or(0),
        7 | 22 => a.map.lock().unwrap().private_bytes(),
        8 => 0,
        9 => KERNEL.lock().add(Obj::ResourceLimit) as u64,
        10 => 0,
        11 => {
            let mut b = [0u8; 8];
            let _ = getrandom(&mut b);
            u64::from_le_bytes(b) ^ id1
        }
        12 => a.base,
        13 => a.layout.total(),
        14 => a.stack(),
        15 => a.layout.stack_size,
        16 => SYSTEM_RESOURCE_SIZE.get().copied().unwrap_or(0),
        // Native mappings need no pages from Horizon's dedicated metadata pool.
        17 => 0,
        18 => PROGRAM_ID,
        20 => a.process_local(),
        23 => 1,
        24 => 200,
        25 | 0xF000_0002 => 0,
        26..=28 => 0,
        _ => {
            eprintln!("[svc] GetInfo unknown id {id0} (handle {handle:#x}, sub {id1})");
            return rc::INVALID_ENUM;
        }
    };
    f.x[1] = v;
    rc::OK
}

fn getrandom(b: &mut [u8]) -> bool {
    unsafe { libc::getentropy(b.as_mut_ptr() as *mut _, b.len()) == 0 }
}
