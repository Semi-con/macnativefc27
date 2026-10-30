use crate::stubs::Args;
use std::sync::atomic::{AtomicU32,Ordering};

fn read64(p:u64) -> u64 { unsafe { (p as *const u64).read_unaligned() } }

pub fn initialize(a:&Args) -> u64 {
    let (obj,builder) = (a.0[0],a.0[1]);
    let (pool,offset) = (read64(builder),read64(builder+8));
    let Some(p) = super::objects::STATE.lock().unwrap().pools.get(&pool).copied() else { return 0; };
    if p.cpu == 0 || offset%4 != 0 || offset.checked_add(4).is_none_or(|end|end>p.size) { return 0; }
    let address = p.cpu+offset;
    // SDK 22 EventInitialize writes 0x28 bytes. Its pure object queries read
    // address+0, CPU pointer+8, pool+0x10 and offset+0x18.
    unsafe {
        std::ptr::write_bytes(obj as *mut u8,0,0x28);
        for (i,v) in [address,address,pool,offset].into_iter().enumerate() { (obj as *mut u64).add(i).write_unaligned(v); }
    }
    1
}

pub fn storage(obj:u64) -> u64 {
    let address = read64(obj+8);
    let (pool,offset) = (read64(obj+0x10),read64(obj+0x18));
    let valid = super::objects::STATE.lock().unwrap().pools.get(&pool).is_some_and(|p| p.cpu!=0 && offset%4==0 && offset.checked_add(4).is_some_and(|end|end<=p.size) && address==p.cpu+offset);
    if !valid { crate::diag::report_and_exit("NVN event has unavailable or invalid physical pool storage"); }
    address
}

pub fn signal_memory(address:u64,mode:u32,value:u32) -> Result<(), &'static str> {
    if address%4 != 0 { return Err("unaligned NVN event storage"); }
    let word = unsafe { AtomicU32::from_ptr(address as *mut u32) };
    match mode { 0=>word.store(value,Ordering::Release),1=>{word.fetch_add(value,Ordering::AcqRel);},_=>return Err("unsupported NVN event signal mode") }
    Ok(())
}

pub fn satisfied(current:u32,mode:u32,value:u32) -> Result<bool,&'static str> {
    Ok(match mode { 0=>current==value,1=>(current.wrapping_sub(value) as i32)>=0,_=>return Err("unsupported NVN event wait mode") })
}

pub fn wait_memory(address:u64,mode:u32,value:u32) -> Result<(), &'static str> {
    let word = unsafe { AtomicU32::from_ptr(address as *mut u32) };
    if satisfied(word.load(Ordering::Acquire),mode,value)? { return Ok(()); }
    crate::kernel::blocking(|| {
        let deadline = std::time::Instant::now()+std::time::Duration::from_secs(10);
        loop {
            let current = word.load(Ordering::Acquire);
            if satisfied(current,mode,value)? { return Ok(()); }
            if std::time::Instant::now()>=deadline {
                eprintln!("[nvn] event {address:#x} wait mode {mode}, expected {value:#x}, actual {current:#x}");
                return Err("NVN event wait remained unsatisfied after ordered GPU completion; cross-queue deferred event waits require an asynchronous replay path");
            }
            std::thread::sleep(std::time::Duration::from_micros(100));
        }
    })
}

pub fn host_impl(name:&str) -> Option<fn(&Args)->u64> {
    Some(match name {
        "nvnEventInitialize"=>initialize,
        "nvnEventFinalize"=>|a| { unsafe {std::ptr::write_bytes(a.0[0] as *mut u8,0,0x28)};0 },
        "nvnEventGetValue"=>|a| unsafe {AtomicU32::from_ptr(storage(a.0[0]) as *mut u32)}.load(Ordering::Acquire) as u64,
        "nvnEventGetMemoryPool"=>|a|read64(a.0[0]+0x10),
        "nvnEventGetMemoryOffset"=>|a|read64(a.0[0]+0x18),
        "nvnEventSignal"=>|a| {signal_memory(storage(a.0[0]),a.0[1] as u32,a.0[2] as u32).unwrap_or_else(|e|crate::diag::report_and_exit(e));0},
        _=>return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn physical_event_object_bounds_copy_and_cpu_signals() {
        let mut backing = [0u32;16]; let base = backing.as_mut_ptr() as u64;
        let pool = 0xfcee0120;
        super::super::objects::STATE.lock().unwrap().pools.insert(pool,super::super::objects::MemoryPool{cpu:base,size:64,flags:0x22});
        let builder = [pool,12u64]; let mut obj = [0xa5u64;8];
        assert_eq!(initialize(&Args([obj.as_mut_ptr() as u64,builder.as_ptr() as u64,0,0,0,0,0,0],[0;8])),1);
        assert!(obj[5..].iter().all(|v|*v==0xa5));
        let copied = obj; let p = storage(copied.as_ptr() as u64);
        signal_memory(p,0,0xfffffffe).unwrap(); signal_memory(p,1,4).unwrap();
        assert_eq!(backing[3],2);
        assert!(satisfied(2,1,0xfffffffe).unwrap()); assert!(!satisfied(0xfffffffe,1,2).unwrap());
        assert!(satisfied(2,0,2).unwrap()); assert!(!satisfied(3,0,2).unwrap());
        assert!(signal_memory(p,2,0).is_err()); assert!(satisfied(0,2,0).is_err());
        super::super::objects::STATE.lock().unwrap().pools.remove(&pool);
    }
    #[test]
    fn event_wait_observes_delayed_cpu_signal() {
        let word = std::sync::Arc::new(AtomicU32::new(0)); let producer = word.clone();
        let t = std::thread::spawn(move || {std::thread::sleep(std::time::Duration::from_millis(2));producer.store(17,Ordering::Release);});
        wait_memory(std::sync::Arc::as_ptr(&word) as u64,0,17).unwrap(); t.join().unwrap();
    }
}
