// Mush8 - kernel.rs
// Constantin Clerc (c22dev) - 0.1
use crate::guest::{self, Entry, GuestThread};
use crate::mem::{self, AddressSpace, Block, perm, state};
use anyhow::Result;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU32, AtomicU64, Ordering::*};
use std::sync::{Arc, Condvar, LazyLock, Mutex, MutexGuard, OnceLock};
use std::time::{Duration, Instant};

pub mod rc {
    pub const OK: u32 = 0;
    pub const INVALID_SIZE: u32 = 0xCA01;
    pub const INVALID_ADDRESS: u32 = 0xCC01;
    pub const INVALID_HANDLE: u32 = 0xE401;
    pub const TIMED_OUT: u32 = 0xEA01;
    pub const CANCELLED: u32 = 0xEC01;
    pub const INVALID_ENUM: u32 = 0xF001;
    pub const NOT_FOUND: u32 = 0xF201;
    pub const INVALID_STATE: u32 = 0xFA01;
}

pub const CURRENT_PROCESS: u32 = 0xFFFF_8001;
pub const CURRENT_THREAD: u32 = 0xFFFF_8000;
const HAS_WAITERS: u32 = 0x4000_0000;

pub struct Thread {
    pub handle: u32,
    pub id: u64,
    pub entry: u64,
    pub arg: u64,
    pub stack_top: u64,
    pub tlr: u64,
    pub prio: AtomicI32,
    pub core: AtomicI32,
    pub affinity: AtomicU64,
    active_core: AtomicI32,
    pub started: AtomicBool,
    pub exited: AtomicBool,
    pub cancel: AtomicBool,
    runnable: AtomicBool,
    ready_seq: AtomicU64,
    cv: Condvar,
    diagnostic_svc: AtomicU32,
    diagnostic_args: [AtomicU64; 31],
    diagnostic_fp: AtomicU64,
}

impl Thread {
    fn sched_core(&self) -> i32 {
        self.active_core.load(Relaxed)
    }
}

#[derive(Default)]
pub struct Event {
    pub signaled: AtomicBool,
}

pub type SharedMemory = crate::shm::Shm;

#[derive(Clone)]
pub enum Obj {
    Process,
    Thread(Arc<Thread>),
    ReadableEvent(Arc<Event>),
    WritableEvent(Arc<Event>),
    Session(Arc<crate::ipc::Session>),
    ResourceLimit,
    SharedMemory(Arc<SharedMemory>),
    TransferMemory {address:u64,size:u64,permission:u32},
}

struct Waiter {
    thread: Option<Arc<Thread>>,
    tag: u32,
    prio: i32,
    mutex: u64,
    woken: AtomicBool,
}

#[derive(Default)]
pub struct State {
    handles: HashMap<u32, Obj>,
    next_handle: u32,
    next_thread_id: u64,
    mutex_waiters: HashMap<u64, Vec<Arc<Waiter>>>,
    cv_waiters: HashMap<u64, Vec<Arc<Waiter>>>,
    addr_waiters: HashMap<u64, Vec<Arc<Waiter>>>,
    tlr_next: u64,
    threads: Vec<Arc<Thread>>,
    next_seq: u64,
    sync_waiters: Vec<(Arc<Thread>, Vec<usize>)>,
}

pub struct Kernel {
    st: Mutex<State>,
    cv: Condvar,
    pub aspace: OnceLock<AddressSpace>,
}

pub static KERNEL: LazyLock<Kernel> = LazyLock::new(|| Kernel {
    st: Mutex::new(State { next_handle: 0x10, next_thread_id: 0x50, ..Default::default() }),
    cv: Condvar::new(),
    aspace: OnceLock::new(),
});

pub fn aspace() -> &'static AddressSpace {
    KERNEL.aspace.get().expect("address space not initialized")
}

fn atomic32(addr: u64) -> &'static AtomicU32 {
    unsafe { AtomicU32::from_ptr(addr as *mut u32) }
}

impl Kernel {
    pub fn lock(&self) -> MutexGuard<'_, State> {
        self.st.lock().unwrap()
    }

    fn wake_all(&self, g: &State) {
        for t in &g.threads {
            t.cv.notify_one();
        }
        self.cv.notify_all();
    }

    fn wake_core(&self, g: &State, _core: i32) {
        // Affinity permits a ready thread to migrate when another core is
        // occupied. A change can therefore affect candidates on every core.
        self.wake_all(g);
    }

    fn wake(&self, w: &Waiter) {
        match &w.thread {
            Some(t) => t.cv.notify_one(),
            None => self.cv.notify_all(),
        }
    }

    fn wait_until<'a>(&self, g: MutexGuard<'a, State>, timeout_ns: i64, mut done: impl FnMut(&mut State) -> bool) -> (MutexGuard<'a, State>, bool) {
        let me = current_kthread(&g);
        let (mut g, ok) = self.wait_until_raw(g, timeout_ns, &mut done, me.as_deref());
        if let Some(t) = me {
            g.make_runnable(&t);
            self.wake_core(&g, t.sched_core());
            g = self.wait_scheduled(g, &t);
        }
        (g, ok)
    }

    fn wait_until_raw<'a>(
        &self,
        mut g: MutexGuard<'a, State>,
        timeout_ns: i64,
        done: &mut impl FnMut(&mut State) -> bool,
        me: Option<&Thread>,
    ) -> (MutexGuard<'a, State>, bool) {
        let deadline = (timeout_ns >= 0).then(|| Instant::now() + Duration::from_nanos(timeout_ns as u64));
        if done(&mut g) {
            return (g, true);
        }
        if let Some(t) = me {
            t.runnable.store(false, Relaxed);
            self.wake_core(&g, t.sched_core());
        }
        let cv = me.map_or(&self.cv, |t| &t.cv);
        loop {
            if done(&mut g) {
                return (g, true);
            }
            let now = Instant::now();
            if deadline.is_some_and(|d| now >= d) {
                return (g, false);
            }
            let wait = deadline.map_or(SAFETY_NET, |d| (d - now).min(SAFETY_NET));
            g = cv.wait_timeout(g, wait).unwrap().0;
        }
    }
}

const SAFETY_NET: Duration = Duration::from_millis(50);

// Priority-ordered bipartite matching of ready threads to virtual cores.
// Affinity is a set of permitted cores; it must not collapse DontCare threads
// onto core zero and starve lower-priority work while cores 1/2 are idle.
fn assign_cores(candidates: &[(u64,i32,i32)]) -> [Option<usize>;4] {
    fn place(i:usize, cs:&[(u64,i32,i32)], slots:&mut [Option<usize>;4], seen:&mut [bool;4]) -> bool {
        let (mask,ideal,previous) = cs[i];
        let order = [ideal,previous,0,1,2,3];
        for core in order {
            if !(0..4).contains(&core) { continue; }
            let c = core as usize;
            if mask & (1 << c) == 0 || seen[c] { continue; }
            seen[c] = true;
            if slots[c].is_none_or(|j| place(j,cs,slots,seen)) {
                slots[c] = Some(i); return true;
            }
        }
        false
    }
    let mut slots = [None;4];
    for i in 0..candidates.len() { place(i,candidates,&mut slots,&mut [false;4]); }
    slots
}

fn sched_enabled() -> bool {
    static E: OnceLock<bool> = OnceLock::new();
    *E.get_or_init(|| std::env::var_os("NX_NO_SCHED").is_none())
}

impl State {
    fn make_runnable(&mut self, t: &Thread) {
        self.next_seq += 1;
        t.ready_seq.store(self.next_seq, Relaxed);
        t.runnable.store(true, Relaxed);
    }

    fn is_scheduled(&self, t: &Thread) -> bool {
        let key = |x: &Thread| (x.prio.load(Relaxed), x.ready_seq.load(Relaxed));
        let mut ready:Vec<_> = self.threads.iter().filter(|o| o.runnable.load(Relaxed)).collect();
        // Tests may install a kernel object without starting a guest entry.
        if !ready.iter().any(|o| std::ptr::eq(o.as_ref(),t)) { return true; }
        ready.sort_by_key(|o| key(o));
        let candidates:Vec<_> = ready.iter().map(|o| (o.affinity.load(Relaxed),o.core.load(Relaxed),o.active_core.load(Relaxed))).collect();
        let assigned = assign_cores(&candidates);
        let mut scheduled = false;
        for (core,index) in assigned.iter().enumerate() {
            if let Some(i) = index {
                ready[*i].active_core.store(core as i32,Relaxed);
                if std::ptr::eq(ready[*i].as_ref(),t) { scheduled = true; }
                ready[*i].cv.notify_one();
            }
        }
        scheduled
    }
}

impl Kernel {
    fn wait_scheduled<'a>(&self, mut g: MutexGuard<'a, State>, t: &Thread) -> MutexGuard<'a, State> {
        if !sched_enabled() {
            return g;
        }
        while !g.is_scheduled(t) {
            let r = t.cv.wait_timeout(g, SAFETY_NET).unwrap();
            g = r.0;
        }
        g
    }
}

thread_local! {
    static KTHREAD: std::cell::RefCell<Option<Arc<Thread>>> = const { std::cell::RefCell::new(None) };
}

fn current_kthread(g: &State) -> Option<Arc<Thread>> {
    if let Some(t) = KTHREAD.with(|k| k.borrow().clone()) {
        return Some(t);
    }
    let h = guest::try_current()?.handle;
    match g.handles.get(&h) {
        Some(Obj::Thread(t)) => {
            KTHREAD.with(|k| *k.borrow_mut() = Some(t.clone()));
            Some(t.clone())
        }
        _ => None,
    }
}

pub fn sched_point() {
    let g = KERNEL.lock();
    if let Some(t) = current_kthread(&g) {
        let _g = KERNEL.wait_scheduled(g, &t);
    }
}

pub fn blocking<R>(f: impl FnOnce() -> R) -> R {
    let me = {
        let g = KERNEL.lock();
        let me = current_kthread(&g);
        if let Some(t) = &me {
            t.runnable.store(false, Relaxed);
            KERNEL.wake_core(&g, t.sched_core());
        }
        me
    };
    let r = f();
    if let Some(t) = me {
        let mut g = KERNEL.lock();
        g.make_runnable(&t);
        KERNEL.wake_core(&g, t.sched_core());
        drop(KERNEL.wait_scheduled(g, &t));
    }
    r
}

pub fn yield_thread() {
    let mut g = KERNEL.lock();
    if let Some(t) = current_kthread(&g) {
        g.make_runnable(&t);
        KERNEL.wake_core(&g, t.sched_core());
        let _g = KERNEL.wait_scheduled(g, &t);
    }
}

impl State {
    pub fn add(&mut self, o: Obj) -> u32 {
        let h = self.next_handle;
        self.next_handle += 1;
        self.handles.insert(h, o);
        h
    }

    pub fn get(&self, h: u32) -> Option<Obj> {
        match h {
            CURRENT_PROCESS => Some(Obj::Process),
            CURRENT_THREAD => self.handles.get(&guest::current().handle).cloned(),
            _ => self.handles.get(&h).cloned(),
        }
    }

    pub fn close(&mut self, h: u32) -> Option<Obj> {
        self.handles.remove(&h)
    }

    fn signaled(&self, h: u32) -> Option<bool> {
        Some(match self.get(h)? {
            Obj::ReadableEvent(e) | Obj::WritableEvent(e) => e.signaled.load(Relaxed),
            Obj::Thread(t) => t.exited.load(Relaxed),
            Obj::Session(_) => false,
            _ => false,
        })
    }

    fn current_thread(&self) -> Arc<Thread> {
        match self.get(CURRENT_THREAD) {
            Some(Obj::Thread(t)) => t,
            _ => panic!("current thread has no kernel object"),
        }
    }

    fn alloc_tlr(&mut self) -> Result<u64> {
        let a = aspace();
        let base = a.base + mem::TLR_OFF;
        let off = self.tlr_next;
        anyhow::ensure!(off+0x200<=mem::PROCESS_LOCAL_OFF-mem::TLR_OFF,"thread-local arena exhausted");
        self.tlr_next += 0x200;
        if off % mem::PAGE == 0 {
            a.set(base + off, mem::PAGE, Block { state: state::THREAD_LOCAL, perm: perm::RW, attr: 0 })?;
        }
        Ok(base + off)
    }

    fn unlock_mutex(&mut self, addr: u64) {
        let list = self.mutex_waiters.entry(addr).or_default();
        if list.is_empty() {
            atomic32(addr).store(0, Release);
            mlog(addr, "unlock(no waiters)", guest::current().handle, 0);
            return;
        }
        let i = best(list);
        let w = list.remove(i);
        let v = w.tag | if list.is_empty() { 0 } else { HAS_WAITERS };
        atomic32(addr).store(v, Release);
        w.woken.store(true, Relaxed);
        KERNEL.wake(&w);
        mlog(addr, "unlock(hand-off)", w.tag, v);
    }
}

fn best(list: &[Arc<Waiter>]) -> usize {
    let mut b = 0;
    for (i, w) in list.iter().enumerate() {
        if w.prio < list[b].prio {
            b = i;
        }
    }
    b
}

fn remove(map: &mut HashMap<u64, Vec<Arc<Waiter>>>, key: u64, w: &Arc<Waiter>) -> bool {
    if let Some(l) = map.get_mut(&key) {
        if let Some(i) = l.iter().position(|x| Arc::ptr_eq(x, w)) {
            l.remove(i);
            return true;
        }
    }
    false
}

pub fn create_thread(entry: u64, arg: u64, stack_top: u64, prio: i32, core: i32) -> Result<u32> {
    let core = if core == -2 { process_core() } else { core };
    let allowed = crate::svc::THREAD_PERMISSIONS.get().map_or(0xf, |p| p.0);
    anyhow::ensure!((0..4).contains(&core) && allowed & (1 << core) != 0, "invalid thread core {core}");
    let mut g = KERNEL.lock();
    let tlr = g.alloc_tlr()?;
    let id = g.next_thread_id;
    g.next_thread_id += 1;
    let handle = g.next_handle;
    // SDK 22 InternalCriticalSection reads the kernel-owned current handle
    // directly from TLR+0x110. Leaving it zero makes an unlocked mutex appear
    // recursively owned. This field exists before any user startup code runs.
    unsafe {((tlr+0x110) as *mut u32).write(handle);}
    let t = Arc::new(Thread {
        handle,
        id,
        entry,
        arg,
        stack_top,
        tlr,
        prio: AtomicI32::new(prio),
        core: AtomicI32::new(core),
        affinity: AtomicU64::new(1 << core),
        active_core: AtomicI32::new(core),
        started: AtomicBool::new(false),
        exited: AtomicBool::new(false),
        cancel: AtomicBool::new(false),
        runnable: AtomicBool::new(false),
        ready_seq: AtomicU64::new(0),
        cv: Condvar::new(),
        diagnostic_svc: AtomicU32::new(0),
        diagnostic_args: std::array::from_fn(|_| AtomicU64::new(0)),
        diagnostic_fp: AtomicU64::new(0),
    });
    let h = g.add(Obj::Thread(t));
    debug_assert_eq!(h, handle);
    if std::env::var_os("NX_TRACE_THREADS").is_some() {
        eprintln!("[thread] create {h:#x}: prio {prio} core {core} entry {}", crate::diag::symbolize(entry));
    }
    Ok(h)
}

pub fn thread(h: u32) -> Option<Arc<Thread>> {
    match KERNEL.lock().get(h)? {
        Obj::Thread(t) => Some(t),
        _ => None,
    }
}

pub fn process_core() -> i32 { *crate::svc::DEFAULT_CORE.get().unwrap_or(&0) }

fn normalize_core_mask(core: i32, mask: u64, old_core: i32, default: i32, allowed: u64) -> Result<(i32,u64),u32> {
    if core == -2 { return Ok((default, 1 << default)); }
    if mask & !allowed != 0 || !(-3..4).contains(&core) { return Err(0xe201); }
    if mask == 0 { return Err(0xe801); }
    let core = if core == -3 { old_core } else { core };
    if core >= 0 && mask & (1 << core) == 0 { return Err(0xe801); }
    Ok((core, mask))
}

pub fn set_core_mask(t: &Thread, core: i32, mask: u64) -> u32 {
    let g = KERNEL.lock();
    let allowed = crate::svc::THREAD_PERMISSIONS.get().map_or(0xf, |p| p.0);
    let (core, mask) = match normalize_core_mask(core,mask,t.core.load(Relaxed),process_core(),allowed) {
        Ok(v) => v, Err(rc) => return rc,
    };
    t.core.store(core,Relaxed); t.affinity.store(mask,Relaxed);
    KERNEL.wake_all(&g);
    rc::OK
}

pub fn processor_number() -> u64 {
    let g = KERNEL.lock();
    current_kthread(&g).map_or(process_core() as u64, |t| t.sched_core() as u64)
}

#[cfg(test)]
mod sdk22_thread_abi_tests {
    use super::*;
    #[test]
    fn affinity_distributes_work_without_losing_priority_or_pinned_cores() {
        let assigned = assign_cores(&[(7,-1,0),(7,-1,0),(7,-1,0),(7,-1,0)]);
        let mut chosen:Vec<_> = assigned.into_iter().flatten().collect(); chosen.sort();
        assert_eq!(chosen,vec![0,1,2]);
        let assigned = assign_cores(&[(7,-1,0),(1,0,0),(4,2,2),(2,1,1)]);
        assert_eq!(assigned,[Some(1),Some(0),Some(2),None]);
        let assigned = assign_cores(&[(1,0,0),(1,0,0),(4,2,2)]);
        assert_eq!(assigned,[Some(0),None,Some(2),None]);
        let assigned = assign_cores(&[(2,1,1),(4,2,2)]);
        assert_eq!(assigned,[None,Some(0),Some(1),None]);
    }
    #[test]
    fn default_core_and_affinity_sentinels_have_kernel_semantics() {
        assert_eq!(normalize_core_mask(-2,u64::MAX,0,2,7),Ok((2,4)));
        assert_eq!(normalize_core_mask(-3,6,1,0,7),Ok((1,6)));
        assert_eq!(normalize_core_mask(-1,4,0,0,7),Ok((-1,4)));
        assert!(normalize_core_mask(-3,4,1,0,7).is_err());
        assert!(normalize_core_mask(2,2,0,0,7).is_err());
        assert!(normalize_core_mask(0,8,0,0,7).is_err());
        assert!(normalize_core_mask(-1,0,0,0,7).is_err());
    }
    #[test]
    fn condition_wait_marker_is_visible_to_producer_after_unlock() {
        KERNEL.aspace.get_or_init(|| mem::AddressSpace::reserve().unwrap());
        guest::init_tsd();
        let waiter = create_thread(0,0,0,44,0).unwrap();
        let producer = create_thread(0,0,0,44,0).unwrap();
        let words = Arc::new([AtomicU32::new(0),AtomicU32::new(0)]);
        let epoch = Arc::new(AtomicU32::new(0));
        let seen = Arc::new(AtomicBool::new(true));
        let pw = words.clone(); let pe = epoch.clone(); let ps = seen.clone();
        let worker = std::thread::spawn(move || {
            let t = thread(producer).unwrap(); let mut ctx = GuestThread::new(producer,t.tlr); ctx.install();
            let (mutex,key) = (&pw[0] as *const _ as u64,&pw[1] as *const _ as u64);
            for i in 1..=256 {
                while pe.load(Acquire) < i { std::thread::yield_now(); }
                while pw[0].compare_exchange(0,producer,AcqRel,Acquire).is_err() { std::thread::yield_now(); }
                let has_waiter = pw[1].load(Acquire) != 0;
                ps.fetch_and(has_waiter,Relaxed);
                if has_waiter { assert_eq!(signal_process_wide_key(key,1),rc::OK); }
                assert_eq!(arbitrate_unlock(mutex),rc::OK);
            }
        });
        let ww = words.clone(); let we = epoch.clone();
        let consumer = std::thread::spawn(move || {
            let t = thread(waiter).unwrap(); let mut ctx = GuestThread::new(waiter,t.tlr); ctx.install();
            let (mutex,key) = (&ww[0] as *const _ as u64,&ww[1] as *const _ as u64);
            let mut results = Vec::new();
            for i in 1..=256 {
                ww[0].store(waiter,Release); we.store(i,Release);
                let result = wait_process_wide_key(mutex,key,waiter,50_000_000);
                results.push(result);
                if result == rc::OK { assert_eq!(ww[0].load(Acquire),waiter); arbitrate_unlock(mutex); }
            }
            results
        });
        worker.join().unwrap();
        let results = consumer.join().unwrap();
        assert!(seen.load(Relaxed),"producer observed unlocked mutex without published waiter flag");
        assert!(results.iter().all(|r| *r == rc::OK),"missed wakeup: {results:x?}");
        for h in [waiter,producer] { let removed = {KERNEL.lock().close(h)}; drop(removed); }
    }
    #[test]
    fn address_modify_signal_preserves_counter_until_last_waiter_is_released() {
        KERNEL.aspace.get_or_init(||mem::AddressSpace::reserve().unwrap());
        guest::init_tsd();
        let word=Arc::new(AtomicU32::new(7));let address=Arc::as_ptr(&word)as u64;
        let handles=[create_thread(0,0,0,44,0).unwrap(),create_thread(0,0,0,44,0).unwrap()];
        let (send,receive)=std::sync::mpsc::channel();let mut workers=Vec::new();
        for handle in handles {
            let word=word.clone();let send=send.clone();
            workers.push(std::thread::spawn(move||{
                let thread=thread(handle).unwrap();let mut context=GuestThread::new(handle,thread.tlr);context.install();
                let result=wait_for_address(Arc::as_ptr(&word)as u64,2,7,5_000_000_000);
                send.send(result).unwrap();
            }));
        }
        let deadline=Instant::now()+Duration::from_secs(2);
        while KERNEL.lock().addr_waiters.get(&address).map_or(0,Vec::len)!=2 {
            assert!(Instant::now()<deadline,"consumers did not enter address wait");std::thread::yield_now();
        }
        let first=signal_to_address(address,2,7,1);
        let after_first=word.load(Acquire);
        let remaining=KERNEL.lock().addr_waiters.get(&address).map_or(0,Vec::len);
        let first_result=receive.recv_timeout(Duration::from_secs(1)).unwrap();
        // Always release the second real waiter before checking observations,
        // including when this test runs against the defective implementation.
        let broadcast=signal_to_address(address,2,after_first as i32,0);
        let after_broadcast=word.load(Acquire);
        let second_result=receive.recv_timeout(Duration::from_secs(1)).unwrap();
        for worker in workers {worker.join().unwrap();}
        for handle in handles {let removed={KERNEL.lock().close(handle)};drop(removed);}
        assert_eq!((first,broadcast,first_result,second_result),(rc::OK,rc::OK,rc::OK,rc::OK));
        assert_eq!(remaining,1);assert_eq!(after_first,7,"one waiter still needs the unchanged shared marker");
        assert_eq!(after_broadcast,6,"releasing all remaining waiters decrements exactly once");
        assert_eq!(signal_to_address(address,2,6,1),rc::OK);assert_eq!(word.load(Acquire),7);
        assert_eq!(signal_to_address(address,2,99,-1),rc::INVALID_STATE);assert_eq!(word.load(Acquire),7);
    }
    #[test]
    fn address_wait_decrement_is_atomic_against_userspace_fast_path() {
        const EPOCHS:usize=2048;
        let word=Arc::new(AtomicU32::new(0));let result=Arc::new(AtomicU32::new(0));
        let gate=Arc::new(std::sync::Barrier::new(3));
        let producer={let word=word.clone();let gate=gate.clone();std::thread::spawn(move||{
            for _ in 0..EPOCHS {gate.wait();let _=word.compare_exchange(0,1,AcqRel,Acquire);gate.wait();}
        })};
        let consumer={let word=word.clone();let gate=gate.clone();let result=result.clone();std::thread::spawn(move||{
            for _ in 0..EPOCHS {gate.wait();result.store(wait_for_address(Arc::as_ptr(&word)as u64,1,1,0),Release);gate.wait();}
        })};
        let mut invalid=0;
        for _ in 0..EPOCHS {
            word.store(0,Release);gate.wait();gate.wait();
            let value=word.load(Acquire);let status=result.load(Acquire);
            if !((status==rc::INVALID_STATE && value==1) || (status==rc::TIMED_OUT && value==u32::MAX)) {invalid+=1;}
        }
        producer.join().unwrap();consumer.join().unwrap();
        assert_eq!(invalid,0,"conditional decrement raced past a successful userspace release");
    }
    #[test]
    fn fresh_threads_have_distinct_nonzero_kernel_tls_handles() {
        KERNEL.aspace.get_or_init(|| mem::AddressSpace::reserve().unwrap());
        let a = create_thread(0,0,0,44,0).unwrap();
        let b = create_thread(0,0,0,44,1).unwrap();
        assert_ne!(a,b);
        for h in [a,b] {
            let t = thread(h).unwrap();
            let handle = unsafe {*((t.tlr+0x110) as *const u32)};
            // The SDK compares this word with the mutex owner, zero when unlocked.
            assert_eq!(handle,h); assert_ne!(handle,0);
            let removed = {KERNEL.lock().close(h)}; drop(removed);
        }
        assert_eq!((CURRENT_PROCESS,CURRENT_THREAD),(0xffff8001,0xffff8000));
    }
}

pub fn start_thread(t: Arc<Thread>, args: [u64; 4]) -> std::io::Result<std::thread::JoinHandle<()>> {
    let mut registers = [0; 8];
    registers[..4].copy_from_slice(&args);
    start_thread_with_args(t, registers)
}

pub fn start_thread_with_args(t: Arc<Thread>, args: [u64; 8]) -> std::io::Result<std::thread::JoinHandle<()>> {
    t.started.store(true, Relaxed);
    {
        let mut g = KERNEL.lock();
        g.threads.push(t.clone());
        g.make_runnable(&t);
        KERNEL.wake_core(&g, t.sched_core());
    }
    std::thread::Builder::new().name(format!("guest-{:#x}", t.id)).stack_size(1 << 20).spawn(move || {
        crate::diag::install_altstack();
        let mut ctx = GuestThread::new(t.handle, t.tlr);
        ctx.install();
        let entry = Entry { pc: t.entry, sp: t.stack_top & !15, x: args };
        drop(KERNEL.wait_scheduled(KERNEL.lock(), &t));
        unsafe { guest::nx_enter_guest(&entry, &mut *ctx) };
        let mut g = KERNEL.lock();
        t.exited.store(true, Relaxed);
        t.runnable.store(false, Relaxed);
        g.threads.retain(|x| !Arc::ptr_eq(x, &t));
        let id = Arc::as_ptr(&t) as usize;
        for (w, objs) in &g.sync_waiters {
            if objs.contains(&id) {
                w.cv.notify_one();
            }
        }
        KERNEL.wake_core(&g, t.sched_core());
    })
}

pub fn create_event() -> (u32, u32) {
    let e = Arc::new(Event::default());
    let mut g = KERNEL.lock();
    (g.add(Obj::WritableEvent(e.clone())), g.add(Obj::ReadableEvent(e)))
}

pub fn signal_event(e: &Event) {
    let g = KERNEL.lock();
    e.signaled.store(true, Relaxed);
    let id = std::ptr::from_ref(e) as usize;
    for (t, objs) in &g.sync_waiters {
        if objs.contains(&id) {
            t.cv.notify_one();
        }
    }
    KERNEL.cv.notify_all();
}

pub fn wait_synchronization(handles: &[u32], timeout_ns: i64) -> (u32, u32) {
    let mut index=0;
    let result=observe_long_wait("synchronization",handles.first().copied().unwrap_or(0)as u64,timeout_ns,||{
        let (result,selected)=wait_synchronization_impl(handles,timeout_ns);index=selected;result
    });
    (result,index)
}

fn wait_synchronization_impl(handles: &[u32], timeout_ns: i64) -> (u32, u32) {
    let mut g = KERNEL.lock();
    let mut objs = Vec::with_capacity(handles.len());
    for &h in handles {
        match g.get(h) {
            None => return (rc::INVALID_HANDLE, 0),
            Some(Obj::ReadableEvent(e) | Obj::WritableEvent(e)) => objs.push(Arc::as_ptr(&e) as usize),
            Some(Obj::Thread(t)) => objs.push(Arc::as_ptr(&t) as usize),
            Some(_) => {}
        }
    }
    let me = g.current_thread();
    let mut idx = None;
    let mut cancelled = false;
    g.sync_waiters.push((me.clone(), objs));
    let (mut g, ok) = KERNEL.wait_until(g, timeout_ns, |s| {
        if me.cancel.swap(false, Relaxed) {
            cancelled = true;
            return true;
        }
        idx = handles.iter().position(|&h| s.signaled(h) == Some(true));
        idx.is_some()
    });
    if let Some(i) = g.sync_waiters.iter().position(|(t, _)| Arc::ptr_eq(t, &me)) {
        g.sync_waiters.swap_remove(i);
    }
    if cancelled {
        (rc::CANCELLED, 0)
    } else if !ok {
        (rc::TIMED_OUT, 0)
    } else {
        (rc::OK, idx.unwrap() as u32)
    }
}

pub fn cancel_synchronization(t: &Thread) {
    let _g = KERNEL.lock();
    t.cancel.store(true, Relaxed);
    t.cv.notify_one();
}

static MUTEX_LOG: Mutex<Vec<(u64, &'static str, u32, u32)>> = Mutex::new(Vec::new());

fn mlog(addr: u64, op: &'static str, thread: u32, word: u32) {
    static ON: OnceLock<bool> = OnceLock::new();
    if !*ON.get_or_init(|| std::env::var_os("NX_MUTEX_LOG").is_some()) {
        return;
    }
    let mut l = MUTEX_LOG.lock().unwrap();
    if l.len() > 20000 {
        l.drain(..10000);
    }
    l.push((addr, op, thread, word));
}

fn dump_mutex_log(addr: u64) {
    let l = MUTEX_LOG.lock().unwrap();
    let ev: Vec<_> = l.iter().filter(|e| e.0 == addr).collect();
    eprintln!("[kernel] history of mutex {addr:#x} (last {}):", ev.len().min(30));
    for (_, op, t, w) in ev.iter().rev().take(30).rev() {
        eprintln!("[kernel]   {op:24} thread {t:#x} word after {w:#x}");
    }
}

pub fn arbitrate_lock(owner: u32, addr: u64, tag: u32) -> u32 {
    let mut g = KERNEL.lock();
    if atomic32(addr).load(Acquire) != owner | HAS_WAITERS {
        return rc::OK;
    }
    let me = g.current_thread();
    if owner == tag {
        eprintln!("[kernel] thread {tag:#x} waits for mutex {addr:#x} that it owns");
        dump_mutex_log(addr);
        let fp: u64 = crate::svc::current_guest_fp();
        crate::diag::backtrace(fp);
    }
    mlog(addr, "ArbitrateLock(wait)", tag, atomic32(addr).load(Relaxed));
    let w = Arc::new(Waiter { thread: Some(me.clone()), tag, prio: me.prio.load(Relaxed), mutex: addr, woken: AtomicBool::new(false) });
    g.mutex_waiters.entry(addr).or_default().push(w.clone());
    let _ = KERNEL.wait_until(g, -1, |_| w.woken.load(Relaxed));
    mlog(addr, "ArbitrateLock(acquired)", tag, atomic32(addr).load(Relaxed));
    rc::OK
}

pub fn arbitrate_unlock(addr: u64) -> u32 {
    let mut g = KERNEL.lock();
    g.unlock_mutex(addr);
    rc::OK
}

static MENU_WAIT_ORIGIN:LazyLock<Instant>=LazyLock::new(Instant::now);
static MENU_WAIT_UNTIL_MS:AtomicU64=AtomicU64::new(0);
static MENU_WAIT_EVENTS:AtomicU64=AtomicU64::new(0);
static MENU_WAIT_STACKS:LazyLock<Mutex<HashMap<(u32,String),(u32,u32)>>>=LazyLock::new(Default::default);
pub(crate) fn arm_menu_wait_observation(milliseconds:u64)->bool {
    if !(1000..=30000).contains(&milliseconds) || !std::env::var("NX_FC27_VIDEO_TRACE").is_ok_and(|v|v=="1") || std::env::var_os("NX_TEST_CONTROL").is_none() {return false;}
    MENU_WAIT_STACKS.lock().unwrap().clear();MENU_WAIT_EVENTS.store(0,Relaxed);
    MENU_WAIT_UNTIL_MS.store(MENU_WAIT_ORIGIN.elapsed().as_millis()as u64+milliseconds,Release);true
}
fn menu_wait_active()->bool {
    let until=MENU_WAIT_UNTIL_MS.load(Acquire);until!=0 && (MENU_WAIT_ORIGIN.elapsed().as_millis()as u64)<until
}
fn observe_menu_wait(kind:&str,key:u64,timeout_ns:i64,result:u32,start:Instant) {
    if !menu_wait_active() || start.elapsed()<Duration::from_millis(30) {return;}
    let thread=guest::current().handle;
    // Periodic30/50ms sleepers must not consume the whole15s window.
    // Retain8 ordinary and8 overdue/long samples for each of32 thread/kinds.
    let (emit,stack)={
        let mut seen=MENU_WAIT_STACKS.lock().unwrap();let key=(thread,kind.into());
        if !seen.contains_key(&key) && seen.len()>=32 {return;}
        let counts=seen.entry(key).or_default();let elapsed=start.elapsed();
        let overdue=if timeout_ns<0 {elapsed>=Duration::from_secs(1)}else{elapsed>=Duration::from_nanos(timeout_ns as u64).saturating_add(Duration::from_millis(30))};
        let stack=counts.0==0 && counts.1==0;let count=if overdue{&mut counts.1}else{&mut counts.0};
        let emit=*count<8;*count=count.saturating_add(1);(emit,stack)
    };
    if !emit {return;}let n=MENU_WAIT_EVENTS.fetch_add(1,Relaxed)+1;if n>512 {return;}
    eprintln!("[menu-thread-wait] event={n} kind={kind} key={key:#x} thread={thread:#x} timeout_ns={timeout_ns} result={result:#x} elapsed_ms={:.3} t_ms={}",start.elapsed().as_secs_f64()*1000.,MENU_WAIT_ORIGIN.elapsed().as_millis());
    if stack {for line in crate::diag::backtrace_lines(crate::svc::current_guest_fp()).into_iter().take(8) {eprintln!("[menu-thread-wait] thread={thread:#x} at {line}");}}
}
fn observe_long_wait(kind:&str,key:u64,timeout_ns:i64,call:impl FnOnce()->u32)->u32 {
    static TRACE:LazyLock<bool>=LazyLock::new(||std::env::var("NX_FC27_VIDEO_TRACE").is_ok_and(|v|v=="1") && std::env::var_os("NX_TEST_CONTROL").is_some());
    if !*TRACE || (timeout_ns>=0 && timeout_ns<1_000_000_000 && !menu_wait_active()) {return call();}
    static START:OnceLock<Instant>=OnceLock::new();let origin=START.get_or_init(Instant::now);
    let start=Instant::now();let result=call();let seconds=start.elapsed().as_secs_f64();
    observe_menu_wait(kind,key,timeout_ns,result,start);
    if seconds>=0.5 {
        static EVENTS:AtomicU64=AtomicU64::new(0);let n=EVENTS.fetch_add(1,Relaxed)+1;
        if n<=128 || (n<=4096 && n.is_power_of_two()) {
            eprintln!("[slow-thread-wait] event={n} kind={kind} key={key:#x} thread={:#x} timeout_ns={timeout_ns} result={result:#x} elapsed_ms={:.3} t_ms={:.3}",guest::current().handle,seconds*1000.,origin.elapsed().as_secs_f64()*1000.);
            if n<=16 {for line in crate::diag::backtrace_lines(crate::svc::current_guest_fp()).into_iter().take(8) {eprintln!("[slow-thread-wait] at {line}");}}
        }
    }
    result
}

pub fn wait_process_wide_key(mutex: u64, key: u64, tag: u32, timeout_ns: i64) -> u32 {
    observe_long_wait("condition",key,timeout_ns,||wait_process_wide_key_impl(mutex,key,tag,timeout_ns))
}

fn wait_process_wide_key_impl(mutex: u64, key: u64, tag: u32, timeout_ns: i64) -> u32 {
    let mut g = KERNEL.lock();
    let me = g.current_thread();
    mlog(mutex, "cv wait: release", tag, atomic32(mutex).load(Relaxed));
    let w = Arc::new(Waiter { thread: Some(me.clone()), tag, prio: me.prio.load(Relaxed), mutex, woken: AtomicBool::new(false) });
    g.cv_waiters.entry(key).or_default().push(w.clone());
    // The producer tests this word in user space after taking the mutex.
    // Publishing it after unlock can lose a signal even with State locked.
    atomic32(key).store(1, Release);
    g.unlock_mutex(mutex);
    let (mut g, ok) = KERNEL.wait_until(g, timeout_ns, |_| w.woken.load(Relaxed));
    if ok {
        return rc::OK;
    }
    if remove(&mut g.cv_waiters, key, &w) {
        if g.cv_waiters.get(&key).is_none_or(|l| l.is_empty()) {
            atomic32(key).store(0, Release);
        }
        mlog(mutex, "cv wait: timed out", tag, atomic32(mutex).load(Relaxed));
        return rc::TIMED_OUT;
    }
    let _ = KERNEL.wait_until(g, -1, |_| w.woken.load(Relaxed));
    rc::OK
}

fn acquire_or_enqueue(g: &mut State, w: &Arc<Waiter>) -> bool {
    let m = atomic32(w.mutex);
    loop {
        let v = m.load(Acquire);
        if v == 0 {
            if m.compare_exchange(0, w.tag, AcqRel, Acquire).is_ok() {
                w.woken.store(true, Relaxed);
                KERNEL.wake(w);
                mlog(w.mutex, "cv reacquire(free)", w.tag, w.tag);
                return true;
            }
        } else if v & HAS_WAITERS != 0 || m.compare_exchange(v, v | HAS_WAITERS, AcqRel, Acquire).is_ok() {
            g.mutex_waiters.entry(w.mutex).or_default().push(w.clone());
            mlog(w.mutex, "cv reacquire(queued)", w.tag, m.load(Relaxed));
            return false;
        }
    }
}

pub fn signal_process_wide_key(key: u64, count: i32) -> u32 {
    let mut g = KERNEL.lock();
    let mut n = 0;
    while count <= 0 || n < count {
        let Some(list) = g.cv_waiters.get_mut(&key).filter(|l| !l.is_empty()) else { break };
        let w = list.remove(best(list));
        acquire_or_enqueue(&mut g, &w);
        n += 1;
    }
    if g.cv_waiters.get(&key).is_none_or(|l| l.is_empty()) {
        atomic32(key).store(0, Release);
    }
    rc::OK
}

pub fn wait_for_address(addr: u64, kind: u32, value: i32, timeout_ns: i64) -> u32 {
    observe_long_wait("address",addr,timeout_ns,||wait_for_address_impl(addr,kind,value,timeout_ns))
}

fn wait_for_address_impl(addr: u64, kind: u32, value: i32, timeout_ns: i64) -> u32 {
    let mut g = KERNEL.lock();
    let a = atomic32(addr);
    let cur = a.load(Acquire) as i32;
    match kind {
        0 if cur < value => {}
        1 => {
            let mut observed=cur;
            loop {
                if observed>=value {return rc::INVALID_STATE;}
                match a.compare_exchange_weak(observed as u32,observed.wrapping_sub(1)as u32,AcqRel,Acquire) {
                    Ok(_)=>break,
                    Err(actual)=>observed=actual as i32,
                }
            }
        }
        2 if cur == value => {}
        0..=2 => return rc::INVALID_STATE,
        _ => return rc::INVALID_ENUM,
    }
    if timeout_ns == 0 {
        return rc::TIMED_OUT;
    }
    let me = g.current_thread();
    let w = Arc::new(Waiter { thread: Some(me.clone()), tag: 0, prio: me.prio.load(Relaxed), mutex: 0, woken: AtomicBool::new(false) });
    g.addr_waiters.entry(addr).or_default().push(w.clone());
    let (mut g, ok) = KERNEL.wait_until(g, timeout_ns, |_| w.woken.load(Relaxed));
    if ok {
        rc::OK
    } else {
        remove(&mut g.addr_waiters, addr, &w);
        rc::TIMED_OUT
    }
}

pub fn signal_to_address(addr: u64, kind: u32, value: i32, count: i32) -> u32 {
    let mut g = KERNEL.lock();
    let a = atomic32(addr);
    let waiting = g.addr_waiters.get(&addr).map_or(0, |l| l.len()) as i32;
    match kind {
        0 => {}
        1 => {
            if a.compare_exchange(value as u32, value.wrapping_add(1) as u32, AcqRel, Acquire).is_err() {
                return rc::INVALID_STATE;
            }
        }
        2 => {
            let new = if waiting == 0 {
                value.wrapping_add(1)
            } else if count <= 0 || waiting <= count {
                value.wrapping_sub(1)
            } else {
                value
            };
            let matched=if new==value {a.load(Acquire)==value as u32}
                else{a.compare_exchange(value as u32,new as u32,AcqRel,Acquire).is_ok()};
            if !matched {
                return rc::INVALID_STATE;
            }
            static TRACE:LazyLock<bool>=LazyLock::new(||std::env::var("NX_FC27_VIDEO_TRACE").is_ok_and(|v|v=="1") && std::env::var_os("NX_TEST_CONTROL").is_some());
            if *TRACE {
                static EVENTS:AtomicU64=AtomicU64::new(0);let n=EVENTS.fetch_add(1,Relaxed)+1;
                if n<=128 || (n<=32768 && n.is_power_of_two()) {eprintln!("[address-signal] event={n} address={addr:#x} waiting={waiting} count={count} before={value} after={new}");}
            }
        }
        _ => return rc::INVALID_ENUM,
    }
    let list = g.addr_waiters.entry(addr).or_default();
    let mut n = 0;
    while !list.is_empty() && (count <= 0 || n < count) {
        let w = list.remove(best(list));
        w.woken.store(true, Relaxed);
        KERNEL.wake(&w);
        n += 1;
    }
    rc::OK
}

pub fn notify_all() {
    let g = KERNEL.lock();
    KERNEL.wake_all(&g);
}

pub fn sleep(ns: i64) {
    if ns <= 0 {
        yield_thread();
        return;
    }
    let _=observe_long_wait("sleep",0,ns,||{
        let g=KERNEL.lock();let _=KERNEL.wait_until(g,ns,|_|false);rc::OK
    });
}

pub fn describe_current_thread() {
    let h = crate::guest::try_current().map(|c| c.handle);
    if let Some(h) = h {
        if let Ok(st) = KERNEL.st.try_lock() {
            if let Some(Obj::Thread(t)) = st.handles.get(&h) {
                eprintln!("        guest thread {h:#x} (id {:#x}) prio {} core {} entry {}", t.id, t.prio.load(Relaxed), t.core.load(Relaxed), crate::diag::symbolize(t.entry));
            }
        }
    }
}

pub fn record_svc(f: &crate::guest::Frame) {
    static ON: OnceLock<bool> = OnceLock::new();
    if !*ON.get_or_init(|| std::env::var_os("NX_WAIT_SNAPSHOTS").is_some()) { return; }
    KTHREAD.with(|k| {
        if let Some(t) = k.borrow().as_ref() {
            for (out, &value) in t.diagnostic_args.iter().zip(&f.x) { out.store(value, Relaxed); }
            t.diagnostic_fp.store(f.x[29], Relaxed);
            t.diagnostic_svc.store(f.x[16] as u32, Release);
        }
    });
}

pub fn start_wait_diagnostics() {
    let Ok(value) = std::env::var("NX_WAIT_SNAPSHOTS") else { return; };
    let mut times: Vec<u64> = value.split(',').filter_map(|s| s.parse().ok()).filter(|s| *s <= 600).take(8).collect();
    times.sort();
    std::thread::spawn(move || {
        let start = Instant::now();
        for sec in times {
            std::thread::sleep(Duration::from_secs(sec).saturating_sub(start.elapsed()));
            let g = KERNEL.lock();
            eprintln!("[wait] snapshot {sec}s, {} threads", g.threads.len());
            for t in &g.threads {
                let svc = t.diagnostic_svc.load(Acquire);
                let args = t.diagnostic_args.each_ref().map(|a| a.load(Relaxed));
                eprintln!("[wait] thread {:#x} id {:#x}, prio {} core {} runnable {} svc {svc:#x} args {:x?}", t.handle, t.id, t.prio.load(Relaxed), t.core.load(Relaxed), t.runnable.load(Relaxed), &args[..4]);
                if !t.runnable.load(Relaxed) {
                    eprintln!("[wait] x19..x29 {:x?}", &args[19..30]);
                    crate::diag::backtrace(t.diagnostic_fp.load(Relaxed));
                    for (i, &p) in args[19..29].iter().enumerate() {
                        if p % 8 == 0 && aspace().readable(p,0x100) {
                            let Some(words) = crate::diag::read_words::<32>(p) else { continue };
                            eprintln!("[wait] x{} {p:#x} memory {words:x?}", i+19);
                        }
                    }
                }
            }
            for (kind, map) in [("mutex", &g.mutex_waiters), ("condition", &g.cv_waiters), ("address", &g.addr_waiters)] {
                for (&address, waiters) in map.iter().filter(|(_, w)| !w.is_empty()).take(128) {
                    let list: Vec<_> = waiters.iter().map(|w| (w.tag, w.mutex, w.woken.load(Relaxed))).collect();
                    let word = crate::diag::read_words::<1>(address).map(|w|w[0] as u32);
                    eprintln!("[wait] {kind} {address:#x} word {word:x?} waiters {list:x?}");
                }
            }
        }
    });
}
