// Mush8 - time.rs
// Constantin Clerc (c22dev) - 0.1
use crate::ipc::{Request, Response, Service, Stub, new_event, write_buffer};
use std::sync::{Arc, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

fn source_id() -> &'static [u8;16] {
    static ID: OnceLock<[u8;16]> = OnceLock::new();
    ID.get_or_init(|| {
        let mut boot: libc::timeval = unsafe {std::mem::zeroed()};
        let mut len = std::mem::size_of_val(&boot);
        let rc = unsafe {libc::sysctlbyname(c"kern.boottime".as_ptr(),&mut boot as *mut _ as *mut _,&mut len,std::ptr::null_mut(),0)};
        let mut id = [0u8;16];
        if rc == 0 {
            id[..8].copy_from_slice(&(boot.tv_sec as i64).to_le_bytes());
            id[8..12].copy_from_slice(&(boot.tv_usec as i32).to_le_bytes());
            id[12..].copy_from_slice(b"NXTM");
        } else {unsafe {libc::arc4random_buf(id.as_mut_ptr() as *mut _,id.len());}}
        id
    })
}

fn now_posix() -> i64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs() as i64)
}

fn steady_seconds() -> i64 {
    (crate::svc::system_tick()/nxabi::GUEST_TICK_HZ) as i64
}

fn time_point(out: &mut Vec<u8>) {
    out.extend_from_slice(&steady_seconds().to_le_bytes());
    out.extend_from_slice(source_id());
}

fn shared_clock_bytes(posix: i64,steady: i64,id: &[u8;16]) -> Vec<u8> {
    let mut data = vec![0u8;0x1000];
    let mut publish = |offset: usize,payload: &[u8]| {
        // Counter zero selects the first copy. Both copies are initialized and
        // invariant; guest counter reads advance time without a writer thread.
        for i in 0..2 {data[offset+8+i*payload.len()..offset+8+(i+1)*payload.len()].copy_from_slice(payload);}
    };
    let mut steady_context = vec![0u8;8]; steady_context.extend_from_slice(id);
    publish(0,&steady_context);
    let mut context = Vec::new(); context.extend_from_slice(&(posix-steady).to_le_bytes());
    context.extend_from_slice(&steady.to_le_bytes()); context.extend_from_slice(id);
    publish(0x38,&context); publish(0x80,&context);
    // SDK 22 uses ContinuousAdjustmentTimePoint at 0xd0 (counter, padding,
    // two 56-byte copies). Its native calculation at sdk+0x43f980 applies
    // ((counter_ns-origin)*rate >> shift)+offset, clamps to a limit, and
    // requires the same source ID as SystemClockContext. Zero coefficients
    // mean the host counter runs without a slew correction.
    let mut adjustment=vec![0u8;40];adjustment.extend_from_slice(id);
    publish(0xd0,&adjustment);
    // The bool atomic has 4-byte, rather than 8-byte, payload alignment.
    data[0xcc]=1;data[0xcd]=1;
    data
}

fn shared_memory() -> &'static Arc<crate::kernel::SharedMemory> {
    static SHM: OnceLock<Arc<crate::kernel::SharedMemory>> = OnceLock::new();
    SHM.get_or_init(|| {
        let mut sm = crate::shm::create(0x1000).expect("time shared memory");
        sm.snapshot = true;
        let data = shared_clock_bytes(now_posix(),steady_seconds(),source_id());
        unsafe {std::ptr::copy_nonoverlapping(data.as_ptr(),sm.host as *mut u8,data.len());}
        Arc::new(sm)
    })
}

fn clock_context(out: &mut Vec<u8>) {
    out.extend_from_slice(&(now_posix() - steady_seconds()).to_le_bytes());
    time_point(out);
}

pub struct TimeService;

impl Service for TimeService {
    fn name(&self) -> String {
        "time:u".into()
    }
    fn call(&self, req: &Request, rsp: &mut Response) -> u32 {
        match req.cmd {
            0 | 1 | 4 | 5 => rsp.objects.push(Arc::new(SystemClock)),
            2 => rsp.objects.push(Arc::new(SteadyClock)),
            3 => rsp.objects.push(Arc::new(TimeZone)),
            20 => rsp.copy_handles.push(crate::kernel::KERNEL.lock().add(crate::kernel::Obj::SharedMemory(shared_memory().clone()))),
            30 | 31 => rsp.copy_handles.push(new_event(false)),
            100 => rsp.data.push(1),
            200 => rsp.data.push(1),
            300 => rsp.push_u64((now_posix() - steady_seconds()) as u64),
            _ => return Stub(self.name()).call(req, rsp),
        }
        0
    }
}

struct SystemClock;

impl Service for SystemClock {
    fn name(&self) -> String {
        "time::ISystemClock".into()
    }
    fn call(&self, req: &Request, rsp: &mut Response) -> u32 {
        match req.cmd {
            0 => rsp.push_u64(now_posix() as u64),
            1 | 3 => return Stub(self.name()).call(req,rsp),
            2 => clock_context(&mut rsp.data),
            4 => rsp.copy_handles.push(new_event(false)),
            _ => return Stub(self.name()).call(req, rsp),
        }
        0
    }
}

struct SteadyClock;

impl Service for SteadyClock {
    fn name(&self) -> String {
        "time::ISteadyClock".into()
    }
    fn call(&self, req: &Request, rsp: &mut Response) -> u32 {
        match req.cmd {
            0 => time_point(&mut rsp.data),
            2 | 200 => rsp.push_u64(0),
            100 => rsp.push_u64(now_posix() as u64),
            101 => rsp.data.push(0),
            102 => rsp.push_u32(0),
            _ => return Stub(self.name()).call(req, rsp),
        }
        0
    }
}

struct TimeZone;

fn calendar(t: i64, out: &mut Vec<u8>) {
    let mut tm: libc::tm = unsafe { std::mem::zeroed() };
    unsafe { libc::localtime_r(&t, &mut tm) };
    out.extend_from_slice(&((tm.tm_year + 1900) as i16).to_le_bytes());
    out.extend([(tm.tm_mon + 1) as u8, tm.tm_mday as u8, tm.tm_hour as u8, tm.tm_min as u8, tm.tm_sec as u8, 0]);
    out.extend_from_slice(&(tm.tm_wday as u32).to_le_bytes());
    out.extend_from_slice(&(tm.tm_yday as u32).to_le_bytes());
    let mut name = [0u8; 8];
    let zone = if tm.tm_zone.is_null() { "UTC".into() } else { unsafe { std::ffi::CStr::from_ptr(tm.tm_zone) }.to_string_lossy().into_owned() };
    name[..zone.len().min(7)].copy_from_slice(&zone.as_bytes()[..zone.len().min(7)]);
    out.extend_from_slice(&name);
    out.extend_from_slice(&(tm.tm_isdst.max(0) as u32).to_le_bytes());
    out.extend_from_slice(&(tm.tm_gmtoff as i32).to_le_bytes());
}

impl Service for TimeZone {
    fn name(&self) -> String {
        "time::ITimeZoneService".into()
    }
    fn call(&self, req: &Request, rsp: &mut Response) -> u32 {
        match req.cmd {
            0 => {
                let mut n = [0u8; 0x24];
                let zone = std::fs::read_link("/etc/localtime").ok().and_then(|p| p.to_str().and_then(|s| s.split_once("zoneinfo/").map(|(_,z)|z.to_owned()))).unwrap_or_else(|| "UTC".into());
                let bytes = zone.as_bytes(); let count = bytes.len().min(n.len()-1);
                n[..count].copy_from_slice(&bytes[..count]);
                rsp.data.extend_from_slice(&n);
            }
            2 => rsp.push_u32(1),
            4 => return Stub(self.name()).call(req,rsp),
            100 | 201 => return Stub(self.name()).call(req,rsp),
            101 => calendar(req.u64(0) as i64, &mut rsp.data),
            202 => {
                let d = &req.data;
                if d.len() < 8 {return crate::ipc::rc::UNKNOWN_COMMAND;}
                let mut tm: libc::tm = unsafe { std::mem::zeroed() };
                tm.tm_year = i16::from_le_bytes([d[0], d[1]]) as i32 - 1900;
                tm.tm_mon = d[2] as i32 - 1;
                tm.tm_mday = d[3] as i32;
                tm.tm_hour = d[4] as i32;
                tm.tm_min = d[5] as i32;
                tm.tm_sec = d[6] as i32;
                tm.tm_isdst = -1;
                let t = unsafe { libc::mktime(&mut tm) };
                rsp.push_u32(1);
                if let Some(b) = req.out_buffer() {
                    write_buffer(&b, &(t as i64).to_le_bytes());
                }
            }
            _ => return Stub(self.name()).call(req, rsp),
        }
        0
    }
}

#[cfg(test)]
mod shared_clock_tests {
    use super::*;
    #[test]
    fn shared_clock_matches_guest_counter_units_and_epoch() {
        let id = *b"clock-test-id-01";
        let b = shared_clock_bytes(1800000000,123456,&id);
        let read = |at| i64::from_le_bytes(b[at..at+8].try_into().unwrap());
        assert_eq!(read(8),0); assert_eq!(&b[16..32],&id);
        for at in [0x38usize,0x80] {
            for copy in 0..2 {
                let start = at+8+copy*32;
                assert_eq!(read(start)+123456,1800000000);
                assert_eq!(&b[start+16..start+32],&id);
            }
        }
        assert_eq!((b[0xcc],b[0xcd]),(1,1));
        for copy in 0..2 {
            let start=0xd8+copy*56;
            assert!(b[start..start+40].iter().all(|b|*b==0));
            assert_eq!(&b[start+40..start+56],&id);
            // Original SDK calculation: offset_s*1e9 + counter_ns + slew_ns.
            let counter_ns=123456i128*1_000_000_000+987_654_321;
            let system_offset=read(0x40) as i128*1_000_000_000;
            assert_eq!((system_offset+counter_ns)/1_000_000_000,1_800_000_000);
            assert_eq!((system_offset+counter_ns)%1_000_000_000,987_654_321);
        }
    }
}
