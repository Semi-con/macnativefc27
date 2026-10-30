// Mush8 - binder.rs
// Constantin Clerc (c22dev) - 0.1
use crate::ipc::{Buffer, Request, Response, Service, Stub, new_event, write_buffer};

pub struct HosBinderDriver;

const REQUEST_BUFFER: u32 = 1;
const SET_BUFFER_COUNT: u32 = 2;
const DEQUEUE_BUFFER: u32 = 3;
const QUEUE_BUFFER: u32 = 7;
const CANCEL_BUFFER: u32 = 8;
const QUERY: u32 = 9;
const CONNECT: u32 = 10;
const DISCONNECT: u32 = 11;
const SET_PREALLOCATED_BUFFER: u32 = 14;
const GET_BUFFER_HISTORY:u32=17;

impl Service for HosBinderDriver {
    fn name(&self) -> String {
        "vi::IHOSBinderDriver".into()
    }
    fn call(&self, req: &Request, rsp: &mut Response) -> u32 {
        match req.cmd {
            0 | 3 => {
                let (id, code) = (req.u32(0), req.u32(4));
                let auto = |a: Option<&Buffer>, b: Option<&Buffer>| [a, b].into_iter().flatten().find(|x| x.size > 0 && x.addr != 0).copied();
                let input = if req.cmd == 0 { req.a.first().copied() } else { auto(req.a.first(), req.x.first()) };
                let output = if req.cmd == 0 { req.b.first().copied() } else { auto(req.b.first(), req.c.first()) };
                let data = input.map(|b| read_parcel(&b)).unwrap_or_default();
                let reply = transact(id, code, &data);
                if let Some(o) = output {
                    write_buffer(&o, &serialize(&reply));
                }
            }
            1 => {}
            2 => rsp.copy_handles.push(new_event(false)),
            _ => return Stub(self.name()).call(req, rsp),
        }
        0
    }
}

fn read_parcel(b: &Buffer) -> Vec<u8> {
    let raw = unsafe { std::slice::from_raw_parts(b.addr as *const u8, b.size as usize) };
    if raw.len() < 0x10 {
        return Vec::new();
    }
    let rd = |o: usize| u32::from_le_bytes(raw[o..o + 4].try_into().unwrap()) as usize;
    let (size, off) = (rd(0), rd(4));
    raw.get(off..off + size).map(|d| d.to_vec()).unwrap_or_default()
}

fn serialize(data: &[u8]) -> Vec<u8> {
    let mut p = Vec::new();
    for w in [data.len() as u32, 0x10, 0, 0x10 + data.len() as u32] {
        p.extend_from_slice(&w.to_le_bytes());
    }
    p.extend_from_slice(data);
    p
}

struct Out(Vec<u8>);
impl Out {
    fn i32(&mut self, v: i32) -> &mut Self {
        self.0.extend_from_slice(&v.to_le_bytes());
        self
    }
}

fn transact(id: u32, code: u32, _data: &[u8]) -> Vec<u8> {
    let mut o = Out(Vec::new());
    match code {
        GET_BUFFER_HISTORY=> {
            let count=i32::from_le_bytes(_data.get(_data.len().saturating_sub(4)..).and_then(|b|b.try_into().ok()).unwrap_or([0;4]));
            let history=if count>0 {super::vi::frame_history(id as u64,count as usize)} else {None};
            let Some(history)=history else {o.i32(-22).i32(0);return o.0;};
            o.i32(0).i32(history.len() as i32);
            // SDK22 reads three parcel u64s plus one u32 per entry (28 bytes),
            // storing them in an aligned 32-byte in-process BufferInfo.
            for frame in history {
                for value in [frame.frame,frame.queued_ns,frame.presented_ns] {o.0.extend_from_slice(&value.to_le_bytes());}
                o.i32(frame.state as i32);
            }
        }
        QUERY => {
            let what = i32::from_le_bytes(_data.get(_data.len().saturating_sub(4)..).and_then(|b| b.try_into().ok()).unwrap_or([0; 4]));
            let (w, h) = super::vi::resolution();
            let v = match what {
                0 => w as i32,
                1 => h as i32,
                2 => 1,
                3 => 2,
                _ => { o.i32(0).i32(-22); return o.0; }
            };
            o.i32(v).i32(0);
        }
        CONNECT => {
            let (w, h) = super::vi::resolution();
            o.i32(w as i32).i32(h as i32).i32(0).i32(0).i32(0);
        }
        SET_PREALLOCATED_BUFFER | DISCONNECT | SET_BUFFER_COUNT | CANCEL_BUFFER => {
            o.i32(-38);
        }
        REQUEST_BUFFER | DEQUEUE_BUFFER | QUEUE_BUFFER => {
            eprintln!("[binder] buffer queue transaction {code} not implemented (presentation goes through NVN)");
            o.i32(-38);
        }
        _ => {
            eprintln!("[binder] binder {id} unknown transaction {code}");
            o.i32(-38);
        }
    }
    o.0
}
