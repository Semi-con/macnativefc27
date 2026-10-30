//! PCM output queues consumed by the real CoreAudio callback.
use crate::audio::output::{self, Output};
use crate::ipc::{Request, Response, Service, Stub, new_event, write_buffer};
use crate::kernel::{KERNEL, Obj};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::sync::atomic::Ordering::Relaxed;

const BLOCK: usize = 256;
const UNSUPPORTED: u32 = crate::ipc::rc::UNKNOWN_COMMAND;
const NAME: &[u8] = b"DeviceOut\0";

pub struct Manager;
pub struct InputManager;

fn name_buffer(req: &Request) -> u32 {
    let Some(buf) = req.out_buffer().filter(|b| b.size >= 0x100) else { return 0 };
    let mut name = [0u8;0x100]; name[..NAME.len()].copy_from_slice(NAME);
    write_buffer(&buf,&name); 1
}

impl Service for Manager {
    fn name(&self) -> String { "audout:u".into() }
    fn call(&self, req: &Request, rsp: &mut Response) -> u32 {
        match req.cmd {
            0 | 2 => rsp.push_u32(name_buffer(req)),
            1 | 3 => {
                let rate = if req.u32(0) == 0 {48000} else {req.u32(0)};
                let channels = req.u32(4) & 0xffff;
                let channels = if channels == 0 {2} else {channels};
                if ![32000,44100,48000].contains(&rate) || !(1..=6).contains(&channels) {
                    eprintln!("[audout] unsupported format {rate} Hz / {channels} channels");
                    return UNSUPPORTED;
                }
                name_buffer(req);
                rsp.push_u32(rate); rsp.push_u32(channels); rsp.push_u32(2); rsp.push_u32(1);
                rsp.objects.push(Stream::open(rate,channels as usize));
            }
            _ => return Stub(self.name()).call(req,rsp),
        }
        0
    }
}
impl Service for InputManager {
    fn name(&self) -> String { "audin:u".into() }
    fn call(&self, req: &Request, rsp: &mut Response) -> u32 {
        // No microphone route is exposed by this offline port. Enumeration is
        // empty; opening/capturing remains an explicit unsupported operation.
        match req.cmd { 0 | 2 => {rsp.push_u32(0);0}, _ => Stub(self.name()).call(req,rsp) }
    }
}

struct Packet { tag: u64, pcm: Vec<i16>, cursor: usize }
struct State {
    started: bool,
    channels: usize,
    volume: f32,
    pending: VecDeque<Packet>,
    released: VecDeque<u64>,
    played: u64,
}
impl State {
    fn new(channels: usize) -> Self {
        Self {started:false,channels,volume:1.0,pending:VecDeque::new(),released:VecDeque::new(),played:0}
    }
    fn append(&mut self, tag: u64, pcm: Vec<i16>) -> bool {
        if self.pending.len() >= 64 || pcm.is_empty() || pcm.len()%self.channels != 0
            || self.pending.iter().any(|p| p.tag == tag) || self.released.contains(&tag) { return false; }
        self.pending.push_back(Packet {tag,pcm,cursor:0}); true
    }
    fn render(&mut self, out: &mut [f32]) -> bool {
        out.fill(0.0);
        if !self.started { return false; }
        let mut released = false;
        for block in out.chunks_exact_mut(BLOCK*output::CHANNELS) {
            for frame in 0..BLOCK {
                let Some(p) = self.pending.front_mut() else { break };
                for c in 0..self.channels {
                    block[c*BLOCK+frame] = p.pcm[p.cursor+c] as f32/32768.0*self.volume;
                }
                if self.channels == 1 { block[BLOCK+frame] = block[frame]; }
                p.cursor += self.channels; self.played += 1;
                if p.cursor == p.pcm.len() {
                    self.released.push_back(self.pending.pop_front().unwrap().tag); released = true;
                }
            }
        }
        released
    }
    fn flush(&mut self) -> bool {
        let flushed = !self.pending.is_empty();
        self.released.extend(self.pending.drain(..).map(|p| p.tag)); flushed
    }
}

struct Stream { state: Arc<Mutex<State>>, event: u32, _output: Output }
impl Stream {
    fn open(rate: u32, channels: usize) -> Arc<dyn Service> {
        let event = new_event(false);
        let Some(Obj::ReadableEvent(signal)) = KERNEL.lock().get(event) else { unreachable!() };
        let state = Arc::new(Mutex::new(State::new(channels)));
        let s = state.clone();
        let output = output::start(rate,BLOCK,Box::new(move |out| {
            let released = s.lock().unwrap().render(out);
            if released { crate::kernel::signal_event(&signal); }
        }));
        eprintln!("[audout] opened PCM16 {rate} Hz / {channels} channels, callback-driven release");
        Arc::new(Self {state,event,_output:output})
    }
    fn update_event(&self, signaled: bool) {
        let object = { KERNEL.lock().get(self.event) };
        if let Some(Obj::ReadableEvent(e)) = object {
            if signaled { crate::kernel::signal_event(&e); } else { e.signaled.store(false,Relaxed); }
        }
    }
}
impl Service for Stream {
    fn name(&self) -> String { "audio::IAudioOut".into() }
    fn call(&self, req: &Request, rsp: &mut Response) -> u32 {
        match req.cmd {
            0 => rsp.push_u32(!self.state.lock().unwrap().started as u32),
            1 => self.state.lock().unwrap().started = true,
            2 => self.state.lock().unwrap().started = false,
            3 | 7 => {
                let Some(b) = req.in_buffer().filter(|b| b.size >= 0x28) else { return UNSUPPORTED };
                let desc = unsafe { std::slice::from_raw_parts(b.addr as *const u8,0x28) };
                let read = |o| u64::from_le_bytes(desc[o..o+8].try_into().unwrap());
                let (ptr,capacity,size,offset) = (read(8),read(16),read(24),read(32));
                if ptr == 0 || size == 0 || size > 16*1024*1024 || size%2 != 0
                    || offset.checked_add(size).is_none_or(|e| e > capacity)
                    || ptr.checked_add(offset).and_then(|p| p.checked_add(size)).is_none() { return UNSUPPORTED; }
                let data = unsafe { std::slice::from_raw_parts((ptr+offset) as *const u8,size as usize) };
                let pcm = data.chunks_exact(2).map(|b| i16::from_le_bytes([b[0],b[1]])).collect();
                if !self.state.lock().unwrap().append(req.u64(0),pcm) { return UNSUPPORTED; }
            }
            4 => {
                let mut k=KERNEL.lock();
                let Some(event)=k.get(self.event) else{return crate::kernel::rc::INVALID_HANDLE};
                rsp.copy_handles.push(k.add(event));
            }
            5 | 8 => {
                let Some(b) = req.out_buffer() else { return UNSUPPORTED };
                let mut s = self.state.lock().unwrap();
                let n = s.released.len().min((b.size/8) as usize);
                let tags: Vec<u8> = s.released.drain(..n).flat_map(u64::to_le_bytes).collect();
                write_buffer(&b,&tags); rsp.push_u32(n as u32);
                self.update_event(!s.released.is_empty());
            }
            6 => rsp.data.push(self.state.lock().unwrap().pending.iter().any(|p| p.tag == req.u64(0)) as u8),
            9 => rsp.push_u32(self.state.lock().unwrap().pending.len() as u32),
            10 => rsp.push_u64(self.state.lock().unwrap().played),
            11 => {
                let mut s = self.state.lock().unwrap(); rsp.data.push(s.flush() as u8); self.update_event(!s.released.is_empty());
            }
            12 => {
                let volume = f32::from_bits(req.u32(0));
                if !volume.is_finite() || !(0.0..=1.0).contains(&volume) { return UNSUPPORTED; }
                self.state.lock().unwrap().volume = volume;
            }
            13 => rsp.data.extend_from_slice(&self.state.lock().unwrap().volume.to_le_bytes()),
            _ => return Stub(self.name()).call(req,rsp),
        }
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn releases_only_consumed_pcm_and_preserves_stereo_order() {
        let mut s = State::new(2); assert!(s.append(1,vec![32767,-32768,16384,-16384]));
        let mut out = vec![9.0;BLOCK*6];
        assert!(!s.render(&mut out)); assert_eq!(s.pending.len(),1); assert_eq!(s.played,0);
        s.started = true; assert!(s.render(&mut out));
        assert_eq!((out[0],out[1],out[BLOCK],out[BLOCK+1]),(32767.0/32768.0,0.5,-1.0,-0.5));
        assert_eq!(s.released,VecDeque::from([1])); assert_eq!(s.played,2);
        assert!(out[2..BLOCK].iter().all(|&s| s == 0.0));
    }
    #[test]
    fn partial_buffers_stop_resume_and_flush_keep_tag_lifetimes() {
        let mut s = State::new(2); s.started = true;
        assert!(s.append(7,vec![100;BLOCK*4])); assert!(!s.append(7,vec![1,2]));
        let mut out = vec![0.0;BLOCK*6]; assert!(!s.render(&mut out));
        assert!(s.released.is_empty()); assert_eq!(s.played,BLOCK as u64);
        s.started = false; assert!(!s.render(&mut out)); assert_eq!(s.played,BLOCK as u64);
        assert!(s.flush()); assert_eq!(s.released,VecDeque::from([7])); assert!(!s.flush());
    }
}
