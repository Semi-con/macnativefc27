// Horizon hwopus IPC backed by the native libopus reference codec (BSD-3-Clause).
// The guest reserves the reported transfer-memory capacity; decoder state and
// PCM scratch belong to this service, as they do to the external audio service.
use crate::ipc::{Request,Response,Service,Stub};
use crate::kernel::{KERNEL,Obj};
use std::ffi::c_void;
use std::sync::{Arc,Mutex};

unsafe extern "C" {
    fn opus_decoder_get_size(channels:i32)->i32;
    fn opus_decoder_init(state:*mut c_void,rate:i32,channels:i32)->i32;
    fn opus_decode(state:*mut c_void,data:*const u8,len:i32,pcm:*mut i16,frame_size:i32,fec:i32)->i32;
    fn opus_multistream_decoder_get_size(streams:i32,coupled:i32)->i32;
    fn opus_multistream_decoder_init(state:*mut c_void,rate:i32,channels:i32,streams:i32,coupled:i32,mapping:*const u8)->i32;
    fn opus_multistream_decode(state:*mut c_void,data:*const u8,len:i32,pcm:*mut i16,frame_size:i32,fec:i32)->i32;
}
const BAD_DATA:u32=0x226f; // SDK 22 maps module111/description17 to invalid Opus data.
const BAD_SIZE:u32=crate::kernel::rc::INVALID_SIZE;
const BAD_PARAMETER:u32=crate::kernel::rc::INVALID_ENUM;

#[derive(Clone)]
struct Parameters {rate:i32,channels:i32,streams:i32,coupled:i32,mapping:[u8;256],multi:bool,large:bool}
impl Parameters {
    fn parse(data:&[u8],multi:bool,extended:bool)->Result<Self,u32> {
        let minimum=if multi {if extended {0x118}else{0x110}} else {if extended {16}else{8}};
        if data.len()<minimum {return Err(BAD_SIZE);}
        let word=|at|i32::from_le_bytes(data[at..at+4].try_into().unwrap());
        let (rate,channels)=(word(0),word(4));
        if ![8000,12000,16000,24000,48000].contains(&rate) || !(1..=if multi {255}else{2}).contains(&channels) {return Err(BAD_PARAMETER);}
        let (streams,coupled)=if multi {(word(8),word(12))} else {(1,channels-1)};
        if !(1..=255).contains(&streams) || coupled<0 || coupled>streams || streams+coupled>255 {return Err(BAD_PARAMETER);}
        let large=if extended {let flag=word(if multi {16}else{8});if !(0..=1).contains(&flag) {return Err(BAD_PARAMETER);}flag!=0} else {false};
        let mut mapping=[0u8;256];if multi {let at=if extended {24}else{16};mapping.copy_from_slice(&data[at..at+256]);} else {mapping[1]=1;}
        Ok(Self{rate,channels,streams,coupled,mapping,multi,large})
    }
    fn state_size(&self)->Result<usize,u32> {
        let n=unsafe {if self.multi {opus_multistream_decoder_get_size(self.streams,self.coupled)} else {opus_decoder_get_size(self.channels)}};
        if n<=0 {Err(BAD_PARAMETER)} else {Ok(n as usize)}
    }
    fn samples(&self)->usize {self.rate as usize*if self.large {120}else{40}/1000}
    fn workspace(&self)->Result<usize,u32> {Ok(self.state_size()?.next_multiple_of(16)+self.samples()*self.channels as usize*2)}
    fn initialize(&self,state:*mut c_void)->i32 {unsafe {if self.multi {opus_multistream_decoder_init(state,self.rate,self.channels,self.streams,self.coupled,self.mapping.as_ptr())} else {opus_decoder_init(state,self.rate,self.channels)}}}
}

pub struct Manager;
impl Service for Manager {
    fn name(&self)->String {"hwopus::IHardwareOpusDecoderManager".into()}
    fn call(&self,req:&Request,rsp:&mut Response)->u32 {
        let (multi,extended,open)=match req.cmd {
            0=>(false,false,true),1=>(false,false,false),2=>(true,false,true),3=>(true,false,false),
            4=>(false,true,true),5|8=>(false,true,false),6=>(true,true,true),7|9=>(true,true,false),
            _=>return Stub(self.name()).call(req,rsp),
        };
        let input=if multi {
            let Some(b)=req.in_buffer() else{return BAD_SIZE};
            if b.size>4096 {return BAD_SIZE;}
            unsafe {std::slice::from_raw_parts(b.addr as *const u8,b.size as usize)}
        } else {&req.data};
        let params=match Parameters::parse(input,multi,extended) {Ok(p)=>p,Err(e)=>return e};
        let needed=match params.workspace() {Ok(n)=>n,Err(e)=>return e};
        if !open {rsp.push_u32(needed as u32);return 0;}
        let size_at=if multi {0}else if extended {16}else{8};
        if req.data.len()<size_at+4 {return BAD_SIZE;}
        let size=req.u32(size_at) as u64;
        let memory=req.copy_handles.first().and_then(|h|KERNEL.lock().get(*h));
        let Some(memory@Obj::TransferMemory{size:available,permission:0,..})=memory else{return crate::kernel::rc::INVALID_HANDLE};
        if size<needed as u64 || available<size {return BAD_SIZE;}
        let mut state=vec![0u128;params.state_size().unwrap().div_ceil(16)];
        if params.initialize(state.as_mut_ptr().cast())!=0 {return BAD_PARAMETER;}
        eprintln!("[hwopus] {}Hz channels{} streams{} coupled{} large{} workspace{needed}",params.rate,params.channels,params.streams,params.coupled,params.large);
        rsp.objects.push(Arc::new(Decoder{params,state:Mutex::new(state),_memory:memory,decoded:std::sync::atomic::AtomicU32::new(0)}));0
    }
}

struct Decoder {params:Parameters,state:Mutex<Vec<u128>>,_memory:Obj,decoded:std::sync::atomic::AtomicU32}
impl Service for Decoder {
    fn name(&self)->String {"hwopus::IHardwareOpusDecoder".into()}
    fn call(&self,req:&Request,rsp:&mut Response)->u32 {
        // These two historical commands carry an unused decoder-context buffer.
        if matches!(req.cmd,1|3) {return 0;}
        let (multi,perf,reset)=match req.cmd {
            0=>(false,false,false),2=>(true,false,false),4=>(false,true,false),5=>(true,true,false),
            6|8=>(false,true,true),7|9=>(true,true,true),_=>return Stub(self.name()).call(req,rsp),
        };
        if multi!=self.params.multi {return BAD_PARAMETER;}
        if reset && req.data.is_empty() {return BAD_SIZE;}
        let Some(input)=req.in_buffer() else{return BAD_SIZE};let Some(output)=req.out_buffer() else{return BAD_SIZE};
        if !(8..=4*1024*1024).contains(&input.size) {return BAD_SIZE;}
        let input=unsafe {std::slice::from_raw_parts(input.addr as *const u8,input.size as usize)};
        let bytes=u32::from_be_bytes(input[..4].try_into().unwrap()) as usize;
        if bytes==0 || bytes>input.len()-8 {return BAD_DATA;}
        let channels=self.params.channels as usize;
        let frames=(output.size as usize/(channels*2)).min(self.params.samples());
        if frames==0 {return BAD_SIZE;}
        // Temporary PCM also prevents a malformed packet or a short output
        // buffer from exposing partial decoder writes to the caller.
        let mut pcm=vec![0i16;frames*channels];let start=std::time::Instant::now();
        let count=crate::kernel::blocking(|| {
            let mut state=self.state.lock().unwrap();let state=state.as_mut_ptr().cast();
            if reset && req.data[0]!=0 && self.params.initialize(state)!=0 {return -1;}
            unsafe {if multi {opus_multistream_decode(state,input[8..8+bytes].as_ptr(),bytes as i32,pcm.as_mut_ptr(),frames as i32,0)} else {opus_decode(state,input[8..8+bytes].as_ptr(),bytes as i32,pcm.as_mut_ptr(),frames as i32,0)}}
        });
        if count<0 {eprintln!("[hwopus] decode error {count}, packet{bytes} outputFrames{frames}");return BAD_DATA;}
        let length=count as usize*channels;
        unsafe {std::ptr::copy_nonoverlapping(pcm.as_ptr().cast::<u8>(),output.addr as *mut u8,length*2);}
        rsp.push_u32((bytes+8) as u32);rsp.push_u32(count as u32);
        if perf {rsp.push_u64(start.elapsed().as_micros().min(u64::MAX as u128) as u64);}
        if self.decoded.fetch_add(1,std::sync::atomic::Ordering::Relaxed)<2 {eprintln!("[hwopus] decoded {bytes} bytes -> {count} frames x {channels} channels");}
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ipc::Buffer;
    unsafe extern "C" {
        fn opus_encoder_get_size(channels:i32)->i32;
        fn opus_encoder_init(state:*mut c_void,rate:i32,channels:i32,application:i32)->i32;
        fn opus_encode(state:*mut c_void,pcm:*const i16,frame:i32,output:*mut u8,max:i32)->i32;
        fn opus_multistream_encoder_get_size(streams:i32,coupled:i32)->i32;
        fn opus_multistream_encoder_init(state:*mut c_void,rate:i32,channels:i32,streams:i32,coupled:i32,mapping:*const u8,application:i32)->i32;
        fn opus_multistream_encode(state:*mut c_void,pcm:*const i16,frame:i32,output:*mut u8,max:i32)->i32;
    }
    #[test]
    fn real_opus_single_and_six_channel_ipc_decode_preserves_mapping_reset_and_bounds() {
        for multi in [false,true] {unsafe {
            let (channels,streams,coupled)=if multi {(6,4,2)} else {(2,1,1)};
            let mut parameters=vec![0u8;if multi {0x118}else{16}];
            parameters[..4].copy_from_slice(&48000i32.to_le_bytes());parameters[4..8].copy_from_slice(&(channels as i32).to_le_bytes());
            if multi {parameters[8..12].copy_from_slice(&(streams as i32).to_le_bytes());parameters[12..16].copy_from_slice(&(coupled as i32).to_le_bytes());parameters[24..30].copy_from_slice(&[0,4,1,2,3,5]);}
            let params=Parameters::parse(&parameters,multi,true).unwrap();let map=params.mapping;
            let make_request=|cmd:u32|Request{cmd,data:if multi {vec![]}else{parameters.clone()},x:if multi {vec![Buffer{addr:parameters.as_ptr() as u64,size:parameters.len() as u64}]}else{vec![]},..Default::default()};
            let mut size_response=Response::default();assert_eq!(Manager.call(&make_request(if multi {9}else{8}),&mut size_response),0);let size=u32::from_le_bytes(size_response.data[..4].try_into().unwrap()).next_multiple_of(4096);
            let handle=KERNEL.lock().add(Obj::TransferMemory{address:0x100000,size:size as u64,permission:0});let mut open=make_request(if multi {6}else{4});open.data.extend_from_slice(&size.to_le_bytes());open.copy_handles.push(handle);let mut opened=Response::default();assert_eq!(Manager.call(&open,&mut opened),0);let service=opened.objects.pop().unwrap();
            let n=if multi {opus_multistream_encoder_get_size(streams,coupled)} else {opus_encoder_get_size(channels)};let mut encoder=vec![0u128;(n as usize).div_ceil(16)];let state=encoder.as_mut_ptr().cast();
            assert_eq!(if multi {opus_multistream_encoder_init(state,48000,channels,streams,coupled,map.as_ptr(),2049)} else {opus_encoder_init(state,48000,channels,2049)},0);
            let signal=(0..960*channels).map(|i|{let t=(i/channels) as f32;let hz=220.0+110.0*(i%channels) as f32;(10000.0*(t*hz*std::f32::consts::TAU/48000.0).sin()) as i16}).collect::<Vec<_>>();let mut packet=vec![0u8;16384];
            let n=if multi {opus_multistream_encode(state,signal.as_ptr(),960,packet.as_mut_ptr(),packet.len() as i32)} else {opus_encode(state,signal.as_ptr(),960,packet.as_mut_ptr(),packet.len() as i32)};assert!(n>0);packet.truncate(n as usize);
            let mut framed=(packet.len() as u32).to_be_bytes().to_vec();framed.extend([0;4]);framed.extend_from_slice(&packet);let mut reference=vec![0u128;params.state_size().unwrap().div_ceil(16)];assert_eq!(params.initialize(reference.as_mut_ptr().cast()),0);
            let mut first=Vec::new();
            for reset in [false,false,true] {
                if reset {assert_eq!(params.initialize(reference.as_mut_ptr().cast()),0);}
                let mut expected=vec![0i16;960*channels as usize];let count=if multi {opus_multistream_decode(reference.as_mut_ptr().cast(),packet.as_ptr(),packet.len() as i32,expected.as_mut_ptr(),960,0)} else {opus_decode(reference.as_mut_ptr().cast(),packet.as_ptr(),packet.len() as i32,expected.as_mut_ptr(),960,0)};assert_eq!(count,960);
                let mut output=vec![0x5555i16;expected.len()+8];let req=Request{cmd:if multi {9}else{8},data:vec![reset as u8],a:vec![Buffer{addr:framed.as_ptr() as u64,size:framed.len() as u64}],b:vec![Buffer{addr:output.as_mut_ptr() as u64,size:(expected.len()*2) as u64}],..Default::default()};let mut response=Response::default();assert_eq!(service.call(&req,&mut response),0);assert_eq!(response.data.len(),16);assert_eq!(u32::from_le_bytes(response.data[..4].try_into().unwrap()),framed.len() as u32);assert_eq!(u32::from_le_bytes(response.data[4..8].try_into().unwrap()),960);assert_eq!(&output[..expected.len()],&expected);assert!(output[expected.len()..].iter().all(|v|*v==0x5555));assert!(expected.iter().any(|v|*v!=0));
                if first.is_empty() {first=expected.clone();}if reset {assert_eq!(expected,first);}
                let mut malformed=framed.clone();malformed[..4].copy_from_slice(&(framed.len() as u32).to_be_bytes());let mut bad=req;bad.a[0].addr=malformed.as_ptr() as u64;output.fill(0x5555);assert_eq!(service.call(&bad,&mut Response::default()),BAD_DATA);assert!(output.iter().all(|v|*v==0x5555));
            }
            let closed={KERNEL.lock().close(handle)};drop(closed);
        }}
    }
}
