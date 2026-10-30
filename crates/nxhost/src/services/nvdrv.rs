//! NV driver session metadata for the native NVN/Metal route. Tegra device
//! ioctls remain explicit unsupported requests; GPU commands go through NVN.
use crate::{ipc::{Request,Response,Service,Stub},kernel::{self,Obj,KERNEL}};
use std::sync::{Arc,Mutex};

#[derive(Default)]
struct State {transfer:Option<Obj>,aruid:Option<u64>,firmware_margin:bool}
#[derive(Default)]
pub struct NvDrv {state:Mutex<State>}
pub fn open()->Arc<dyn Service> {Arc::new(NvDrv::default())}
impl Service for NvDrv {
    fn name(&self)->String {"nvdrv".into()}
    fn call(&self,req:&Request,rsp:&mut Response)->u32 {
        let mut s=self.state.lock().unwrap();
        match req.cmd {
            3 => {
                if s.transfer.is_some(){return kernel::rc::INVALID_STATE;}
                if req.copy_handles.len()!=2 {return kernel::rc::INVALID_HANDLE;}
                let k=KERNEL.lock();
                if !matches!(k.get(req.copy_handles[0]),Some(Obj::Process)){return kernel::rc::INVALID_HANDLE;}
                let Some(transfer@Obj::TransferMemory {address,size,permission})=k.get(req.copy_handles[1]) else{return kernel::rc::INVALID_HANDLE};
                if req.u32(0) as u64!=size || address%0x1000!=0 || permission!=0{return kernel::rc::INVALID_SIZE;}
                s.transfer=Some(transfer);rsp.push_u32(0);0
            }
            8 => {
                if s.transfer.is_none(){return kernel::rc::INVALID_STATE;}
                let id=req.u64(0);
                if id!=0x100{return kernel::rc::INVALID_HANDLE;}
                s.aruid=Some(id);rsp.push_u32(0);0
            }
            13 => {
                if s.transfer.is_none(){return kernel::rc::INVALID_STATE;}
                match req.u64(0){0=>s.firmware_margin=false,1=>s.firmware_margin=true,_=>return kernel::rc::INVALID_ENUM}0
            }
            _=>Stub(self.name()).call(req,rsp),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn initialization_retains_typed_transfer_after_client_handle_close() {
        let service=NvDrv::default();
        let h=KERNEL.lock().add(Obj::TransferMemory {address:0x10000,size:0x600000,permission:0});
        let mut rsp=Response::default();
        assert_eq!(service.call(&Request {cmd:3,data:0x600000u32.to_le_bytes().to_vec(),copy_handles:vec![kernel::CURRENT_PROCESS,h],..Default::default()},&mut rsp),0);
        let closed={KERNEL.lock().close(h)};drop(closed);
        assert!(service.state.lock().unwrap().transfer.is_some());
        assert_eq!(service.call(&Request {cmd:8,data:0x100u64.to_le_bytes().to_vec(),..Default::default()},&mut Response::default()),0);
        assert_eq!(service.call(&Request {cmd:13,data:1u64.to_le_bytes().to_vec(),..Default::default()},&mut Response::default()),0);
        assert_eq!(service.call(&Request {cmd:1,..Default::default()},&mut Response::default()),crate::ipc::rc::UNKNOWN_COMMAND);
    }
}
