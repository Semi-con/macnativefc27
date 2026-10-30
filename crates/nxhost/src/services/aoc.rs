// Mush8 - aoc.rs
// Constantin Clerc (c22dev) - 0.1
use crate::ipc::{Act, Request, Response, Service, Stub, Table, new_event, write_buffer};
use std::sync::{Arc,LazyLock,Mutex};
use std::collections::HashMap;

const INVALID_BUFFER_SIZE: u32 = (200 << 9) | 166;
const NO_PURCHASED_PRODUCT_INFO: u32 = (400 << 9) | 164;

#[derive(Default)]
struct Mounts(HashMap<u64,u32>);
impl Mounts {
    fn mount(&mut self,id:u64,available:&[u64])->u32 {
        if !available.contains(&id){return crate::kernel::rc::NOT_FOUND;}
        let count=self.0.entry(id).or_default();
        let Some(next)=count.checked_add(1) else{return crate::kernel::rc::INVALID_STATE};
        *count=next;0
    }
    fn unmount(&mut self,id:u64)->u32 {
        let Some(count)=self.0.get_mut(&id) else{return crate::kernel::rc::NOT_FOUND};
        *count-=1;if *count==0{self.0.remove(&id);}0
    }
    fn check(&self,available:&[u64])->u32 {
        if self.0.keys().all(|id|available.contains(id)){0}else{crate::kernel::rc::NOT_FOUND}
    }
}
static MOUNTS:LazyLock<Mutex<Mounts>>=LazyLock::new(||Mutex::new(Mounts::default()));
fn available_ids()->Vec<u64> {
    super::fs::CONFIG.get().map(|c|c.dlc.iter().filter(|(_,s)|s.size()>0).map(|(id,_)|*id).collect()).unwrap_or_default()
}

fn base_id() -> u64 {
    crate::svc::PROGRAM_ID + 0x1000
}

fn indices() -> Vec<u32> {
    let mut v: Vec<u32> = super::fs::CONFIG
        .get()
        .map(|c| c.dlc.iter().filter_map(|(id, _)| id.checked_sub(base_id()).map(|i| i as u32)).collect())
        .unwrap_or_default();
    v.sort();
    v
}

pub struct Aoc;

impl Aoc {
    fn list(req: &Request, rsp: &mut Response, start: usize, count: usize) -> u32 {
        let Some(b) = req.out_buffer() else { return INVALID_BUFFER_SIZE };
        if count > b.size as usize / 4 {
            return INVALID_BUFFER_SIZE;
        }
        let list: Vec<u32> = indices().into_iter().skip(start).take(count).collect();
        let bytes: Vec<u8> = list.iter().flat_map(|i| i.to_le_bytes()).collect();
        write_buffer(&b, &bytes);
        rsp.push_u32(list.len() as u32);
        0
    }
}

impl Service for Aoc {
    fn name(&self) -> String {
        "aoc:u".into()
    }
    fn call(&self, req: &Request, rsp: &mut Response) -> u32 {
        match req.cmd {
            0 | 2 => rsp.push_u32(indices().len() as u32),
            3 => return Self::list(req, rsp, req.u32(0) as usize, req.u32(4) as usize),
            1 => return Self::list(req, rsp, req.u32(8) as usize, req.u32(12) as usize),
            4 | 5 => rsp.push_u64(base_id()),
            6 | 7 => {
                if !indices().contains(&req.u32(0)){return crate::kernel::rc::NOT_FOUND;}
            }
            11 => return MOUNTS.lock().unwrap().mount(req.u64(0),&available_ids()),
            12 => return MOUNTS.lock().unwrap().unmount(req.u64(0)),
            13 => rsp.data.push(MOUNTS.lock().unwrap().0.contains_key(&req.u64(0)) as u8),
            50 => return MOUNTS.lock().unwrap().check(&available_ids()),
            8 | 10 => rsp.copy_handles.push(new_event(false)),
            9 => rsp.push_u64(0),
            100 | 101 => rsp.objects.push(Arc::new(Table {
                name: "ec::IPurchaseEventManager",
                cmds: &[
                    (0, Act::Ok),
                    (1, Act::Ok),
                    (2, Act::Event(false)),
                    (3, Act::Err(NO_PURCHASED_PRODUCT_INFO)),
                    (4, Act::Err(NO_PURCHASED_PRODUCT_INFO)),
                ],
            })),
            _ => return Stub(self.name()).call(req, rsp),
        }
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn mount_notifications_reference_count_only_available_content() {
        let available=[crate::game::TITLE_ID+0x1002];let id=available[0];let mut m=Mounts::default();
        assert_eq!(m.mount(id+1,&available),crate::kernel::rc::NOT_FOUND);
        assert_eq!(m.mount(id,&available),0);assert_eq!(m.mount(id,&available),0);
        assert_eq!(m.check(&available),0);assert_eq!(m.check(&[]),crate::kernel::rc::NOT_FOUND);
        assert_eq!(m.unmount(id),0);assert!(m.0.contains_key(&id));
        assert_eq!(m.unmount(id),0);assert!(!m.0.contains_key(&id));
        assert_eq!(m.unmount(id),crate::kernel::rc::NOT_FOUND);
    }
}
