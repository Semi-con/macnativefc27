// Mush8 - account.rs
// Constantin Clerc (c22dev) - 0.1
use crate::ipc::{Request, Response, Service, Stub, write_buffer};
use crate::services::am::USER_ID;
use std::sync::{Arc, LazyLock, Mutex};

const NICKNAME: &str = "Player";

pub struct Account;

#[derive(Default)]
struct ApplicationInfo { initialized: bool, version: u32, pid: Option<u64>, program_id: u64, open_context_stored: bool }
static APPLICATION: LazyLock<Mutex<ApplicationInfo>> = LazyLock::new(Default::default);

fn user_matches(req: &Request) -> bool { req.data.get(..16) == Some(&USER_ID) }

fn user_list(req: &Request, include: bool) -> u32 {
    let Some(b) = req.out_buffer() else { return crate::kernel::rc::INVALID_SIZE; };
    if b.size < 16 || b.size % 16 != 0 || b.size > 0x10000 { return crate::kernel::rc::INVALID_SIZE; }
    let mut list = vec![0u8; b.size as usize];
    if include { list[..16].copy_from_slice(&USER_ID); }
    write_buffer(&b, &list); 0
}

fn profile_base(out: &mut Vec<u8>) {
    out.extend_from_slice(&USER_ID);
    out.extend_from_slice(&0u64.to_le_bytes());
    let mut n = [0u8; 0x20];
    n[..NICKNAME.len()].copy_from_slice(NICKNAME.as_bytes());
    out.extend_from_slice(&n);
}

impl Service for Account {
    fn name(&self) -> String {
        "acc:u0".into()
    }
    fn call(&self, req: &Request, rsp: &mut Response) -> u32 {
        match req.cmd {
            0 => rsp.push_u32(1),
            1 => rsp.data.push(user_matches(req) as u8),
            2 | 3 | 141 => return user_list(req, true),
            131 => return user_list(req, APPLICATION.lock().unwrap().open_context_stored),
            4 => rsp.data.extend_from_slice(&USER_ID),
            5 => { if !user_matches(req) { return crate::kernel::rc::NOT_FOUND; } rsp.objects.push(Arc::new(Profile)); }
            50 => rsp.data.push(0),
            51 | 52 => {
                // One local offline profile; a request requiring a linked
                // network account has no eligible user and returns invalid Uid.
                rsp.data.extend_from_slice(if req.data.first().copied().unwrap_or(0) == 0 { &USER_ID } else { &[0u8; 16] });
            }
            100 | 140 | 160 => {
                if req.data.len() < 8 { return crate::kernel::rc::INVALID_SIZE; }
                let mut info = APPLICATION.lock().unwrap();
                info.initialized = true; info.pid = req.pid; info.program_id = crate::game::TITLE_ID;
                info.version = match req.cmd {100 => 0, 140 => 1, _ => 2};
            }
            110 | 111 | 130 => return Stub(self.name()).call(req, rsp),
            101 => { if !user_matches(req) { return crate::kernel::rc::NOT_FOUND; } rsp.objects.push(Arc::new(ManagerForApplication)); }
            150 => rsp.data.push(0),
            _ => return Stub(self.name()).call(req, rsp),
        }
        0
    }
}

struct Profile;

impl Service for Profile {
    fn name(&self) -> String {
        "account::IProfile".into()
    }
    fn call(&self, req: &Request, rsp: &mut Response) -> u32 {
        match req.cmd {
            0 => {
                if let Some(b) = req.out_buffer() {
                    write_buffer(&b, &[0u8; 0x80]);
                }
                profile_base(&mut rsp.data);
            }
            1 => profile_base(&mut rsp.data),
            10 => rsp.push_u32(0),
            11 => rsp.push_u32(0),
            _ => return Stub(self.name()).call(req, rsp),
        }
        0
    }
}

const NOT_LINKED: u32 = 124 | 341 << 9;

struct ManagerForApplication;

impl Service for ManagerForApplication {
    fn name(&self) -> String {
        "account::IManagerForApplication".into()
    }
    fn call(&self, req: &Request, rsp: &mut Response) -> u32 {
        match req.cmd {
            0..=4 | 130 | 150 | 170 => NOT_LINKED,
            160 => { APPLICATION.lock().unwrap().open_context_stored = true; 0 },
            _ => Stub(self.name()).call(req, rsp),
        }
    }
}

#[cfg(test)]
mod local_account_tests {
    use super::*;
    #[test]
    fn v2_application_registration_and_local_profile_are_consistent() {
        let mut rsp = Response::default();
        assert_eq!(Account.call(&Request {cmd:160, data:vec![0;8], pid:Some(7), ..Default::default()}, &mut rsp), 0);
        let info = APPLICATION.lock().unwrap();
        assert!(info.initialized); assert_eq!(info.program_id, crate::game::TITLE_ID); assert_eq!(info.version, 2); assert_eq!(info.pid, Some(7)); drop(info);
        let mut chosen = Response::default();
        assert_eq!(Account.call(&Request {cmd:52, data:vec![0], ..Default::default()}, &mut chosen), 0);
        assert_eq!(chosen.data, USER_ID);
        let mut profile = Response::default();
        assert_eq!(Account.call(&Request {cmd:5, data:chosen.data, ..Default::default()}, &mut profile), 0);
        let mut base = Response::default();
        assert_eq!(profile.objects[0].call(&Request {cmd:1, ..Default::default()}, &mut base), 0);
        assert_eq!(&base.data[..16], &USER_ID);
        assert_eq!(&base.data[24..30], b"Player");
        assert_ne!(Account.call(&Request {cmd:5, data:vec![0;16], ..Default::default()}, &mut Response::default()), 0);
    }

    #[test]
    fn bounded_user_lists_and_offline_network_results() {
        let mut short = [0xa5u8;8];
        let request = |buffer: &mut [u8]| Request {cmd:2, b:vec![crate::ipc::Buffer {addr:buffer.as_mut_ptr() as u64, size:buffer.len() as u64}], ..Default::default()};
        assert_eq!(Account.call(&request(&mut short), &mut Response::default()), crate::kernel::rc::INVALID_SIZE);
        assert_eq!(short, [0xa5;8]);
        let mut users = [0xa5u8;128];
        assert_eq!(Account.call(&request(&mut users), &mut Response::default()), 0);
        assert_eq!(&users[..16], &USER_ID); assert!(users[16..].iter().all(|v| *v == 0));
        let mut chosen = Response::default();
        assert_eq!(Account.call(&Request {cmd:52, data:vec![1], ..Default::default()}, &mut chosen), 0);
        assert_eq!(chosen.data, [0;16]);
        for cmd in [0,1,2,3,4] { assert_eq!(ManagerForApplication.call(&Request {cmd, ..Default::default()}, &mut Response::default()), NOT_LINKED); }
    }
}
