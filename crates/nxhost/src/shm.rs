// Mush8 - shm.rs
// Constantin Clerc (c22dev) - 0.1
use anyhow::{Result, ensure};
use std::sync::atomic::{AtomicU32, Ordering};

pub struct Shm {
    pub fd: i32,
    pub host: u64,
    pub size: u64,
    /// Read-only providers whose contents are invariant for the process may be
    /// copied into a 4 KiB guest subpage without losing shared update semantics.
    pub snapshot: bool,
}

pub fn create(size: u64) -> Result<Shm> {
    let backing_size = size.next_multiple_of(crate::mem::PAGE);
    static N: AtomicU32 = AtomicU32::new(0);
    let name = format!("/nxhost.{}.{}\0", std::process::id(), N.fetch_add(1, Ordering::Relaxed));
    let mut fd = unsafe { libc::shm_open(name.as_ptr() as *const _, libc::O_RDWR | libc::O_CREAT | libc::O_EXCL, 0o600) };
    if fd >= 0 {
        unsafe { libc::shm_unlink(name.as_ptr() as *const _) };
    } else {
        let path = std::env::temp_dir().join(format!("nxhost.{}.{}", std::process::id(), N.fetch_add(1, Ordering::Relaxed)));
        let c = std::ffi::CString::new(path.as_os_str().as_encoded_bytes())?;
        fd = unsafe { libc::open(c.as_ptr(), libc::O_RDWR | libc::O_CREAT | libc::O_EXCL, 0o600) };
        ensure!(fd >= 0, "shared memory file {}: {}", path.display(), std::io::Error::last_os_error());
        unsafe { libc::unlink(c.as_ptr()) };
    }
    ensure!(unsafe { libc::ftruncate(fd, backing_size as i64) } == 0, "ftruncate: {}", std::io::Error::last_os_error());
    let p = unsafe { libc::mmap(std::ptr::null_mut(), backing_size as usize, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_SHARED, fd, 0) };
    ensure!(p != libc::MAP_FAILED, "mmap shm: {}", std::io::Error::last_os_error());
    Ok(Shm { fd, host: p as u64, size, snapshot:false })
}

impl Drop for Shm {
    fn drop(&mut self) {
        unsafe {libc::munmap(self.host as *mut _,self.size.next_multiple_of(crate::mem::PAGE) as usize);libc::close(self.fd);}
    }
}
