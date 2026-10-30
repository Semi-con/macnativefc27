use std::path::Path;
use std::sync::Mutex;

#[derive(Clone,Copy)]
pub struct Policy {initial:[u64;2],maximum:[u64;2]}
static LOCK:Mutex<()>=Mutex::new(());
impl Policy {
    pub fn from_nacp(data:&[u8])->anyhow::Result<Self> {
        anyhow::ensure!(data.len()>=0x3168,"truncated save policy");
        let read=|at|u64::from_le_bytes(data[at..at+8].try_into().unwrap());
        let initial=[read(0x3080),read(0x3088)];
        let maximum=[read(0x3148).max(initial[0]),read(0x3150).max(initial[1])];
        anyhow::ensure!(maximum.iter().all(|v|*v<=i64::MAX as u64),"invalid save policy size");
        Ok(Self{initial,maximum})
    }
    fn read(&self,root:&Path)->std::io::Result<[u64;2]> {
        match std::fs::read(root.join(".host-save-capacity")) {
            Ok(b) if b.len()==16=> {
                let sizes=[u64::from_le_bytes(b[..8].try_into().unwrap()),u64::from_le_bytes(b[8..].try_into().unwrap())];
                if sizes.iter().zip(self.maximum).all(|(n,max)|*n<=max) {Ok(sizes)} else {Err(std::io::Error::other("save capacity exceeds NACP maximum"))}
            }
            Ok(_)=>Err(std::io::Error::other("invalid persisted save capacity")),
            Err(e) if e.kind()==std::io::ErrorKind::NotFound=>Ok(self.initial),
            Err(e)=>Err(e),
        }
    }
    pub fn get(&self,root:&Path)->std::io::Result<[u64;2]> {let _lock=LOCK.lock().unwrap();self.read(root)}
    pub fn maximum(&self)->[u64;2] {self.maximum}
    pub fn ensure(&self,root:&Path)->std::io::Result<()> {
        let _lock=LOCK.lock().unwrap();std::fs::create_dir_all(root.join("type1-index0"))?;
        let sizes=self.read(root)?;self.write(root,sizes)
    }
    fn write(&self,root:&Path,sizes:[u64;2])->std::io::Result<()> {
        use std::io::Write;
        std::fs::create_dir_all(root)?;
        let temp=root.join(".host-save-capacity.tmp");
        let mut f=std::fs::File::create(&temp)?;for v in sizes {f.write_all(&v.to_le_bytes())?;}f.sync_all()?;
        std::fs::rename(temp,root.join(".host-save-capacity"))?;std::fs::File::open(root)?.sync_all()
    }
    pub fn extend(&self,root:&Path,requested:[u64;2])->std::io::Result<()> {
        let _lock=LOCK.lock().unwrap();let old=self.read(root)?;
        if requested.iter().zip(self.maximum).any(|(n,max)|*n>max) {return Err(std::io::Error::new(std::io::ErrorKind::InvalidInput,"save extension exceeds NACP policy"));}
        let new=[old[0].max(requested[0]),old[1].max(requested[1])];self.write(root,new)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn nacp_capacity_extension_survives_reopen_and_rejects_invalid_metadata() {
        let mut nacp=vec![0u8;0x4000];
        for (at,v) in [(0x3080,0x3900000u64),(0x3088,0x1000000),(0x3148,0x16e00000),(0x3150,0x1400000)] {nacp[at..at+8].copy_from_slice(&v.to_le_bytes());}
        let policy=Policy::from_nacp(&nacp).unwrap();let root=std::env::temp_dir().join(format!("fc27-save-policy-{}",std::process::id()));assert!(!root.exists());
        assert_eq!(policy.get(&root).unwrap(),[0x3900000,0x1000000]);policy.ensure(&root).unwrap();assert!(root.join("type1-index0").is_dir());
        policy.extend(&root,[0x4000000,0x1200000]).unwrap();let reopened=Policy::from_nacp(&nacp).unwrap();assert_eq!(reopened.get(&root).unwrap(),[0x4000000,0x1200000]);
        reopened.extend(&root,[0,0]).unwrap();assert_eq!(reopened.get(&root).unwrap(),[0x4000000,0x1200000]);assert!(reopened.extend(&root,[0x20000000,0]).is_err());
        std::fs::write(root.join(".host-save-capacity"),b"broken").unwrap();assert!(policy.get(&root).is_err());std::fs::remove_dir_all(root).unwrap();
    }
}
