// Logical GPU addresses only: no CPU mappings or physical allocations change.
// A guard band keeps the conservative STG high-word domain away from unrelated
// pools. Finalized reservations can be reused; command buffers retain their
// own immutable mapping tables and physical Metal buffers.
use std::sync::LazyLock;
pub(super) const START:u64=1<<39;
pub(super) const END:u64=1<<40;
const BAND:u64=1<<33;
const PREFIX:u64=1<<28;
pub(super) fn enabled()->bool {static E:LazyLock<bool>=LazyLock::new(||std::env::var_os("NX_GPU_POOL_DOMAINS").is_some());*E}
fn span(size:u64)->Option<u64> {size.checked_add(PREFIX)?.checked_add(BAND-1).map(|n|n/BAND*BAND)}
pub(super) struct Arena {next:u64,free:Vec<(u64,u64)>}
impl Default for Arena {fn default()->Self {Self{next:START,free:Vec::new()}}}
impl Arena {
    pub(super) fn reserve(&mut self,size:u64)->Option<u64> {
        if size==0 {return None;}
        let bytes=span(size)?;
        let base=if let Some(n)=self.free.iter().position(|&(_,len)|len>=bytes) {
            let (at,len)=self.free.remove(n);if len>bytes {self.free.push((at+bytes,len-bytes));}at
        } else {let at=self.next;let end=at.checked_add(bytes)?;if end>END {return None;}self.next=end;at};
        base.checked_add(PREFIX)
    }
    pub(super) fn release(&mut self,address:u64,size:u64) {
        let base=address.checked_sub(PREFIX).expect("GPU domain prefix");let bytes=span(size).expect("GPU domain size");
        assert!(base>=START && base%BAND==0 && base.checked_add(bytes).is_some_and(|end|end<=self.next));
        self.free.push((base,bytes));self.free.sort_unstable();let mut merged:Vec<(u64,u64)>=Vec::new();
        for (at,len) in self.free.drain(..) {if let Some(last)=merged.last_mut() {assert!(last.0+last.1<=at,"double-freed GPU domain");if last.0+last.1==at {last.1+=len;continue;}}merged.push((at,len));}self.free=merged;
    }
}
#[cfg(test)] mod tests {
    use super::*;
    #[test] fn domains_guard_high_word_windows_and_reuse_finalized_spans() {
        let mut a=Arena::default();let v=(0..60).map(|_|a.reserve(64*1024*1024).unwrap()).collect::<Vec<_>>();
        for pair in v.windows(2) {assert!(pair[1]<(1<<40));assert_eq!(pair[1]-pair[0],BAND);let p=nxshader::WriteWindowFootprint{storage_slot:0,carry:true,offset:0x7fffff,bytes:16,masked_high:false};for (at,len) in p.windows(pair[0]) {assert!(at+len<pair[1]);}}
        a.release(v[2],64*1024*1024);a.release(v[3],64*1024*1024);assert_eq!(a.reserve(BAND),Some(v[2]));
        let n=a.next;assert_eq!(a.reserve(u64::MAX),None);assert_eq!(a.next,n);assert_eq!(a.reserve(0),None);
    }
}
