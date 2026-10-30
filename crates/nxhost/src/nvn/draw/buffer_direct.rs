// Sampled-buffer views may carry an immutable retained GPU pointer when the
// whole logical view fits one physical chunk. Storage images never use tags.
use crate::gfx;
use std::sync::{Arc,LazyLock};
pub(super) fn enabled()->bool {
    static ON:LazyLock<bool>=LazyLock::new(||std::env::var_os("NX_DIRECT_BUFFER_READ").is_some());
    super::shaders::buffer_read_enabled() && *ON
}
pub(super) fn metadata(pools:&[Arc<gfx::PoolMemory>],heap:Option<&gfx::HeapBuffer>,meta:[u32;4],format:[u32;4])->Option<[u32;4]> {
    let address=meta[0] as u64|((meta[1] as u64)<<32);
    let bytes=(format[0] as u64).checked_mul(format[1] as u64)?/8;
    if !(1..=16).contains(&bytes)||meta[2]==0||!(1..=48).contains(&(meta[3]&255))||meta[3]&0x80000000!=0{return None;}
    let length=bytes.checked_mul(meta[2] as u64)?;let end=address.checked_add(length)?;
    // gmem_gpu gives the heap priority. Even a partial heap intersection must
    // fit its single chunk; do not substitute a pool pointer across that edge.
    let range=if let Some(h)=heap.filter(|h|address<h.base.saturating_add(h.size)&&end>h.base) {h.buffer_range(address,length)}else{gfx::pool_buffer_range(pools,address,length)};
    let (chunk,offset)=range?;let gpu=chunk.gpu.checked_add(offset as u64)?;gpu.checked_add(length)?;
    if gpu==0 || (gpu^address)&3!=0 {return None;}
    Some([gpu as u32,(gpu>>32) as u32,meta[2],meta[3]|0x80000000])
}

#[cfg(test)]mod tests {
    // Real mappings, chunk boundaries and GPU alias lifetimes are covered in
    // buffer_read_tests and the existing buffer-texture read/write GPU test.
    #[test]fn zero_or_invalid_descriptors_never_become_direct_pointers(){
        assert!(super::metadata(&[],None,[0;4],[0;4]).is_none());
        assert!(super::metadata(&[],None,[0,1,0,46],[32,4,2,1]).is_none());
        assert!(super::metadata(&[],None,[0,1,1,49],[32,4,2,1]).is_none());
    }
}
