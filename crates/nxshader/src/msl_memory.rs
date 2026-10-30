// Included only by translations requesting the contiguous multiword path.
// Keep the common prelude/default translations byte-identical. Named AOT
// functions can share these helpers in one library through the include guard.
pub(super) const MULTIWORD:&str=r#"
#ifndef NX_MULTIWORD_GMEM_HELPERS
#define NX_MULTIWORD_GMEM_HELPERS
static inline ulong nx_gmem_contiguous(constant Res& res,ulong ga,ulong length) {
    if(length==0ul || ga+length-1ul<ga) return 0ul;
    ulong offset=ga-res.heap_base;
    if(offset<res.heap_size && res.heap_chunk_size!=0ul) {
        if(length>res.heap_size-offset || length>res.heap_chunk_size-offset%res.heap_chunk_size) return 0ul;
        return res.heap_gpu[offset/res.heap_chunk_size]+offset%res.heap_chunk_size;
    }
#if NX_PAGED_GMEM
    if(length<=4096ul-(ga&4095ul)) {
        ulong pointer=gmem_page(res,ga);
        if(pointer!=0ul) return pointer;
    }
#endif
    ulong lo=0ul,hi=res.pool_span_count;
    while(lo<hi) {
        ulong mid=(lo+hi)/2ul;
        ulong base=res.pool_spans[mid*3ul],size=res.pool_spans[mid*3ul+1ul];
        if(ga<base) hi=mid;
        else if(ga-base>=size) lo=mid+1ul;
        else return length<=size-(ga-base)?res.pool_spans[mid*3ul+2ul]+ga-base:0ul;
    }
    return 0ul;
}
static inline uint2 nx_gmem_ld2(constant Res& res,ulong ga) {
    ulong pointer=nx_gmem_contiguous(res,ga,8ul);
    if(pointer!=0ul) {device const uint* p=reinterpret_cast<device const uint*>(pointer);return uint2(p[0],p[1]);}
    return uint2(gmem(res,ga)[0],gmem(res,ga+4ul)[0]);
}
static inline uint4 nx_gmem_ld4(constant Res& res,ulong ga) {
    ulong pointer=nx_gmem_contiguous(res,ga,16ul);
    if(pointer!=0ul) {device const uint* p=reinterpret_cast<device const uint*>(pointer);return uint4(p[0],p[1],p[2],p[3]);}
    return uint4(gmem(res,ga)[0],gmem(res,ga+4ul)[0],gmem(res,ga+8ul)[0],gmem(res,ga+12ul)[0]);
}
#endif
"#;
