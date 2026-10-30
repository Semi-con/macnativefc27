// Explicit Maxwell convergence tokens for compute programs. All lanes stay in
// the outer loop until the entire SIMD group exits; a lane at SYNC/BRK waits
// for the other surviving holders of that token before running its join block.
pub const INIT:&str=r#"
    bool live=true, waiting=false;
    uint token_sp=0u, simt_steps=0u;
    uint token_id[32], token_target[32], token_mask[32], token_kind[32];
    for(uint n=0u;n<32u;n++) {token_id[n]=0u;token_target[n]=0u;token_mask[n]=0u;token_kind[n]=0u;}
"#;
pub const HEADER:&str=r#"
    while(simd_any(live)) {
        if(++simt_steps>100000u) {if(live && shader_fault(res,64u|(uint(waiting)<<16u)|(token_sp<<20u),pc)) {@FAULT_CONTEXT@}break;}
        simdgroup_barrier(mem_flags::mem_threadgroup);
        uint waiters=uint((simd_vote::vote_t)simd_ballot(live && waiting));
        while(waiters!=0u) {
            uint leader=ctz(waiters);
            uint top=token_sp==0u?0u:token_sp-1u;
            uint id=simd_shuffle(token_id[top],ushort(leader));
            uint target=simd_shuffle(token_target[top],ushort(leader));
            uint mask=simd_shuffle(token_mask[top],ushort(leader));
            bool holder=false;
            for(uint n=0u;n<32u;n++) {
                if(n<token_sp && token_id[n]==id && token_target[n]==target && token_mask[n]==mask) holder=true;
            }
            uint participants=uint((simd_vote::vote_t)simd_ballot(live && holder));
            bool arrived=live && waiting && token_sp!=0u && token_id[top]==id && token_target[top]==target && token_mask[top]==mask;
            uint arrivals=uint((simd_vote::vote_t)simd_ballot(arrived));
            if((arrivals&participants)==participants && arrived) {token_sp--;waiting=false;}
            waiters&=~arrivals;
        }
        if(!live || waiting) continue;
        switch(pc) {
"#;
pub const TAIL:&str=r#"
        default: shader_fault(res,2u,pc);live=false;continue;
        }
    }
    return epilogue();
}
"#;
