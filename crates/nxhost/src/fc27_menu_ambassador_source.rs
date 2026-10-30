//! Isolated experiment: connect the selected original ambassador record to an
//! otherwise empty 3D-menu provider result. No invented player or asset IDs.
//! Leaves the original provider's construction, strings, vectors, defaults,
//! return value, and every later scene/preparation callback in charge.
use anyhow::{Result,ensure};

// Original3058600..14 copies PlayerId54 -> runtime0 and TeamId48 -> runtime8.
// Original30585d0,305860c,3058668 and30586a8 map the optional overrides.
// Original5cac8f8..99c copies these cached words into reflected PlayerInfo.
const FIELDS:[(usize,usize);8]=[
    (0,0),(8,4),(0x18,0x2c),(0x1c,0x30),(0x68,0x34),
    (0x80,0x40),(0x84,0x44),(0xc0,0x48),
];
const OUTPUT_BYTES:usize=0x910;
const SOURCE_BYTES:usize=0xc4;
fn copy(at:u64,out:&mut[u8])->bool{
    crate::input::test_control::config_inspection::snapshot(at,out)
}
fn qword(at:u64)->Option<u64>{let mut b=[0;8];copy(at,&mut b).then(||u64::from_le_bytes(b))}
fn integer(bytes:&[u8],at:usize)->Option<i32>{Some(i32::from_le_bytes(bytes.get(at..at+4)?.try_into().ok()?))}
fn selected_in_array(start:u64,end:u64,selected:u64)->bool{
    start!=0&&end>start&&(end-start)%0x180==0&&(end-start)/0x180<=64
        &&selected>=start&&selected<end&&(selected-start)%0x180==0
}
fn plan(output:&[u8],source:&[u8])->Option<[(usize,i32);8]>{
    if output.len()!=OUTPUT_BYTES||source.len()!=SOURCE_BYTES||source[0x24]!=1{return None;}
    // All eleven cached entries must have the original invalid-ID sentinel.
    if (0..11).any(|i|integer(output,0x10+i*0xd0)!=Some(-1)){return None;}
    if !(1..=10_000_000).contains(&integer(source,0)?)||!(1..=1_000_000).contains(&integer(source,8)?){return None;}
    let mut result=[(0,0);8];
    for (slot,(from,to))in result.iter_mut().zip(FIELDS){
        let value=integer(source,from)?;
        if value< -1{return None;}
        *slot=(0x10+to,value);
    }
    Some(result)
}
pub(crate) fn verify(base:u64)->Result<()>{
    if !enabled(){return Ok(());}
    ensure!(std::env::var("NX_FC27_MENU_PRIME_PROBE").is_ok_and(|v|v=="1"),"ambassador source experiment requires isolated original sequence probe");
    for (at,expected)in [
        (0x3058600,0xb90103e8294a21e9),(0x3058608,0xb9011be9b94049e8),
        (0x3058668,0xb9016be8b9404d28),(0x30586a8,0xb90183e8b9406928),
        (0x30586b0,0xb90187e8b9406528),(0x30586b8,0xb901c3e8b9405928),
        (0x5cac8fc,0xb94002e8b9008728),(0x5cac904,0xb94006e8b9006b28),
        (0x5cac910,0xb9006328b9402ee8),(0x5cac918,0xb9006f28b94032e8),
    ]{
        ensure!(crate::diag::read_words::<1>(base+at)==Some([expected]),"ambassador source experiment: original conversion guard mismatch at{at:#x}");
    }
    ensure!(crate::diag::read_words::<4>(base+0xc8deb38)==Some([base+0xc3a4873,0x6225b6fc,0x68,base+0xdea7d00])
        &&crate::diag::read_words::<4>(base+0xc8dec78)==Some([base+0xc3a4865,0x3c331dc6,0x84,base+0xdea7d00]),
        "ambassador source experiment: original PlayerId/TeamId reflection mismatch");
    Ok(())
}
fn enabled()->bool{std::env::var("NX_FC27_MENU_AMBASSADOR_SOURCE").is_ok_and(|v|v=="1")}
fn output_block(state:u32,perm:u32)->bool{
    matches!(state,crate::mem::state::NORMAL|crate::mem::state::ALIAS|crate::mem::state::STACK)
        &&(perm&(crate::mem::perm::R|crate::mem::perm::W))==(crate::mem::perm::R|crate::mem::perm::W)
}
fn writable(at:u64,len:usize)->bool{
    let Some(end)=at.checked_add(len as u64)else{return false;};
    // Original methods called through the compatibility stubs can construct
    // this output on the current GuestThread's owned Vec-backed gate stack.
    // That host allocation is intentionally outside the guest VAMM map.
    if crate::guest::try_current().is_some_and(|g|g.owns_gate_stack_span(at,len)){return true;}
    let map=crate::kernel::aspace().map.lock().unwrap();let mut cursor=at;
    while cursor<end{
        let(start,size,block)=map.query(cursor);
        let Some(next)=start.checked_add(size).filter(|v|*v>cursor)else{return false;};
        if !output_block(block.state,block.perm){return false;}
        cursor=next.min(end);
    }
    true
}
pub(crate) fn apply_empty_result(base:u64,output:u64,returned:u64){
    if !enabled()||returned as u32!=1{return;}
    let Some(service)=qword(base+0xdc557e8).filter(|v|*v!=0)else{return;};
    let Some(start)=qword(service+0x20)else{return;};let Some(end)=qword(service+0x28)else{return;};
    let Some(selected)=qword(service+0x1b80)else{return;};
    if !selected_in_array(start,end,selected){eprintln!("[mainmenu-source] declined: selected record not in bounded original array");return;}
    let mut source=[0;SOURCE_BYTES];let mut before=[0;OUTPUT_BYTES];
    if !copy(selected,&mut source)||!copy(output,&mut before){return;}
    let Some(changes)=plan(&before,&source)else{eprintln!("[mainmenu-source] declined: populated result or invalid original source");return;};
    if !writable(output,OUTPUT_BYTES){eprintln!("[mainmenu-source] declined: caller output is not a readable/writable guest data or stack span");return;}
    // Only integer words in the caller-owned, originally constructed output.
    // No writes to the service/cache, original asset, strings or pointer fields.
    for (offset,value)in changes{
        unsafe{((output+offset as u64)as*mut i32).write_unaligned(value);}
    }
    eprintln!("[mainmenu-source] supplied current original PlayerId={} TeamId={} to one empty entry; original defaults/return and completion gates retained",changes[0].1,changes[1].1);
}

#[cfg(test)]mod tests{
    use super::*;
    #[test]fn preserve_opaque_fields_and_pointer_lifetimes(){
        let mut output=[0x5au8;OUTPUT_BYTES];for i in 0..11{output[0x10+i*0xd0..0x14+i*0xd0].copy_from_slice(&(-1i32).to_le_bytes());}
        let before=output;let mut source=[0u8;SOURCE_BYTES];source[0x24]=1;
        for ((from,_),value)in FIELDS.into_iter().zip([211110i32,52,-1,-1,-1,156,-1,-1]){source[from..from+4].copy_from_slice(&value.to_le_bytes());}
        let changes=plan(&output,&source).unwrap();for (at,v)in changes{output[at..at+4].copy_from_slice(&v.to_le_bytes());}
        for i in 0..OUTPUT_BYTES{if !changes.iter().any(|(at,_)|(*at..*at+4).contains(&i)){assert_eq!(before[i],output[i]);}}
        assert_eq!(integer(&output,0x10),Some(211110));assert_eq!(integer(&output,0x14),Some(52));
        assert_eq!(integer(&output,0x50),Some(156));
        assert!(plan(&output,&source).is_none()); // A populated result is never overwritten.
        source[0x24]=0;assert!(plan(&before,&source).is_none());
        source[0x24]=1;source[..4].copy_from_slice(&(-1i32).to_le_bytes());assert!(plan(&before,&source).is_none());
    }
    #[test]fn reject_unbounded_or_misaligned_selected_sources(){
        assert!(selected_in_array(0x10000,0x10780,0x10300));
        for (start,end,selected)in [(0,0,0),(0x10000,0x10000,0x10000),(0x10000,0x10780,0x10304),(0x10000,0x10780,0x10780),(0x10000,0x20000,0x10000)]{assert!(!selected_in_array(start,end,selected));}
    }
    #[test]fn accept_original_stack_output_but_reject_code_and_read_only_pages(){
        use crate::mem::{state,perm};
        for state in [state::STACK,state::NORMAL,state::ALIAS]{
            assert!(output_block(state,perm::R|perm::W));
            assert!(!output_block(state,perm::R));assert!(!output_block(state,0));
        }
        assert!(!output_block(state::CODE,perm::R|perm::W));
        let guest=crate::guest::GuestThread::new(101,0);
        assert!(guest.owns_gate_stack_span(guest.gate_stack_top-OUTPUT_BYTES as u64,OUTPUT_BYTES));
        assert!(!guest.owns_gate_stack_span(guest.gate_stack_top+0x1000,4));
        assert!(!guest.owns_gate_stack_span(0,OUTPUT_BYTES));
        assert!(!guest.owns_gate_stack_span(u64::MAX-1,OUTPUT_BYTES));
    }
}
