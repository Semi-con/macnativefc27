//! Bounded observation of the original render-time expression callbacks.
//! Registration receives a delegating stub. A separate isolated experiment can
//! resolve absent shadow-view time through its original parent view binding.
//! Both modes default off and are restricted to the isolated test.
use anyhow::{ensure, Result};
use std::sync::{OnceLock, atomic::{AtomicBool,AtomicU32, Ordering::Relaxed}};
use crate::stubs::Args;

const SITES: [(u64, [u32; 2], u64, &str); 3] = [
    (0x9993540, [0xd503201f, 0x10003ce2], 0x9993ce0, "renderTime"),
    (0x9993580, [0xd503201f, 0x10003ae2], 0x9993ce0, "prevRenderTime"),
    (0x99935c4, [0xd503201f, 0x10003cc2], 0x9993d60, "deltaRenderTime"),
];
static BASE: OnceLock<u64> = OnceLock::new();
static CALLS: [AtomicU32; 3] = [const {AtomicU32::new(0)}; 3];
static MISSING: [AtomicU32; 3] = [const {AtomicU32::new(0)}; 3];
static PARENT_TIME:AtomicBool=AtomicBool::new(false);
static PARENT_CALLS:AtomicU32=AtomicU32::new(0);

// ADRP x2, literal-page; LDR x2, [x2, literal-offset]. The original
// registration then stores the callback using its normal hash table path.
fn literal_load(pc: u64, literal: u64) -> Result<[u32; 2]> {
    ensure!(pc % 4 == 0 && literal % 8 == 0, "unaligned menu time literal");
    let pages = ((literal & !4095) as i128 - (pc & !4095) as i128) >> 12;
    ensure!((-(1 << 20)..(1 << 20)).contains(&pages), "menu time literal out of ADRP range");
    let immediate = (pages as i64 as u32) & 0x1fffff;
    Ok([0x90000002 | ((immediate & 3) << 29) | ((immediate >> 2) << 5),
        0xf9400042 | (((literal as u32 & 4095) / 8) << 10)])
}
fn snapshot(address: u64, bytes: &mut [u8]) -> bool {
    crate::input::test_control::config_inspection::snapshot(address, bytes)
}
fn word(address: u64) -> Option<u64> {
    let mut b = [0; 8];
    snapshot(address, &mut b).then(|| u64::from_le_bytes(b))
}
fn interned_descriptor(base:u64,id:u16)->Option<[u64;2]> {
    // Original 8dd5d68..8dd5d9c resolves a source-set member ID to its
    // registered 16-byte descriptor. The descriptor's low16 are not that ID.
    let key=id as u64+16;
    let leading=key.leading_zeros();
    let chunk=59u32.checked_sub(leading)?;
    let registry=word(base+0xd91f630).filter(|v|*v!=0)?;
    let table=word(registry+0x328+chunk as u64*8).filter(|v|*v!=0)?;
    let index=(0x7fffffffffffffffu64>>leading)&key;
    let at=table.checked_add(index*16)?;
    Some([word(at)?,word(at+8)?])
}
fn original_context(base:u64,args:&Args) {
    // The existing stub ABI places saved guest FP/LR immediately before Args.
    // Inspect via the OS copy helper, then follow only the guarded original
    // a1d7660 -> a1d95b0 call chain; no live register or guest data is edited.
    let at=args as *const Args as u64;
    let frame=word(at-16); let lr=word(at-8);
    let caller=frame.and_then(|v|word(v+8));
    eprintln!("[menu-time-context] original_lr={lr:x?} frame_caller={caller:x?} gprs8={:x?}",args.0);
    if lr!=Some(base+0xa1d76c8)||caller!=Some(base+0xa1d969c){return;}
    let frame=frame.unwrap();
    let object=word(frame+0x58).unwrap_or(0);
    let context=word(frame+0x48).unwrap_or(0);
    let program_entry=word(frame+0x18).unwrap_or(0);
    let definition=word(program_entry).unwrap_or(0);
    let mut header=[0u8;0x38];
    if definition==0||!snapshot(definition,&mut header){return;}
    let source_count=u16::from_le_bytes(header[0x10..0x12].try_into().unwrap()) as usize;
    let callback_count=u16::from_le_bytes(header[0x12..0x14].try_into().unwrap());
    let descriptors=u64::from_le_bytes(header[0x20..0x28].try_into().unwrap());
    let targets=u64::from_le_bytes(header[0x28..0x30].try_into().unwrap());
    let sources=(0..source_count.min(16)).map(|i|[word(descriptors+i as u64*16),word(descriptors+i as u64*16+8)]).collect::<Vec<_>>();
    let object_words=(0..11).map(|i|word(object+i*8)).collect::<Vec<_>>();
    let context_words=[0,0xa8,0xb0,0xc8,0xd0,0xd8,0xe8,0xf0,0x148].map(|offset|word(context+offset));
    eprintln!("[menu-time-context] object={object:#x} context={context:#x} program_entry={program_entry:#x} definition={definition:#x} header={header:02x?} source_count={source_count} callback_count={callback_count} source_descriptors={sources:x?} object_words={object_words:x?} context_selected_words={context_words:x?}");
    for i in 0..source_count.min(512) {
        let hash=word(descriptors+i as u64*16+8);
        if matches!(hash,Some(0x3e0d33cb9a2cb51d|0x04fcba25316fc68a)) {
            eprintln!("[menu-time-target] descriptor_index={i} hash={hash:x?} target={:x?}",word(targets+i as u64*8));
        }
    }
    let outer_frame=word(frame).unwrap_or(0);
    let outer_lr=word(outer_frame+8);
    let render_frame=word(outer_frame).unwrap_or(0);
    let render_lr=word(render_frame+8);
    let roots=word(frame+0x40).unwrap_or(0); // saved caller X22, original X5
    let mut count=[0;4];
    let count=snapshot(outer_frame.saturating_sub(0x3c),&mut count).then(||u32::from_le_bytes(count));
    let source_handle=word(program_entry+8).unwrap_or(0);
    let handle_words=(0..6).map(|i|word(source_handle+i*8)).collect::<Vec<_>>();
    eprintln!("[menu-time-roots] outer_lr={outer_lr:x?} render_lr_rva={:x?} roots={roots:#x} count={count:?} source_handle={source_handle:#x} handle_words={handle_words:x?}",render_lr.and_then(|v|v.checked_sub(base)));
    if render_lr==Some(base+0xb33b860) {
        // 98ee310 saves its caller's X22 at FP+40; b33b860's X22 is the
        // original view whose current/previous source roots it passes.
        let view=word(render_frame+0x40).unwrap_or(0);
        let parent=word(view).unwrap_or(0);
        for (label,address) in [("view",view),("parent",parent)] {
            if address==0 {continue;}
            let fields=[0,8,0x10,0x18,0x1bd0,0x1bd8,0x1be0,0x1be8,0x1d90].map(|offset|word(address+offset));
            let name=word(address+8).filter(|v| (base+0xb99b000..base+0xc416000).contains(v));
            let mut bytes=[0;96];
            let name=name.filter(|p|snapshot(*p,&mut bytes)).and_then(|_|bytes.iter().position(|v|*v==0))
                .filter(|len|*len>0&&bytes[..*len].iter().all(|v|(32..127).contains(v)))
                .map(|len|String::from_utf8_lossy(&bytes[..len]).into_owned());
            eprintln!("[menu-time-view] label={label} address={address:#x} fields={fields:x?} original_static_name={name:?}");
        }
    }
    for i in 0..count.unwrap_or(0).min(8) {
        let root=word(roots+i as u64*8).unwrap_or(0);
        let root_words=(0..6).map(|i|word(root+i*8)).collect::<Vec<_>>();
        let set=word(root+8).unwrap_or(0)&0xffffffffffff;
        let set_words=(0..6).map(|i|word(set+i*8)).collect::<Vec<_>>();
        eprintln!("[menu-time-roots] index={i} root={root:#x} root_words={root_words:x?} set={set:#x} set_words={set_words:x?}");
        let groups=(word(set+0x18).unwrap_or(0)>>32)&0xffff;
        if groups>8 {continue;}
        for group in 0..groups {
            let packed=word(set+0x20+group*8).unwrap_or(0);
            let pointer=(packed&0x7ffffffffff)<<4;
            let key=word(pointer+0x20).unwrap_or(0);
            let mut count=[0;2];
            if !snapshot(key+0x14,&mut count){continue;}
            let source_count=u16::from_le_bytes(count) as usize &0x7fff;
            if source_count>512 {continue;}
            let mut bytes=vec![0;source_count*2];
            if source_count==0||!snapshot(key+0x16,&mut bytes){continue;}
            let ids=bytes.chunks_exact(2).map(|b|u16::from_le_bytes(b.try_into().unwrap())).collect::<Vec<_>>();
            let mut failures=0;
            let matches=ids.into_iter().filter_map(|id| {
                let descriptor=interned_descriptor(base,id);
                if descriptor.is_none(){failures+=1;}
                descriptor.filter(|d|matches!(d[1],0x3e0d33cb9a2cb51d|0x04fcba25316fc68a)).map(|d|(id,d))
            }).collect::<Vec<_>>();
            eprintln!("[menu-time-root-members] root_index={i} group={group} source_count={source_count} descriptor_read_failures={failures} matched_time_descriptors={matches:x?}");
        }
    }
}
unsafe fn delegate(address: u64, args: &Args) {
    // Original callbacks take three pointer arguments, write through X2 and
    // return void. No floating-point argument or return is reinterpreted.
    let call: extern "C" fn(u64, u64, u64) = unsafe {std::mem::transmute(address as usize)};
    call(args.0[0], args.0[1], args.0[2]);
}
unsafe fn delegate_private(address:u64,args:&Args,index:usize,source:u64) {
    let mut pointers=[0u64;256];pointers[index]=source;
    let mut forwarded=Args(args.0,args.1);forwarded.0[0]=pointers.as_ptr()as u64;
    unsafe{delegate(address,&forwarded);}
}
fn parent_time_source(base:u64,slot:usize,args:&Args,index:u32)->Option<u64> {
    if !PARENT_TIME.load(Relaxed)||index>=256
        ||crate::fc27_menu_event_observation::master_readiness()!=(true,true){return None;}
    let at=args as *const Args as u64;
    let frame=word(at.checked_sub(16)?)?;
    if word(at-8)!=Some(base+0xa1d76c8)||word(frame+8)!=Some(base+0xa1d969c){return None;}
    let outer_frame=word(frame)?;
    if word(outer_frame+8)!=Some(base+0x98eee84){return None;}
    let render_frame=word(outer_frame)?;
    if word(render_frame+8)!=Some(base+0xb33b860){return None;}
    let view=word(render_frame+0x40).filter(|v|*v!=0)?;
    let parent=word(view).filter(|v|*v!=0&&*v!=view)?;
    let name=word(view+8).filter(|v|(base+0xb99b000..base+0xc416000).contains(v))?;
    let mut bytes=[0;22];
    if !snapshot(name,&mut bytes)||bytes!=*b"WorldShadowSliceAtlas\0"{return None;}
    let mut kind=[0;4];
    if !snapshot(view+0x10,&mut kind)||u32::from_le_bytes(kind)!=3{return None;}
    let empty=base+0xd91e240;
    if ![0x1bd0,0x1bd8,0x1be0,0x1be8].into_iter().all(|off|word(view+off)==Some(empty)){return None;}
    let root=word(parent+0x1bd0).filter(|v|*v!=0&&*v!=empty)?;
    let program_entry=word(frame+0x18).filter(|v|*v!=0)?;
    let definition=word(program_entry).filter(|v|*v!=0)?;
    let mut header=[0;0x38];if !snapshot(definition,&mut header){return None;}
    let count=u16::from_le_bytes(header[0x10..0x12].try_into().ok()?) as usize;
    if count==0||count>512{return None;}
    let descriptors=u64::from_le_bytes(header[0x20..0x28].try_into().ok()?);
    let targets=u64::from_le_bytes(header[0x28..0x30].try_into().ok()?);
    let global=base+if slot==1{0xda0e490}else{0xda0e440};
    let expected=[word(global)?,word(global+8)?];
    let mut found=None;
    for i in 0..count {
        if [word(descriptors+i as u64*16),word(descriptors+i as u64*16+8)]!=expected.map(Some){continue;}
        let target=word(targets+i as u64*8)?;
        if (target>>32)&0xffff!=index as u64 {return None;}
        found=Some(i);break;
    }
    let descriptor_index=found?;
    let handle=word(program_entry+8).filter(|v|*v!=0)?;
    let key=word(handle+8).filter(|v|*v!=0)?;
    // Reuse the original typed binding resolver with one real parent root.
    // It retains its normal source matching and cache semantics. No frame
    // duration, renderer matrices, readiness flags or guest fields are written.
    let roots=[root];
    let resolve:extern "C" fn(u64,u32,u64)->u64=unsafe{std::mem::transmute((base+0x8dd48a0)as usize)};
    let mapping=resolve(key,1,roots.as_ptr()as u64);
    if mapping==0{return None;}
    for i in 0..count+1 {
        let address=mapping.checked_add(i as u64*12)?;
        let mut entry=[0;12];if !snapshot(address,&mut entry){return None;}
        let root_index=u16::from_le_bytes(entry[..2].try_into().ok()?);
        if root_index==0xffff {return None;}
        let mapped_descriptor=u16::from_le_bytes(entry[6..8].try_into().ok()?) as usize;
        if mapped_descriptor!=descriptor_index {continue;}
        if root_index!=0 {return None;}
        let pointer_offset=u16::from_le_bytes(entry[2..4].try_into().ok()?) as u64;
        let value_offset=u32::from_le_bytes(entry[8..12].try_into().ok()?) as u64;
        if pointer_offset>0x80||pointer_offset%8!=0{return None;}
        let values=word(root.checked_add(pointer_offset)?).filter(|v|*v!=0)?;
        let value=values.checked_add(value_offset)?;
        let mut data=[0;24];if !snapshot(value,&mut data){return None;}
        if !data.chunks_exact(4).all(|b|f32::from_le_bytes(b.try_into().unwrap()).is_finite()){return None;}
        let calls=PARENT_CALLS.fetch_add(1,Relaxed)+1;
        if calls<=8||(calls<=1024&&calls.is_power_of_two()) {
            eprintln!("[menu-shadow-time] operation={} count={calls} child={view:#x} parent={parent:#x} descriptor_index={descriptor_index} original_source={value:#x}; original parent binding and getter, private input table only",SITES[slot].3);
        }
        return Some(value);
    }
    None
}
fn dispatch(tag: u64, args: &Args) -> u64 {
    let slot = tag as usize;
    let count = CALLS[slot].fetch_add(1, Relaxed) + 1;
    let base = *BASE.get().unwrap();
    let mut index = [0; 4];
    let index = snapshot(args.0[1], &mut index).then(|| u32::from_le_bytes(index));
    let source = index.filter(|v| *v < 256)
        .and_then(|v| args.0[0].checked_add(v as u64 * 8)).and_then(word);
    let missing=if source==Some(0){MISSING[slot].fetch_add(1,Relaxed)+1}else{0};
    if count <= 4 || (count <= 1024 && count.is_power_of_two()) || (missing>0&&missing<=2) {
        let mut values = [0; 24];
        let values = source.filter(|v| *v != 0).filter(|v| snapshot(*v, &mut values))
            .map(|_| std::array::from_fn::<_, 6, _>(|i| f32::from_le_bytes(values[i*4..i*4+4].try_into().unwrap())));
        eprintln!("[menu-time] operation={} count={count} table={:#x} parameters={:#x} output={:#x} index={index:?} source={source:x?} floats6={values:?}",
            SITES[slot].3, args.0[0], args.0[1], args.0[2]);
        if count == 1 || missing==1 {
            let pointers = (0..32).map(|i| args.0[0].checked_add(i*8).and_then(word)).collect::<Vec<_>>();
            let descriptors = [0xda0e440, 0xda0e490].map(|rva| [word(base+rva), word(base+rva+8)]);
            eprintln!("[menu-time] operation={} sources32={pointers:x?} original_time_descriptors={descriptors:x?}", SITES[slot].3);
            original_context(base,args);
        }
    }
    if source==Some(0) {
        if let Some(value)=index.and_then(|i|parent_time_source(base,slot,args,i)) {
            // Only this callback receives a private pointer table. The original
            // getter computes its complete output using the original parent
            // world time; the caller's table and all other inputs stay intact.
            unsafe {delegate_private(base+SITES[slot].2,args,index.unwrap() as usize,value);}
            return 0;
        }
    }
    unsafe {delegate(base + SITES[slot].2, args);}
    0
}

pub(crate) fn apply(modules: &[crate::loader::Module]) -> Result<()> {
    let parent_time=std::env::var("NX_FC27_MENU_SHADOW_PARENT_TIME").is_ok_and(|v|v=="1");
    if !std::env::var("NX_FC27_MENU_TIME_TRACE").is_ok_and(|v| v == "1") {
        ensure!(!parent_time,"shadow parent-time experiment requires menu time trace");return Ok(());
    }
    ensure!(std::env::var_os("NX_TEST_CONTROL").is_some()
        && std::env::var("NX_FC27_VIDEO_TRACE").is_ok_and(|v| v == "1")
        && std::env::current_exe()?.file_name().is_some_and(|v| v == "FC27-test"),
        "menu time observation requires isolated test executable and video trace");
    let m = modules.iter().find(|m| m.name == "main").ok_or_else(|| anyhow::anyhow!("missing FC27 main"))?;
    let id: String = m.module_id.iter().map(|v| format!("{v:02x}")).collect();
    ensure!(id == "fb81983aa1cc3a2e97402497e2ca652800000000000000000000000000000000",
        "menu time observation unknown game build");
    ensure!(0x9993d78 <= m.text && nxabi::keypage::FC27_TIME_CALLBACKS[2] as u64 + 8 <= nxabi::keypage::SIZE,
        "menu time observation invalid layout");
    for (index, &(at, original, _, _)) in SITES.iter().enumerate() {
        let actual = unsafe {((m.base + at) as *const [u32; 2]).read_unaligned()};
        ensure!(actual == original, "menu time observation registration guard mismatch");
        ensure!(word(m.base + m.keypage + nxabi::keypage::FC27_TIME_CALLBACKS[index] as u64) == Some(0),
            "menu time observation literal slot already used");
    }
    ensure!(unsafe {((m.base+0x9993ce0) as *const [u32;2]).read_unaligned()} == [0xb9400028,0xf8687808]
        && unsafe {((m.base+0x9993d6c) as *const [u32;2]).read_unaligned()} == [0xb9400028,0xf8687813],
        "menu time observation callback guard mismatch");
    if parent_time {
        ensure!(std::env::var("NX_FC27_MENU_AMBASSADOR_SOURCE").is_ok_and(|v|v=="1")
            &&std::env::var("NX_FC27_MENU_PRIME_PROBE").is_ok_and(|v|v=="1"),"shadow parent-time experiment requires isolated original 3D-menu source/probe");
        ensure!(unsafe{((m.base+0x8dd48a0)as*const[u32;2]).read_unaligned()}==[0xa9ba7bfd,0xa9016ffc]
            &&unsafe{((m.base+0xa1d7898)as*const[u32;2]).read_unaligned()}==[0x79800b68,0xb100051f],"shadow parent-time original resolver/target layout mismatch");
    }
    let patches = SITES.iter().enumerate().map(|(i,s)| literal_load(m.base+s.0,
        m.base+m.keypage+nxabi::keypage::FC27_TIME_CALLBACKS[i] as u64)).collect::<Result<Vec<_>>>()?;
    ensure!(BASE.set(m.base).is_ok(), "menu time observation already installed");
    PARENT_TIME.store(parent_time,Relaxed);
    // All guards pass before mutations; loader will apply R/RX protection and
    // flush instruction caches before any guest execution starts.
    for (i, &(at, _, _, _)) in SITES.iter().enumerate() {
        let stub = crate::stubs::make(dispatch, i as u64);
        unsafe {
            ((m.base+m.keypage+nxabi::keypage::FC27_TIME_CALLBACKS[i] as u64) as *mut u64).write(stub);
            ((m.base+at) as *mut [u32;2]).write_unaligned(patches[i]);
        }
    }
    eprintln!("[menu-time] observing three original expression callbacks; bounded reads; isolated_shadow_parent_time={parent_time}");
    Ok(())
}

#[cfg(test)] mod tests {
    use super::*;
    #[test] fn literal_load_preserves_register_and_roundtrips_signed_page_range() {
        let pc = 0x7109993540;
        for delta in [-(1i64<<20), -1, 0, 1, (1<<20)-1] {
            let literal = ((pc & !4095) as i64 + (delta<<12)) as u64 + 0x68;
            let [a,l] = literal_load(pc,literal).unwrap();
            assert_eq!(a & 0x9f00001f,0x90000002);
            let raw = ((a>>29)&3)|(((a>>5)&0x7ffff)<<2);
            let signed = ((raw as i32)<<11)>>11;
            assert_eq!(signed as i64,delta);
            assert_eq!(l & 0xffc003ff,0xf9400042);
            assert_eq!(((l>>10)&4095)*8,literal as u32&4095);
        }
        assert!(literal_load(pc,pc+1).is_err());
        assert!(literal_load(pc,(pc&!4095)+(1u64<<32)).is_err());
    }
    #[test] fn callback_forwards_three_full_width_pointers_and_original_output_once() {
        static N:AtomicU32=AtomicU32::new(0);
        extern "C" fn original(table:u64,parameters:u64,output:u64) {
            assert_eq!(table,0x7123456789); assert_eq!(parameters,0x7abcdef012);
            N.fetch_add(1,Relaxed);
            unsafe {(output as *mut [u32;4]).write([0x3c888889,0x3c888889,0x3f800000,0x40000000]);}
        }
        let mut output=[0u32;4];
        let args=Args([0x7123456789,0x7abcdef012,output.as_mut_ptr() as u64,0,0,0,0,0],[0;8]);
        unsafe {delegate(original as *const () as usize as u64,&args);}
        assert_eq!(N.load(Relaxed),1);
        assert_eq!(output,[0x3c888889,0x3c888889,0x3f800000,0x40000000]);
    }
    #[test]fn private_source_table_preserves_caller_table_and_full_original_result_once(){
        static N:AtomicU32=AtomicU32::new(0);
        extern "C" fn original(table:u64,index:u64,output:u64){
            N.fetch_add(1,Relaxed);
            unsafe{
                let index=(index as*const u32).read() as usize;
                let source=(table as*const u64).add(index).read();
                (output as*mut[f32;6]).write((source as*const[f32;6]).read());
            }
        }
        let caller=[0u64;256];let index=255u32;let time=[19.5f32,19.5,19.5,0.0167,0.0167,0.0167];let mut output=[0f32;6];
        let args=Args([caller.as_ptr()as u64,&index as*const u32 as u64,output.as_mut_ptr()as u64,0,0,0,0,0],[0;8]);
        unsafe{delegate_private(original as*const()as usize as u64,&args,index as usize,time.as_ptr()as u64);}
        assert_eq!(N.load(Relaxed),1);assert_eq!(output,time);assert_eq!(caller,[0u64;256]);
        assert_eq!(args.0[0],caller.as_ptr()as u64);
    }
}
