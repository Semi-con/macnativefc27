//! Isolated original parent-view source inheritance for empty shadow views.
//! Preserve every original mapping; resolve only absent descriptors through the
//! original binding engine, then delegate the complete original evaluation once.
use anyhow::{ensure,Result};
use std::sync::{OnceLock,atomic::{AtomicU32,Ordering::Relaxed}};
use crate::stubs::Args;
static BASE:OnceLock<u64>=OnceLock::new();
static RESUME:OnceLock<u64>=OnceLock::new();
static CALLS:AtomicU32=AtomicU32::new(0);
const ENTRY:u64=0xa1d7660;

#[repr(C)]
#[derive(Clone,Copy,Debug,PartialEq,Eq)]
struct Mapping{root:u16,pointer_offset:u16,flags:u16,descriptor:u16,value_offset:u32}
const END:Mapping=Mapping{root:0xffff,pointer_offset:0,flags:0,descriptor:0,value_offset:0};
fn snapshot(address:u64,bytes:&mut[u8])->bool{crate::input::test_control::config_inspection::snapshot(address,bytes)}
fn word(address:u64)->Option<u64>{let mut b=[0;8];snapshot(address,&mut b).then(||u64::from_le_bytes(b))}
fn read_map(address:u64,descriptors:usize,roots:usize)->Option<Vec<Mapping>>{
    let mut out=Vec::new();
    for i in 0..=descriptors {
        let mut b=[0;12];if !snapshot(address.checked_add(i as u64*12)?,&mut b){return None;}
        let half=|at|u16::from_le_bytes(b[at..at+2].try_into().unwrap());
        let entry=Mapping{root:half(0),pointer_offset:half(2),flags:half(4),descriptor:half(6),value_offset:u32::from_le_bytes(b[8..12].try_into().unwrap())};
        if entry.root==0xffff{return Some(out);}
        if entry.root as usize>=roots||entry.descriptor as usize>=descriptors{return None;}
        out.push(entry);
    }
    None
}
fn merge(original:&[Mapping],parent:&[Mapping],parent_index:u16)->(Vec<Mapping>,Vec<u16>){
    let mut merged=original.to_vec();let mut added=Vec::new();
    for entry in parent {
        if merged.iter().any(|v|v.descriptor==entry.descriptor){continue;}
        let mut entry=*entry;entry.root=parent_index;added.push(entry.descriptor);merged.push(entry);
    }
    merged.push(END);(merged,added)
}
fn replacement(base:u64,args:&Args)->Option<(Vec<u64>,Vec<Mapping>,Vec<u16>,u64,u64)>{
    // The original frontend's next scene can still render this exact empty
    // shadow child after menu UnPrime returns (run256). Readiness belongs to
    // playback sequencing, not to view-parameter ownership. The view name,
    // type, four empty roots, parent, original call chain and binding layout
    // below determine eligibility throughout that view's lifetime.
    let at=args as*const Args as u64;
    if word(at.checked_sub(8)?)!=Some(base+0xa1d969c){return None;}
    let frame=word(at-16)?;
    if word(frame+8)!=Some(base+0x98eee84){return None;}
    let render=word(frame)?;
    if word(render+8)!=Some(base+0xb33b860){return None;}
    let view=word(render+0x40).filter(|v|*v!=0)?;
    let parent=word(view).filter(|v|*v!=0&&*v!=view)?;
    let name=word(view+8).filter(|v|(base+0xb99b000..base+0xc416000).contains(v))?;
    let mut label=[0;22];let mut kind=[0;4];
    if !snapshot(name,&mut label)||label!=*b"WorldShadowSliceAtlas\0"
        ||!snapshot(view+0x10,&mut kind)||u32::from_le_bytes(kind)!=3{return None;}
    let empty=base+0xd91e240;
    if ![0x1bd0,0x1bd8,0x1be0,0x1be8].into_iter().all(|offset|word(view+offset)==Some(empty)){return None;}
    let parent_root=word(parent+0x1bd0).filter(|v|*v!=0&&*v!=empty)?;
    let mut count=[0;4];if !snapshot(frame.checked_sub(0x3c)?,&mut count){return None;}
    let count=u32::from_le_bytes(count) as usize;if !(1..=16).contains(&count){return None;}
    let mut bytes=vec![0;count*8];if !snapshot(args.0[3],&mut bytes){return None;}
    let mut roots=bytes.chunks_exact(8).map(|b|u64::from_le_bytes(b.try_into().unwrap())).collect::<Vec<_>>();
    if roots.contains(&parent_root){return None;}
    let definition=word(args.0[0]).filter(|v|*v!=0)?;
    let mut descriptor_count=[0;2];if !snapshot(definition+0x10,&mut descriptor_count){return None;}
    let descriptors=u16::from_le_bytes(descriptor_count) as usize;if !(1..=512).contains(&descriptors){return None;}
    let original=read_map(args.0[2],descriptors,count)?;
    let handle=word(args.0[0]+8).filter(|v|*v!=0)?;
    let key=word(handle+8).filter(|v|*v!=0)?;
    let parent_roots=[parent_root];
    let resolve:extern "C" fn(u64,u32,u64)->u64=unsafe{std::mem::transmute((base+0x8dd48a0)as usize)};
    let mapping=resolve(key,1,parent_roots.as_ptr()as u64);
    if mapping==0{return None;}
    let parent_map=read_map(mapping,descriptors,1)?;
    let (mapping,added)=merge(&original,&parent_map,count as u16);
    if added.is_empty(){return None;}
    roots.push(parent_root);
    Some((roots,mapping,added,view,parent))
}
unsafe fn delegate(address:u64,args:&Args){
    let call:extern "C" fn(u64,u64,u64,u64,u64,u64)=unsafe{std::mem::transmute(address as usize)};
    call(args.0[0],args.0[1],args.0[2],args.0[3],args.0[4],args.0[5]);
}
fn dispatch(_:u64,args:&Args)->u64{
    if let Some((roots,mapping,added,view,parent))=replacement(*BASE.get().unwrap(),args){
        let count=CALLS.fetch_add(1,Relaxed)+1;
        if count<=8||(count<=1024&&count.is_power_of_two()){
            eprintln!("[menu-shadow-bindings] count={count} child={view:#x} parent={parent:#x} inherited_descriptor_indices={added:?}; all original bindings retained, original evaluator called once");
        }
        let mut forwarded=Args(args.0,args.1);forwarded.0[2]=mapping.as_ptr()as u64;forwarded.0[3]=roots.as_ptr()as u64;
        unsafe{delegate(*RESUME.get().unwrap(),&forwarded);}
    }else{unsafe{delegate(*RESUME.get().unwrap(),args);}}
    0
}
fn branch(from:u64,to:u64)->Result<u32>{
    let offset=to as i128-from as i128;
    ensure!(offset%4==0&&(-(1i128<<27)..(1i128<<27)).contains(&offset),"shadow binding branch out of range");
    Ok(0x14000000|((offset/4)as u32&0x3ffffff))
}
pub(crate)fn apply(modules:&[crate::loader::Module])->Result<()>{
    if !std::env::var("NX_FC27_MENU_SHADOW_PARENT_BINDINGS").is_ok_and(|v|v=="1"){return Ok(());}
    ensure!(std::env::current_exe()?.file_name().is_some_and(|v|v=="FC27-test")
        &&std::env::var_os("NX_TEST_CONTROL").is_some(),"shadow binding experiment requires isolated test executable");
    for variable in ["NX_FC27_MENU_TIME_TRACE","NX_FC27_VIDEO_TRACE","NX_FC27_MENU_AMBASSADOR_SOURCE","NX_FC27_MENU_PRIME_PROBE"]{
        ensure!(std::env::var(variable).is_ok_and(|v|v=="1"),"shadow binding experiment missing {variable}");
    }
    ensure!(!std::env::var("NX_FC27_MENU_SHADOW_PARENT_TIME").is_ok_and(|v|v=="1"),"shadow binding and scalar-time experiments must be isolated");
    let m=modules.iter().find(|m|m.name=="main").ok_or_else(||anyhow::anyhow!("missing FC27 main"))?;
    ensure!(m.module_id.iter().map(|v|format!("{v:02x}")).collect::<String>()=="fb81983aa1cc3a2e97402497e2ca652800000000000000000000000000000000","shadow binding unknown build");
    ensure!(ENTRY+8<=m.text&&0xa1d78ac<=m.text,"shadow binding invalid text layout");
    ensure!(unsafe{((m.base+ENTRY)as*const[u32;2]).read_unaligned()}==[0xa9ba7bfd,0xa9016ffc]
        &&unsafe{((m.base+0x8dd48a0)as*const[u32;2]).read_unaligned()}==[0xa9ba7bfd,0xa9016ffc]
        &&unsafe{((m.base+0xa1d7898)as*const[u32;2]).read_unaligned()}==[0x79800b68,0xb100051f],"shadow binding original evaluator/resolver guard mismatch");
    // nxpatch allocates only through tramp.start+tramp.length; the loader maps
    // the remaining final page as zero padding. Reserve 32 bytes strictly past
    // the used bank, within that one page, with a zero guard before mutation.
    let end=m.tramp.0.checked_add(m.tramp.1).ok_or_else(||anyhow::anyhow!("trampoline overflow"))?;
    let space=(end+15)&!15;let page_end=(end+crate::mem::PAGE-1)&!(crate::mem::PAGE-1);
    ensure!(space+32<=page_end&&page_end<=m.size,"shadow binding lacks mapped trampoline tail padding");
    let mut padding=[0;32];ensure!(snapshot(m.base+space,&mut padding)&&padding==[0;32],"shadow binding trampoline padding not empty");
    let bridge=m.base+space;let resume=bridge+16;
    let entry_branch=branch(m.base+ENTRY,bridge)?;
    let resume_branch=branch(resume+4,m.base+ENTRY+4)?;
    ensure!(BASE.set(m.base).is_ok()&&RESUME.set(resume).is_ok(),"shadow binding observer already installed");
    let stub=crate::stubs::make(dispatch,0);
    unsafe{
        // LDR X16,literal(+8); BR X16; full host stub pointer. The resume
        // executes the displaced, PC-independent STP and branches to entry+4.
        (bridge as*mut[u32;2]).write([0x58000050,0xd61f0200]);
        ((bridge+8)as*mut u64).write(stub);
        (resume as*mut[u32;2]).write([0xa9ba7bfd,resume_branch]);
        ((m.base+ENTRY)as*mut u32).write(entry_branch);
    }
    eprintln!("[menu-shadow-bindings] guarded original evaluator interposed; optional exact empty shadow child inherits only absent parent descriptors; trampoline tail main+{space:#x}");
    Ok(())
}
#[cfg(test)]mod tests{
    use super::*;
    #[test]fn fallback_merge_preserves_all_original_fields_and_duplicates(){
        let old=Mapping{root:2,pointer_offset:32,flags:0xabcd,descriptor:7,value_offset:124};
        let other=Mapping{root:4,descriptor:19,..old};
        let fallback=Mapping{root:0,pointer_offset:40,flags:0x1234,descriptor:11,value_offset:68};
        let originals=[old,old,other];let parent=[Mapping{value_offset:999,..old},fallback,fallback];
        let (merged,added)=merge(&originals,&parent,7);
        assert_eq!(&merged[..3],&originals);assert_eq!(merged[3],Mapping{root:7,..fallback});
        assert_eq!(merged[4],END);assert_eq!(added,[11]);
    }
    #[test]fn branch_range_and_signed_displacements_preserve_target(){
        let from=0x710a1d7660;
        for delta in [-(1i64<<27),-4,0,4,(1<<27)-4]{
            let instruction=branch(from,(from as i64+delta)as u64).unwrap();
            assert_eq!(instruction&0xfc000000,0x14000000);
            let immediate=(((instruction&0x3ffffff)as i32)<<6)>>6;
            assert_eq!(immediate as i64*4,delta);
        }
        assert!(branch(from,from+2).is_err());assert!(branch(from,from+(1<<27)).is_err());
        assert_eq!(std::mem::size_of::<Mapping>(),12);assert_eq!(std::mem::align_of::<Mapping>(),4);
    }
    #[test]fn original_binding_evaluator_receives_six_full_width_pointers_once(){
        static COUNT:AtomicU32=AtomicU32::new(0);
        extern "C" fn original(a:u64,b:u64,c:u64,d:u64,e:u64,f:u64){
            assert_eq!([a,b,c,d,e,f],[0x7111111111,0x7222222222,0x7333333333,0x7444444444,0x7555555555,0x7666666666]);COUNT.fetch_add(1,Relaxed);
        }
        let args=Args([0x7111111111,0x7222222222,0x7333333333,0x7444444444,0x7555555555,0x7666666666,0,0],[0;8]);
        unsafe{delegate(original as*const()as usize as u64,&args);}
        assert_eq!(COUNT.load(Relaxed),1);
    }
}
