//! Bounded observations of original frontend menu load/unload commands.
//! No playback requests, guest data edits, readiness or clock changes.
use anyhow::{ensure,Result};
use std::sync::{OnceLock,atomic::{AtomicU32,Ordering::Relaxed}};
use crate::stubs::Args;
static BASE:OnceLock<u64>=OnceLock::new();
static RESUMES:OnceLock<[u64;3]>=OnceLock::new();
static CALLS:[AtomicU32;3]=[const{AtomicU32::new(0)};3];
const LOAD:u64=0x40f8560;
const UNLOAD:u64=0x6912880;
const UNPRIME:u64=0x1b9f600;
const LOAD_WORDS:[u32;4]=[0xa9bb7bfd,0xf9000bf9,0xa9025ff8,0xa90357f6];
const UNPRIME_WORDS:[u32;4]=[0xa9bc7bfd,0xf9000bf7,0xa90257f6,0xa9034ff4];
fn word(address:u64)->Option<u64>{let mut b=[0;8];crate::input::test_control::config_inspection::snapshot(address,&mut b).then(||u64::from_le_bytes(b))}
fn selected(base:u64)->Option<u32>{
    let manager=word(base+0xd44cac0)?;if word(manager)!=Some(base+0xc885828){return None;}
    let mut b=[0;4];crate::input::test_control::config_inspection::snapshot(manager+0x1884,&mut b).then(||u32::from_le_bytes(b))
}
unsafe fn delegate(address:u64,args:&Args)->u64{
    let call:extern "C" fn(u64,u64,u64,u64,u64,u64,u64,u64)->u64=unsafe{std::mem::transmute(address as usize)};
    call(args.0[0],args.0[1],args.0[2],args.0[3],args.0[4],args.0[5],args.0[6],args.0[7])
}
fn dispatch(tag:u64,args:&Args)->u64{
    let i=tag as usize;let count=CALLS[i].fetch_add(1,Relaxed)+1;let base=*BASE.get().unwrap();
    let log=count<=16||(count<=1024&&count.is_power_of_two());
    if i==2{
        // This original method consumes one runtime instance pointer, whose
        // asset and manager are validated independently against the original
        // typed index. Observe its actual return, never synthesize completion.
        let owner=word(args.0[0]+0x80);let asset=word(args.0[0]);
        let master=owner.zip(asset).filter(|(manager,asset)|crate::menu_cinematic_diagnostics::registered_master_asset(*manager,base)==Some(*asset));
        let result=unsafe{delegate(RESUMES.get().unwrap()[2],args)};
        if let Some((manager,asset))=master{
            if log{eprintln!("[menu-lifecycle] original master instance UnPrime returned; manager={manager:#x} asset={asset:#x}");}
            crate::fc27_menu_auto_lifecycle::unprime_returned(manager,asset);
        }
        return result;
    }
    if log{eprintln!("[menu-lifecycle] command={} count={count} stage=before selected={:?} caller_rva={:?}",["loadMainMenuScreen","unloadMainMenuStadium"][i],selected(base),word(args as*const Args as u64-8).and_then(|v|v.checked_sub(base)).filter(|v|*v<0xb99a7c0));}
    let result=unsafe{delegate(RESUMES.get().unwrap()[i],args)};
    crate::fc27_menu_auto_lifecycle::command(i);
    if log{eprintln!("[menu-lifecycle] command={} count={count} stage=after selected={:?}",["loadMainMenuScreen","unloadMainMenuStadium"][i],selected(base));}
    result
}
fn absolute(target:u64)->[u32;4]{[0x58000050,0xd61f0200,target as u32,(target>>32)as u32]}
fn branch(from:u64,to:u64)->Result<u32>{
    let offset=to as i128-from as i128;ensure!(offset%4==0&&(-(1i128<<27)..(1i128<<27)).contains(&offset),"menu lifecycle branch out of range");Ok(0x14000000|((offset/4)as u32&0x3ffffff))
}
pub(crate)fn apply(modules:&[crate::loader::Module])->Result<()>{
    if !std::env::var("NX_FC27_MENU_EVENT_TRACE").is_ok_and(|v|v=="1"){return Ok(());}
    ensure!(std::env::current_exe()?.file_name().is_some_and(|v|v=="FC27-test")&&std::env::var_os("NX_TEST_CONTROL").is_some(),"menu lifecycle observation requires isolated test executable");
    let m=modules.iter().find(|m|m.name=="main").ok_or_else(||anyhow::anyhow!("missing main"))?;
    ensure!(m.module_id.iter().map(|v|format!("{v:02x}")).collect::<String>()=="fb81983aa1cc3a2e97402497e2ca652800000000000000000000000000000000","menu lifecycle unknown build");
    ensure!(LOAD+24<=m.text&&UNLOAD+16<=m.text&&UNPRIME+24<=m.text,"menu lifecycle text bounds");
    ensure!(unsafe{((m.base+LOAD)as*const[u32;4]).read_unaligned()}==LOAD_WORDS&&unsafe{((m.base+LOAD+16)as*const[u32;2]).read_unaligned()}==[0xa9044ff4,0x910003fd]
        &&unsafe{((m.base+UNLOAD)as*const[u32;4]).read_unaligned()}==[0xa9be7bfd,0xf9000bf3,0x910003fd,0xd0036508]
        &&unsafe{((m.base+UNPRIME)as*const[u32;4]).read_unaligned()}==UNPRIME_WORDS
        &&unsafe{((m.base+UNPRIME+16)as*const[u32;2]).read_unaligned()}==[0x910003fd,0xaa0003f3],"menu lifecycle original entry guard mismatch");
    let end=m.tramp.0.checked_add(m.tramp.1).ok_or_else(||anyhow::anyhow!("trampoline overflow"))?;
    // The preceding32 bytes are reserved for the shadow binding adapter.
    let space=((end+15)&!15)+32;let page_end=(end+crate::mem::PAGE-1)&!(crate::mem::PAGE-1);
    ensure!(space+96<=page_end&&page_end<=m.size,"menu lifecycle trampoline padding bounds");
    let mut zero=[0;96];ensure!(crate::input::test_control::config_inspection::snapshot(m.base+space,&mut zero)&&zero==[0;96],"menu lifecycle trampoline padding occupied");
    let far_resume=m.base+space;let bridge=far_resume+32;let near_resume=bridge+16;
    let unprime_resume=far_resume+64;
    let enter=branch(m.base+UNLOAD,bridge)?;let leave=branch(near_resume+4,m.base+UNLOAD+4)?;
    ensure!(BASE.set(m.base).is_ok()&&RESUMES.set([far_resume,near_resume,unprime_resume]).is_ok(),"menu lifecycle observer already installed");
    let load=crate::stubs::make(dispatch,0);let unload=crate::stubs::make(dispatch,1);
    let unprime=crate::stubs::make(dispatch,2);
    unsafe{
        // LOAD lies outside the near island's B range. Its four displaced
        // instructions only save registers relative to SP: copy them exactly,
        // then use an absolute X16 veneer back to its original entry+16.
        (far_resume as*mut[u32;4]).write(LOAD_WORDS);((far_resume+16)as*mut[u32;4]).write(absolute(m.base+LOAD+16));
        (bridge as*mut[u32;4]).write(absolute(unload));(near_resume as*mut[u32;2]).write([0xa9be7bfd,leave]);
        (unprime_resume as*mut[u32;4]).write(UNPRIME_WORDS);((unprime_resume+16)as*mut[u32;4]).write(absolute(m.base+UNPRIME+16));
        ((m.base+LOAD)as*mut[u32;4]).write_unaligned(absolute(load));((m.base+UNLOAD)as*mut u32).write(enter);
        ((m.base+UNPRIME)as*mut[u32;4]).write_unaligned(absolute(unprime));
    }
    eprintln!("[menu-lifecycle] two exact original command entries and instance UnPrime observed; delegated once, bounded logs, no lifecycle requests");Ok(())
}
#[cfg(test)]mod tests{
    use super::*;
    #[test]fn near_and_far_veneers_keep_full_addresses_and_displaced_prologues(){
        let target=0xfedcba9876543210;let words=absolute(target);assert_eq!(u64::from(words[2])|(u64::from(words[3])<<32),target);
        assert!(branch(LOAD,0xe1c51a0).is_err());assert!(branch(UNLOAD,0xe1c51a0).is_ok());
        assert!(branch(0x10000000,0x08000000).is_ok());assert!(branch(0x10000000,0x18000000).is_err());assert!(branch(0,3).is_err());
        assert_eq!(LOAD_WORDS,[0xa9bb7bfd,0xf9000bf9,0xa9025ff8,0xa90357f6]);
        assert_eq!(UNPRIME_WORDS,[0xa9bc7bfd,0xf9000bf7,0xa90257f6,0xa9034ff4]);assert!(branch(UNPRIME,0xe1c51c0).is_err());
    }
    #[test]fn lifecycle_delegate_preserves_all_gpr_arguments_and_return_once(){
        static COUNT:AtomicU32=AtomicU32::new(0);
        extern "C" fn original(a:u64,b:u64,c:u64,d:u64,e:u64,f:u64,g:u64,h:u64)->u64{assert_eq!([a,b,c,d,e,f,g,h],[1,2,3,4,5,6,7,8]);COUNT.fetch_add(1,Relaxed);0xfedcba9876543210}
        assert_eq!(unsafe{delegate(original as*const()as usize as u64,&Args([1,2,3,4,5,6,7,8],[0;8]))},0xfedcba9876543210);assert_eq!(COUNT.load(Relaxed),1);
    }
}
