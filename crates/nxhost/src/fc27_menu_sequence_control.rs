//! Isolated original-menu sequence experiment; disabled unless explicitly requested.
//! Applied before guest code becomes executable. Never changes source/cache files.
use anyhow::{Result,ensure};
const AT:u64=0x5e14c58;
const ORIGINAL:[u32;3]=[0x71001d1f,0x52800088,0x1a9f0508];
fn replace(words:&mut[u32;3])->Result<()>{
    ensure!(*words==ORIGINAL,"FC27 menu sequence instruction guard mismatch");
    words[1]=0x52800068; // mov w8,#3: original G4_00, only platform==7 branch.
    Ok(())
}
pub(crate) fn apply(modules:&[crate::loader::Module])->Result<()>{
    let Ok(mode)=std::env::var("NX_FC27_MENU_SEQUENCE")else{return Ok(());};
    ensure!(mode=="g4","Unsupported FC27 menu sequence experiment");
    ensure!(std::env::var_os("NX_TEST_CONTROL").is_some()&&std::env::var("NX_FC27_VIDEO_TRACE").is_ok_and(|v|v=="1"),"FC27 menu sequence experiment requires isolated test control and video trace");
    ensure!(std::env::current_exe()?.file_name().is_some_and(|n|n=="FC27-test"),"FC27 menu sequence experiment only accepts the local test executable");
    let m=modules.iter().find(|m|m.name=="main").ok_or_else(||anyhow::anyhow!("Missing FC27 main module"))?;
    let id:String=m.module_id.iter().map(|v|format!("{v:02x}")).collect();
    ensure!(id=="fb81983aa1cc3a2e97402497e2ca652800000000000000000000000000000000","FC27 menu sequence build ID mismatch");
    ensure!(m.prefix>0&&AT+12<=m.text&&0xc01eed8+26<m.size,"FC27 menu sequence layout mismatch");
    for (at,label)in [(0xc01eebcu64,b"MainMenuSequenceTypes_G4_00\0".as_slice()),(0xc01eed8,b"MainMenuSequenceTypes_NX_00\0".as_slice())]{
        let actual=unsafe{std::slice::from_raw_parts((m.base+at)as*const u8,label.len())};
        ensure!(actual==label,"FC27 original menu sequence enum name mismatch");
    }
    // Loader has completed relocations but has not set RX permissions or started guest execution.
    let words=unsafe{&mut*((m.base+AT)as*mut[u32;3])};replace(words)?;
    eprintln!("[menu-sequence-control] isolated original NX_00(4) -> G4_00(3) at main+0x5e14c5c; one instruction changed");
    Ok(())
}
#[cfg(test)]mod tests{
    use super::*;
    #[test]fn changed_version_never_modifies_any_instruction(){
        for index in 0..3{for bit in 0..32{
            let mut words=ORIGINAL;words[index]^=1<<bit;let before=words;
            assert!(replace(&mut words).is_err());assert_eq!(words,before);
        }}
    }
    #[test]fn exact_original_sequence_change_is_one_mov_immediate_and_cannot_reapply(){
        let mut words=ORIGINAL;replace(&mut words).unwrap();
        assert_eq!(words[0],ORIGINAL[0]);assert_eq!(words[2],ORIGINAL[2]);
        assert_eq!(words[1]&!(0xffff<<5),ORIGINAL[1]&!(0xffff<<5));
        assert_eq!((words[1]>>5)&0xffff,3);assert_eq!(words[1]&31,8);
        let before=words;assert!(replace(&mut words).is_err());assert_eq!(words,before);
    }
}
