//! Read-only identification of original menu still-frame selection containers.
//! Exact constructor, reflected Boolean-field and typed-object identities only.
pub(crate) fn version(base:u64)->bool{
    crate::diag::read_words::<4>(base+0xc8defa8)==Some([base+0xbaf8112,0x17ab1b7b,0x38,base+0xdeaf300])&&
    crate::diag::read_words::<4>(base+0xc8de720)==Some([base+0xbaf8112,0x17ab1b7b,0x26,base+0xdeaf300])&&
    crate::diag::read_words::<2>(base+0x5cb07ac)==Some([0x91214108f003d308,0xfd001800a902280a])&&
    crate::diag::read_words::<2>(base+0x5cb07bc)==Some([0xb803b01fb900381f,0xa8c17bfda9002009])&&
    crate::diag::read_words::<2>(base+0x5f72fac)==Some([0xa900a01fb802301f,0x9119e10890034b68])&&
    crate::diag::read_words::<3>(base+0xc8deec0)==Some([base+0x9c7610,base+0x5f71600,base+0x5f71760])&&
    crate::diag::read_words::<3>(base+0xc8de678)==Some([base+0x9c7610,base+0x10,base+0x957ee0])&&
    crate::diag::read_words::<3>(base+0xc8af5c8)==Some([base+0x5e14ac0,base+0x176fee0,base+0x147b110])&&
    crate::diag::read_words::<1>(base+0x5e14ac0)==Some([0x9120c000d003c700])&&
    crate::diag::read_words::<2>(base+0x5e14b60)==Some([0x90040356f9401c13,0xf94012a891009674])
}
fn word(b:&[u8],at:usize)->Option<u64>{Some(u64::from_le_bytes(b.get(at..at+8)?.try_into().ok()?))}
fn integer(b:&[u8],at:usize)->Option<i32>{Some(i32::from_le_bytes(b.get(at..at+4)?.try_into().ok()?))}
// Candidate counts are independent of the stronger data-container validator.
// A matching vtable alone does not establish a live or active object.
pub(crate) const RUNTIME_CLASSES:[(&str,u64);4]=[
    ("CinematicMainMenuSequenceSelectionEntity",0xc8af5c8),
    ("FIFAHubEntity",0xc8b5018),
    ("ClientCinematicInEntity",0xc5c3a20),
    ("PlatformSplitterEntity",0xcaf1ea8),
];
pub(crate) fn runtime_candidate(b:&[u8],base:u64)->Option<usize>{
    let vtable=word(b,0)?;
    RUNTIME_CLASSES.iter().position(|(_,rva)|base.checked_add(*rva)==Some(vtable))
}
pub(crate) fn row(b:&[u8],address:u64,base:u64)->Option<String>{
    if b.len()>=72&&word(b,0)==Some(base+0xc8af5c8){
        let data=word(b,0x38)?;let mut fields=[0;40];
        if !crate::input::test_control::config_inspection::snapshot(data,&mut fields)||word(&fields,0)!=Some(base+0xc8de678)||word(&fields,8)!=Some(base+0xd713460)||fields[0x24..=0x26].iter().any(|v|*v>1){return None;}
        let platform=crate::diag::read_words::<1>(base+0xdd7ea10).and_then(|v|{
            let mut data=[0;4];if v[0]==0||!crate::input::test_control::config_inspection::snapshot(v[0]+0x18,&mut data){return None;}
            let value=i32::from_le_bytes(data);(0..=10).contains(&value).then_some(value)
        }).map_or("null".into(),|v|v.to_string());
        return Some(format!(r#"{{"class":"CinematicMainMenuSequenceSelectionEntity","address":"{address:#x}","data":"{data:#x}","asset_UseStillFrames":{},"asset_UseFirstBoot":{},"asset_RestrictTelstarSets":{},"selector_platform_value":{platform},"output_property_pointer":"{:#x}","limits":"Original runtime entity exists; asset values do not include property overrides. No event execution or selected output inferred from existence."}}"#,fields[0x26],fields[0x25],fields[0x24],word(b,0x40)?));
    }
    if b.len()>=64 && word(b,0)==Some(base+0xc8deec0) && word(b,8)==Some(base+0xd713850) && b[0x38..=0x3e].iter().all(|v|*v<=1){
        Some(format!(r#"{{"class":"MainMenuInitializeData","address":"{address:#x}","UseStillFrames":{},"UseFirstBoot":{},"NumPopulatedAmbassadors":{},"AvatarIndex":{},"RestrictTelstarSets":{},"RandomizeEntryPoint":{}}}"#,b[0x38],b[0x39],integer(b,0x30)?,integer(b,0x34)?,b[0x3c],b[0x3e]))
    }else if b.len()>=40 && word(b,0)==Some(base+0xc8de678) && word(b,8)==Some(base+0xd713460) && b[0x24..=0x26].iter().all(|v|*v<=1){
        Some(format!(r#"{{"class":"CinematicMainMenuSequenceSelectionEntityData","address":"{address:#x}","UseStillFrames":{},"UseFirstBoot":{},"RestrictTelstarSets":{},"StadiumId":{}}}"#,b[0x26],b[0x25],b[0x24],integer(b,0x20)?))
    }else{None}
}

#[cfg(test)]mod runtime_candidate_tests{
    use super::*;
    #[test]fn vtable_candidates_are_independent_of_asset_layout_and_reject_neighbors(){
        let base=0x70c0804000;
        for (index,(_,rva))in RUNTIME_CLASSES.iter().enumerate(){
            let bytes=(base+rva).to_le_bytes();
            assert_eq!(runtime_candidate(&bytes,base),Some(index));
            assert_eq!(row(&bytes,0,base),None,"candidate is not a fully validated runtime object");
            assert_eq!(runtime_candidate(&(base+rva+8).to_le_bytes(),base),None);
        }
        assert_eq!(runtime_candidate(&[0;7],base),None);
        assert_eq!(runtime_candidate(&[0;8],u64::MAX),None);
    }
}
