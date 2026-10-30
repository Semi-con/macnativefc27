//! Bounded, read-only inspection of original menu cinematic candidates.
//! Invoked only by the existing opt-in, version-checked menu heap scan.
use serde_json::{Value, json};

const NAMES: [(&str, u32); 9] = [
    ("f26_mm_default_proto_01", 1319562555),
    ("f26_mm_master_01", 610392389),
    ("f26_mm_mlkr_vanity_01", 2656017155),
    ("f26_mm_mlkr_plyr_01", 3085253161),
    ("f26_mm_cor_plyr_01", 647754511),
    ("f26_mm_gtun_plyr_01", 1356496057),
    ("f27_mm_mod_pitch_gen_dribble_01", 3755401501),
    ("f27_mm_mod_pitch_bespk_dribble_01", 2190595582),
    ("f27_mm_mod_club_gen_juggle_01", 1962520931),
];

fn word(bytes: &[u8], at: usize) -> Option<u64> {
    Some(u64::from_le_bytes(bytes.get(at..at + 8)?.try_into().ok()?))
}
fn snapshot(address: u64, bytes: &mut [u8]) -> bool {
    crate::input::test_control::config_inspection::snapshot(address, bytes)
}
fn read_word(address: u64) -> Option<u64> {
    let mut bytes = [0; 8];
    snapshot(address, &mut bytes).then(|| u64::from_le_bytes(bytes))
}
pub(crate) fn version(base: u64) -> bool {
    crate::diag::read_words::<2>(base + 0x1ba52e0) == Some([0x91288108d00550e8, 0xf9409660f9000288])
        && crate::diag::read_words::<2>(base + 0x1ba53d8) == Some([0xf005c528f9401c15, 0xf9456114b9400821])
        && crate::diag::read_words::<1>(base + 0x1ba5470) == Some([0xa9454ff4f9408105])
        && crate::diag::read_words::<4>(base + 0xc5c5a20) == Some([base + 0xbb6d77e, 0x6d71be83, 0x30, base + 0xdea8bd8])
}
pub(crate) fn selected_pointer_name(name: &[u8]) -> Option<&'static str> {
    let end = name.iter().position(|b| *b == 0)?;
    NAMES.iter().find_map(|(expected, _)| (expected.as_bytes() == &name[..end]).then_some(*expected))
}
fn selected_name(data: &[u8], name: &[u8]) -> Option<usize> {
    let hash = u32::from_le_bytes(data.get(0x28..0x2c)?.try_into().ok()?);
    let end = name.iter().position(|b| *b == 0)?;
    NAMES.iter().position(|(expected, id)| *id == hash && expected.as_bytes() == &name[..end])
}
fn original_object(address: u64, base: u64) -> Value {
    let vtable = (address != 0).then(|| read_word(address)).flatten();
    let rva = vtable.and_then(|v| v.checked_sub(base)).filter(|v| (0xb99b000..0xd308570).contains(v));
    json!({"address": format!("{address:#x}"), "vtable_rva": rva.map(|v| format!("{v:#x}"))})
}

fn index_entry(mut node:u64,key:u32,mut read:impl FnMut(u64,&mut[u8])->bool)->Result<Option<u64>,&'static str>{
    let mut visited=[0;32];
    for index in 0..visited.len(){
        if node==0{return Ok(None);}
        if visited[..index].contains(&node){return Err("cycle");}
        visited[index]=node;
        let mut bytes=[0;48];
        if !read(node,&mut bytes){return Err("unreadable");}
        let found=u32::from_le_bytes(bytes[0x20..0x24].try_into().unwrap());
        if found==key{return Ok(word(&bytes,0x28));}
        // Original 1baa494..1baa4a8 uses unsigned keys, child0 for
        // node-key<target-key, child8 otherwise (not a guessed tree layout).
        node=word(&bytes,if found<key{0}else{8}).unwrap();
    }
    Err("depth_bound")
}
fn manager_index_version(address:u64,base:u64)->bool{
    read_word(address)==Some(base+0xc885828)
        &&crate::diag::read_words::<3>(base+0x1baa478)==Some([0xb4001108f9505668,0x2a0003f752841309,0x8b0902695280010a])
        &&crate::diag::read_words::<3>(base+0x1baa494)==Some([0x6b17017fb940210b,0x9a88337b9a8a33eb,0xb5ffff68f86b6908])
}
pub(crate) fn registered_master_asset(address:u64,base:u64)->Option<u64>{
    registered_menu_asset(address,base,610392389)
}
pub(crate) fn registered_menu_asset(address:u64,base:u64,hash:u32)->Option<u64>{
    if !NAMES.iter().any(|(_,key)|*key==hash){return None;}
    if !manager_index_version(address,base){return None;}
    let asset=index_entry(read_word(address+0x20a8)?,hash,snapshot).ok()??;
    let mut bytes=[0;16];
    (snapshot(asset,&mut bytes)&&word(&bytes,0)==Some(base+0xc886208)&&word(&bytes,8)==Some(base+0xd6d9dc8)).then_some(asset)
}
#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub(crate) struct MenuRuntimeState{pub(crate) registrations:u32,pub(crate) list_empty:bool}
fn runtime_fields(bytes:&[u8],address:u64,manager:u64,asset:u64)->Option<MenuRuntimeState>{
    if word(bytes,0)!=Some(asset)||word(bytes,0x80)!=Some(manager){return None;}
    let registrations=u32::from_le_bytes(bytes.get(0x128..0x12c)?.try_into().ok()?);
    if registrations>4096{return None;}
    Some(MenuRuntimeState{registrations,list_empty:word(bytes,0xe8)==Some(address+0xe8)&&word(bytes,0xf0)==Some(address+0xe8)})
}
// The original RequestUnPrime handler finds runtime instances in manager+62400,
// then1b9f600 clears its registration list and restores empty links/count.
// Run255 shows this list can already be empty during visible playback; it is
// not an animation-state or resource-release completion test.
// Absence is distinct from an unreadable/mismatched entry; no calls or writes.
pub(crate) fn menu_runtime_state(manager:u64,base:u64,hash:u32)->Result<Option<MenuRuntimeState>,&'static str>{
    if crate::diag::read_words::<2>(base+0x1babbb4)!=Some([0x2a0203f491418808,0x910fc115aa0003f3])
        ||crate::diag::read_words::<2>(base+0x1b9f6e8)!=Some([0x9103a268f9400269,0xa90ed668b9012a7f]){return Err("runtime_version");}
    let asset=registered_menu_asset(manager,base,hash).ok_or("typed_asset")?;
    let root=read_word(manager+0x62400).ok_or("runtime_root")?;
    let Some(address)=index_entry(root,hash,snapshot)?else{return Ok(None);};
    let mut bytes=[0;0x130];if !snapshot(address,&mut bytes){return Err("runtime_unreadable");}
    runtime_fields(&bytes,address,manager,asset).map(Some).ok_or("runtime_identity_or_count")
}
pub(crate) fn master_name(manager:u64,base:u64)->Option<(u64,u64)>{
    let asset=registered_master_asset(manager,base)?;
    let name=read_word(asset+0x60)?;let mut bytes=[0;96];
    (snapshot(name,&mut bytes)&&selected_pointer_name(&bytes)==Some("f26_mm_master_01"))
        .then_some((asset,name))
}
fn manager_index(address:u64,base:u64)->Value{
    if !manager_index_version(address,base){return json!({"version_verified":false});}
    let mut selected=[0;4];
    let selected=snapshot(address+0x1884,&mut selected).then(||u32::from_le_bytes(selected));
    let root=read_word(address+0x20a8);
    let automatic_prep_version=crate::diag::read_words::<4>(base+0xc5c5350)==Some([base+0xbe5b0d5,0xd0cd313,0x9d,base+0xdeaf300]);
    let rows:Vec<_>=NAMES.iter().map(|(name,key)|{
        let result=root.ok_or("unreadable_root").and_then(|node|index_entry(node,*key,snapshot));
        match result{
            Ok(Some(asset))=>{
                let mut bytes=[0;0xf0];
                let typed=snapshot(asset,&mut bytes)&&word(&bytes,0)==Some(base+0xc886208)&&word(&bytes,8)==Some(base+0xd6d9dc8);
                json!({"instance_name":name,"instance_hash":key,"in_index":true,
                    "asset":original_object(asset,base),"typed_original_asset":typed,
                    "IsSlideshow":typed.then_some(bytes[0xe4]).filter(|v|*v<=1),
                    "AutomaticPrepSchematics":(typed&&automatic_prep_version).then_some(bytes[0x9d]).filter(|v|*v<=1),
                    "FullSuspendBCEngineWhileRunning":typed.then_some(bytes[0xe2]).filter(|v|*v<=1),
                    "runtime_registrations":match menu_runtime_state(address,base,*key){
                        Ok(Some(s))=>json!({"present":true,"count":s.registrations,"list_empty":s.list_empty,"limits":"May be empty during playback; not animation or resource-release completion."}),
                        Ok(None)=>json!({"present":false}),Err(error)=>json!({"error":error}),
                    }})
            },
            Ok(None)=>json!({"instance_name":name,"instance_hash":key,"in_index":false}),
            Err(error)=>json!({"instance_name":name,"instance_hash":key,"in_index":null,"error":error}),
        }
    }).collect();
    json!({"version_verified":true,"selected_instance_hash_1884":selected,
        "selected_menu_name":selected.and_then(|hash|NAMES.iter().find(|(_,key)|*key==hash).map(|(name,_)|*name)),
        "original_index_root_is_null":root.map(|node|node==0),"menu_entries":rows,
        "limits":"Read-only original manager index. Nine known keys, max32 nodes/key,48bytes/node,cycle/readability guards; no calls or writes. Not locked against original game changes. Index membership/selected hash do not prove active playback."})
}

pub(crate) struct Observation {
    base: u64,
    verified: bool,
    matched: u64,
    rows: Vec<Value>,
}
impl Observation {
    pub(crate) fn new(base: u64) -> Self {
        Self { base, verified: version(base), matched: 0, rows: Vec::new() }
    }
    pub(crate) fn observe(&mut self, bytes: &[u8], address: u64) {
        if !self.verified || word(bytes, 0) != Some(self.base + 0xc5c3a20) { return; }
        let Some(data) = word(bytes, 0x38).filter(|v| *v != 0) else { return; };
        let mut fields = [0; 64];
        if !snapshot(data, &mut fields) { return; }
        let Some(pointer) = word(&fields, 0x30).filter(|v| *v != 0) else { return; };
        let mut name = [0; 96];
        if !snapshot(pointer, &mut name) { return; }
        let Some(index) = selected_name(&fields, &name) else { return; };
        self.matched += 1;
        if self.rows.len() < 32 {
            self.rows.push(json!({
                "address": format!("{address:#x}"), "data": original_object(data, self.base),
                "data_type_info_rva": word(&fields, 8).and_then(|v| v.checked_sub(self.base)).map(|v| format!("{v:#x}")),
                "instance_name": NAMES[index].0, "instance_hash": NAMES[index].1,
                "owner": original_object(word(bytes, 0x30).unwrap_or(0), self.base),
                "limits": "Exact original runtime vtable and original asset name/hash pair; may still be a stale candidate. No activation or playback state inferred."
            }));
        }
    }
    pub(crate) fn report(&self) -> Value {
        let manager = if self.verified {
            let address = read_word(self.base + 0xd44cac0).unwrap_or(0);
            let vtable = (address != 0).then(|| read_word(address)).flatten();
            let method = vtable.filter(|v| v.checked_sub(self.base).is_some_and(|r| (0xb99b000..0xd308570).contains(&r)))
                .and_then(|v| v.checked_add(0x100)).and_then(read_word)
                .and_then(|v| v.checked_sub(self.base)).filter(|r| *r < 0xb99a7c0);
            json!({"global_rva": "0xd44cac0", "object": original_object(address, self.base),
                "original_event_method_slot": "0x100", "method_rva": method.map(|v| format!("{v:#x}")),
                "index":manager_index(address,self.base),
                "limits": "Read-only original CinematicIn event-dispatch manager pointer. Interface presence is not a startup or playback observation; no functions invoked."})
        } else { json!({"version_verified": false}) };
        json!({"version_verified": self.verified, "matching_named_candidates": self.matched,
            "rows": self.rows, "manager": manager,
            "limits": "Existing bounded menu heap scan only; max32 rows, exact nine original menu names/hashes,96byte name reads. No arbitrary strings, raw heap dumps, writes or virtual calls. Candidate existence does not establish an active cinematic."})
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]fn runtime_registration_observation_requires_original_asset_owner_and_both_empty_links(){
        let address=0x7400000000u64;let manager=0x7500000000u64;let asset=0x7600000000u64;
        let mut bytes=[0;0x130];bytes[..8].copy_from_slice(&asset.to_le_bytes());bytes[0x80..0x88].copy_from_slice(&manager.to_le_bytes());
        for at in [0xe8,0xf0]{bytes[at..at+8].copy_from_slice(&(address+0xe8).to_le_bytes());}
        assert_eq!(runtime_fields(&bytes,address,manager,asset),Some(MenuRuntimeState{registrations:0,list_empty:true}));
        bytes[0x128..0x12c].copy_from_slice(&2u32.to_le_bytes());assert_eq!(runtime_fields(&bytes,address,manager,asset).unwrap().registrations,2);
        bytes[0xe8]^=8;assert!(!runtime_fields(&bytes,address,manager,asset).unwrap().list_empty);
        assert!(runtime_fields(&bytes,address,manager+8,asset).is_none());assert!(runtime_fields(&bytes,address,manager,asset+8).is_none());
        bytes[0x128..0x12c].copy_from_slice(&4097u32.to_le_bytes());assert!(runtime_fields(&bytes,address,manager,asset).is_none());
        assert!(runtime_fields(&bytes[..0x128],address,manager,asset).is_none());
    }
    #[test]
    fn names_require_exact_original_hash_and_bounded_terminated_name() {
        for (index, (name, hash)) in NAMES.iter().enumerate() {
            let mut data = [0; 64];
            data[0x28..0x2c].copy_from_slice(&hash.to_le_bytes());
            let mut text = name.as_bytes().to_vec(); text.push(0);
            assert_eq!(selected_pointer_name(&text), Some(*name));
            assert_eq!(selected_name(&data, &text), Some(index));
            assert_eq!(selected_name(&data, name.as_bytes()), None);
            data[0x28] ^= 1;
            assert_eq!(selected_name(&data, &text), None);
        }
        assert_eq!(selected_name(&[0; 16], b"f26_mm_master_01\0"), None);
        assert_eq!(selected_name(&[0; 64], b"unrelated_private_text\0"), None);
        assert_eq!(selected_name(&[0; 64], &[255, 0]), None);
        assert_eq!(selected_pointer_name(b"unrelated_private_text\0"), None);
        assert_eq!(selected_pointer_name(b"f26_mm_master_01"), None);
    }
    #[test]
    fn original_index_unsigned_child_order_and_malformed_bounds(){
        let node=|key:u32,right:u64,left:u64,value:u64|{
            let mut bytes=[0;48];bytes[..8].copy_from_slice(&right.to_le_bytes());
            bytes[8..16].copy_from_slice(&left.to_le_bytes());bytes[0x20..0x24].copy_from_slice(&key.to_le_bytes());
            bytes[0x28..0x30].copy_from_slice(&value.to_le_bytes());bytes
        };
        let nodes=[(0x1000,node(0x80000000,0x2000,0x3000,1)),(0x2000,node(0xf0000000,0,0,2)),(0x3000,node(1,0,0,3))];
        let read=|address,output:&mut[u8]|{
            if let Some((_,bytes))=nodes.iter().find(|(at,_)|*at==address){output.copy_from_slice(bytes);true}else{false}
        };
        for (key,value)in [(0x80000000,1),(0xf0000000,2),(1,3)]{assert_eq!(index_entry(0x1000,key,read),Ok(Some(value)));}
        assert_eq!(index_entry(0x1000,2,read),Ok(None));assert_eq!(index_entry(0,2,read),Ok(None));
        assert_eq!(index_entry(0x4000,2,read),Err("unreadable"));
        let cycle=node(0,0x1000,0,0);
        assert_eq!(index_entry(0x1000,1,|_,b|{b.copy_from_slice(&cycle);true}),Err("cycle"));
        assert_eq!(index_entry(0x1000,1,|at,b|{b.copy_from_slice(&node(0,at+0x1000,0,0));true}),Err("depth_bound"));
    }
}
