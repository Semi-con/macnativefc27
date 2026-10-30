//! Read-only, opt-in observation of loaded menu configuration JSON metadata.
//! Called only from the original version-checked bounded menu heap inspection.
use std::{collections::HashSet,path::{Path,PathBuf}};
use serde_json::{Value,json};
const ANCHORS:[&[u8];6]=[b"MainMenuSequenceTypes_",b"stadium_backgrounds/",b"main_menu_config.json",b"charVizStreamed",b"MainMenuVideo",b"use3DMainMenu"];
const LIMIT:usize=64*1024;
fn sensitive(name:&str)->bool{
    let lower=name.to_ascii_lowercase();
    ["account","auth","token","password","secret","credential","profile","gamertag","friend","email","user"].iter().any(|s|lower.contains(s))
}
fn contains_sensitive(v:&Value,depth:usize)->bool{
    if depth>16{return true;}
    match v{Value::Object(m)=>m.iter().any(|(k,v)|sensitive(k)||contains_sensitive(v,depth+1)),Value::Array(a)=>a.iter().any(|v|contains_sensitive(v,depth+1)),_=>false}
}
fn selected_string(s:&str)->bool{
    if s.len()>256||sensitive(s)||!s.bytes().all(|c|c.is_ascii_alphanumeric()||b"_-/ .".contains(&c)){return false;}
    let lower=s.to_ascii_lowercase();
    lower.contains("mainmenu")||lower.contains("main_menu")||lower.contains("stadium_backgrounds/")||lower.starts_with("content/cinematic/")||lower.starts_with("cinematicdynamicsets_mainmenu_")||
        ["nx","nx_00","g4","g4_00","g5","g5_00","basiclowcost","common_nomotion_00","common_notelstar_00","true","false","on","off","disabled","enabled"].contains(&lower.as_str())
}
fn metadata(v:&Value,depth:usize,budget:&mut usize)->Value{
    if depth>16||*budget==0{return json!({"omitted":"size/depth bound"});}*budget-=1;
    match v{
        Value::Object(m)=>Value::Object(m.iter().take(128).filter(|(k,_)|k.len()<=128&&!sensitive(k)).map(|(k,v)|(k.clone(),metadata(v,depth+1,budget))).collect()),
        Value::Array(a)=>json!({"count":a.len(),"values":a.iter().take(128).map(|v|metadata(v,depth+1,budget)).collect::<Vec<_>>(),"truncated":a.len()>128}),
        Value::String(s)=>if selected_string(s){v.clone()}else{json!({"omitted":"unselected string","bytes":s.len()})},
        _=>v.clone()
    }
}
fn candidate(bytes:&[u8],start:usize,anchor:usize)->Option<(usize,Value)>{
    let tail=bytes.get(start..bytes.len().min(start+LIMIT))?;
    let mut stream=serde_json::Deserializer::from_slice(tail).into_iter::<Value>();
    let value=stream.next()?.ok()?;let len=stream.byte_offset();
    if !value.is_object()||anchor>=start+len||contains_sensitive(&value,0){return None;}
    Some((len,value))
}
pub(crate) struct Observation{output:PathBuf,seen:HashSet<u64>,documents:Vec<Vec<u8>>,rows:Vec<Value>,attempts:u32,rejected:u32}
impl Observation{
    pub(crate) fn new(output:&Path)->Self{Self{output:output.to_owned(),seen:HashSet::new(),documents:Vec::new(),rows:Vec::new(),attempts:0,rejected:0}}
    pub(crate) fn observe(&mut self,bytes:&[u8],address:u64,step:usize){
        if self.documents.len()>=8||self.attempts>=256{return;}
        for needle in ANCHORS{
            for anchor in memchr::memmem::find_iter(bytes,needle).take(32){
                if anchor>=step{continue;}
                let left=anchor.saturating_sub(LIMIT);let mut starts=memchr::memrchr_iter(b'{',&bytes[left..anchor]).take(16).map(|v|left+v).collect::<Vec<_>>();starts.reverse();
                for start in starts{
                    if self.attempts>=256||self.documents.len()>=8{return;}
                    let first=bytes[start+1..bytes.len().min(start+64)].iter().find(|c|!c.is_ascii_whitespace());
                    if first!=Some(&b'"'){continue;}
                    if !self.seen.insert(address+start as u64){continue;}self.attempts+=1;
                    let Some((len,value))=candidate(bytes,start,anchor)else{self.rejected+=1;continue;};
                    let original=&bytes[start..start+len];
                    if self.documents.iter().any(|old|old==original){continue;}
                    let mut budget=2048;let selected=metadata(&value,0,&mut budget);
                    self.rows.push(json!({"address":format!("{:#x}",address+start as u64),"bytes":len,"anchor":String::from_utf8_lossy(needle),"metadata":selected}));self.documents.push(original.to_vec());
                }
            }
        }
    }
    pub(crate) fn report(&self)->String{
        let path=self.output.with_extension("menu-configs.json");
        let report=json!({"candidate_attempts":self.attempts,"invalid_sensitive_or_unselected_candidates":self.rejected,"documents":self.rows,"limits":"Original version-checked read-only menu heap scan only. Valid JSON objects must enclose an exact menu anchor; max64KiB per document,8 unique documents,256 attempts,16 nesting,2048 retained nodes. Sensitive property names reject entire object; only menu/platform/scene strings survive. No raw heap, unrestricted file content, keys, guest writes or virtual calls. Absence does not establish a missing configuration."});
        let written=std::fs::write(&path,serde_json::to_vec_pretty(&report).unwrap()).is_ok();
        json!({"candidate_attempts":self.attempts,"documents":self.rows.len(),"metadata_file":path.file_name().unwrap().to_string_lossy(),"written":written}).to_string()
    }
}
#[cfg(test)]mod tests{
    use super::*;
    #[test]fn reject_anchor_outside_object_and_sensitive_document(){
        let b=br#"{"valid":true} MainMenuSequenceTypes_NX_00"#;
        assert!(candidate(b,0,14).is_none());
        let b=br#"{"sequence":"MainMenuSequenceTypes_NX_00","token":"private"}"#;
        assert!(candidate(b,0,20).is_none());
    }
    #[test]fn preserve_menu_parameters_without_unselected_text(){
        let b=br#"{"sequence":"MainMenuSequenceTypes_NX_00","duration":5,"label":"unrelated text"} trailing"#;
        let (_,v)=candidate(b,0,20).unwrap();let mut budget=2048;let v=metadata(&v,0,&mut budget);
        assert_eq!(v["duration"],5);assert_eq!(v["sequence"],"MainMenuSequenceTypes_NX_00");assert!(v["label"].is_object());assert!(!v.to_string().contains("unrelated text"));
    }
}
