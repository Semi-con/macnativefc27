//! Bounded read-only recovery of original RIFF/EBXT reflection metadata.
//! Invoked only by the version-checked isolated menu observer. No media or keys.
use std::{path::{Path,PathBuf},collections::HashSet};
#[derive(Clone,Copy)]struct Shape{signatures:u32,types:u32,fields:u32}
fn word(b:&[u8],at:usize)->Option<u32>{Some(u32::from_le_bytes(b.get(at..at+4)?.try_into().ok()?))}
fn skip_table(b:&[u8],at:&mut usize,stride:usize,max:u32)->Option<u32>{
    let n=word(b,*at)?;*at=at.checked_add(4)?;if n>max{return None;}
    *at=at.checked_add((n as usize).checked_mul(stride)?)?;(*at<=b.len()).then_some(n)
}
fn shape(b:&[u8])->Option<Shape>{
    if b.get(..4)?!=b"RIFF" || word(b,4)?as usize+8!=b.len() || b.get(8..12)?!=b"EBXT" {return None;}
    let mut at=12;let mut found=None;
    while at<b.len(){
        let tag=b.get(at..at+4)?;let len=word(b,at+4)?as usize;let start=at+8;let data=b.get(start..start.checked_add(len)?)?;
        if !matches!(tag,b"REFL"|b"RFL2") || found.is_some(){return None;}
        let mut p=0;let signatures=skip_table(data,&mut p,20,65536)?;let types=skip_table(data,&mut p,16,65536)?;
        let fields=skip_table(data,&mut p,12,1_048_576)?;
        skip_table(data,&mut p,12,1_048_576)?;skip_table(data,&mut p,8,1_048_576)?;
        if tag==b"RFL2"{let _=word(data,p)?;p+=4;}
        if p!=data.len() || signatures==0 || types==0{return None;}
        found=Some(Shape{signatures,types,fields});at=start.checked_add(len)?.checked_add(len&1)?;
    }
    (at==b.len()).then_some(found).flatten()
}
fn copy(address:u64,data:&mut[u8])->bool{
    for (i,chunk)in data.chunks_mut(1<<20).enumerate(){
        if !crate::input::test_control::config_inspection::snapshot(address+(i*(1<<20))as u64,chunk){return false;}
    }true
}
pub(crate) struct Observation{output:PathBuf,seen:HashSet<u64>,files:Vec<Vec<u8>>,rows:Vec<String>,candidates:u32,invalid:u32,copy_failures:u32}
impl Observation{
    pub(crate) fn new(output:&Path)->Self{Self{output:output.to_owned(),seen:HashSet::new(),files:Vec::new(),rows:Vec::new(),candidates:0,invalid:0,copy_failures:0}}
    pub(crate) fn observe(&mut self,bytes:&[u8],address:u64,step:usize,bound:u64){
        if self.files.len()>=2 || self.candidates>=16{return;}
        for at in 0..step.min(bytes.len().saturating_sub(11)){
            if &bytes[at..at+4]!=b"RIFF" || &bytes[at+8..at+12]!=b"EBXT"{continue;}
            let start=address+at as u64;if !self.seen.insert(start){continue;}self.candidates+=1;
            if self.candidates>16{return;}
            let len=word(bytes,at+4).unwrap()as usize+8;
            if !(32..=16*1024*1024).contains(&len) || start.checked_add(len as u64).is_none_or(|end|end>bound){self.invalid+=1;continue;}
            let mut data=vec![0;len];
            if !copy(start,&mut data){self.copy_failures+=1;continue;}
            let Some(s)=shape(&data)else{self.invalid+=1;continue;};
            if self.files.iter().any(|old|old==&data){continue;}
            let path=self.output.with_extension(format!("shared-types-{}.ebxt",self.files.len()));
            if std::fs::write(&path,&data).is_err(){continue;}
            let name=path.file_name().unwrap().to_string_lossy();
            self.rows.push(format!(r#"{{"file":"{name}","bytes":{len},"signatures":{},"types":{},"fields":{}}}"#,s.signatures,s.types,s.fields));
            self.files.push(data);if self.files.len()>=2{break;}
        }
    }
    pub(crate) fn report(&self)->String{format!(r#"{{"header_candidates":{},"invalid_or_unsupported_metadata":{},"copy_failures":{},"files":[{}],"limits":"Only complete RIFF/EBXT REFL or RFL2 integer/GUID descriptor-table grammar,2 unique files,max16MiB each,16header candidates. Unknown formats rejected. No arbitrary memory export, media, keys or game writes."}}"#,self.candidates,self.invalid,self.copy_failures,self.rows.join(","))}
}
