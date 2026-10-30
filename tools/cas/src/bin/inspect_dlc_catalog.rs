//! Read only DLC file catalog from the original XCI, without payload extraction.
use std::{env,path::Path,fs};
use serde_json::json;
fn main()->Result<(),Box<dyn std::error::Error>>{
 let a=env::args().collect::<Vec<_>>();if a.len()!=4{return Err("usage: inspect_dlc_catalog XCI KEY_FILE NEW_OUTPUT_JSON".into());}let out=Path::new(&a[3]);if out.exists(){return Err("Refusing existing output".into());}let keys=nxfs::keys::Keys::load(Path::new(&a[2]))?;let nsp=nxfs::Nsp::open(Path::new(&a[1]),&keys)?;let mut rows=Vec::new();
 for nca in &nsp.ncas{if !matches!(nca.content_type,nxfs::nca::ContentType::Data|nxfs::nca::ContentType::PublicData){continue;}let s=nca.sections.iter().find(|s|s.fs_type==nxfs::nca::FsType::RomFs).ok_or("DLC has no RomFS")?;let romfs=nxfs::romfs::RomFs::open(nca.fs_storage(s)?)?;let files=romfs.files();if files.len()>150000{return Err("DLC catalog bound".into());}rows.push(json!({"title_id":format!("{:016x}",nca.program_id),"content_type":format!("{:?}",nca.content_type),"nca_name":nca.name,"files":files.iter().map(|f|json!({"path":f.path,"size":f.size,"offset":f.offset})).collect::<Vec<_>>()}));}
 let result=json!({"source_modified":false,"payload_extracted":false,"dlc":rows});fs::write(out,serde_json::to_vec_pretty(&result)?)?;println!("{}",serde_json::to_string_pretty(&result)?);Ok(())
}
