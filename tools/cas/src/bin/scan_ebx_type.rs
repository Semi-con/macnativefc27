//! Bounded read-only original CAS/EBX type inventory. Does not run game code.
use std::{collections::{HashMap,HashSet},env,fs::{self,File,OpenOptions},io::{BufRead,BufReader,Write},path::Path,time::{Duration,Instant}};
use serde_json::{Value,json};
use sha1::{Digest,Sha1};
type Error=Box<dyn std::error::Error>;
fn u32_at(b:&[u8],at:usize)->Result<u32,Error>{Ok(u32::from_le_bytes(b.get(at..at+4).ok_or("EBX integer bound")?.try_into()?))}
fn classify(b:&[u8],targets:&HashSet<Vec<u8>>)->Result<Option<Value>,Error>{
    if b.get(..4)!=Some(b"RIFF")||u32_at(b,4)?as usize+8!=b.len()||!matches!(b.get(8..12),Some(b"EBX\0")|Some(b"EBXS")){return Err("Unsupported or invalid original EBX RIFF".into());}
    let(mut at,mut efix,mut payload)=(12,None,None);
    while at<b.len(){let n=u32_at(b,at+4)?as usize;let start=at+8;let end=start.checked_add(n).ok_or("RIFF overflow")?;let bytes=b.get(start..end).ok_or("RIFF chunk bound")?;
        if &b[at..at+4]==b"EFIX"{if efix.is_some(){return Err("Duplicate EFIX".into());}efix=Some(bytes);}
        if &b[at..at+4]==b"EBXD"{payload=Some((start+15)&!15);}
        at=end.checked_add(n&1).ok_or("RIFF padding overflow")?;
    }
    if at!=b.len(){return Err("RIFF total length bound".into());}let data=efix.ok_or("Missing EFIX")?;let n=u32_at(data,16)?as usize;if n>65536{return Err("EFIX type count bound".into());}
    let guids=data.get(20..20+n*16).ok_or("EFIX GUID table bound")?;let matches=guids.chunks_exact(16).enumerate().filter(|(_,g)|targets.contains(*g)).map(|(i,g)|(i,g.to_vec())).collect::<Vec<_>>();
    if matches.is_empty(){return Ok(None);}
    let mut at=20+n*16;let signatures=u32_at(data,at)?as usize;at+=4;if signatures>65536{return Err("EFIX signature count bound".into());}at=at.checked_add(signatures*4).ok_or("EFIX overflow")?;
    let _exported=u32_at(data,at)?;let instances=u32_at(data,at+4)?as usize;at+=8;if instances==0||instances>65536{return Err("EFIX instance count bound".into());}let first=u32_at(data,at)?as usize;let root=payload.ok_or("Missing EBXD")?.checked_add(first).ok_or("EBXD overflow")?;let root_ref=u16::from_le_bytes(b.get(root..root+2).ok_or("EBXD root type bound")?.try_into()?)as usize;if root_ref>=n{return Err("Root type exceeds EFIX table".into());}
    Ok(Some(json!({"matched_type_refs":matches.iter().map(|(i,g)|json!({"index":i,"guid_bytes":g.iter().map(|v|format!("{v:02x}")).collect::<String>(),"is_root_type":*i==root_ref})).collect::<Vec<_>>(),"root_type_ref":root_ref,"instance_count":instances,"type_count":n})))
}
fn text<'a>(v:&'a Value,key:&str)->Result<&'a str,Error>{v.get(key).and_then(Value::as_str).ok_or_else(||format!("Missing manifest string {key}").into())}
fn number(v:&Value,key:&str)->Result<u64,Error>{v.get(key).and_then(Value::as_u64).ok_or_else(||format!("Missing manifest integer {key}").into())}
fn main()->Result<(),Error>{
    let args=env::args().collect::<Vec<_>>();if args.len()!=7{return Err("usage: scan_ebx_type XCI KEY_FILE METADATA_JSONL GUID_BYTES_CSV NEW_OUTPUT_DIR MAX_SECONDS".into());}
    let limit=args[6].parse::<u64>()?;if !(1..=180).contains(&limit){return Err("Time bound must be1..180 seconds".into());}let output=Path::new(&args[5]);fs::create_dir(output)?;
    let mut targets=HashSet::new();for hex in args[4].split(','){if hex.len()!=32{return Err("Target GUID must have32 hex digits".into());}let guid=(0..16).map(|i|u8::from_str_radix(&hex[2*i..2*i+2],16)).collect::<Result<Vec<_>,_>>()?;targets.insert(guid);}if targets.is_empty()||targets.len()>16{return Err("Target GUID count bound".into());}
    let keys=nxfs::keys::Keys::load(Path::new(&args[2]))?;let base=nxfs::Nsp::open(Path::new(&args[1]),&keys)?;let romfs=nxfs::romfs::RomFs::open(base.program(true)?.patched_romfs(base.program(false)?)?)?;let files=romfs.files().into_iter().map(|f|(f.path,(f.offset,f.size))).collect::<HashMap<_,_>>();
    let start=Instant::now();let mut seen=HashSet::new();let mut scanned=0u64;let mut duplicates=0u64;let mut compressed_bytes=0u64;let mut decoded_bytes=0u64;let mut errors=Vec::new();let mut error_count=0u64;let mut matches=Vec::new();let mut stopped=None;let mut lines=0u64;
    for line in BufReader::new(File::open(&args[3])?).lines(){
        if start.elapsed()>Duration::from_secs(limit){stopped=Some("time bound");break;}if lines>=150000||compressed_bytes>=2*1024*1024*1024{stopped=Some("record/byte bound");break;}lines+=1;let line=line?;if line.len()>4096{return Err("Manifest line bound".into());}let v:Value=serde_json::from_str(&line)?;let expected_hash=text(&v,"sha1")?;if seen.contains(expected_hash){duplicates+=1;continue;}
        let result=(||->Result<Option<Value>,Error>{
            let name=text(&v,"name")?;if name.len()>512{return Err("Asset name bound".into());}let loc=v.get("location").ok_or("Missing asset location")?;let path=text(loc,"file")?;if !(path.starts_with("/Data/")||path.starts_with("/Patch/"))||!path.ends_with(".cas"){return Err("Manifest path outside original CAS files".into());}let off=number(loc,"offset")?;let size=number(loc,"size")?;let expected=number(&v,"original_size")?;
            if !(1..=16*1024*1024).contains(&size)||!(1..=16*1024*1024).contains(&expected){return Err("Asset size bound".into());}let(fbase,flen)=files.get(path).ok_or("CAS file missing from merged RomFS")?;if off.checked_add(size).is_none_or(|end|end>*flen){return Err("CAS slice outside file".into());}
            let source=romfs.storage().read_vec(fbase+off,size as usize)?;if format!("{:x}",Sha1::digest(&source))!=expected_hash{return Err("Compressed original SHA1 mismatch".into());}seen.insert(expected_hash.to_owned());compressed_bytes+=size;
            let decoded=fc27_local_cas_decode::decode(&source,expected as usize)?;decoded_bytes+=decoded.len()as u64;scanned+=1;let Some(types)=classify(&decoded,&targets)?else{return Ok(None);};let dest=output.join(format!("{}.ebx",expected_hash));OpenOptions::new().create_new(true).write(true).open(&dest)?.write_all(&decoded)?;
            Ok(Some(json!({"name":name,"bytes":decoded.len(),"local":dest,"compressed_sha1_verified":expected_hash,"location":loc,"types":types})))
        })();match result{Ok(Some(row))=>{eprintln!("matched {}",row["name"]);matches.push(row);},Ok(None)=>{},Err(e)=>{error_count+=1;if errors.len()<16{errors.push(json!({"name":v.get("name"),"error":e.to_string()}));}}}
        if lines%5000==0{eprintln!("records={lines} scanned={scanned} matches={} errors={error_count} seconds={:.2}",matches.len(),start.elapsed().as_secs_f64());}
    }
    let result=json!({"records":lines,"scanned_valid_checksum_cas_assets":scanned,"duplicate_compressed_hashes":duplicates,"compressed_bytes":compressed_bytes,"decoded_bytes":decoded_bytes,"seconds":start.elapsed().as_secs_f64(),"error_count":error_count,"first_errors":errors,"stopped":stopped,"matches":matches,"complete_without_errors":stopped.is_none()&&error_count==0,"limits":"Read-only original merged RomFS, verified compressed SHA1 and bounded CAS/EBX structure. Max150000records,2GiB compressed,16MiB per asset,16target GUIDs,caller time bound<=180sec. Only exact matching decoded metadata written. No key contents exported, no source mutation, no guest execution, no absence conclusion on partial/errors."});fs::write(output.join("scan-result.json"),serde_json::to_vec_pretty(&result)?)?;println!("{}",json!({"records":lines,"scanned":scanned,"errors":error_count,"matches":result["matches"].as_array().unwrap().len(),"seconds":start.elapsed().as_secs_f64(),"stopped":stopped}));Ok(())
}
