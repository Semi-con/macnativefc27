//! Bounded local original-file scan. No game execution or source mutation.
use std::{collections::HashMap,env,fs::{self,File,OpenOptions},io::{BufRead,BufReader,Write},path::Path,time::{Duration,Instant}};
use serde_json::{Value,json};
use sha1::{Digest,Sha1};
type Error=Box<dyn std::error::Error>;
fn text<'a>(v:&'a Value,k:&str)->Result<&'a str,Error>{v[k].as_str().ok_or_else(||format!("Missing string {k}").into())}
fn num(v:&Value,k:&str)->Result<u64,Error>{v[k].as_u64().ok_or_else(||format!("Missing integer {k}").into())}
fn header(b:&[u8])->Result<(u64,u64),Error>{
 let h=u64::from_be_bytes(b.try_into()?);let n=(h>>32)&0xffffff;let codec=(h>>24)&255;
 if h>>56!=0||(h>>20)&15!=7||n==0||n>16*1024*1024||!matches!(codec,0|25){return Err("Unsupported CAS header".into());}
 let size=if codec==0{n}else{h&0xfffff};if size==0{return Err("Zero CAS size".into());}Ok((size,n))
}
fn total(b:&[u8])->Result<usize,Error>{
 let(mut at,mut total,mut blocks)=(0usize,0u64,0);
 while at<b.len(){let(size,n)=header(b.get(at..at+8).ok_or("Window header bound")?)?;at=at.checked_add(8+size as usize).ok_or("CAS overflow")?;if at>b.len(){return Err("Window payload bound".into());}total+=n;blocks+=1;if total>16*1024*1024||blocks>256{return Err("Window decode bound".into());}}
 Ok(total as usize)
}
fn main()->Result<(),Error>{
 let a=env::args().collect::<Vec<_>>();if !matches!(a.len(),6|7){return Err("usage: scan_legacy_menu XCI KEY_FILE METADATA_JSONL NEW_OUTPUT_DIR MAX_SECONDS [PATTERNS_JSON]".into());}let limit=a[5].parse::<u64>()?;if !(1..=180).contains(&limit){return Err("Time bound".into());}let output=Path::new(&a[4]);fs::create_dir(output)?;
 let keys=nxfs::keys::Keys::load(Path::new(&a[2]))?;let base=nxfs::Nsp::open(Path::new(&a[1]),&keys)?;let romfs=nxfs::romfs::RomFs::open(base.program(true)?.patched_romfs(base.program(false)?)?)?;let files=romfs.files().into_iter().map(|f|(f.path,(f.offset,f.size))).collect::<HashMap<_,_>>();
 let patterns:Vec<String>=if a.len()==7{let bytes=fs::read(&a[6])?;if bytes.len()>8192{return Err("Pattern file size bound".into());}serde_json::from_slice(&bytes)?}else{["stadium_backgrounds","backgroundstadiums","use3dmainmenu","main_menu_config","mainfeflow","mainmenuvideo","mainmenumovieplayer","fchubfluxcontext","usestillframes","slideshow","cinematicmainmenu"].into_iter().map(str::to_owned).collect()};
 if patterns.is_empty()||patterns.len()>32||patterns.iter().any(|p|!(4..=64).contains(&p.len())||!p.bytes().all(|b|b.is_ascii_graphic())||p.bytes().any(|b|b.is_ascii_uppercase())){return Err("Pattern count/ASCII/lowercase bound".into());}
 let start=Instant::now();let(mut records,mut scanned,mut compressed_bytes,mut decoded_bytes,mut errors)=(0u64,0u64,0u64,0u64,0u64);let mut first_errors=Vec::new();let mut matches=Vec::new();let mut lua_files=Vec::new();let mut magics:HashMap<String,u64>=HashMap::new();let mut prefixes:HashMap<(String,u64,u64),Vec<(u64,u64)>>=HashMap::new();let mut stopped=None;
 for line in BufReader::new(File::open(&a[3])?).lines(){
  if start.elapsed()>Duration::from_secs(limit){stopped=Some("time bound");break;}if records>=150000||compressed_bytes>=2*1024*1024*1024{stopped=Some("record/byte bound");break;}let line=line?;if line.len()>4096{return Err("Metadata line bound".into());}let v:Value=serde_json::from_str(&line)?;records+=1;
  let result=(||->Result<(),Error>{
   let name=text(&v,"name")?;if name.len()!=16||!name.bytes().all(|c|c.is_ascii_hexdigit()){return Err("Filename hash bound".into());}let loc=&v["location"];let path=text(loc,"file")?;if !(path.starts_with("/Data/")||path.starts_with("/Patch/"))||!path.ends_with(".cas"){return Err("Original CAS path bound".into());}let offset=num(loc,"offset")?;let chunk_size=num(loc,"size")?;let first=num(&v,"compressed_start")?;let last=num(&v,"compressed_end")?;let file_size=num(&v,"file_size")?;
   if first>=last||last>chunk_size||last-first>16*1024*1024||!(32..=15*1024*1024).contains(&file_size){return Err("Original file size/range bound".into());}let(fbase,flen)=files.get(path).ok_or("CAS file missing")?;if offset.checked_add(chunk_size).is_none_or(|end|end>*flen){return Err("Chunk outside CAS file".into());}
   let positions=prefixes.entry((path.to_owned(),offset,chunk_size)).or_insert_with(||vec![(0,0)]);
   while positions.last().unwrap().0<first{let(pos,logical)=*positions.last().unwrap();if pos+8>chunk_size{return Err("Prefix header bound".into());}let b=romfs.storage().read_vec(fbase+offset+pos,8)?;let(size,n)=header(&b)?;let next=pos+8+size;if next>chunk_size||positions.len()>4096{return Err("Prefix payload/block bound".into());}positions.push((next,logical+n));}
   let at=positions.binary_search_by_key(&first,|r|r.0).map_err(|_|"File compressed start is not a CAS block boundary")?;let logical=positions[at].1;
   let source=romfs.storage().read_vec(fbase+offset+first,(last-first)as usize)?;compressed_bytes+=source.len()as u64;let expected=total(&source)?;let body=fc27_local_cas_decode::decode(&source,expected)?;decoded_bytes+=body.len()as u64;let relative=num(&v,"uncompressed_offset")?.checked_sub(logical).ok_or("File precedes window")?as usize;let end=relative.checked_add(file_size as usize).ok_or("File range overflow")?;let file=body.get(relative..end).ok_or("File exceeds decoded window")?;scanned+=1;
   if let Some(expected)=v["expected_file_sha1"].as_str(){if format!("{:x}",Sha1::digest(file))!=expected{return Err("Independent fixture content mismatch".into());}}
   let magic=file[..4].iter().map(|x|format!("{x:02x}")).collect::<String>();*magics.entry(magic).or_default()+=1;
   if file.starts_with(b"\x1bLua"){lua_files.push(json!({"hash":name,"bytes":file.len(),"collector":v["collector"]}));}
   let retain=v["retain"].as_bool().unwrap_or(false);
   if file.starts_with(b"DDS ")&&!retain{return Ok(());}let mut lower=file.to_vec();lower.make_ascii_lowercase();let hits=patterns.iter().filter(|p|lower.windows(p.len()).any(|w|w==p.as_bytes())).cloned().collect::<Vec<_>>();
   if retain||!hits.is_empty(){if matches.len()>=256{return Err("Matching file output bound".into());}let dest=output.join(format!("{name}.bin"));OpenOptions::new().create_new(true).write(true).open(&dest)?.write_all(file)?;matches.push(json!({"hash":name,"bytes":file.len(),"collector":v["collector"],"patterns":hits,"explicitly_selected":retain,"local":dest,"original_metadata":v,"logical_prefix":logical,"window_expected":expected,"relative_offset":relative}));}
   Ok(())
  })();if let Err(e)=result{errors+=1;if first_errors.len()<16{first_errors.push(json!({"name":v.get("name"),"error":e.to_string()}));}}if records%2000==0{eprintln!("records={records} scanned={scanned} matches={} errors={errors} seconds={:.2}",matches.len(),start.elapsed().as_secs_f64());}
 }
 let result=json!({"patterns":patterns,"records":records,"scanned_valid_file_ranges":scanned,"compressed_bytes":compressed_bytes,"decoded_bytes":decoded_bytes,"seconds":start.elapsed().as_secs_f64(),"error_count":errors,"first_errors":first_errors,"stopped":stopped,"matches":matches,"lua_files":lua_files,"magic_counts":magics,"complete_without_errors":stopped.is_none()&&errors==0,"limits":"Read-only bounded original root/launch file windows; no chunk checksum supplied by original TOC, so no checksum verification claimed. Max150000 records,2GiB compressed,16MiB decode/window,180seconds. Files absent from this scope or not matching ASCII anchors remain unassessed."});fs::write(output.join("scan-result.json"),serde_json::to_vec_pretty(&result)?)?;println!("{}",json!({"records":records,"scanned":scanned,"errors":errors,"matches":result["matches"].as_array().unwrap().len(),"lua_files":result["lua_files"].as_array().unwrap().len(),"seconds":start.elapsed().as_secs_f64()}));Ok(())
}
