// Bounded, opt-in inspection of this test process's guest heap. Only selected
// ASCII engine-setting lines are retained; no raw memory or key data is saved.
use std::{path::Path,sync::atomic::{AtomicU32,Ordering::Relaxed},time::{Duration,Instant}};

const CHUNK:usize=1<<20;
const MAX_MATCHES:usize=256;
const MAX_LINE:usize=384;

fn relevant(line:&[u8])->bool {
    if line.is_empty() || line.len()>MAX_LINE || !line.iter().all(|b|(32..127).contains(b)||*b==b'\t') {return false;}
    let text=String::from_utf8_lossy(line).to_ascii_lowercase();
    if ["key","token","secret","password","credential","aes"].iter().any(|s|text.contains(s)) {return false;}
    ["gametime.","renderdevice.","frameratecap","maxsimfps","targetfpsis60","framepacing","preferredframerate"].iter().any(|s|text.contains(s))
}

fn lines(bytes:&[u8])->Vec<(usize,String)> {
    let mut result=Vec::new();let mut start=0;
    for end in 0..=bytes.len() {
        if end<bytes.len() && ((32..127).contains(&bytes[end])||bytes[end]==b'\t') {continue;}
        if relevant(&bytes[start..end]) {result.push((start,String::from_utf8_lossy(&bytes[start..end]).into_owned()));if result.len()==MAX_MATCHES {break;}}
        start=end+1;
    }
    result
}

pub(super) fn quote_ascii(line:&str)->String {
    let mut out=String::from("\"");
    for c in line.chars() {match c {'\\'=>out.push_str("\\\\"),'"'=>out.push_str("\\\""),'\t'=>out.push_str("\\t"),'\r'=>out.push_str("\\r"),'\n'=>out.push_str("\\n"),_=>out.push(c)}}
    out.push('"');out
}

fn variable(bytes:&[u8],cursor:&mut usize)->Option<usize> {
    let mut value=0u32;
    for shift in (0..35).step_by(7) {
        let b=*bytes.get(*cursor)?;*cursor+=1;
        if shift==28 && b>15 {return None;}
        value|=((b&127) as u32)<<shift;
        if b<128 {return Some(value as usize);}
    }
    None
}

// Frostbite DbObject named dictionary: only the file name and payload are
// returned. Length-delimited fields are bounded by the owned OS copy.
fn file_record(bytes:&[u8])->Option<(&str,&[u8])> {
    if !bytes.starts_with(b"\x02$file\0") {return None;}
    let mut cursor=7;let size=variable(bytes,&mut cursor)?;
    let body=bytes.get(cursor..cursor.checked_add(size)?)?;let mut at=0;let mut name:Option<&str>=None;let mut payload:Option<&[u8]>=None;
    for _ in 0..16 {
        let tag=*body.get(at)?;at+=1;
        if tag==0 {
            if at!=body.len() {return None;}
            let name=name?;
            if name.len()>255 || name.bytes().any(|b|!(32..127).contains(&b)) || name.to_ascii_lowercase().contains("key") {return None;}
            return Some((name,payload?));
        }
        if tag&128!=0 {return None;}
        let end=at+body.get(at..)?.iter().take(32).position(|b|*b==0)?;
        let field=&body[at..end];at=end+1;
        let length=match tag {1|2|7|19=>variable(body,&mut at)?,6=>1,8|11=>4,9|12=>8,15=>16,16=>20,_=>return None};
        let value=body.get(at..at.checked_add(length)?)?;at+=length;
        if tag==7 && field==b"name" {name=Some(std::str::from_utf8(value).ok()?.trim_end_matches('\0'));}
        if tag==19 && field==b"payload" {payload=Some(value);}
    }
    None
}

fn document_rows(bytes:&[u8])->Vec<String> {
    let mut documents=Vec::new();
    for (offset,_) in bytes.windows(7).enumerate().filter(|(_,w)|*w==b"\x02$file\0") {
        let Some((name,payload))=file_record(&bytes[offset..]) else {continue;};
        if payload.len()>CHUNK {continue;}
        let Ok(text)=std::str::from_utf8(payload) else {continue;};
        let rows=text.lines().collect::<Vec<_>>();let mut selected=std::collections::BTreeSet::new();
        for (index,line) in rows.iter().enumerate() {
            let lower=line.to_ascii_lowercase();
            if ["presentinterval","forceusesleeptimer","gametime.","frameratecap","maxsimfps","targetfpsis60","vsyncenable"].iter().any(|s|lower.contains(s)) && relevant(line.as_bytes()) {
                for n in index.saturating_sub(2)..=(index+2).min(rows.len()-1) {selected.insert(n);}
            }
        }
        let mut result=Vec::new();
        for n in selected.into_iter().take(120) {
            let line=rows[n];let lower=line.to_ascii_lowercase();
            if line.len()<=MAX_LINE && line.bytes().all(|b|(32..127).contains(&b)||b==b'\t') && !["key","token","secret","password","credential","aes"].iter().any(|s|lower.contains(s)) {
                result.push(format!(r#"{{"line":{},"text":{}}}"#,n+1,quote_ascii(line)));
            }
        }
        if !result.is_empty() {documents.push(format!(r#"{{"file":{},"rows":[{}]}}"#,quote_ascii(name),result.join(",")));}
        if documents.len()==16 {break;}
    }
    documents
}

fn text_contexts(bytes:&[u8])->Vec<String> {
    let allowed=|b:u8|(32..127).contains(&b)||matches!(b,b'\t'|b'\r'|b'\n');
    let safe=|s:&str|s.len()<=MAX_LINE && !["key","token","secret","password","credential","aes"].iter().any(|t|s.to_ascii_lowercase().contains(t));
    let mut contexts=Vec::new();let mut start=0;
    for end in 0..=bytes.len() {
        if end<bytes.len() && allowed(bytes[end]) {continue;}
        let text=std::str::from_utf8(&bytes[start..end]).unwrap();
        if text.contains('\n') {
            let rows=text.lines().collect::<Vec<_>>();let mut selected=std::collections::BTreeSet::new();
            for (index,line) in rows.iter().enumerate() {
                let lower=line.to_ascii_lowercase();
                if ["presentinterval","forceusesleeptimer","gametime.","frameratecap","maxsimfps","targetfpsis60"].iter().any(|t|lower.contains(t)) && safe(line) {
                    for n in index.saturating_sub(12)..=(index+3).min(rows.len()-1) {selected.insert(n);}
                    for n in 0..index {let head=rows[n].trim_start();if head.starts_with('[')||head.starts_with("if ")||head.starts_with("if(")||head.starts_with("#if")||head.starts_with("elseif")||head.starts_with("else") {selected.insert(n);}}
                }
            }
            if !selected.is_empty() {
                for n in 0..rows.len().min(6) {selected.insert(n);}
                let result=selected.into_iter().filter(|n|safe(rows[*n])).take(240).map(|n|format!(r#"{{"line":{},"text":{}}}"#,n+1,quote_ascii(rows[n]))).collect::<Vec<_>>();
                contexts.push(format!(r#"{{"rows":[{}],"source":"Printable decoded text segment; filename not established"}}"#,result.join(",")));
                if contexts.len()==16 {break;}
            }
        }
        start=end+1;
    }
    contexts
}

pub(crate) fn snapshot(address:u64,output:&mut [u8])->bool {
    if output.is_empty() || output.len()>CHUNK || address.checked_add(output.len() as u64).is_none() {return false;}
    unsafe extern "C" {
        static mach_task_self_:u32;
        fn mach_vm_read_overwrite(task:u32,address:u64,size:u64,output:u64,copied:*mut u64)->i32;
    }
    let mut copied=0u64;
    let code=unsafe {mach_vm_read_overwrite(mach_task_self_,address,output.len() as u64,output.as_mut_ptr() as u64,&mut copied)};
    code==0 && copied==output.len() as u64
}

fn inspect(output:&Path) {
    let started=Instant::now();let space=crate::kernel::aspace();let first=space.heap();
    // Both previous complete scans placed every matched setting below75MiB.
    // Keep generous headroom while avoiding rescanning unrelated late heaps.
    let bound=first+space.layout.heap_size.min(256u64<<20);
    let mut cursor=first;let mut bytes_scanned=0u64;let mut failed_chunks=0usize;let mut matches=Vec::new();let mut documents=Vec::new();let mut contexts=Vec::new();let mut buffer=vec![0u8;CHUNK];
    while cursor<bound && matches.len()<MAX_MATCHES && started.elapsed()<Duration::from_secs(15) {
        let (base,size,block)=space.map.lock().unwrap().query(cursor);
        let Some(end)=base.checked_add(size).map(|v|v.min(bound)).filter(|v|*v>cursor) else {break;};
        if block.state!=crate::mem::state::NORMAL || block.perm&crate::mem::perm::R==0 {cursor=end;continue;}
        let amount=(end-cursor).min(CHUNK as u64) as usize;
        // The OS copies into owned memory. A concurrent guest unmap produces
        // an error rather than a host dereference; a scan is not a snapshot.
        if snapshot(cursor,&mut buffer[..amount]) {
            bytes_scanned+=amount as u64;
            if documents.len()<16 {for document in document_rows(&buffer[..amount]) {if !documents.contains(&document) {documents.push(document);}if documents.len()==16 {break;}}}
            if contexts.len()<16 {for context in text_contexts(&buffer[..amount]) {if !contexts.contains(&context) {contexts.push(context);}if contexts.len()==16 {break;}}}
            for (offset,text) in lines(&buffer[..amount]) {
                matches.push(format!(r#"{{"heap_offset":{},"text":{}}}"#,cursor-first+offset as u64,quote_ascii(&text)));
                if matches.len()==MAX_MATCHES {break;}
            }
        } else {failed_chunks+=1;}
        cursor+=amount as u64;
    }
    let report=format!(r#"{{"bytes_scanned":{bytes_scanned},"seconds":{},"failed_chunks":{failed_chunks},"matches":[{}],"documents":[{}],"contexts":[{}],"range_complete":{},"limits":"Guest NORMAL readable heap only, first256MiB,15seconds,256matching lines,384bytes per line. Up to16 validated DbObject files/120 rows and16 printable contexts/240 rows; sensitive-key words excluded. Chunk-boundary records may be missed. Concurrent changes are not a coherent heap snapshot. No raw memory retained."}}"#,started.elapsed().as_secs_f64(),matches.join(","),documents.join(","),contexts.join(","),cursor>=bound);
    let _=std::fs::write(output,report);
    eprintln!("[config-inspection] finished bytes={bytes_scanned} matches={} output={}",matches.len(),output.display());
}

pub(super) fn poll(control:&Path) {
    let request=control.with_extension("inspect");
    let Some(raw)=super::read(&request) else {return;};
    let _=std::fs::remove_file(&request);
    let Ok(sequence)=raw.trim().parse::<u64>() else {return;};
    static COUNT:AtomicU32=AtomicU32::new(0);
    if sequence==0 || COUNT.fetch_add(1,Relaxed)>=2 {return;}
    let output=control.with_extension(format!("inspect-{sequence}.json"));
    let _=std::thread::Builder::new().name("config-inspection".into()).spawn(move ||inspect(&output));
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn only_bounded_frame_settings_are_retained() {
        let result=lines(b"unrelated user data\0GameTime.MaxSimFps 60\r\nRenderDevice.VSyncEnable 0\0GameTime.SecretKey secret\0");
        assert_eq!(result.iter().map(|(_,s)|s.as_str()).collect::<Vec<_>>(),["GameTime.MaxSimFps 60","RenderDevice.VSyncEnable 0"]);
        assert!(!relevant(b"GameTime.password abc"));assert!(!relevant(b"RenderDevice.token abc"));assert!(!relevant(b"GameTime.\xff"));
        let long=format!("GameTime.{}","x".repeat(MAX_LINE));assert!(!relevant(long.as_bytes()));
        let many=b"MaxSimFps\0".repeat(MAX_MATCHES+20);assert_eq!(lines(&many).len(),MAX_MATCHES);
    }
    #[test]
    fn self_task_snapshot_copies_owned_bytes_and_rejects_bad_ranges() {
        let original=*b"settings";let mut copy=[0u8;8];
        assert!(snapshot(original.as_ptr() as u64,&mut copy));assert_eq!(copy,original);
        assert!(!snapshot(0x121300000500,&mut copy));assert!(!snapshot(u64::MAX-3,&mut copy));assert!(!snapshot(0,&mut []));
    }
    #[test]
    fn setting_text_quotes_cannot_escape_json_strings() {
        assert_eq!(quote_ascii("GameTime.\"a\\b\"\t60"),"\"GameTime.\\\"a\\\\b\\\"\\t60\"");
        assert_eq!(quote_ascii("GameTime.Foo\r\n"),"\"GameTime.Foo\\r\\n\"");
    }
    #[test]
    fn validated_file_records_associate_settings_with_their_source_name() {
        fn var(mut n:usize,out:&mut Vec<u8>) {loop {let b=(n&127) as u8;n>>=7;out.push(b|if n>0 {128} else {0});if n==0 {break;}}}
        let mut body=b"\x07name\0".to_vec();let name=b"Scripts/Platform.ini\0";var(name.len(),&mut body);body.extend_from_slice(name);
        body.extend_from_slice(b"\x13payload\0");let text=b"if IsNX()\nRenderDevice.PresentInterval 2\nend\n";var(text.len(),&mut body);body.extend_from_slice(text);body.push(0);
        let mut data=b"\x02$file\0".to_vec();var(body.len(),&mut data);data.extend_from_slice(&body);
        assert_eq!(file_record(&data),Some(("Scripts/Platform.ini",&text[..])));
        let documents=document_rows(&data);assert_eq!(documents.len(),1);assert!(documents[0].contains("Scripts/Platform.ini"));assert!(documents[0].contains("if IsNX()"));
        data.pop();assert!(file_record(&data).is_none());let mut at=0;assert!(variable(&[255;8],&mut at).is_none());
    }
    #[test]
    fn printable_context_keeps_platform_guards_and_excludes_secrets() {
        let data=b"\0# configuration\nif Platform == 'NX'\nGameTime.ForceUseSleepTimer true\nGameTime.SecretKey hidden\nend\0";
        let rows=text_contexts(data);assert_eq!(rows.len(),1);assert!(rows[0].contains("Platform == 'NX'"));assert!(rows[0].contains("ForceUseSleepTimer true"));assert!(!rows[0].contains("hidden"));assert!(!rows[0].contains("SecretKey"));
        assert!(text_contexts(b"GameTime.ForceUseSleepTimer true\0").is_empty());
    }
}
