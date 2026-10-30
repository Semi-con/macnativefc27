pub fn decode(source:&[u8],expected:usize)->Result<Vec<u8>,Box<dyn std::error::Error>> {
    if source.len()>16*1024*1024 || !(1..=16*1024*1024).contains(&expected){return Err("bounded source/output size exceeded".into());}
    let mut at=0usize;let mut output=Vec::with_capacity(expected);let mut blocks=0;
    while at<source.len() {
        blocks+=1;if blocks>256{return Err("CAS block count limit".into());}
        let header=source.get(at..at+8).ok_or("truncated CAS header")?;at+=8;
        let packed=u64::from_be_bytes(header.try_into()?);
        let flags=(packed>>56)as u8;let count=((packed>>32)&0xff_ffff)as usize;let codec=(packed>>24)as u8;let size=(packed&0xf_ffff)as usize;
        if flags!=0 || (packed>>20)&15!=7 || count==0 || output.len()+count>expected{return Err("CAS header size/flags differ from verified format".into());}
        let n=if codec==0{count}else{size};let bytes=source.get(at..at+n).ok_or("truncated CAS payload")?;at+=n;
        let first=output.len();output.resize(first+count,0);
        match codec {
            0=>output[first..].copy_from_slice(bytes),
            25=>{let n=oozextract::Extractor::new().read_from_slice(bytes,&mut output[first..])?;if n!=count{return Err("Oodle output length mismatch".into());}},
            _=>return Err(format!("unsupported CAS codec {codec}").into()),
        }
    }
    if output.len()!=expected{return Err("total decompressed size mismatch".into());}
    Ok(output)
}
