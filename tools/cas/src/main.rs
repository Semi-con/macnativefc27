use std::{env,fs::{self,OpenOptions},io::Write};

fn main()->Result<(),Box<dyn std::error::Error>> {
    let args:Vec<_>=env::args().collect();if args.len()!=4{return Err("usage: fc27-local-cas-decode INPUT NEW_OUTPUT EXPECTED_SIZE".into());}
    if fs::metadata(&args[1])?.len()>16*1024*1024{return Err("input file exceeds bound".into());}
    let expected=args[3].parse()?;let source=fs::read(&args[1])?;let output=fc27_local_cas_decode::decode(&source,expected)?;
    OpenOptions::new().write(true).create_new(true).open(&args[2])?.write_all(&output)?;
    println!("{}",output.len());Ok(())
}
