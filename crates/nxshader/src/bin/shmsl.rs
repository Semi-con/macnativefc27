// Mush8 - shmsl.rs
// Constantin Clerc (c22dev) - 0.1
fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let compute = args.first().is_some_and(|a| a == "--compute");
    if compute {
        args.remove(0);
    }
    let constants=if let Some(index)=args.iter().position(|s|s=="--constants") {
        let bytes=std::fs::read(&args[index+1]).expect("read constants");args.drain(index..index+2);Some(bytes)
    } else {None};
    let blob = std::fs::read(&args[0]).expect("read");
    let control=std::path::Path::new(&args[0]).with_extension("control");
    let memory=std::fs::read(control).ok().filter(|c|c.len()>=0x758).map(|c| {
        (u32::from_le_bytes(c[0x710..0x714].try_into().unwrap()),u32::from_le_bytes(c[0x754..0x758].try_into().unwrap()))
    }).unwrap_or((0,0));
    let r = if compute {
        if let Some(bytes)=constants {nxshader::translate_compute_with_constants(&blob,memory.0,memory.1,&bytes)} else {nxshader::translate_compute_with_memory(&blob,memory.0,memory.1)}
    } else {
        if let Some(bytes)=constants {nxshader::translate_with_constants(&blob,0,0,0,&bytes)} else {nxshader::translate(&blob)}
    };
    match r {
        Ok(t) => {
            if let Some(o) = args.get(1) {
                std::fs::write(o, &t.msl).unwrap();
            } else {
                print!("{}", t.msl);
            }
            eprintln!("{:?} {} interface {:?}", t.stage, t.entry, t.interface);
        }
        Err(e) => {
            eprintln!("error: {e:#}");
            std::process::exit(1);
        }
    }
}
