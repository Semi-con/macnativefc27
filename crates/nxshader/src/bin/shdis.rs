// Mush8 - shdis.rs
// Constantin Clerc (c22dev) - 0.1
use nxshader::table::{InstName, Op, decode as decode_op};

fn decode(blob: &[u8]) -> Vec<(usize, u64, InstName)> {
    let code = &blob[0x80.min(blob.len())..];
    nxshader::program::Program::decode(code).expect("decode shader").insts.into_iter()
        .map(|i|(i.addr as usize,i.raw,i.name)).collect()
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().map(String::as_str) == Some("--stats") {
        let mut counts: std::collections::HashMap<Op, usize> = std::collections::HashMap::new();
        let mut invalid = 0;
        for f in &args[1..] {
            for (_, w, n) in decode(&std::fs::read(f).unwrap()) {
                *counts.entry(decode_op(w).1).or_insert(0) += 1;
                if n == InstName::Invalid {
                    invalid += 1;
                }
            }
        }
        let mut v: Vec<_> = counts.into_iter().collect();
        v.sort_by(|a, b| b.1.cmp(&a.1));
        println!("{} distinct opcodes, {invalid} invalid", v.len());
        for (n, c) in v {
            println!("{c:7} {n:?}");
        }
    } else {
        for (off, w, n) in decode(&std::fs::read(&args[0]).unwrap()) {
            println!("{off:#06x}: {w:016x} {n:?}");
        }
    }
}
