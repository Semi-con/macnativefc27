// Mush8 - program.rs
// Constantin Clerc (c22dev) - 0.1
use crate::table::{InstName, InstProps, Op, decode};
use anyhow::{Result, bail};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, Debug)]
pub struct Inst {
    pub addr: u32,
    pub raw: u64,
    pub name: InstName,
    pub op: Op,
    pub props: InstProps,
}

pub struct Program {
    pub insts: Vec<Inst>,
    pub targets: BTreeSet<u32>,
    pub converge: BTreeMap<u32, u32>,
    pub indirect_constant_bytes:Option<u32>,
    pub indirect_targets:BTreeMap<u32,BTreeSet<u32>>,
}

pub fn branch_target(i: &Inst) -> u32 {
    let rel = (((i.raw >> 20) as i32) << 8) >> 8;
    let t = (i.addr as i64 + rel as i64 + 8) as u32;
    if t % 32 == 0 { t + 8 } else { t }
}

impl Program {
    pub fn decode(code: &[u8]) -> Result<Program> {
        Self::decode_with_constants(code,None)
    }

    pub fn decode_with_constants(code:&[u8],constants:Option<&[u8]>)->Result<Program> {
        let mut insts = Vec::new();
        for (i, c) in code.chunks_exact(8).enumerate() {
            if i % 4 == 0 {
                continue;
            }
            let raw = u64::from_le_bytes(c.try_into().unwrap());
            let (name, op, props) = decode(raw);
            let inst = Inst { addr: (i * 8) as u32, raw, name, op, props };
            let rel = ((((raw >> 20) as i32) << 8) >> 8) + 8;
            if name == InstName::Bra && (-32..=0).contains(&rel) && (raw>>16)&15==7 && raw&31==15 {
                break;
            }
            insts.push(inst);
        }
        if insts.is_empty() {
            bail!("empty program");
        }
        let mut targets = BTreeSet::new();
        targets.insert(insts[0].addr);
        let mut converge = BTreeMap::new();
        let (mut ssy, mut pbk): (Vec<u32>, Vec<u32>) = (Vec::new(), Vec::new());
        for i in &insts {
            while ssy.last() == Some(&i.addr) {
                ssy.pop();
            }
            while pbk.last() == Some(&i.addr) {
                pbk.pop();
            }
            match i.name {
                InstName::Bra => {
                    targets.insert(branch_target(i));
                }
                InstName::Ssy => {
                    let t = branch_target(i);
                    targets.insert(t);
                    ssy.push(t);
                }
                InstName::Pbk => {
                    let t = branch_target(i);
                    targets.insert(t);
                    pbk.push(t);
                }
                InstName::Sync => {
                    if let Some(&t) = ssy.last() {
                        converge.insert(i.addr, t);
                    }
                }
                InstName::Brk => {
                    if let Some(&t) = pbk.last() {
                        converge.insert(i.addr, t);
                    }
                }
                _ => {}
            }
        }
        // Indirect jump-table entries may point to any instruction, including
        // addresses without a direct branch reference in this code section.
        let mut indirect_constant_bytes=None;
        let mut indirect_targets=BTreeMap::new();
        if insts.iter().any(|i|i.op==Op::Brx) {
            if let Some((additional,size))=constants.and_then(|c|constant_jump_targets(&insts,&targets,c)) {
                targets.extend(additional.values().flatten().copied());indirect_constant_bytes=Some(size);indirect_targets=additional;
            } else {targets.extend(insts.iter().map(|i|i.addr));}
        }
        {
            // Shared-memory accesses yield to the SIMD scheduler at their
            // successor. Ordinary ALU instructions do not need switch labels;
            // thousands of redundant cases make Metal's pipeline optimizer
            // build an unnecessarily large register/PHI graph.
            for pair in insts.windows(2) {
                if matches!(pair[0].op,Op::Lds|Op::Sts) {targets.insert(pair[1].addr);}
            }
        }
        Ok(Program { insts, targets, converge,indirect_constant_bytes,indirect_targets })
    }
}

// A BRX loaded directly from the immutable embedded C1 bank can only select
// a value present in that bank. Keep every such valid destination, not a
// guessed table length/index or a game-specific address. Reset provenance
// at every possible incoming edge and reject unknown/multi-register writes.
fn constant_jump_targets(insts:&[Inst],direct:&BTreeSet<u32>,constants:&[u8])->Option<(BTreeMap<u32,BTreeSet<u32>>,u32)> {
    use crate::inst::{InstBrx,InstLdc};
    if constants.is_empty() || constants.len()>65536 || constants.len()%4!=0 {return None;}
    let valid=insts.iter().map(|i|i.addr).collect::<BTreeSet<_>>();
    let mut indirect=BTreeSet::new();
    for i in insts.iter().filter(|i|i.op==Op::Brx) {
        let o=InstBrx(i.raw);if o.ca()||o.lmt() {return None;}
        let base=i.addr.wrapping_add(8).wrapping_add(((o.imm24()<<8)>>8) as u32);
        let mut found=false;
        for bytes in constants.chunks_exact(4) {
            let t=base.wrapping_add(u32::from_le_bytes(bytes.try_into().unwrap()));
            let t=if t%32==0 {t.wrapping_add(8)} else {t};
            if valid.contains(&t) {indirect.insert(t);found=true;}
        }
        if !found {return None;}
    }
    let mut loaded=BTreeMap::<i32,BTreeSet<u32>>::new();
    let mut bounds=BTreeMap::<i32,u32>::new();
    let mut targets=BTreeMap::new();
    for i in insts {
        if direct.contains(&i.addr)||indirect.contains(&i.addr) {loaded.clear();bounds.clear();}
        if i.op==Op::Brx {
            let o=InstBrx(i.raw);let base=i.addr.wrapping_add(8).wrapping_add(((o.imm24()<<8)>>8)as u32);
            let next=loaded.get(&o.src_a())?.iter().map(|w|base.wrapping_add(*w)).map(|t|if t%32==0 {t.wrapping_add(8)}else{t}).filter(|t|valid.contains(t)).collect::<BTreeSet<_>>();
            if next.is_empty(){return None;}
            targets.insert(i.addr,next);
        }
        // Bounds are proven only within a block, for unconditional unsigned
        // min-immediate and non-overflowing left shifts. Unknown inputs may
        // take every u32 value; unknown writes and incoming edges erase facts.
        let bound=jump_index_bound(i,&bounds);
        let words=if i.op==Op::Ldc {
            let o=InstLdc(i.raw);
            let maximum=if o.src_a()==255 {Some(0)}else{bounds.get(&o.src_a()).copied()};
            let offset=o.cbuf_offset()as u16 as i16;
            let range=maximum.filter(|_|offset>=0).and_then(|max|max.checked_add(offset as u32))
                .filter(|&end|(end as usize&!3)+4<=constants.len())
                .map(|end|(offset as usize&!3)..(end as usize&!3)+4);
            let bytes=range.map_or(constants,|range|&constants[range]);
            Some(bytes.chunks_exact(4).map(|b|u32::from_le_bytes(b.try_into().unwrap())).collect::<BTreeSet<_>>())
        }else{None};
        if i.props.has(InstProps::RD2) {loaded.clear();bounds.clear();}
        if i.props.has(InstProps::RD) {
            let dest=(i.raw&255) as i32;
            let scalar=matches!(i.op,
                Op::FaddR|Op::FaddI|Op::FaddC|Op::Fadd32i|
                Op::FmulR|Op::FmulI|Op::FmulC|Op::Fmul32i|
                Op::FfmaR|Op::FfmaI|Op::FfmaC|Op::FfmaRc|Op::Ffma32i|
                Op::FmnmxR|Op::FmnmxI|Op::FmnmxC|Op::Mufu|
                Op::MovR|Op::MovI|Op::MovC|Op::Mov32i|
                Op::SelR|Op::SelI|Op::SelC|Op::ImnmxR|Op::ImnmxI|Op::ImnmxC|
                Op::ShlR|Op::ShlI|Op::ShlC);
            let scalar=scalar||(i.op==Op::Ldc&&InstLdc(i.raw).ls_size().0==4);
            if scalar {loaded.remove(&dest);bounds.remove(&dest);} else {loaded.clear();bounds.clear();}
            if let Some(max)=bound.filter(|_|dest!=255){bounds.insert(dest,max);}
            if i.op==Op::Ldc {
                let o=InstLdc(i.raw);
                if o.cbuf_slot()==1 && o.address_mode().0<2 && o.ls_size().0==4 && o.pred()==7 && !o.pred_inv() && dest!=255 {loaded.insert(dest,words.unwrap());}
            }
        }
        if matches!(i.op,Op::Bra|Op::Brx|Op::Sync|Op::Brk|Op::Exit|Op::Ssy|Op::Pbk) {loaded.clear();bounds.clear();}
    }
    Some((targets,constants.len() as u32))
}

fn jump_index_bound(i:&Inst,bounds:&BTreeMap<i32,u32>)->Option<u32>{
    use crate::inst::{InstImnmxI,InstShlI};
    if (i.raw>>16)&15!=7{return None;}
    match i.op{
        Op::ImnmxI=>{
            let o=InstImnmxI(i.raw);
            (!o.signed()&&o.src_pred()==7&&!o.src_pred_inv()&&o.x_mode().0==0)
                .then_some(((o.imm20()<<12)>>12)as u32)
        },
        Op::ShlI=>{
            let o=InstShlI(i.raw);if o.x(){return None;}
            let shift=((o.imm20()<<12)>>12)as u32;
            let shift=if o.m(){shift&31}else if shift>=32{return Some(0);}else{shift};
            let max=if o.src_a()==255{0}else{*bounds.get(&o.src_a())?};
            (max<=u32::MAX>>shift).then(||max<<shift)
        },
        _=>None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn address(n:usize)->u32 {((n/3*4+1+n%3)*8) as u32}
    fn fixture(load:u64,clobber:u64)->Vec<u8> {
        let mut words=vec![0x010000000007f000,load,clobber,0xe25000000007040f,0x010000000017f005,0xe30000000007000f,0x010000000027f005,0xe30000000007000f];
        let delta=-(address(3) as i64+8);words[3]|=((delta as u64)&0xffffff)<<20;
        let mut raw=Vec::new();for group in words.chunks(3) {raw.extend_from_slice(&0u64.to_le_bytes());for w in group {raw.extend_from_slice(&w.to_le_bytes());}}raw
    }
    #[test]
    fn embedded_jump_table_keeps_all_valid_destinations_and_normalizes_control_words() {
        let code=fixture(0xef94001000070004,0x50b0000000070f00);
        let c=[address(4),address(6)-8,0xfffffff0].iter().flat_map(|v|v.to_le_bytes()).collect::<Vec<_>>();
        let p=Program::decode_with_constants(&code,Some(&c)).unwrap();assert_eq!(p.indirect_constant_bytes,Some(12));
        assert!(p.targets.contains(&address(4)));assert!(p.targets.contains(&address(6)));assert!(!p.targets.contains(&address(1)));
        assert!(Program::decode(&code).unwrap().targets.contains(&address(1)));
    }
    #[test]
    fn uncertain_jump_sources_preserve_the_full_instruction_dispatch() {
        let c=address(6).to_le_bytes();
        for (load,clobber) in [
            (0xef94001000000004,0x50b0000000070f00), // predicated load
            (0xef94003000070004,0x50b0000000070f00), // mutable C3
            (0xef95001000070004,0x50b0000000070f00), // 64-bit destination
            (0xef94001000070004,0x010000000017f004), // overwritten register
            (0xef94001000070004,0xeed6200000070003), // unknown overlapping vector write
        ] {
            let p=Program::decode_with_constants(&fixture(load,clobber),Some(&c)).unwrap();assert_eq!(p.indirect_constant_bytes,None);assert!(p.targets.contains(&address(1)));
        }
        // An indirect edge can enter at BRX without executing its preceding
        // LDC; lexical proximity is not sufficient proof of dominance.
        let code=fixture(0xef94001000070004,0x50b0000000070f00);
        assert_eq!(Program::decode_with_constants(&code,Some(&address(3).to_le_bytes())).unwrap().indirect_constant_bytes,None);
    }
    #[test]
    fn unsigned_clamped_table_excludes_padding_without_losing_any_case(){
        let make=|clamp:u64,shift:u64|{
            let mut words=vec![clamp,shift,0xef94001000075e5e,0xe250000000075e0f];
            let delta=-(address(3)as i64+8);words[3]|=((delta as u64)&0xffffff)<<20;
            for n in 0..5 {words.extend([0x010000000007f000|(n<<20),0xe30000000007000f]);}
            words.push(0xe2400fffff87000f);
            words.chunks(3).flat_map(|g|std::iter::once(0u64).chain(g.iter().copied())).flat_map(u64::to_le_bytes).collect::<Vec<_>>()
        };
        let clamp=0x3820038000473320;let shift=0x384800000027205e;
        let destinations=(0..5).map(|n|address(4+n*2)).collect::<BTreeSet<_>>();
        let mut c=destinations.iter().flat_map(|v|v.to_le_bytes()).collect::<Vec<_>>();c.resize(256,0);
        let p=Program::decode_with_constants(&make(clamp,shift),Some(&c)).unwrap();
        assert_eq!(p.indirect_targets.get(&address(3)),Some(&destinations));
        assert!(!p.indirect_targets[&address(3)].contains(&8),"zero padding must not create an entry backedge");
        for (a,b)in [(clamp^(1<<16),shift),(clamp|(1<<48),shift),(clamp|(1<<42),shift),(clamp,shift^(1<<16)),(clamp,shift|(1<<43))]{
            let p=Program::decode_with_constants(&make(a,b),Some(&c)).unwrap();
            assert!(p.indirect_targets[&address(3)].contains(&8),"unproved bound must keep every bank word: {a:x}/{b:x}");
        }
    }
    #[test]
    fn index_bounds_reject_shift_overflow_and_do_not_cross_incoming_edges(){
        let decode_inst=|raw|{let(name,op,props)=decode(raw);Inst{addr:8,raw,name,op,props}};
        let mut bounds=BTreeMap::from([(32,4)]);
        assert_eq!(jump_index_bound(&decode_inst(0x384800000027205e),&bounds),Some(16));
        bounds.insert(32,u32::MAX);assert_eq!(jump_index_bound(&decode_inst(0x384800000027205e),&bounds),None);
        assert_eq!(jump_index_bound(&decode_inst(0x3820038000473320),&BTreeMap::new()),Some(4));
        // If an indirect edge can enter at the load, its index no longer has
        // a block-local clamp proof. The full table including entry8 remains.
        let mut words=vec![0x3820038000473320u64,0x384800000027205e,0xef94001000075e5e,0xe250000000075e0f,0xe30000000007000f];
        words[3]|=((-(address(3)as i64+8)as u64)&0xffffff)<<20;
        let raw=words.chunks(3).flat_map(|g|std::iter::once(0u64).chain(g.iter().copied())).flat_map(u64::to_le_bytes).collect::<Vec<_>>();
        let c=[address(2),address(4),0].iter().flat_map(|v|v.to_le_bytes()).collect::<Vec<_>>();
        let p=Program::decode_with_constants(&raw,Some(&c)).unwrap();assert!(p.indirect_targets[&address(3)].contains(&8));
    }
}
