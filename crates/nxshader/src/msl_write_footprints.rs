// Proof metadata only: this does not rewrite instructions or generated MSL.
// Recognize a bounded grid-index STG128 address idiom, with actual generated
// register assignments so multi-register TEX/LD writes cannot escape the proof.
use crate::{inst::*,program::Program,table::Op};

#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub struct GridWriteFootprint {
    pub storage_slot:usize,
    pub row_bank:usize,
    pub row_word:usize,
}

impl GridWriteFootprint {
    pub fn bytes(self,base:u64,row:u32,groups:[u32;3])->Option<(u64,u64)> {
        if groups.contains(&0) || base%4!=0 {return None;}
        let index=(groups[1] as u64-1).checked_mul(row as u64)?.checked_add(groups[0] as u64-1)?;
        // The XMAD triplet and ISCADD operate in32 bits. Reject wrapping and
        // low-address carry rather than treating a wrapped range as contiguous.
        let last=index.checked_mul(16)?;
        if last>u32::MAX as u64 || (base as u32 as u64).checked_add(last)?.checked_add(15)?>u32::MAX as u64 {return None;}
        let len=last.checked_add(16)?;base.checked_add(len)?;
        Some((base,len))
    }
}

type Writes=Vec<(i32,String)>;
fn unpred(raw:u64)->bool {raw>>16&15==7}
fn last_def(writes:&[Writes],reg:i32,end:usize)->Option<usize> {(0..end).rev().find(|&n|writes[n].iter().any(|(r,_)|*r==reg))}
fn expr(writes:&[Writes],n:usize,reg:i32,want:String)->bool {writes[n].as_slice()==[(reg,want)]}

pub(super) fn analyze(p:&Program,w:&[Writes],body:&str)->Option<GridWriteFootprint> {
    if p.insts.len()!=w.len() || body.contains("cf =") {return None;}
    // Every additional global writer or indirect edge keeps the conservative
    // path. Images are excluded separately by the runtime resource interface.
    if p.insts.iter().any(|i|matches!(i.op,Op::St|Op::Atom|Op::AtomCas|Op::Red|Op::Brx)) {return None;}
    let stores=p.insts.iter().enumerate().filter(|(_,i)|i.op==Op::Stg).map(|(n,_)|n).collect::<Vec<_>>();
    let &[store]=stores.as_slice() else {return None;};
    let end=p.insts.last()?;
    if store+2!=p.insts.len() || end.op!=Op::Exit || !unpred(end.raw) || InstExit(end.raw).ccc()!=Ccc::T {return None;}
    let st=InstStg(p.insts[store].raw);
    if !st.e() || st.ls_size().0!=6 || st.imm24()!=0 || st.src_a()>=254 {return None;}
    let lo=st.src_a();let hi=lo+1;
    let low=last_def(w,lo,store)?;let high=last_def(w,hi,store)?;
    if p.insts[low].op!=Op::IscaddC || p.insts[high].op!=Op::IaddC || low>=high {return None;}
    let l=InstIscaddC(p.insts[low].raw);let h=InstIaddC(p.insts[high].raw);
    let word=l.cbuf_offset();
    if l.cbuf_slot()!=0 || word<0xc4 || (word-0xc4)%4!=0 || (word-0xc4)/4>=16 || l.imm5()!=4 || l.avg_mode().0!=0 {return None;}
    if h.cbuf_slot()!=0 || h.cbuf_offset()!=word+1 || h.src_a()!=255 || !h.x() || h.write_cc() || h.avg_mode().0!=0 || h.sat() {return None;}
    let flat=l.src_a();let third=last_def(w,flat,low)?;
    if p.insts[third].op!=Op::XmadR {return None;}
    let t=InstXmadR(p.insts[third].raw);let y=t.src_a();let merge=t.src_b();let x=t.src_c();
    if [x,y,merge,flat].contains(&255) || x==y {return None;}
    let first=last_def(w,x,third)?;let second=last_def(w,merge,third)?;
    if first>=second || second>=third || p.insts[first].op!=Op::XmadC || p.insts[second].op!=Op::XmadC {return None;}
    let f=InstXmadC(p.insts[first].raw);let b=InstXmadC(p.insts[second].raw);
    let bank=f.cbuf_slot();let row=f.cbuf_offset();
    if !(3..18).contains(&bank) || row<0 || b.cbuf_slot()!=bank || b.cbuf_offset()!=row {return None;}
    // Exact emitted expressions also check signedness, half-word selections,
    // carry modifiers and every source/destination, including overlapping regs.
    for (n,r,e) in [
        (first,x,format!("xmad(r{y}, res.cb[{bank}][{row}], r{x}, 0u, false, false, false, false, false, false, 0xffffffffu)")),
        (second,merge,format!("xmad(r{y}, res.cb[{bank}][{row}], 0u, 0u, false, false, false, true, false, true, 0xffffffffu)")),
        (third,flat,format!("xmad(r{y}, r{merge}, r{x}, 4u, false, false, true, true, true, false, 0xffffffffu)")),
        (low,lo,format!("((r{flat} << 4u) + res.cb[0][{word}])")),
        (high,hi,format!("(0u + res.cb[0][{}] + uint(cf))",word+1)),
    ] {if !unpred(p.insts[n].raw) || !expr(w,n,r,e) {return None;}}
    // The roots are unconditional CTAID loads in the straight-line entry
    // prefix. No incoming edge may bypass/re-execute them or enter the final
    // address sequence. No control operation is allowed within that sequence.
    let xi=last_def(w,x,first)?;let yi=last_def(w,y,first)?;
    for (n,r,sreg,e) in [(xi,x,0x25,"ctaid.x"),(yi,y,0x26,"ctaid.y")] {
        if p.insts[n].op!=Op::S2r || InstS2r(p.insts[n].raw).s_reg().0!=sreg || !expr(w,n,r,e.into()) {return None;}
    }
    let prefix=xi.max(yi);
    if prefix>=first || p.insts[..=prefix].iter().any(|i|!unpred(i.raw) || !matches!(i.op,Op::MovR|Op::MovI|Op::MovC|Op::Mov32i|Op::S2r)) {return None;}
    for (n,defs) in w.iter().enumerate() {
        if defs.iter().any(|(r,_)|*r==x) && n!=xi && n!=first {return None;}
        if defs.iter().any(|(r,_)|*r==y) && n!=yi {return None;}
    }
    if p.targets.iter().chain(p.converge.values()).any(|&target|
        (target<=p.insts[prefix].addr && target!=p.insts[0].addr) || (target>=p.insts[first].addr && target<=p.insts[store].addr)) {return None;}
    if p.insts[first..=store].iter().any(|i|matches!(i.op,Op::Bra|Op::Ssy|Op::Pbk|Op::Sync|Op::Brk|Op::Exit|Op::Brx)) {return None;}
    Some(GridWriteFootprint{storage_slot:((word-0xc4)/4) as usize,row_bank:bank as usize,row_word:row as usize})
}

#[cfg(test)]
mod tests {
    use super::*;
    fn instructions(extra:&[u64])->Vec<u64> {
        let mut v=vec![0xf0c800000257000fu64,0xf0c8000002670010];v.extend_from_slice(extra);
        v.extend([0x4e00078c0107100f,0x4f107f8c01071001,0x5b30079800171000,0x4c1882000c470000,0x4c1008000c57ff01,0xeede200000070004,0xe30000000007000f]);v
    }
    fn code(v:&[u64])->Vec<u8> {v.chunks(3).flat_map(|g|std::iter::once(0u64).chain(g.iter().copied())).flat_map(u64::to_le_bytes).collect()}
    fn proof(v:&[u64])->Option<GridWriteFootprint> {crate::translate_compute_with_memory(&code(v),0,0).unwrap().grid_write_footprint}
    #[test]
    fn grid_footprint_is_proved_from_instructions_and_rejects_control_and_vector_clobbers() {
        let v=instructions(&[]);let p=proof(&v).expect("canonical unsigned grid store");
        assert_eq!(p,GridWriteFootprint{storage_slot:0,row_bank:3,row_word:16});
        assert_eq!(p.bytes(0x3000000000,8,[8,5,1]),Some((0x3000000000,640)));
        // A second original STG, predicated address definition, shifted address
        // formula, root clobber by an overlapping LDG128, or incoming jump into
        // the tail cannot inherit the proof from a nearby canonical sequence.
        let mut extra_store=v.clone();extra_store.insert(2,0xeede200000070004);assert!(proof(&extra_store).is_none());
        let mut pred=v.clone();pred[2]&=!(7<<16);assert!(proof(&pred).is_none());
        let mut shift=v.clone();shift[5]^=1<<39;assert!(proof(&shift).is_none());
        let clobber=instructions(&[0xeed620000007000e]);assert!(proof(&clobber).is_none()); //R14..17 includes both roots
        let mut branch=v.clone();branch.insert(2,0xe24000000007000f);assert!(proof(&branch).is_none());
        let carry=instructions(&[0x5c10800000070004]);assert!(proof(&carry).is_none());
        let mut root=v.clone();root[0]&=!(7<<16);assert!(proof(&root).is_none());
    }
    #[test]
    fn grid_ranges_reject_wrap_and_keep_full_mapping_extent() {
        let p=proof(&instructions(&[])).unwrap();
        assert_eq!(p.bytes(0x3000000000,5,[2,3,1]),Some((0x3000000000,192)));
        assert!(p.bytes(0x30fffffff0,1,[2,1,1]).is_none());
        assert!(p.bytes(0x3000000000,u32::MAX,[1,2,1]).is_none());
        assert!(p.bytes(0xfffffffffffffff0,1,[2,1,1]).is_none());
        assert!(p.bytes(0x3000000001,1,[1,1,1]).is_none());
        assert!(p.bytes(0x3000000000,1,[0,1,1]).is_none());
    }
}

#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub struct DescriptorWriteFootprint {
    pub storage_slot:usize,
    pub offset:i64,
    pub bytes:u64,
}

impl DescriptorWriteFootprint {
    pub fn range(self,base:u64)->Option<(u64,u64)> {
        if base%4!=0 || self.offset%4!=0 {return None;}
        let address=base.checked_add_signed(self.offset)?;address.checked_add(self.bytes)?;
        Some((address,self.bytes))
    }
}

pub(super) fn descriptor_stores(p:&Program,w:&[Writes])->Option<Vec<DescriptorWriteFootprint>> {
    if p.insts.len()!=w.len() || p.insts.iter().any(|i|matches!(i.op,Op::St|Op::Atom|Op::AtomCas|Op::Red|Op::Brx)) {return None;}
    let mut stores=Vec::new();
    for (n,i) in p.insts.iter().enumerate().filter(|(_,i)|i.op==Op::Stg) {
        let st=InstStg(i.raw);let lo=st.src_a();let hi=lo+1;
        if !st.e() || lo>=254 || st.ls_size().0<4 {return None;}
        let low=last_def(w,lo,n)?;let high=last_def(w,hi,n)?;
        if p.insts[low].op!=Op::MovC || p.insts[high].op!=Op::MovC {return None;}
        let l=InstMovC(p.insts[low].raw);let h=InstMovC(p.insts[high].raw);let word=l.cbuf_offset();
        if l.cbuf_slot()!=0 || word<0xc4 || (word-0xc4)%4!=0 || (word-0xc4)/4>=16 || h.cbuf_slot()!=0 || h.cbuf_offset()!=word+1 {return None;}
        if !expr(w,low,lo,format!("res.cb[0][{word}]")) || !expr(w,high,hi,format!("res.cb[0][{}]",word+1)) {return None;}
        // Earlier ALU/loads may reuse these registers. What matters is that
        // every path executing this store executes both most-recent roots.
        // Keep that proof local: no control operation inside the region and
        // no incoming edge may bypass its first instruction. An edge to the
        // first root is safe because it reinitializes both address words.
        if !unpred(p.insts[low].raw) || !unpred(p.insts[high].raw) {return None;}
        let prefix=low.max(high);
        let preserved_entry=!w.iter().enumerate().any(|(at,defs)|defs.iter().any(|(reg,_)|(*reg==lo && at!=low)||(*reg==hi && at!=high)))
            && !p.insts[..=prefix].iter().any(|i|!unpred(i.raw) || !matches!(i.op,Op::MovR|Op::MovI|Op::MovC|Op::Mov32i|Op::S2r))
            && !p.targets.iter().chain(p.converge.values()).any(|&target|target<=p.insts[prefix].addr && target!=p.insts[0].addr);
        let start=low.min(high);
        let local_region=!p.insts[start..=n].iter().any(|i|matches!(i.op,Op::Bra|Op::Ssy|Op::Pbk|Op::Sync|Op::Brk|Op::Exit|Op::Brx))
            && !p.targets.iter().chain(p.converge.values()).any(|&target|target>p.insts[start].addr && target<=i.addr);
        if !preserved_entry && !local_region {return None;}
        let bytes=match st.ls_size().0 {4=>4,5=>8,6|7=>16,_=>return None,};
        let offset=((st.imm24()<<8)>>8) as i64;
        stores.push(DescriptorWriteFootprint{storage_slot:((word-0xc4)/4) as usize,offset,bytes});
        if stores.len()>64 {return None;}
    }
    if stores.is_empty() {None} else {Some(stores)}
}

#[cfg(test)]
mod descriptor_tests {
    use super::*;
    fn fixture(extra:&[u64])->Vec<u8> {
        let mut v=vec![0x4c9807800c470000u64,0x4c9807800c570001];v.extend_from_slice(extra);
        v.extend([0xeedc200000470002,0xeedc200000070003,0xe30000000007000f]);
        v.chunks(3).flat_map(|g|std::iter::once(0u64).chain(g.iter().copied())).flat_map(u64::to_le_bytes).collect()
    }
    fn proof(extra:&[u64])->Option<Vec<DescriptorWriteFootprint>> {crate::translate_compute_with_memory(&fixture(extra),0,0).unwrap().descriptor_write_footprint}
    fn word_code(words:&[u64])->Vec<u8> {words.chunks(3).flat_map(|g|std::iter::once(0u64).chain(g.iter().copied())).flat_map(u64::to_le_bytes).collect()}
    fn late_words()->Vec<u64> {
        vec![0x010000000017f000,0x010000000027f001,0x5c10800000070004,
             0x4c9807800c470000,0x4c9807800c570001,
             0xeedc200000470002,0xeedc200000070003,0xe30000000007000f]
    }
    fn late_proof(words:&[u64])->Option<Vec<DescriptorWriteFootprint>> {crate::translate_compute_with_memory(&word_code(words),0,0).unwrap().descriptor_write_footprint}
    fn branch_to(words:&mut Vec<u64>,target:usize) {
        words.insert(0,0xe24000000000000f); // P0 branch, retaining fallthrough
        let address=|n:usize|((n/3*4+1+n%3)*8) as i64;
        let delta=address(target)-address(0)-8;
        words[0]|=((delta as u64)&0xffffff)<<20;
    }
    #[test]
    fn late_descriptor_roots_require_an_uninterrupted_region_without_bypass_edges() {
        let words=late_words();let expected=proof(&[]).unwrap();
        assert_eq!(late_proof(&words),Some(expected.clone()));
        let mut to_start=words.clone();branch_to(&mut to_start,4);
        assert_eq!(late_proof(&to_start),Some(expected));
        for target in [5,6,7] {
            let mut bypass=words.clone();branch_to(&mut bypass,target);
            assert!(late_proof(&bypass).is_none(),"edge bypassed one or both roots: {target}");
        }
        let mut pred=words.clone();pred[3]&=!(7<<16);assert!(late_proof(&pred).is_none());
        let mut vector=words.clone();vector.insert(5,0xeed6200000070000);assert!(late_proof(&vector).is_none());
        let mut control=words;control.insert(4,0xe24000000000000f);assert!(late_proof(&control).is_none());
    }

    #[test]
    fn descriptor_register_reuse_keeps_each_store_bound_to_its_own_latest_roots() {
        let words=[0x4c9807800c470000,0x4c9807800c570001,0xeedc200000070002,
                   0x4c9807800c870000,0x4c9807800c970001,0xeedc200000070003,0xe30000000007000f];
        assert_eq!(late_proof(&words),Some(vec![DescriptorWriteFootprint{storage_slot:0,offset:0,bytes:4},DescriptorWriteFootprint{storage_slot:1,offset:0,bytes:4}]));
    }
    #[test]
    fn descriptor_stores_require_preserved_entry_address_roots_and_cover_every_store() {
        let v=proof(&[]).unwrap();assert_eq!(v,vec![DescriptorWriteFootprint{storage_slot:0,offset:4,bytes:4},DescriptorWriteFootprint{storage_slot:0,offset:0,bytes:4}]);
        for extra in [[0xeed6200000070000],[0xeedc200000070602],[0xe2400fffff07000f]] {assert!(proof(&extra).is_none(),"unproved address source: {extra:x?}");}
        assert!(proof(&[0xe24000000007000f]).is_some(),"a branch after complete address initialization is bounded");
        let mut bytes=fixture(&[]);bytes[8+2]&=!7; //predicate belongs to the raw instruction's third byte
        assert!(crate::translate_compute_with_memory(&bytes,0,0).unwrap().descriptor_write_footprint.is_none());
        let p=DescriptorWriteFootprint{storage_slot:0,offset:-4,bytes:8};
        assert_eq!(p.range(0x1000),Some((0xffc,8)));assert!(p.range(0).is_none());assert!(p.range(u64::MAX-3).is_none());
    }
}

// A conservative address domain, not a declaration-length assumption. The low
// address word can be arbitrary GPU data. Only the latest high-word definition
// must be dominated by an immutable CB0 storage-descriptor high word (+CF bool).
#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub struct WriteWindowFootprint {pub storage_slot:usize,pub carry:bool,pub offset:i64,pub bytes:u64,pub masked_high:bool}
impl WriteWindowFootprint {
    fn interval(self,high:u64,width:u64)->Vec<(u64,u64)> {
        let start=((high as i128+self.offset as i128-3).rem_euclid(1i128<<64)) as u64;
        let len=width+3+self.bytes-1;let end=start as u128+len as u128;
        if end<=1u128<<64 {vec![(start,len)]} else {let first=((1u128<<64)-start as u128) as u64;vec![(start,first),(0,len-first)]}
    }
    pub fn windows(self,base:u64)->Vec<(u64,u64)> {
        if !self.masked_high {return self.interval((base>>32)<<32,(1u64<<32)*if self.carry {2}else{1});}
        // ATOM/RED mask the high register to8 bits before adding the signed
        // immediate. Split a carry from255 to0 at the40-bit boundary; applying
        // the signed immediate still follows the original64-bit addition.
        let mut out=self.interval(((base>>32)&255)<<32,1u64<<32);
        if self.carry {out.extend(self.interval((((base>>32)+1)&255)<<32,1u64<<32));}out
    }
}

fn high_root(p:&Program,w:&[Writes],at:usize,high:i32,first:i32)->Option<(usize,bool)> {
    let root=&p.insts[at];let (word,carry)=match root.op {
        Op::MovC=>{let m=InstMovC(root.raw);if m.cbuf_slot()!=0 || !expr(w,at,high,format!("res.cb[0][{}]",m.cbuf_offset())) {return None;}(m.cbuf_offset(),false)},
        Op::IaddC=>{let m=InstIaddC(root.raw);if m.cbuf_slot()!=0 || m.src_a()!=255 || !m.x() || !expr(w,at,high,format!("(0u + res.cb[0][{}] + uint(cf))",m.cbuf_offset())) {return None;}(m.cbuf_offset(),true)},
        _=>return None,
    };
    (word>=first && (word-first)%4==0 && (word-first)/4<16).then_some((((word-first)/4) as usize,carry))
}
fn predecessors(p:&Program)->Option<Vec<Vec<usize>>> {
    use std::collections::BTreeMap;
    let indices=p.insts.iter().enumerate().map(|(n,i)|(i.addr,n)).collect::<BTreeMap<_,_>>();let mut reverse=vec![Vec::new();p.insts.len()];let mut edges=0;
    let ssy=p.insts.iter().filter(|i|i.op==Op::Ssy).map(crate::program::branch_target).collect::<Vec<_>>();
    let pbk=p.insts.iter().filter(|i|i.op==Op::Pbk).map(crate::program::branch_target).collect::<Vec<_>>();
    for (n,i) in p.insts.iter().enumerate() {
        // Include fallthrough even for unconditional branches. This is an
        // overapproximation; it can reject a proof but cannot invent dominance.
        let stop=i.op==Op::Exit && unpred(i.raw) && InstExit(i.raw).ccc()==Ccc::T;
        if n+1<p.insts.len() && !stop {reverse[n+1].push(n);edges+=1;}
        let targets=match i.op {Op::Bra=>vec![crate::program::branch_target(i)],Op::Brx=>p.targets.iter().copied().collect(),Op::Sync=>{if ssy.is_empty(){return None;}ssy.clone()},Op::Brk=>{if pbk.is_empty(){return None;}pbk.clone()},_=>Vec::new()};
        // Reconvergence stacks are lane-dependent. Every statically pushed
        // target is included rather than trusting one lexical stack instance.
        for target in targets {reverse[*indices.get(&target)?].push(n);edges+=1;if edges>131072{return None;}}
    }
    Some(reverse)
}
pub(super) fn descriptor_high_windows(p:&Program,w:&[Writes],first:i32)->Option<Vec<WriteWindowFootprint>> {
    if p.insts.len()!=w.len() || p.insts.len()>32768 || !p.insts.iter().any(|i|matches!(i.op,Op::Stg|Op::Atom|Op::Red))
        || p.insts.iter().any(|i|matches!(i.op,Op::St|Op::AtomCas|Op::SustB|Op::SuatomB)) {return None;}
    let reverse=predecessors(p)?;let mut out=Vec::new();
    for (n,i) in p.insts.iter().enumerate().filter(|(_,i)|matches!(i.op,Op::Stg|Op::Atom|Op::Red)) {
        let (low,offset,bytes,masked_high)=match i.op {
            Op::Stg=>{let st=InstStg(i.raw);if !st.e(){return None;}let bytes=match st.ls_size().0{4=>4,5=>8,6|7=>16,_=>return None};(st.src_a(),((st.imm24()<<8)>>8) as i64,bytes,false)},
            Op::Atom=>{let a=InstAtom(i.raw);if !a.e() || a.size().0>1{return None;}(a.src_a(),((a.imm20()<<12)>>12) as i64,4,true)},
            Op::Red=>{let a=InstRed(i.raw);if !a.e() || a.red_size().0>1{return None;}(a.src_a(),((a.imm20()<<12)>>12) as i64,4,true)},
            _=>unreachable!(),
        };if low>=254{return None;}let high=low+1;
        let mut roots=Vec::new();
        if let Some(def)=last_def(w,high,n) {
            let root=&p.insts[def];
            let fixed_pred=!unpred(root.raw) && root.raw>>16&15==i.raw>>16&15 && root.op==Op::MovC
                && p.insts[def..=n].iter().all(|j|matches!(j.op,Op::MovC|Op::MovR|Op::Stg|Op::Nop))
                && !p.targets.iter().chain(p.converge.values()).any(|&target|target>root.addr && target<=i.addr);
            if fixed_pred {roots.push(high_root(p,w,def,high,first)?);}
        }
        if roots.is_empty() {
            // All reaching high-word definitions must be proved. Stop at an
            // unconditional definition, follow both paths of a predicated one,
            // and account for later loop iterations/register-vector clobbers.
            let mut work=reverse[n].clone();let mut visited=vec![false;p.insts.len()];
            if work.is_empty() {return None;}
            while let Some(at)=work.pop() {
                if visited[at] {continue;}visited[at]=true;
                if w[at].iter().any(|(reg,_)|*reg==high) {
                    let root=high_root(p,w,at,high,first)?;if !roots.contains(&root) {roots.push(root);}
                    if unpred(p.insts[at].raw) {continue;}
                }
                if at==0 {return None;}
                work.extend_from_slice(&reverse[at]);
            }
            if roots.is_empty() {return None;}
        }
        for (storage_slot,carry) in roots {
            let proof=WriteWindowFootprint{storage_slot,carry,offset,bytes,masked_high};
            if !out.contains(&proof) {out.push(proof);}if out.len()>64 {return None;}
        }
    }
    (!out.is_empty()).then_some(out)
}

#[cfg(test)]
mod high_window_tests {
    use super::*;
    fn code(v:&[u64])->Vec<u8> {v.chunks(3).flat_map(|g|std::iter::once(0u64).chain(g.iter().copied())).flat_map(u64::to_le_bytes).collect()}
    fn jump(v:&mut [u64],from:usize,to:usize) {
        let address=|n:usize|((n/3*4+1+n%3)*8) as i64;let delta=address(to)-address(from)-8;
        v[from]=0xe24000000000000f|(((delta as u64)&0xffffff)<<20);
    }
    #[test]
    fn high_definitions_follow_loops_and_reject_entry_bypass_and_later_iteration_clobber() {
        let mut safe=vec![0x4c9807800c570001,0x010000000017f004,0,0xeedc200000070002,0xe30000000007000f];jump(&mut safe,2,1);
        assert!(crate::translate_compute_with_memory(&code(&safe),0,0).unwrap().write_window_footprint.is_some());
        let mut bypass=vec![0,0x4c9807800c570001,0x010000000017f004,0xeedc200000070002,0xe30000000007000f];jump(&mut bypass,0,3);
        assert!(crate::translate_compute_with_memory(&code(&bypass),0,0).unwrap().write_window_footprint.is_none());
        let mut later=vec![0x4c9807800c570001,0xeedc200000070002,0x010000000027f001,0,0xe30000000007000f];jump(&mut later,3,1);
        assert!(crate::translate_compute_with_memory(&code(&later),0,0).unwrap().write_window_footprint.is_none());
    }
    #[test]
    fn high_windows_cover_signed_immediates_alignment_carry_and_u64_wrap() {
        for high in [0u64,1,0x7f,u32::MAX as u64] {for carry in [false,true] {for offset in [-8388608i64,-17,-1,0,1,17,8388607] {for bytes in [4u64,8,16] {
            let p=WriteWindowFootprint{storage_slot:0,carry,offset,bytes,masked_high:false};let windows=p.windows(high<<32);
            for low in [0u64,1,2,3,4,0x12345678,u32::MAX as u64-3,u32::MAX as u64] {for cf in 0..=u64::from(carry) {
                let address=(((high+cf) as u32 as u64)<<32|low).wrapping_add_signed(offset)&!3;
                for k in 0..bytes {let at=address.wrapping_add(k);assert!(windows.iter().any(|&(a,n)|at as u128>=a as u128 && (at as u128)<a as u128+n as u128));}
            }}
        }}}}
    }
    #[test]
    fn atomic_domains_mask_high_before_signed_offset_and_cover40bit_carry_wrap() {
        for high in [0u64,1,254,255,256,u32::MAX as u64] {for carry in [false,true] {for offset in [-524288i64,-1,0,1,524287] {
            let p=WriteWindowFootprint{storage_slot:0,carry,offset,bytes:4,masked_high:true};let ranges=p.windows(high<<32);
            for cf in 0..=u64::from(carry) {for low in [0u64,1,3,4,0xfffffffcu64,u32::MAX as u64] {
                let address=((((high+cf)&255)<<32)|low).wrapping_add_signed(offset);
                for k in 0..4 {let byte=address.wrapping_add(k);assert!(ranges.iter().any(|&(a,n)|byte as u128>=a as u128 && (byte as u128)<a as u128+n as u128));}
            }}
        }}}
    }
    #[test]
    fn compute_atomic_and_fragment_reduction_use_their_own_descriptor_banks() {
        let atom=(0xed01000000500606u64&!(255u64|(255<<8)|(255<<20)|(7<<16)))|8|(8<<8)|(6<<20)|(7<<16);
        let raw=code(&[0x4c9807800c470008,0x4c9807800c570009,atom,0xe30000000007000f,0xe2400fffff87000f]);
        assert!(crate::translate_compute_with_memory(&raw,0,0).unwrap().write_window_footprint.unwrap().iter().all(|p|p.masked_high));
        let red=(0xebf9000003070205u64&!(255u64|(255<<8)))|10|(8<<8);
        let fragment=|word:u64|{let mut b=vec![0u8;128];b[..4].copy_from_slice(&0x12345678u32.to_le_bytes());b[0x30..0x34].copy_from_slice(&(5u32<<10).to_le_bytes());b[0x78..0x7c].copy_from_slice(&15u32.to_le_bytes());b.extend(code(&[0x4c98078000070008|((word-1)<<20),0x4c98078000070009|(word<<20),red,0xe30000000007000f,0xe2400fffff87000f]));b};
        let t=crate::translate(&fragment(0x145)).unwrap();assert_eq!(t.write_window_footprint.unwrap(),vec![WriteWindowFootprint{storage_slot:0,carry:false,offset:0,bytes:4,masked_high:true}]);
        assert!(crate::translate(&fragment(0xc5)).unwrap().write_window_footprint.is_none());
    }
    #[test]
    fn high_word_proof_allows_unknown_gpu_low_word_but_rejects_clobber_and_bypass() {
        let code=|v:&[u64]|v.chunks(3).flat_map(|g|std::iter::once(0u64).chain(g.iter().copied())).flat_map(u64::to_le_bytes).collect::<Vec<_>>();
        let v=[0x4c9807800c570001,0xeedc200000070002,0xe30000000007000f];
        let t=crate::translate_compute_with_memory(&code(&v),0,0).unwrap();assert_eq!(t.write_window_footprint,Some(vec![WriteWindowFootprint{storage_slot:0,carry:false,offset:0,bytes:4,masked_high:false}]));
        let mut clobber=v.to_vec();clobber.insert(1,0xeed6200000070000);assert!(crate::translate_compute_with_memory(&code(&clobber),0,0).unwrap().write_window_footprint.is_none());
        let mut wrong=v;wrong[0]^=1<<20;assert!(crate::translate_compute_with_memory(&code(&wrong),0,0).unwrap().write_window_footprint.is_none());
        let mut pred=v;pred[0]&=!(7<<16);assert!(crate::translate_compute_with_memory(&code(&pred),0,0).unwrap().write_window_footprint.is_none());
    }
}
