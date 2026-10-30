// The SCC condensation is a DAG. Emit its common postdominator joins as
// nested branches, leaving a local PC dispatcher only inside genuine cycles.
use super::graphics_regions::Plan;
use crate::program::Program;
use std::collections::{BTreeSet,VecDeque};
use std::fmt::Write;

struct Flow<'a> {
    plan:&'a Plan,text:Vec<&'a str>,joins:Vec<usize>,out:String,limit:usize,
}

impl Flow<'_> {
    fn condition(&self,node:usize)->String {
        self.plan.components[node].1.iter().map(|&block|format!("pc == {}u",self.plan.blocks[block])).collect::<Vec<_>>().join(" || ")
    }

    fn component(&mut self,node:usize) {
        let (cyclic,blocks)=&self.plan.components[node];
        if *cyclic {self.out.push_str("    for (;;) {\n    switch (pc) {\n");}
        for (at,&block) in blocks.iter().enumerate() {
            if *cyclic {writeln!(self.out,"    case {}u: {{",self.plan.blocks[block]).unwrap();}
            else {self.out.push_str("    do {\n");}
            self.out.push_str(self.text[block]);
            if let Some(&next)=self.plan.blocks.get(block+1) {
                writeln!(self.out,"        pc = {next}u;").unwrap();
                if *cyclic && blocks.get(at+1)!=Some(&(block+1)) {self.out.push_str("        continue;\n");}
            } else if *cyclic {self.out.push_str("        return epilogue();\n");}
            if *cyclic {self.out.push_str("    }\n");}
            else {self.out.push_str("    } while (false);\n");}
        }
        if *cyclic {self.out.push_str("    default: break;\n    }\n    break;\n    }\n");}
    }

    fn walk(&mut self,mut node:usize,stop:usize,depth:usize)->Option<()> {
        if depth>128 {return None;}
        let exit=self.plan.components.len();
        while node!=stop {
            if self.out.len()>self.limit {return None;}
            if node==exit {self.out.push_str("    return epilogue();\n");return Some(());}
            self.component(node);
            let successors=self.plan.successors[node].clone();
            match successors.len() {
                0=>node=exit,
                1=>{
                    // An over-approximated BRX table may also contain invalid
                    // destinations. Preserve the original default epilogue,
                    // even when the graph has only one valid successor SCC.
                    let next=successors[0];
                    let condition=self.condition(next);
                    writeln!(self.out,"    if (!({condition})) return epilogue();").unwrap();
                    node=next;
                }
                _=>{
                    let join=self.joins[node];
                    for (i,next) in successors.into_iter().enumerate() {
                        let condition=self.condition(next);
                        writeln!(self.out,"    {}if ({condition}) {{",if i==0 {""} else {"else "}).unwrap();
                        self.walk(next,join,depth+1)?;self.out.push_str("    }\n");
                    }
                    self.out.push_str("    else { return epilogue(); }\n");node=join;
                }
            }
        }
        Some(())
    }
}

pub(super) fn emit(body:&str,prog:&Program,plan:&Plan)->Option<String> {
    let count=plan.components.len();
    if count>1024 {return None;}
    let mut spans=Vec::new();let mut offset=0;
    for line in body.split_inclusive('\n') {
        if let Some(addr)=line.strip_prefix("    case ").and_then(|s|s.strip_suffix("u:\n")).and_then(|s|s.parse::<u32>().ok()) {
            spans.push((addr,offset,offset+line.len()));
        }
        offset+=line.len();
    }
    if spans.iter().map(|p|p.0).collect::<Vec<_>>()!=plan.blocks {return None;}
    let text=(0..spans.len()).map(|n|&body[spans[n].2..spans.get(n+1).map_or(body.len(),|p|p.1)]).collect();
    let mut incoming=vec![0usize;count];
    for next in &plan.successors {for &target in next {incoming[target]+=1;}}
    let mut ready=(0..count).filter(|&n|incoming[n]==0).collect::<VecDeque<_>>();let mut order=Vec::new();
    while let Some(node)=ready.pop_front() {
        order.push(node);
        for &next in &plan.successors[node] {incoming[next]-=1;if incoming[next]==0 {ready.push_back(next);}}
    }
    if order.len()!=count {return None;}
    let mut rank=vec![count;count+1];for (at,&node) in order.iter().enumerate() {rank[node]=at;}
    let mut post=vec![BTreeSet::new();count+1];post[count].insert(count);
    let mut joins=vec![count;count];
    for &node in order.iter().rev() {
        let successors=&plan.successors[node];
        let mut common=post[successors.first().copied().unwrap_or(count)].clone();
        for &next in successors.iter().skip(1) {common=common.intersection(&post[next]).copied().collect();}
        joins[node]=*common.iter().min_by_key(|&&n|rank[n])?;
        common.insert(node);post[node]=common;
    }
    let out=format!("    // structured graphics: {count} SCCs / {} original blocks\n    uint pc = {}u;\n",plan.blocks.len(),prog.insts[0].addr);
    let mut flow=Flow{plan,text,joins,out,limit:body.len().saturating_mul(4).saturating_add(65536)};
    flow.walk(plan.entry,count,0)?;flow.out.push_str("    return epilogue();\n}\n");
    if flow.out.len()>flow.limit {return None;}
    Some(flow.out)
}
