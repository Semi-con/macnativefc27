// Hierarchical graphics dispatch. Acyclic blocks need one topological sweep;
// each genuine cycle keeps its own dispatcher, with every entry preserved.
use crate::program::{Inst,Program,branch_target};
use crate::inst::{Ccc,InstBra,InstBrx,InstExit};
use crate::table::{Op,InstProps};
use std::collections::{BTreeMap,BTreeSet,VecDeque};
use std::fmt::Write;

pub(super) struct Plan {
    pub(super) blocks:Vec<u32>,groups:Vec<(bool,Vec<usize>)>,
    pub(super) components:Vec<(bool,Vec<usize>)>,
    pub(super) successors:Vec<Vec<usize>>,
    pub(super) entry:usize,
}

fn unconditional(i:&Inst)->bool {
    let pred=i.props.has(InstProps::NOPRED) || (i.raw>>16)&15==7;
    pred && match i.op {
        Op::Bra=>InstBra(i.raw).ccc()==Ccc::T,
        Op::Brx=>InstBrx(i.raw).ccc()==Ccc::T,
        Op::Exit=>InstExit(i.raw).ccc()==Ccc::T,
        _=>true,
    }
}

pub(super) fn prepare(prog:&mut Program,constants:Option<&[u8]>)->Option<Plan> {
    // Unknown BRX provenance can enter any instruction. Keep its original
    // complete dispatcher instead of inventing a smaller graph.
    if prog.insts.iter().any(|i|i.op==Op::Brx) && (prog.indirect_constant_bytes.is_none() || constants.is_none()) {return None;}
    let original=prog.targets.clone();
    for pair in prog.insts.windows(2) {
        if matches!(pair[0].op,Op::Bra|Op::Brx|Op::Sync|Op::Brk|Op::Exit) {prog.targets.insert(pair[1].addr);}
    }
    let starts=prog.insts.iter().enumerate().filter(|(_,i)|prog.targets.contains(&i.addr)).map(|(n,_)|n).collect::<Vec<_>>();
    if starts.len()<2 || starts.len()>4096 {prog.targets=original;return None;}
    let blocks=starts.iter().map(|&n|prog.insts[n].addr).collect::<Vec<_>>();
    let index=blocks.iter().enumerate().map(|(n,&a)|(a,n)).collect::<BTreeMap<_,_>>();
    let valid=prog.insts.iter().map(|i|i.addr).collect::<BTreeSet<_>>();
    let mut edges=vec![Vec::new();blocks.len()];
    for n in 0..blocks.len() {
        let end=starts.get(n+1).copied().unwrap_or(prog.insts.len());
        let i=&prog.insts[end-1];
        let mut fallthrough=true;
        match i.op {
            Op::Bra=>{if let Some(&t)=index.get(&branch_target(i)) {edges[n].push(t);}fallthrough=!unconditional(i);}
            Op::Brx=>{
                for &t in prog.indirect_targets.get(&i.addr)? {
                    if valid.contains(&t) {if let Some(&t)=index.get(&t) {edges[n].push(t);}}
                }
                fallthrough=!unconditional(i);
            }
            Op::Sync|Op::Brk=>{if let Some(t)=prog.converge.get(&i.addr).and_then(|t|index.get(t)) {edges[n].push(*t);}fallthrough=!unconditional(i);}
            Op::Exit=>fallthrough=!unconditional(i),
            _=>{},
        }
        if fallthrough && n+1<blocks.len() {edges[n].push(n+1);}
        edges[n].sort_unstable();edges[n].dedup();
    }
    let count=blocks.len();let mut reverse=vec![Vec::new();count];
    for (a,next) in edges.iter().enumerate() {for &b in next {reverse[b].push(a);}}
    // Iterative Kosaraju avoids a shader-sized Rust call stack.
    let mut visited=vec![false;count];let mut order=Vec::new();
    for start in 0..count {
        if visited[start] {continue;}
        visited[start]=true;let mut stack=vec![(start,0)];
        while let Some(&(node,at))=stack.last() {
            if at<edges[node].len() {
                stack.last_mut().unwrap().1+=1;let next=edges[node][at];
                if !visited[next] {visited[next]=true;stack.push((next,0));}
            } else {stack.pop();order.push(node);}
        }
    }
    let mut component=vec![usize::MAX;count];let mut parts=Vec::new();
    for &start in order.iter().rev() {
        if component[start]!=usize::MAX {continue;}
        let id=parts.len();component[start]=id;let mut stack=vec![start];let mut part=Vec::new();
        while let Some(node)=stack.pop() {
            part.push(node);
            for &next in &reverse[node] {if component[next]==usize::MAX {component[next]=id;stack.push(next);}}
        }
        part.sort_unstable();parts.push(part);
    }
    if parts.len()==1 && (parts[0].len()>1 || edges[0].contains(&0)) {prog.targets=original;return None;}
    let components=parts.iter().map(|part|(part.len()>1 || edges[part[0]].contains(&part[0]),part.clone())).collect::<Vec<_>>();
    let mut successors=vec![Vec::new();parts.len()];
    for (node,next) in edges.iter().enumerate() {
        for &target in next {if component[node]!=component[target] {successors[component[node]].push(component[target]);}}
    }
    for next in &mut successors {next.sort_unstable();next.dedup();}
    let mut acyclic=vec![false;count];let mut groups=Vec::new();
    for part in parts {
        if part.len()==1 && !edges[part[0]].contains(&part[0]) {acyclic[part[0]]=true;}
        else {groups.push((true,part));}
    }
    let mut incoming=vec![0;count];
    for (a,next) in edges.iter().enumerate() {if acyclic[a] {for &b in next {if acyclic[b] {incoming[b]+=1;}}}}
    let mut ready=(0..count).filter(|&n|acyclic[n] && incoming[n]==0).collect::<VecDeque<_>>();let mut linear=Vec::new();
    while let Some(node)=ready.pop_front() {
        linear.push(node);
        for &next in &edges[node] {if acyclic[next] {incoming[next]-=1;if incoming[next]==0 {ready.push_back(next);}}}
    }
    assert_eq!(linear.len(),acyclic.iter().filter(|&&v|v).count());
    if !linear.is_empty() {groups.push((false,linear));}
    groups.sort_by_key(|(_,nodes)|*nodes.iter().min().unwrap());
    Some(Plan{blocks,groups,components,successors,entry:component[0]})
}

pub(super) fn emit(body:&str,prog:&Program,plan:&Plan)->String {
    let mut spans=Vec::new();let mut offset=0;
    for line in body.split_inclusive('\n') {
        if let Some(addr)=line.strip_prefix("    case ").and_then(|s|s.strip_suffix("u:\n")).and_then(|s|s.parse::<u32>().ok()) {
            spans.push((addr,offset,offset+line.len()));
        }
        offset+=line.len();
    }
    assert_eq!(spans.iter().map(|p|p.0).collect::<Vec<_>>(),plan.blocks);
    let text=|node:usize| &body[spans[node].2..spans.get(node+1).map_or(body.len(),|p|p.1)];
    let mut out=format!("    // graphics regions: {} groups / {} original blocks\n    uint pc = {}u;\n    for (;;) {{\n    switch (pc) {{\n",plan.groups.len(),plan.blocks.len(),prog.insts[0].addr);
    for (cyclic,nodes) in &plan.groups {
        for &node in nodes {writeln!(out,"    case {}u:",plan.blocks[node]).unwrap();}
        out.push_str("    {\n");
        if *cyclic {
            out.push_str("    for (;;) {\n    switch (pc) {\n");
            for (n,&node) in nodes.iter().enumerate() {
                writeln!(out,"    case {}u:",plan.blocks[node]).unwrap();out.push_str(text(node));
                if let Some(&next)=plan.blocks.get(node+1) {
                    writeln!(out,"        pc = {next}u;").unwrap();
                    if nodes.get(n+1)!=Some(&(node+1)) {out.push_str("        continue;\n");}
                }
            }
            out.push_str("    default: break;\n    }\n    break;\n    }\n");
        } else {
            // A branch continue exits just this single-iteration block. Every
            // successor inside this DAG appears later in the same sweep.
            for &node in nodes {
                writeln!(out,"    if (pc == {}u) {{\n    do {{",plan.blocks[node]).unwrap();out.push_str(text(node));
                if let Some(next)=plan.blocks.get(node+1) {writeln!(out,"        pc = {next}u;").unwrap();}
                out.push_str("    } while (false);\n    }\n");
            }
        }
        out.push_str("    continue;\n    }\n");
    }
    out.push_str("    default: return epilogue();\n    }\n    }\n}\n");out
}
