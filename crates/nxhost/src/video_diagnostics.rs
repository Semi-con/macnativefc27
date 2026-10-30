//! Optional observation of FC27's original embedded libvpx1.7 decoder.
//! No replacement decoder, frame skipping, simulation changes or asset writes.
use std::sync::{OnceLock,atomic::{AtomicU64,Ordering::Relaxed}};
use crate::{loader::Module,stubs::Args};

#[derive(Clone,Copy)]
struct Interface {name:&'static str,address:u64,string:u64,caps:u64,init:u64,destroy:u64,decode:u64,get_frame:u64}
const INTERFACES:[Interface;2]=[
    Interface{name:"vp8",address:0xcb8d928,string:0xbe4bde9,caps:0x140001,init:0x815e560,destroy:0x815e6a0,decode:0x815e780,get_frame:0x815ec70},
    Interface{name:"vp9",address:0xcb8d690,string:0xbc6e171,caps:0x440005,init:0x8129900,destroy:0x81299f0,decode:0x8129dc0,get_frame:0x812a750},
];
static ORIGINAL:OnceLock<[u64;4]>=OnceLock::new();
static START:OnceLock<std::time::Instant>=OnceLock::new();
static CALLS:[AtomicU64;4]=[const{AtomicU64::new(0)};4];
pub(crate) fn trace_time_ms()->Option<f64> {START.get().map(|v|v.elapsed().as_secs_f64()*1000.)}

pub(crate) fn graphics_path(address:u64)->Option<String> {
    if address==0 {return None;}
    let words=crate::diag::read_words::<32>(address)?;
    let bytes:Vec<u8>=words.into_iter().flat_map(u64::to_le_bytes).collect();
    let end=bytes.iter().position(|b|*b==0)?;let bytes=&bytes[..end];
    if !(8..=255).contains(&bytes.len()) || !bytes.iter().all(|b|b.is_ascii_alphanumeric() || b"_-/\\. ".contains(b)) {return None;}
    let text=String::from_utf8_lossy(bytes).into_owned();let lower=text.to_ascii_lowercase();
    if ["key","token","secret","password","credential","account","profile","auth","user","admin"].into_iter().any(|v|lower.contains(v)) {return None;}
    (lower.contains('/') || lower.contains('\\') || lower.ends_with(".dds") || lower.ends_with(".webm")).then_some(text)
}

fn background_resource(mut fp:u64) {
    let Some(base)=crate::diag::MODULES.get().and_then(|m|m.iter().find(|m|m.name=="main")).map(|m|m.base) else{return;};
    // This exact callee preserves its caller's x20/x19 at FP+0x20/+0x28.
    // Read only while the original synchronous NVN label call is active.
    if crate::diag::read_words::<2>(base+0x97861b0)!=Some([0xa9057bfdd10203ff,0xa9074ff4a90657f6]) ||
       crate::diag::read_words::<1>(base+0x97861c0)!=Some([0xf9405433910143fd]) {return;}
    for _ in 0..32 {
        if fp==0 || fp%8!=0 {return;}
        let Some([next,lr])=crate::diag::read_words::<2>(fp) else{return;};
        if lr==base+0x97677ac {
            let Some([descriptor,resource])=fp.checked_add(0x20).and_then(crate::diag::read_words::<2>) else{return;};
            let Some(header)=crate::diag::read_words::<32>(resource) else{return;};
            let Some(vtable)=header[0].checked_sub(base).filter(|v|(0xb99b000..0xd308570).contains(v)) else{return;};
            let mut names=std::collections::BTreeSet::new();
            let mut pointers=Vec::from(header);
            if let Some(words)=crate::diag::read_words::<24>(descriptor) {pointers.extend(words);}
            for pointer in pointers.iter().copied() {
                if names.len()==8 {break;}
                if let Some(name)=graphics_path(pointer) {names.insert(name);}
            }
            // Only graphics-resource metadata reachable from this synchronous
            // call is sampled. Never export raw words or recurse without bound.
            for pointer in pointers.into_iter().filter(|v|*v!=0).take(48) {
                if names.len()==8 {break;}
                if let Some(words)=crate::diag::read_words::<32>(pointer) {
                    for value in words {
                        if names.len()==8 {break;}
                        if let Some(name)=graphics_path(value) {names.insert(name);}
                    }
                }
            }
            eprintln!("[background-resource] object={resource:#x} descriptor={descriptor:#x} vtable_rva={vtable:#x} names={names:?}");
            return;
        }
        if next<=fp {return;}fp=next;
    }
}

pub(crate) fn texel_write(name:&str,n:u64,args:&Args) {
    if ORIGINAL.get().is_none() {return;}
    let label=name=="nvnTextureSetDebugLabel";
    if !label && !matches!(name,"nvnTextureWriteTexels"|"nvnTextureWriteTexelsStrided"|"nvnTextureFlushTexels") {return;}
    if !label && !(n<512 || (n<16384 && n%32==0) || n.is_power_of_two()) {return;}
    let Some(space)=crate::kernel::KERNEL.aspace.get() else{return;};
    let object=args.0[0];
    if !space.readable(object,0x70) {return;}
    let header=unsafe{(object as *const [u64;14]).read_unaligned()};
    if header[0]!=0x5845_5458_4E56_4E58 {return;}
    let width=header[5]as u32;let height=(header[5]>>32)as u32;
    if (width as u64*height as u64)<500000 {return;}
    if label {
        static LABELS:AtomicU64=AtomicU64::new(0);
        if LABELS.fetch_add(1,Relaxed)>=512 {return;}
        if width==1024 && height==512 && header[7]as u32==80 {
            static ORIGINS:AtomicU64=AtomicU64::new(0);
            if ORIGINS.fetch_add(1,Relaxed)<16 {
                let frame=(args as *const Args as u64).checked_sub(16).and_then(crate::diag::read_words::<2>);
                if let Some([fp,caller])=frame {
                    eprintln!("[background-origin] t_ms={:.3} object={object:#x} thread={:x?} caller={}",START.get().unwrap().elapsed().as_secs_f64()*1000.,crate::guest::try_current().map(|v|v.handle),crate::diag::symbolize(caller));
                    for line in crate::diag::backtrace_lines(fp).into_iter().take(12) {eprintln!("[background-origin] at {line}");}
                    background_resource(fp);
                }
            }
        }
        let address=args.0[1];
        let label=if address!=0 && space.readable(address,128) {
            let bytes=unsafe{std::slice::from_raw_parts(address as *const u8,128)};
            String::from_utf8_lossy(&bytes[..bytes.iter().position(|v|*v==0).unwrap_or(128)]).into_owned()
        }else{String::from("<unreadable-label>")};
        eprintln!("[video-resource] t_ms={:.3} object={object:#x} dimensions={width}x{height} format={} flags={:#x} pool={:#x} offset={:#x} label={label:?}",START.get().unwrap().elapsed().as_secs_f64()*1000.,header[7]as u32,header[4]as u32,header[2],header[3]);
        return;
    }
    let view=args.0[1];let view_words=(view!=0 && space.readable(view,16)).then(||unsafe{(view as *const [u32;4]).read_unaligned()});
    let region=args.0[2];let region_words=(region!=0 && space.readable(region,24)).then(||unsafe{(region as *const [i32;6]).read_unaligned()});
    eprintln!("[video-texture] name={name} count={} t_ms={:.3} object={object:#x} dimensions={width}x{height} view={view_words:x?} region={region_words:?}",n+1,START.get().unwrap().elapsed().as_secs_f64()*1000.);
}

fn matches_interface(base:u64,spec:Interface,words:&[u64;10])->bool {
    words[1]==5 && words[2]==spec.caps &&
    [(0,spec.string),(3,spec.init),(4,spec.destroy),(8,spec.decode),(9,spec.get_frame)]
        .into_iter().all(|(index,offset)|base.checked_add(offset)==Some(words[index]))
}

// These exact C signatures are vpx_codec_decode_fn_t and
// vpx_codec_get_frame_fn_t from libvpx1.7's internal ABI5 (LP64).
unsafe fn delegate(slot:usize,address:u64,args:&Args)->u64 {
    if slot%2==0 {
        let call:extern "C" fn(u64,u64,u32,u64,i64)->i32=unsafe{std::mem::transmute(address as usize)};
        call(args.0[0],args.0[1],args.0[2]as u32,args.0[3],args.0[4]as i64)as u32 as u64
    } else {
        let call:extern "C" fn(u64,u64)->u64=unsafe{std::mem::transmute(address as usize)};
        call(args.0[0],args.0[1])
    }
}

fn dispatch(tag:u64,args:&Args)->u64 {
    let slot=tag as usize;
    let count=CALLS[slot].fetch_add(1,Relaxed)+1;
    let start=std::time::Instant::now();
    let key=if slot==0 && args.0[2]>0 {
        crate::kernel::KERNEL.aspace.get().filter(|a|a.readable(args.0[1],1))
            .map(|_|unsafe{(args.0[1]as *const u8).read()&1==0})
    }else{None};
    let result=unsafe{delegate(slot,ORIGINAL.get().unwrap()[slot],args)};
    if count<=512 || (count<=16384 && count%32==0) || count.is_power_of_two() {
        let codec=INTERFACES[slot/2].name;
        let t=START.get().unwrap().elapsed().as_secs_f64()*1000.;
        if slot%2==0 {
            eprintln!("[video-trace] codec={codec} decode={count} t_ms={t:.3} ctx={:#x} bytes={} vp8_key={key:?} result={result} decode_ms={:.6}",args.0[0],args.0[2]as u32,start.elapsed().as_secs_f64()*1000.);
        } else {
            // libvpx1.7 vpx_image_t ABI4 has12u32 header fields before
            // the plane pointers. Only inspect readable descriptor metadata.
            let header=crate::kernel::KERNEL.aspace.get().filter(|a|result!=0 && a.readable(result,48))
                .map(|_|unsafe{(result as *const [u32;12]).read_unaligned()});
            let dimensions=header.map(|v|[v[6],v[7]]);
            eprintln!("[video-trace] codec={codec} get={count} t_ms={t:.3} ctx={:#x} image={result:#x} dimensions={dimensions:?}",args.0[0]);
        }
    }
    result
}

pub(crate) fn install(modules:&[Module])->anyhow::Result<()> {
    if !std::env::var("NX_FC27_VIDEO_TRACE").is_ok_and(|v|v=="1") {return Ok(());}
    anyhow::ensure!(std::env::var_os("NX_TEST_CONTROL").is_some(),"video observation requires an isolated test process");
    let main=modules.iter().find(|m|m.name=="main").ok_or_else(||anyhow::anyhow!("video observation: main module absent"))?;
    let expected=[0xfb,0x81,0x98,0x3a,0xa1,0xcc,0x3a,0x2e,0x97,0x40,0x24,0x97,0xe2,0xca,0x65,0x28];
    anyhow::ensure!(main.module_id.len()==32 && main.module_id[..16]==expected && main.module_id[16..].iter().all(|v|*v==0),"video observation: unknown module version");
    let mut originals=[0;4];
    for (i,spec)in INTERFACES.into_iter().enumerate() {
        anyhow::ensure!(spec.address>=main.data.0 && spec.address+80<=main.data.0+main.data.1 && spec.decode<main.text && spec.get_frame<main.text,"video observation: interface outside declared data/text");
        let address=main.base+spec.address;
        anyhow::ensure!(crate::kernel::aspace().readable(address,80),"video observation: interface unreadable");
        let words=unsafe{(address as *const [u64;10]).read_unaligned()};
        anyhow::ensure!(matches_interface(main.base,spec,&words),"video observation: {} interface fields differ",spec.name);
        originals[i*2]=words[8];originals[i*2+1]=words[9];
    }
    anyhow::ensure!(ORIGINAL.set(originals).is_ok(),"video observation already installed");
    let _=START.set(std::time::Instant::now());
    // All guards pass before any pointer changes; no guest thread has started.
    for (i,spec)in INTERFACES.into_iter().enumerate() {
        for (field,tag)in [(0x40,i*2),(0x48,i*2+1)] {
            let stub=crate::stubs::make(dispatch,tag as u64);
            unsafe{((main.base+spec.address+field)as *mut u64).write_volatile(stub);}
        }
        eprintln!("[video-trace] installed original {} decode/get observation; ABI5 interface checked",spec.name);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn original_video_interface_guards_all_identity_and_callable_fields() {
        let base=0x7100000000;
        for spec in INTERFACES {
            let mut words=[0;10];words[1]=5;words[2]=spec.caps;
            for (i,v)in [(0,spec.string),(3,spec.init),(4,spec.destroy),(8,spec.decode),(9,spec.get_frame)] {words[i]=base+v;}
            assert!(matches_interface(base,spec,&words));
            for i in [0,1,2,3,4,8,9] {let mut bad=words;bad[i]^=1;assert!(!matches_interface(base,spec,&bad));}
            assert!(!matches_interface(base+0x1000,spec,&words));
        }
    }
    #[test]
    fn original_video_calls_preserve_c_arguments_deadline_and_results() {
        extern "C" fn decode(ctx:u64,data:u64,len:u32,user:u64,deadline:i64)->i32 {
            assert_eq!((ctx,data,len,user,deadline),(0x7100010000,0x7200020000,0xf1234567,0x7300030000,-17));7
        }
        extern "C" fn frame(ctx:u64,iter:u64)->u64 {assert_eq!((ctx,iter),(0x7100010000,0x7200020000));0x7400040000}
        let args=Args([0x7100010000,0x7200020000,0xf1234567,0x7300030000,(-17i64)as u64,0,0,0],[0;8]);
        for slot in [0,2] {assert_eq!(unsafe{delegate(slot,decode as *const ()as u64,&args)},7);}
        for slot in [1,3] {assert_eq!(unsafe{delegate(slot,frame as *const ()as u64,&args)},0x7400040000);}
    }
}
