// Local test transport only. Reflected original rendering settings, with
// version/type/region checks and independent conditional expiry. No file edits.
use std::{path::Path,sync::{LazyLock,Mutex,Condvar,OnceLock,atomic::{AtomicU32,Ordering::Relaxed}},time::{Duration,Instant}};

#[derive(Clone,Copy,Debug,PartialEq)]
enum Kind {Post,WorldBase,World,Mesh,Crowd,Game,FifaRender,FifaCrowd}
impl Kind {
    fn name(self)->&'static str {match self {Self::Post=>"GlobalPostProcessSettings",Self::WorldBase=>"WorldRenderSettingsBase",Self::World=>"WorldRenderSettings",Self::Mesh=>"MeshSettings",Self::Crowd=>"CrowdSettings",Self::Game=>"GameRenderSettings",Self::FifaRender=>"FifaRenderingSettings",Self::FifaCrowd=>"FifaCrowdSettings"}}
    fn shape(self)->(usize,u64) {match self {Self::Post=>(0x350,0xcc4f428),Self::WorldBase=>(0x2f0,0xcdf41e8),Self::World=>(0x8f0,0xcdf6770),Self::Mesh=>(0x80,0xcc4d658),Self::Crowd=>(0x70,0xcd22950),Self::Game=>(0x248,0xcc447d8),Self::FifaRender=>(0x50,0xc422408),Self::FifaCrowd=>(0x128,0xc890530)}}
}
#[derive(Clone,Copy,Debug,PartialEq)]
enum Field {ForceDof,DynamicAo,MotionBlur,Filmic,Subsurface,MeshLod,CrowdLod,CrowdBoostLod,Resolution,ResolutionMin,DynamicResolution,HighresFan,PitchClumps,CrowdSelection,SidelineEnable,SidelineMax,ForceMeshLod}
impl Field {
    fn parse(s:&str)->Option<Self> {match s {"mesh-force-lod"=>Some(Self::ForceMeshLod),"highres-crowd-fan"=>Some(Self::HighresFan),"pitch-clumps"=>Some(Self::PitchClumps),"crowd-selection-max"=>Some(Self::CrowdSelection),"highres-sideline"=>Some(Self::SidelineEnable),"highres-sideline-max"=>Some(Self::SidelineMax),"resolution-scale"=>Some(Self::Resolution),"resolution-scale-min"=>Some(Self::ResolutionMin),"dynamic-resolution"=>Some(Self::DynamicResolution),"force-dof"=>Some(Self::ForceDof),"dynamic-ao"=>Some(Self::DynamicAo),"motion-blur"=>Some(Self::MotionBlur),"filmic-effects"=>Some(Self::Filmic),"subsurface"=>Some(Self::Subsurface),"mesh-lod-scale"=>Some(Self::MeshLod),"crowd-lod-scale"=>Some(Self::CrowdLod),"crowd-boost-lod-scale"=>Some(Self::CrowdBoostLod),_=>None}}
    fn kind(self)->Kind {match self {Self::HighresFan|Self::PitchClumps=>Kind::FifaRender,Self::CrowdSelection|Self::SidelineEnable|Self::SidelineMax=>Kind::FifaCrowd,Self::Resolution|Self::ResolutionMin|Self::DynamicResolution=>Kind::Game,Self::MeshLod|Self::ForceMeshLod=>Kind::Mesh,Self::CrowdLod|Self::CrowdBoostLod=>Kind::Crowd,Self::MotionBlur|Self::Filmic|Self::Subsurface=>Kind::World,_=>Kind::Post}}
    fn offset(self)->u64 {match self {Self::ForceMeshLod=>0x3c,Self::HighresFan=>0x48,Self::PitchClumps=>0x46,Self::CrowdSelection=>0x114,Self::SidelineEnable=>0x11e,Self::SidelineMax=>0x118,Self::Resolution=>0x15c,Self::ResolutionMin=>0x174,Self::DynamicResolution=>0x1fe,Self::ForceDof=>0x24c,Self::DynamicAo=>0x336,Self::MotionBlur=>0x2c7,Self::Filmic=>0x7e7,Self::Subsurface=>0x8c1,Self::MeshLod=>0x50,Self::CrowdLod=>0x44,Self::CrowdBoostLod=>0x40}}
    fn is_float(self)->bool {matches!(self,Self::MeshLod|Self::CrowdLod|Self::CrowdBoostLod|Self::Resolution|Self::ResolutionMin)}
    fn is_count(self)->bool {matches!(self,Self::CrowdSelection|Self::SidelineMax)}
    fn width(self)->usize {if self==Self::ForceDof || self==Self::ForceMeshLod || self.is_float() || self.is_count() {4}else{1}}
    fn valid(self,value:i32)->bool {if self==Self::ForceMeshLod{return (-1..=4).contains(&value);}if self.is_count(){return (0..=65536).contains(&value);}if self.is_float() {let f=f32::from_bits(value as u32);return f.is_finite() && (if self==Self::ResolutionMin {0.0}else{0.125}..=4.0).contains(&f);}if self==Self::ForceDof {(-1..=1).contains(&value)}else{(0..=1).contains(&value)}}
}
fn word(b:&[u8],n:usize)->Option<u64> {Some(u64::from_le_bytes(b.get(n..n+8)?.try_into().ok()?))}
fn float_json(bits:u32)->String {let f=f32::from_bits(bits);if f.is_finite(){f.to_string()}else{"null".into()}}
fn uint(b:&[u8],n:usize)->Option<u32> {Some(u32::from_le_bytes(b.get(n..n+4)?.try_into().ok()?))}
fn value(bytes:&[u8],field:Field)->Option<i32> {let at=field.offset() as usize;Some(if field.width()==4 {i32::from_le_bytes(bytes.get(at..at+4)?.try_into().ok()?)}else{*bytes.get(at)? as i32})}
fn copy(address:u64,bytes:&mut[u8])->bool {super::config_inspection::snapshot(address,bytes)}
fn write(address:u64,field:Field,value:i32)->bool {
    if !field.valid(value)|| (field.width()==4 && address%4!=0) {return false;}
    unsafe extern "C" {static mach_task_self_:u32;fn mach_vm_write(task:u32,address:u64,data:u64,count:u32)->i32;}
    let data=value.to_le_bytes();unsafe {mach_vm_write(mach_task_self_,address,data.as_ptr() as u64,field.width() as u32)==0}
}
const WORLD_VISIBILITY_FIELDS:&[(&str,u64,[u64;4])]=&[
    ("DrawDebugSkyEnvmap",0xcdf5840,[0xbc2e64f,0x7925b439,0x26f,0xdeaf300]),
    ("DrawDebugSkyTextures",0xcdf58e0,[0xbd55de1,0xb10b988,0x2b2,0xdeaf300]),
    ("DrawDebugHalfResEnvironment",0xcdf5880,[0xbac326e,0x572d43dc,0x287,0xdeaf300]),
    ("StaticEnvmapEnable",0xcdf6300,[0xbec31a7,0xbba2802a,0x2c1,0xdeaf300]),
    ("CustomEnvmapEnable",0xcdf6320,[0xbc788b0,0x7284dafc,0x24f,0xdeaf300]),
    ("SkyEnvmapEnable",0xcdf6340,[0xbe56a4e,0x50105bb2,0x269,0xdeaf300]),
    ("SkyEnvmapForceUpdateEnable",0xcdf63a0,[0xbdc3a59,0xfa0385fd,0x2c4,0xdeaf300]),
    ("DynamicEnvmapEnable",0xcdf6460,[0xbce7790,0x81054f4c,0x2b0,0xdeaf300]),
    ("DeferredShadingEnable",0xcdf4240,[0xbac31ea,0x4203dd78,0x27f,0xdeaf300]),
    ("ForwardOpaqueEnable",0xcdf4260,[0xbc2e60e,0x94c3fcda,0x280,0xdeaf300]),
    ("ModelBuildParallelEnable",0xcdf4320,[0xbbbfdd3,0x3c416c0,0x254,0xdeaf300]),
    ("ModelDrawParallelEnable",0xcdf4360,[0xbb0b328,0x23e2aacb,0x295,0xdeaf300]),
    ("SkyEnvmapUpdateCountThreshold",0xcdf63e0,[0xbbe46cf,0x606a29f2,0x1c4,0xdeaf2e0]),
    ("SkyEnvmapUpdateValueThreshold",0xcdf6400,[0xbd55df6,0x2aa84ce8,0xe8,0xdeaf2e0]),
];
fn visibility_layout_matches(base:u64)->bool {
    WORLD_VISIBILITY_FIELDS.iter().all(|(_,at,expected)|crate::diag::read_words::<4>(base+at).is_some_and(|w|
        w[0]==base+expected[0] && w[1]&0xffff_ffff==expected[1] && w[2]==expected[2] && w[3]==base+expected[3]))
}
fn world_visibility_fields(bytes:&[u8])->String {
    WORLD_VISIBILITY_FIELDS.iter().map(|(name,_,w)| {
        let at=w[2]as usize;
        let value=if w[3]==0xdeaf2e0 {float_json(uint(bytes,at).unwrap())}else{bytes[at].to_string()};
        format!("\"{name}\":{value}")
    }).collect::<Vec<_>>().join(",")
}
fn version_base()->Option<u64> {
    let base=crate::diag::MODULES.get()?.iter().find(|m|m.name=="main")?.base;
    let a=crate::diag::read_words::<4>(base+0xcc50300)?;
    let b=crate::diag::read_words::<4>(base+0xcdf5020)?;
    (visibility_layout_matches(base) && a==[base+0xba75bc2,0x2000d144ee34,0x336,base+0xdeaf300]
        && b==[base+0xbd79c7d,0x2000aee79fe8,0x2c7,base+0xdeaf300]
        && crate::diag::read_words::<4>(base+0xcdf6d28)==Some([base+0xbb0b440,0x46cba566,0x7e7,base+0xdeaf300])
        && crate::diag::read_words::<4>(base+0xcdf7768)==Some([base+0xbd0c40b,0x84cab19,0x8c1,base+0xdeaf300])
        && crate::diag::read_words::<4>(base+0xcc4d770)==Some([base+0xba77a75,0x40e36e07,0x50,base+0xdeaf2e0])
        && crate::diag::read_words::<4>(base+0xcd22998)==Some([base+0xbe78405,0x40234869,0x44,base+0xdeaf2e0])
        && crate::diag::read_words::<4>(base+0xcd229b8)==Some([base+0xbb995f9,0x511677b7,0x40,base+0xdeaf2e0])
        && crate::diag::read_words::<4>(base+0xcc44c70)==Some([base+0xbd07448,0x2000af2762b4,0x1fe,base+0xdeaf300])
        && crate::diag::read_words::<4>(base+0xcc44c30)==Some([base+0xbd9edfe,0x200084c6b2aa,0x15c,base+0xdeaf2e0])
        && crate::diag::read_words::<4>(base+0xcc44c50)==Some([base+0xbae2dd6,0x2000633114a7,0x174,base+0xdeaf2e0])
        && crate::diag::read_words::<1>(base+0x9986870)==Some([0x911f6108d00195e8])
        && crate::diag::read_words::<4>(base+0xc9b5340)==Some([base+0xbd6aab0,0xd173ab35,0x46,base+0xdeaf300])
        && crate::diag::read_words::<4>(base+0xc9b53c0)==Some([base+0xbb4474e,0x24e96acf,0x48,base+0xdeaf300])
        && crate::diag::read_words::<4>(base+0xc890558)==Some([base+0xbdf85c1,0x3069d50e,0x11e,base+0xdeaf300])
        && crate::diag::read_words::<4>(base+0xc890578)==Some([base+0xbb87e20,0x4e432976,0x118,base+0xdea7d00])
        && crate::diag::read_words::<4>(base+0xc8905f8)==Some([base+0xba89ffc,0xf2c3924,0x114,base+0xdea7d00])
        && crate::diag::read_words::<1>(base+0xd138848)==Some([base+0xd7a6ce0])
        && crate::diag::read_words::<1>(base+0xd0811a0)==Some([base+0xd6fdb88])
        && crate::diag::read_words::<1>(base+0x65e3fe0)==Some([0x91102108f002f1e8])
        && crate::diag::read_words::<1>(base+0x5c786b0)==Some([0x9114c108900360c8])
        && crate::diag::read_words::<4>(base+0xcc4d6d0)==Some([base+0xbb7831d,0x509483fc,0x3c,base+0xdea7d00])
        && crate::diag::read_words::<1>(base+0x9997b9c)==Some([0x91196108d00195a8])
        && crate::diag::read_words::<1>(base+0xa6b0bd0)==Some([0x91254108d0013388])
        && crate::diag::read_words::<1>(base+0xb77e804)==Some([0x911dc1089000b3c8])
        && crate::diag::read_words::<1>(base+0x9995150)==Some([0xd00195c9b0012b0a])
        && crate::diag::read_words::<1>(base+0x9995420)==Some([0xb0012b8ab9024c0a])).then_some(base)
}
fn identity(bytes:&[u8],base:u64,kind:Kind)->bool {
    let (size,vtable)=kind.shape();if bytes.len()<size || word(bytes,0)!=Some(base+vtable) {return false;}
    let Some(t)=word(bytes,8) else{return false;};
    (match kind {Kind::FifaRender=>t==base+0xd7a6ce0,Kind::FifaCrowd=>t==base+0xd6fdb88,_=>t%8==0 && (base..base+0x2000_0000).contains(&t)})
        && match kind {Kind::FifaRender=>[Field::HighresFan,Field::PitchClumps].iter().all(|f|value(bytes,*f).is_some_and(|v|f.valid(v))),Kind::FifaCrowd=>[Field::CrowdSelection,Field::SidelineEnable,Field::SidelineMax].iter().all(|f|value(bytes,*f).is_some_and(|v|f.valid(v))),Kind::Game=>[Field::Resolution,Field::ResolutionMin,Field::DynamicResolution].iter().all(|f|value(bytes,*f).is_some_and(|v|f.valid(v))) && [0x7c,0xb4,0x19c,0xac,0x154].iter().all(|&at|uint(bytes,at).is_some_and(|v|f32::from_bits(v).is_finite())),Kind::Post=>[Field::ForceDof,Field::DynamicAo].iter().all(|f|value(bytes,*f).is_some_and(|v|f.valid(v))),Kind::WorldBase=>value(bytes,Field::MotionBlur).is_some_and(|v|Field::MotionBlur.valid(v)),Kind::World=>[Field::MotionBlur,Field::Filmic,Field::Subsurface].iter().all(|f|value(bytes,*f).is_some_and(|v|f.valid(v))),Kind::Mesh=>[Field::MeshLod,Field::ForceMeshLod].iter().all(|f|value(bytes,*f).is_some_and(|v|f.valid(v))),Kind::Crowd=>[Field::CrowdLod,Field::CrowdBoostLod].iter().all(|f|value(bytes,*f).is_some_and(|v|f.valid(v)))}
}
// CrowdSettings derives from an asset whose +0x18 is TextureArrayPoolDef,
// not the SubDataContainer flags word. Its exact runtime TypeInfo is verified.
fn editable(bytes:&[u8],base:u64,kind:Kind)->bool {
    identity(bytes,base,kind) && if kind==Kind::Crowd {word(bytes,8)==Some(base+0xdd69530)} else if matches!(kind,Kind::FifaRender|Kind::FifaCrowd) {word(bytes,8)==Some(base+if kind==Kind::FifaRender{0xd7a6ce0}else{0xd6fdb88})} else {uint(bytes,0x18)==Some(11)}
}
fn inspect(output:&Path) {
    let Some(base)=version_base() else {let _=std::fs::write(output,r#"{"version_verified":false,"objects":[],"error":"original render field/constructor signatures differ"}"#);return;};
    let started=Instant::now();let space=crate::kernel::aspace();let first=space.heap();let bound=space.stack();let copy_bound=4096u64<<20;
    let mut cursor=first;let mut rows=Vec::new();let mut scanned=0u64;let mut alias_scanned=0u64;let mut object_count=0u64;let mut counts=std::collections::BTreeMap::new();let mut world_debug_enabled=0u64;let mut failures=0;const CHUNK:usize=1<<20;const TAIL:usize=0x1000;let mut buf=vec![0;CHUNK];
    while cursor<bound && scanned<copy_bound && started.elapsed()<Duration::from_secs(15) {
        let (start,size,block)=space.map.lock().unwrap().query(cursor);let Some(end)=start.checked_add(size).map(|v|v.min(bound)).filter(|v|*v>cursor) else {break;};
        if !matches!(block.state,crate::mem::state::NORMAL|crate::mem::state::ALIAS) || block.perm&crate::mem::perm::R==0 {cursor=end;continue;}
        let step=(end-cursor).min((CHUNK-TAIL) as u64).min(copy_bound-scanned) as usize;let amount=(end-cursor).min((step+TAIL) as u64) as usize;
        if copy(cursor,&mut buf[..amount]) {
            scanned+=step as u64;if cursor>=space.alias() {alias_scanned+=step as u64;}
            for at in (0..step.saturating_sub(7)).step_by(8) {
                let kind=match word(&buf[..amount],at) {Some(v) if v==base+0xc422408=>Kind::FifaRender,Some(v) if v==base+0xc890530=>Kind::FifaCrowd,Some(v) if v==base+0xcc447d8=>Kind::Game,Some(v) if v==base+0xcc4f428=>Kind::Post,Some(v) if v==base+0xcdf41e8=>Kind::WorldBase,Some(v) if v==base+0xcdf6770=>Kind::World,Some(v) if v==base+0xcc4d658=>Kind::Mesh,Some(v) if v==base+0xcd22950=>Kind::Crowd,_=>continue};
                let bytes=&buf[at..amount];if !identity(bytes,base,kind) {continue;}
                object_count+=1;let region=if cursor>=space.alias(){"alias"}else{"heap"};*counts.entry((region,kind.name())).or_insert(0u64)+=1;
                if matches!(kind,Kind::World|Kind::WorldBase) && WORLD_VISIBILITY_FIELDS[..3].iter().any(|(_,_,w)|bytes[w[2]as usize]!=0) {world_debug_enabled+=1;}
                if rows.len()>=64 {continue;}
                let mut fields=if kind==Kind::FifaRender {format!(r#""HighResCrowdFanEnabled":{},"PitchClumpsEnabled":{},"EnableIndexCull":{},"NISViewportScale":{},"NISViewportScaleRuntime":{}"#,value(bytes,Field::HighresFan).unwrap(),value(bytes,Field::PitchClumps).unwrap(),bytes[0x4a],float_json(uint(bytes,0x38).unwrap()),float_json(uint(bytes,0x34).unwrap()))}else if kind==Kind::FifaCrowd {format!(r#""HighresCrowd_MaxSelectionCount":{},"HighresSlc_Enable":{},"HighresSlc_Max":{}"#,value(bytes,Field::CrowdSelection).unwrap(),value(bytes,Field::SidelineEnable).unwrap(),value(bytes,Field::SidelineMax).unwrap())}else if kind==Kind::Game {format!(r#""ResolutionScale":{},"ResolutionScaleMin":{},"DynamicResolutionScaleEnable":{},"DynamicResolutionScaleMin":{},"NxConsoleNormalResolutionScaleMin":{},"NxConsoleNormalResolutionScaleMax":{},"NxConsoleBoostResolutionScaleMin":{},"NxConsoleBoostResolutionScaleMax":{},"InactiveSkipFrameCount":{}"#,f32::from_bits(value(bytes,Field::Resolution).unwrap() as u32),f32::from_bits(value(bytes,Field::ResolutionMin).unwrap() as u32),value(bytes,Field::DynamicResolution).unwrap(),f32::from_bits(uint(bytes,0x7c).unwrap()),f32::from_bits(uint(bytes,0xb4).unwrap()),f32::from_bits(uint(bytes,0x19c).unwrap()),f32::from_bits(uint(bytes,0xac).unwrap()),f32::from_bits(uint(bytes,0x154).unwrap()),uint(bytes,0xe4).unwrap())}else if kind==Kind::Post {format!(r#""ForceDofEnable":{},"DynamicAOEnable":{},"DofMethod":{}"#,value(bytes,Field::ForceDof).unwrap(),value(bytes,Field::DynamicAo).unwrap(),uint(bytes,0x234).unwrap())}else if kind==Kind::WorldBase {format!(r#""MotionBlurEnable":{},"MotionBlurForceOn":{}"#,value(bytes,Field::MotionBlur).unwrap(),bytes[0x2af])}else if kind==Kind::Mesh {format!(r#""GlobalLodScale":{},"ForceLod":{},"RuntimeLodQualityLevel":{}"#,f32::from_bits(value(bytes,Field::MeshLod).unwrap() as u32),uint(bytes,0x3c).unwrap() as i32,uint(bytes,0x60).unwrap())}else if kind==Kind::Crowd {format!(r#""LodScale":{},"LodScaleNxBoostMode":{}"#,f32::from_bits(value(bytes,Field::CrowdLod).unwrap() as u32),f32::from_bits(value(bytes,Field::CrowdBoostLod).unwrap() as u32))}else{format!(r#""MotionBlurEnable":{},"MotionBlurForceOn":{},"FilmicEffectsEnable":{},"SubSurfaceScatteringEnable":{},"TemporalAAQuality":{},"ShadowmapResolution":{},"ShadowmapQuality":{},"PlanarShadowsEnable":{}"#,value(bytes,Field::MotionBlur).unwrap(),bytes[0x2af],value(bytes,Field::Filmic).unwrap(),value(bytes,Field::Subsurface).unwrap(),uint(bytes,0x5dc).unwrap(),uint(bytes,0x60).unwrap(),uint(bytes,0x18c).unwrap(),bytes[0x884])};
                if matches!(kind,Kind::World|Kind::WorldBase) {fields.push(',');fields.push_str(&world_visibility_fields(bytes));}
                let flags=if matches!(kind,Kind::Crowd|Kind::FifaRender|Kind::FifaCrowd) {"null".into()}else{uint(bytes,0x18).unwrap().to_string()};
                let editable_region=cursor+at as u64>=space.heap() && cursor+at as u64+kind.shape().0 as u64<=space.heap()+space.layout.heap_size && block.state==crate::mem::state::NORMAL;
                rows.push(format!(r#"{{"class":"{}","address":"{:#x}","region":"{}","type_pointer":"{:#x}","flags":{flags},"editable":{},"header_word18":"{:#x}","fields":{{{fields}}}}}"#,kind.name(),cursor+at as u64,if cursor>=space.alias(){"alias"}else{"heap"},word(bytes,8).unwrap(),editable_region && editable(bytes,base,kind),word(bytes,0x18).unwrap()));
            }
        }else{failures+=1;}
        cursor+=step as u64;
    }
    let counts=counts.iter().map(|((region,class),count)|format!(r#"{{"region":"{region}","class":"{class}","count":{count}}}"#)).collect::<Vec<_>>().join(",");
    let report=format!(r#"{{"version_verified":true,"bytes_scanned":{scanned},"alias_bytes_scanned":{alias_scanned},"object_count":{object_count},"counts":[{counts}],"world_debug_sky_or_environment_enabled_count":{world_debug_enabled},"seconds":{},"failed_chunks":{failures},"range_complete":{},"scan_first":"{first:#x}","scan_bound":"{bound:#x}","objects":[{}],"limits":"Read-only guest NORMAL/ALIAS data reservation heap through alias, max4096MiB copied/15sec/64 retained objects; counts and world debug checks include all matching objects. Exact original vtable and field-layout signatures. Alias observations cannot be edited. Active vs default instances not established. No raw heap or secrets retained."}}"#,started.elapsed().as_secs_f64(),cursor>=bound,rows.join(","));
    let _=std::fs::write(output,report);eprintln!("[render-quality] inspected objects={} bytes={scanned}",rows.len());
}
enum Command {Read(u64),Reset(u64),Edit{seq:u64,field:Field,object:u64,old:i32,new:i32,hold:Duration}}
fn parse(s:&str)->Option<Command> {
    let mut p=s.split_whitespace();let cmd=p.next()?;let seq=p.next()?.parse::<u64>().ok()?;if seq==0{return None;}
    if cmd=="read" {return p.next().is_none().then_some(Command::Read(seq));}if cmd=="reset" {return p.next().is_none().then_some(Command::Reset(seq));}if cmd!="edit" {return None;}
    let field=Field::parse(p.next()?)?;let object=u64::from_str_radix(p.next()?.strip_prefix("0x")?,16).ok()?;let old=p.next()?.parse().ok()?;let new=p.next()?.parse().ok()?;let ms=p.next()?.parse::<u64>().ok()?;
    if object==0||object%8!=0||object.checked_add(0x8f0).is_none()||!field.valid(old)||!field.valid(new)||old==new||!(1000..=120000).contains(&ms)||p.next().is_some(){return None;}
    Some(Command::Edit{seq,field,object,old,new,hold:Duration::from_millis(ms)})
}
struct Patch {seq:u64,field:Field,object:u64,header:[u8;32],old:i32,new:i32,until:Instant}
fn matches(p:&Patch,wanted:i32)->bool {
    let mut header=[0;32];let mut bytes=[0;4];let n=p.field.width();
    if !copy(p.object,&mut header)||header!=p.header||!copy(p.object+p.field.offset(),&mut bytes[..n]) {return false;}
    (if n==4 {i32::from_le_bytes(bytes)}else{bytes[0] as i32})==wanted
}
fn restore(p:&Patch)->bool {matches(p,p.new)&&write(p.object+p.field.offset(),p.field,p.old)&&matches(p,p.old)}
fn apply(seq:u64,field:Field,object:u64,base:u64,old:i32,new:i32,hold:Duration)->Result<Patch,&'static str> {
    let mut bytes=vec![0;field.kind().shape().0];if !copy(object,&mut bytes)||!editable(&bytes,base,field.kind()) {return Err("original object/type/flags mismatch");}
    let header=bytes[..32].try_into().unwrap();let p=Patch{seq,field,object,header,old,new,until:Instant::now()+hold};
    if !matches(&p,old){return Err("expected typed value changed");}if !write(object+field.offset(),field,new){return Err("OS typed write failed");}
    if !matches(&p,new){let _=restore(&p);return Err("write verification failed");}Ok(p)
}
#[derive(Default)]struct State{last:u64,accepted:u32,patches:Vec<Patch>}
static JOURNAL:LazyLock<(Mutex<State>,Condvar)>=LazyLock::new(||(Mutex::new(State::default()),Condvar::new()));
fn worker()->bool {
    static STARTED:OnceLock<bool>=OnceLock::new();*STARTED.get_or_init(||std::thread::Builder::new().name("render-quality-expiry".into()).spawn(|| {
        let (lock,changed)=&*JOURNAL;let mut s=lock.lock().unwrap();loop {
            let now=Instant::now();let mut at=0;while at<s.patches.len(){if s.patches[at].until<=now {let p=s.patches.remove(at);eprintln!("[render-quality] expired seq={} field={:?} restored={}",p.seq,p.field,restore(&p));}else{at+=1;}}
            s=if let Some(until)=s.patches.iter().map(|p|p.until).min(){changed.wait_timeout(s,until.saturating_duration_since(Instant::now())).unwrap().0}else{changed.wait(s).unwrap()};
        }
    }).is_ok())
}
fn report(control:&Path,seq:u64,accepted:bool,message:&str) {let _=std::fs::write(control.with_extension(format!("render-edit-{seq}.json")),format!(r#"{{"sequence":{seq},"accepted":{accepted},"message":"{message}","scope":"Test-process original rendering fields only; independent conditional rollback. No clocks/game files/saves modified."}}"#));}
pub(super) fn poll(control:&Path) {
    let path=control.with_extension("render-quality");let Some(text)=super::read(&path) else{return;};let _=std::fs::remove_file(&path);let Some(cmd)=parse(&text) else{eprintln!("[render-quality] rejected invalid bounded request");return;};
    if let Command::Read(seq)=cmd {static READS:AtomicU32=AtomicU32::new(0);if READS.fetch_add(1,Relaxed)<4 {let out=control.with_extension(format!("render-{seq}.json"));let _=std::thread::Builder::new().name("render-quality-scan".into()).spawn(move||inspect(&out));}return;}
    let seq=match cmd {Command::Reset(seq)|Command::Edit{seq,..}=>seq,_=>unreachable!()};let (lock,changed)=&*JOURNAL;let mut s=lock.lock().unwrap();if seq<=s.last {report(control,seq,false,"stale sequence");return;}s.last=seq;
    match cmd {
        Command::Reset(_)=>{let mut ok=true;for p in s.patches.drain(..){ok&=restore(&p);}changed.notify_all();report(control,seq,ok,"conditional rollback completed; newer or disappeared objects retained");},
        Command::Edit{field,object,old,new,hold,..}=>{
            if s.accepted>=16||s.patches.len()>=8||s.patches.iter().any(|p|p.object==object&&p.field==field){report(control,seq,false,"bounded mutation limit or duplicate field");return;}
            let Some(base)=version_base() else {report(control,seq,false,"original version signatures mismatch");return;};let space=crate::kernel::aspace();let (start,size,block)=space.map.lock().unwrap().query(object);let n=field.kind().shape().0 as u64;
            if object<space.heap()||object+n>space.heap()+space.layout.heap_size||block.state!=crate::mem::state::NORMAL||block.perm&crate::mem::perm::RW!=crate::mem::perm::RW||object+n>start+size {report(control,seq,false,"not a writable guest heap NORMAL object");return;}
            if !worker(){report(control,seq,false,"expiry worker unavailable; no write");return;}match apply(seq,field,object,base,old,new,hold){Ok(p)=>{s.patches.push(p);s.accepted+=1;changed.notify_all();report(control,seq,true,"typed original rendering field changed; verified and rollback armed");},Err(e)=>report(control,seq,false,e)}
        },_=>unreachable!()
    }
}
#[cfg(test)]mod tests {
    use super::*;
    #[test]fn render_edits_are_whitelisted_typed_and_bounded(){
        assert!(parse("edit 1 force-dof 0x7200000000 -1 0 45000").is_some());assert!(parse("edit 2 dynamic-ao 0x7200000000 1 0 1000").is_some());
        for s in ["read 0","read 1 extra","edit 1 time-scale 0x7200000000 1 0 1000","edit 1 dynamic-ao 0x7200000000 2 0 1000","edit 1 force-dof 0x7200000001 -1 0 1000","edit 1 force-dof 0x7200000000 -2 0 1000","edit 1 motion-blur 0x7200000000 1 0 120001"]{assert!(parse(s).is_none(),"{s}");}
    }
    #[test]fn crowd_type_uses_real_module_base_not_rounded_allocation(){
        let base=0x70c0804000u64;let mut asset=[0u8;0x70];asset[..8].copy_from_slice(&(base+0xcd22950).to_le_bytes());asset[8..16].copy_from_slice(&0x70ce56d530u64.to_le_bytes());asset[0x18..0x20].copy_from_slice(&0x70cc79e840u64.to_le_bytes());asset[0x40..0x44].copy_from_slice(&1.0f32.to_le_bytes());asset[0x44..0x48].copy_from_slice(&1.0f32.to_le_bytes());assert!(editable(&asset,base,Kind::Crowd));assert!(!editable(&asset,0x70c0000000,Kind::Crowd));
    }
    #[test]fn actual_float_lod_edits_are_finite_bounded_and_restore_exact_bytes(){
        let base=0x7100000000u64;#[repr(align(16))]struct Aligned([u8;0xa0]);let mut storage=Aligned([0xabu8;0xa0]);let data=&mut storage.0;let object=unsafe{data.as_mut_ptr().add(16)} as u64;
        for (kind,fields) in [(Kind::Mesh,vec![Field::MeshLod]),(Kind::Crowd,vec![Field::CrowdLod,Field::CrowdBoostLod])]{
            let (size,vtable)=kind.shape();data[16..16+size].fill(0);data[16..24].copy_from_slice(&(base+vtable).to_le_bytes());data[24..32].copy_from_slice(&(base+if kind==Kind::Crowd{0xdd69530}else{0xe6810b0}).to_le_bytes());if kind==Kind::Crowd{data[40..48].copy_from_slice(&(base+0xbf9a840).to_le_bytes());}else{data[40..44].copy_from_slice(&11u32.to_le_bytes());}for f in &fields {let at=16+f.offset() as usize;data[at..at+4].copy_from_slice(&1.0f32.to_le_bytes());}let before=*data;
            for f in fields {let old=1.0f32.to_bits() as i32;let new=0.5f32.to_bits() as i32;assert!(!f.valid(f32::NAN.to_bits() as i32));assert!(!f.valid(f32::INFINITY.to_bits() as i32));assert!(!f.valid(0));assert!(!f.valid(4.01f32.to_bits() as i32));let p=apply(1,f,object,base,old,new,Duration::from_secs(1)).unwrap();let at=16+f.offset() as usize;assert_eq!(&data[..at],&before[..at]);assert_eq!(&data[at+4..],&before[at+4..]);assert!(restore(&p));assert_eq!(&*data,&before);}
        }
    }
    #[test]fn original_fifa_render_controls_require_exact_rtti_and_preserve_refcounts_and_strings(){
        let base=0x7100804000u64;#[repr(align(16))]struct Aligned([u8;0x160]);let mut storage=Aligned([0xabu8;0x160]);let data=&mut storage.0;let object=unsafe{data.as_mut_ptr().add(16)} as u64;
        for (kind,fields) in [(Kind::FifaRender,vec![Field::HighresFan,Field::PitchClumps]),(Kind::FifaCrowd,vec![Field::CrowdSelection,Field::SidelineEnable,Field::SidelineMax])]{
            let (size,vtable)=kind.shape();data[16..16+size].fill(0);data[16..24].copy_from_slice(&(base+vtable).to_le_bytes());data[24..32].copy_from_slice(&(base+if kind==Kind::FifaRender{0xd7a6ce0u64}else{0xd6fdb88}).to_le_bytes());data[32..40].copy_from_slice(&0x0bbf000000000001u64.to_le_bytes());data[40..48].copy_from_slice(&(base+0xdeac4c8).to_le_bytes());
            for f in &fields {let at=16+f.offset() as usize;if f.width()==4 {data[at..at+4].copy_from_slice(&128i32.to_le_bytes());}else{data[at]=1;}}
            let before=*data;assert!(editable(&data[16..16+size],base,kind));
            for f in fields {let old=if f.is_count(){128}else{1};let p=apply(1,f,object,base,old,0,Duration::from_secs(1)).unwrap();let at=16+f.offset() as usize;assert_eq!(&data[..at],&before[..at]);assert_eq!(&data[at+f.width()..],&before[at+f.width()..]);assert!(restore(&p));assert_eq!(&*data,&before);}
            data[24..32].copy_from_slice(&(base+0xd7ab740).to_le_bytes());assert!(!identity(&data[16..16+size],base,kind));assert!(!editable(&data[16..16+size],base,kind));
        }
        assert!(!Field::CrowdSelection.valid(-1));assert!(!Field::CrowdSelection.valid(65537));assert!(parse("edit 1 crowd-selection-max 0x7100000000 128 0 120000").is_some());
    }
    #[test]fn original_mesh_force_lod_is_signed_bounded_and_preserves_every_other_byte(){
        let base=0x7100804000u64;#[repr(align(16))]struct Aligned([u8;0xa0]);let mut storage=Aligned([0xabu8;0xa0]);let data=&mut storage.0;data[16..16+0x80].fill(0);data[16..24].copy_from_slice(&(base+0xcc4d658).to_le_bytes());data[24..32].copy_from_slice(&(base+0xddda678).to_le_bytes());data[40..44].copy_from_slice(&11u32.to_le_bytes());data[16+0x50..16+0x54].copy_from_slice(&1.0f32.to_le_bytes());data[16+0x3c..16+0x40].copy_from_slice(&(-1i32).to_le_bytes());let before=*data;let object=unsafe{data.as_mut_ptr().add(16)} as u64;
        for new in [1,2,4]{let p=apply(1,Field::ForceMeshLod,object,base,-1,new,Duration::from_secs(1)).unwrap();assert_eq!(value(&data[16..],Field::ForceMeshLod),Some(new));assert_eq!(&data[..16+0x3c],&before[..16+0x3c]);assert_eq!(&data[16+0x40..],&before[16+0x40..]);assert!(restore(&p));assert_eq!(&*data,&before);}
        assert!(parse("edit 1 mesh-force-lod 0x7200000000 -1 1 120000").is_some());assert!(!Field::ForceMeshLod.valid(-2));assert!(!Field::ForceMeshLod.valid(5));
    }
    #[test]fn original_resolution_fields_edit_and_restore_independently(){
        let base=0x70c0804000u64;#[repr(align(16))]struct Aligned([u8;0x280]);let mut storage=Aligned([0xabu8;0x280]);let data=&mut storage.0;data[16..16+0x248].fill(0);data[16..24].copy_from_slice(&(base+0xcc447d8).to_le_bytes());data[24..32].copy_from_slice(&(base+0xda06df0).to_le_bytes());data[40..44].copy_from_slice(&11u32.to_le_bytes());data[16+0x15c..16+0x160].copy_from_slice(&1.0f32.to_le_bytes());data[16+0x1fe]=1;let before=*data;let object=unsafe{data.as_mut_ptr().add(16)} as u64;
        for (i,f,old,new) in [(1,Field::DynamicResolution,1,0),(2,Field::Resolution,1.0f32.to_bits() as i32,0.75f32.to_bits() as i32),(3,Field::ResolutionMin,0,1.0f32.to_bits() as i32)] {let p=apply(i,f,object,base,old,new,Duration::from_secs(1)).unwrap();let at=16+f.offset() as usize;assert_eq!(&data[..at],&before[..at]);assert_eq!(&data[at+f.width()..],&before[at+f.width()..]);assert!(restore(&p));assert_eq!(&*data,&before);}
    }
    #[test]fn concrete_world_fields_edit_correct_inherited_and_derived_bytes(){
        let base=0x7100000000u64;#[repr(align(16))]struct Aligned([u8;0x920]);let mut storage=Aligned([0xabu8;0x920]);let data=&mut storage.0;let object=unsafe{data.as_mut_ptr().add(16)} as u64;
        let mut original=[0u8;0x8f0];original[..8].copy_from_slice(&(base+0xcdf6770).to_le_bytes());original[8..16].copy_from_slice(&(base+0xe6810b0).to_le_bytes());original[0x18..0x1c].copy_from_slice(&11u32.to_le_bytes());for f in [Field::MotionBlur,Field::Filmic,Field::Subsurface]{original[f.offset() as usize]=1;}data[16..16+0x8f0].copy_from_slice(&original);let before=*data;
        assert!(identity(&data[16..16+0x8f0],base,Kind::World));assert!(!identity(&data[16..16+0x350],base,Kind::World));
        for (i,f) in [Field::MotionBlur,Field::Filmic,Field::Subsurface].into_iter().enumerate(){let p=apply(i as u64+1,f,object,base,1,0,Duration::from_secs(1)).unwrap();let at=16+f.offset() as usize;assert_eq!(data[at],0);assert_eq!(&data[..at],&before[..at]);assert_eq!(&data[at+1..],&before[at+1..]);assert!(restore(&p));assert_eq!(&*data,&before);}
        let mut asset=[0u8;0x70];asset[..8].copy_from_slice(&(base+0xcd22950).to_le_bytes());asset[8..16].copy_from_slice(&(base+0xdd69530).to_le_bytes());asset[0x18..0x20].copy_from_slice(&(base+0xbf9a840).to_le_bytes());asset[0x40..0x44].copy_from_slice(&1.0f32.to_le_bytes());asset[0x44..0x48].copy_from_slice(&1.0f32.to_le_bytes());assert!(editable(&asset,base,Kind::Crowd));asset[8..16].copy_from_slice(&(base+0xe6810b0).to_le_bytes());assert!(!editable(&asset,base,Kind::Crowd));
        assert!(parse("edit 1 filmic-effects 0x7200000000 1 0 45000").is_some());assert!(parse("edit 1 subsurface 0x7200000000 1 0 45000").is_some());assert!(parse("edit 1 subsurface 0x7200000000 1 2 45000").is_none());
    }
    #[test]fn actual_byte_and_word_edits_preserve_neighbors_and_newer_values(){
        let base=0x7100000000u64;#[repr(align(16))]struct Aligned([u8;0x380]);let mut storage=Aligned([0xabu8;0x380]);let data=&mut storage.0;let object=unsafe{data.as_mut_ptr().add(16)} as u64;let mut original=[0;0x350];original[..8].copy_from_slice(&(base+0xcc4f428).to_le_bytes());original[8..16].copy_from_slice(&(base+0xda0c3c0).to_le_bytes());original[0x18..0x1c].copy_from_slice(&11u32.to_le_bytes());original[0x24c..0x250].copy_from_slice(&(-1i32).to_le_bytes());original[0x336]=1;data[16..16+0x350].copy_from_slice(&original);let before=*data;
        let p=apply(1,Field::DynamicAo,object,base,1,0,Duration::from_secs(1)).unwrap();assert!(matches(&p,0));assert_eq!(&data[..16+0x336],&before[..16+0x336]);assert_eq!(&data[16+0x337..],&before[16+0x337..]);assert!(restore(&p));assert_eq!(&*data,&before);
        let p=apply(2,Field::ForceDof,object,base,-1,0,Duration::from_secs(1)).unwrap();assert!(matches(&p,0));assert!(write(object+0x24c,Field::ForceDof,1));assert!(!restore(&p));assert!(!write(object+0x24d,Field::ForceDof,0));
    }
}
