// Bounded read-only observations of main-menu UI and playback objects.
use std::{path::Path,sync::{LazyLock,atomic::{AtomicU32,Ordering::Relaxed}},time::{Duration,Instant}};
const SIZE:usize=0x2e0;
const VTABLE:u64=0xc730318;
fn resource_path(line:&[u8])->bool {
    if !(8..=256).contains(&line.len()) || !line.iter().all(|b|b.is_ascii_alphanumeric() || b"_-/\\. ".contains(b)) {return false;}
    let s=String::from_utf8_lossy(line).to_ascii_lowercase();
    if ["key","token","secret","password","credential","account","profile","auth","user","admin"].into_iter().any(|v|s.contains(v)) {return false;}
    if s.starts_with("sound/") {return false;}
    if !s.contains('/') && !s.contains('\\') && !s.contains('.') {return false;}
    let movie=[".webm",".vp6",".vp8",".vp9",".mp4",".bk2",".usm"].into_iter().any(|v|s.ends_with(v));
    ["mainmenu","main_menu","main-menu","slideshow"].into_iter().any(|v|s.contains(v)) || (movie && (s.contains("movie") || s.contains("background")))
}
fn collect_paths(bytes:&[u8],address:u64,paths:&mut Vec<String>,seen:&mut std::collections::HashSet<String>) {
    if paths.len()>=96 {return;}
    let mut start=0;
    for end in 0..=bytes.len() {
        if end<bytes.len() && (32..127).contains(&bytes[end]) {continue;}
        let line=&bytes[start..end];
        if resource_path(line) {
            let value=String::from_utf8_lossy(line).into_owned();
            if seen.insert(value.clone()) {
                let text=super::config_inspection::quote_ascii(&value);let at=address+start as u64;
                paths.push(format!(r#"{{"address":"{at:#x}","path":{text}}}"#));if paths.len()==96 {break;}
            }
        }
        start=end+1;
    }
}
fn copy(address:u64,bytes:&mut[u8])->bool {super::config_inspection::snapshot(address,bytes)}
fn word(bytes:&[u8],at:usize)->Option<u64> {Some(u64::from_le_bytes(bytes.get(at..at+8)?.try_into().ok()?))}
fn integer(bytes:&[u8],at:usize)->Option<i32> {Some(i32::from_le_bytes(bytes.get(at..at+4)?.try_into().ok()?))}
fn identity(bytes:&[u8],base:u64)->bool {
    bytes.len()>=SIZE && word(bytes,0)==Some(base+VTABLE) && word(bytes,0x118)==Some(base+VTABLE+0xa8)
}
fn cinematic_identity(bytes:&[u8],base:u64)->bool {
    bytes.len()>=0xf0 && word(bytes,0)==Some(base+0xc886208) && word(bytes,8)==Some(base+0xd6d9dc8)
        && [0xe0,0xe1,0xe2,0xe4,0xe5,0xe6,0xe7,0xe8].into_iter().all(|at|bytes[at]<=1)
}
fn version_base()->Option<u64> {
    let base=crate::diag::MODULES.get()?.iter().find(|m|m.name=="main")?.base;
    (crate::diag::read_words::<3>(base+VTABLE)==Some([base+0x3c45660,base+0x3c459e0,base+0x3c45a50])
        && crate::diag::read_words::<3>(base+VTABLE+0xa8)==Some([base+0x3c4d920,base+0x3c4d930,base+0x3c4d9a0])
        && crate::diag::read_words::<1>(base+0x3c4363c)==Some([0x910c6108b0045768])
        && crate::diag::read_words::<1>(base+0x3c48254)==Some([0xf9409000aa0003f3])
        && crate::diag::read_words::<4>(base+0xc886490)==Some([base+0xbcd1c40,0x657039be,0xe4,base+0xdeaf300])
        && crate::diag::read_words::<2>(base+0xd079c28)==Some([base+0xbd65887,0xf0006359360a8f])
        && crate::diag::read_words::<1>(base+0xd079c10)==Some([base+0xd6d9dc8])).then_some(base)
}
fn object(pointer:u64,base:u64)->String {
    let vtable=(pointer!=0).then(||crate::diag::read_words::<1>(pointer)).flatten().map(|v|v[0]);
    let rva=vtable.and_then(|v|v.checked_sub(base)).filter(|v|(0xb99b000..0xd308570).contains(v));
    format!(r#"{{"address":"{pointer:#x}","vtable_rva":{}}}"#,rva.map_or("null".into(),|v|format!("\"{v:#x}\"")))
}
fn slideshow_version(base:u64)->bool {
    crate::diag::read_words::<2>(base+0xd2c2578)==Some([base+0xbbe231c,0x680063c10f8872]) &&
    crate::diag::read_words::<2>(base+0xd2c7f00)==Some([base+0xbac0f20,0x480063ce57f3d2]) &&
    crate::diag::read_words::<1>(base+0xaa8bdd0)==Some([0x912a6108f00116a8]) &&
    crate::diag::read_words::<3>(base+0xcd62a98)==Some([base+0xaa8be10,base+0xaa8be20,base+0xaa8be80]) &&
    crate::diag::read_words::<1>(base+0xaa8bc00)==Some([0x9120a108900116c8])
}
fn slideshow_data(bytes:&[u8],base:u64)->bool {
    // The asset uses the generic DataContainer vtable; cd63828 belongs
    // to its runtime track factory, not to the serialized asset instance.
    bytes.len()>=0x48 && word(bytes,8)==Some(base+0xdda7aa8) &&
        word(bytes,0).and_then(|v|v.checked_sub(base)).is_some_and(|v|(0xb99b000..0xd308570).contains(&v))
}
fn selected_path(pointer:u64)->String {
    crate::video_diagnostics::graphics_path(pointer).map(|v|super::config_inspection::quote_ascii(&v)).unwrap_or("null".into())
}
fn keyframe_times(pointer:u64)->String {
    let Some(count_address)=pointer.checked_sub(4) else{return "null".into();};
    let mut count=[0;4];if !copy(count_address,&mut count) {return "null".into();}
    let count=u32::from_le_bytes(count)&0x7fffffff;
    if count==0 || count>4096 {return "null".into();}
    let mut data=[0u8;32];let len=count.min(8)as usize*4;
    if !copy(pointer,&mut data[..len]) {return "null".into();}
    let times:Option<Vec<_>>=data[..len].chunks_exact(4).map(|v|{
        let value=f32::from_le_bytes(v.try_into().unwrap());value.is_finite().then(||value.to_string())
    }).collect();
    times.map(|v|format!(r#"{{"count":{count},"first_times":[{}]}}"#,v.join(","))).unwrap_or("null".into())
}
fn ea_identifier(address:u64)->Option<String> {
    let mut header=[0u8;16];if !copy(address,&mut header) {return None;}
    let flag=header[15];let len=if flag&128==0 {15usize.checked_sub(flag as usize)?}else{integer(&header,8)?as usize};
    if len==0 || len>160 {return None;}
    let pointer=if flag&128==0{address}else{word(&header,0).unwrap()};let mut bytes=vec![0;len];
    if !copy(pointer,&mut bytes) || !bytes.iter().all(|b|b.is_ascii_alphanumeric() || b"_-/\\. ".contains(b)) {return None;}
    let text=String::from_utf8_lossy(&bytes).into_owned();let lower=text.to_ascii_lowercase();
    if ["key","token","secret","password","credential","account","profile","auth","user","admin"].into_iter().any(|v|lower.contains(v)) {return None;}
    Some(text)
}
fn frontend_location(address:u64)->String {
    ea_identifier(address).filter(|s|{
        let lower=s.to_ascii_lowercase();
        ["main","front","ui/","content/","cinematic/","felua"].into_iter().any(|v|lower.starts_with(v)) || lower.contains("mainmenu")
    }).map(|v|super::config_inspection::quote_ascii(&v)).unwrap_or("null".into())
}
fn response_string(service:u64,base:u64,id:u32)->Option<String> {
    if !matches!(id,53|54) || crate::diag::read_words::<1>(service)!=Some([base+0xcd76a70]) ||
        crate::diag::read_words::<1>(base+0xcd76aa8)!=Some([base+0xac32590]) ||
        crate::diag::read_words::<3>(base+0xac325a0)!=Some([0x93407c28b9402809,0x9b09a1489ac9090a,0xf8687948f940100a]) {return None;}
    let [buckets]=crate::diag::read_words::<1>(service.checked_add(0x20)?)?;
    let mut count=[0u8;4];if !copy(service+0x28,&mut count) {return None;}
    let count=u32::from_le_bytes(count);if count==0 || count>65536 {return None;}
    let [sentinel]=crate::diag::read_words::<1>(buckets.checked_add(count as u64*8)?)?;
    let [mut node]=crate::diag::read_words::<1>(buckets.checked_add((id%count)as u64*8)?)?;
    let mut seen=std::collections::HashSet::new();
    for _ in 0..32 {
        if node==0 || node==sentinel || !seen.insert(node) {return None;}
        let mut key=[0u8;4];if !copy(node,&mut key) {return None;}
        if u32::from_le_bytes(key)==id {return ea_identifier(node.checked_add(8)?);}
        node=crate::diag::read_words::<1>(node.checked_add(0x20)?)?[0];
    }
    None
}
fn method_rva(pointer:u64,base:u64,slot:u64)->String {
    let address=crate::diag::read_words::<1>(pointer).and_then(|v|v[0].checked_add(slot)).and_then(crate::diag::read_words::<1>).map(|v|v[0]);
    address.and_then(|v|v.checked_sub(base)).filter(|v|*v<0xb99a7c0).map_or("null".into(),|v|format!("\"{v:#x}\""))
}
fn menu_component(service:u64,base:u64)->String {
    if crate::diag::read_words::<1>(service)!=Some([base+0xcd8e5a8]) ||
        crate::diag::read_words::<2>(base+0xad41a00)!=Some([0x7200003ff9400008,0x528006a8f9404502]) ||
        crate::diag::read_words::<1>(base+0x2d84f14)!=Some([0xf9003a75ad0682a0]) {return "null".into();}
    let Some([component])=crate::diag::read_words::<1>(service+0x70) else{return "null".into();};
    if crate::diag::read_words::<1>(component)!=Some([base+0xcd8df80]) {return "null".into();}
    let mut metadata=[0u8;0x218];if !copy(component,&mut metadata) {return "null".into();}
    let evaluator=word(&metadata,0x50).unwrap();let count=integer(&metadata,0xc8).filter(|v|(0..=4096).contains(v));
    format!(r#"{{"component":{},"evaluator":{},"evaluator_method_20_rva":{},"pending_string_count":{},"owner_thread_initialized":{},"filtered_string_70":{},"filtered_string_160":{},"filtered_string_200":{}}}"#,
        object(component,base),object(evaluator,base),method_rva(evaluator,base,0x20),count.map_or("null".into(),|v|v.to_string()),word(&metadata,0xd0).unwrap()!=0,
        frontend_location(component+0x70),frontend_location(component+0x160),frontend_location(component+0x200))
}
fn registry_entry(registry:u64,id:u32)->Option<(u32,u64)> {
    debug_assert!(matches!(id,0x0f695031|0x0ebde53a|0x254ef20a));
    for _ in 0..3 {
        let mut bank_bytes=[0u8;4];if !copy(registry+0x3308,&mut bank_bytes) {return None;}
        let bank=u32::from_le_bytes(bank_bytes);if bank>1 {return None;}
        let mut keys=[0u8;541*4];let start=registry+0x48;
        if !copy(start+if bank==0{0}else{0x1960},&mut keys) {return None;}
        let mut service=0;
        for probe in 0..541u32 {
            let index=id.wrapping_add(probe)%541;
            let key=u32::from_le_bytes(keys[index as usize*4..index as usize*4+4].try_into().unwrap());
            if key==0 {break;}
            if key==id {
                let slot=start+(if bank==0{0x878}else{0x21d8})+index as u64*8;
                service=crate::diag::read_words::<1>(slot).map(|v|v[0]).unwrap_or(0);break;
            }
        }
        let mut after=[0u8;4];if !copy(registry+0x3308,&mut after) || bank_bytes!=after {continue;}
        return Some((bank,service));
    }
    None
}
fn ui_registry(base:u64)->Option<u64> {
    // Original use3DMainMenu query resolves its response bridge0f695031 via a
    // concrete double-buffered registry. Reproduce only its read-only
    // table lookup, never invoke its functions or expose other entries.
    if crate::diag::read_words::<2>(base+0x5aff2f0)!=Some([0xf941f500d003f1c8,0xf9402902f9400008]) ||
       crate::diag::read_words::<2>(base+0x9541860)!=Some([0x52843b0ab9730809,0x52832c0c52810f0b]) ||
       crate::diag::read_words::<2>(base+0x95418a0)!=Some([0xd367fdce9bab7dae,0xb86e594f1b08b5ce]) ||
       crate::diag::read_words::<1>(base+0x95418d4)!=Some([0xd65f03c0f86e5920]) ||
       crate::diag::read_words::<1>(base+0xcc21750)!=Some([base+0x9541860]) {return None;}
    let [registry]=crate::diag::read_words::<1>(base+0xd9393e8)?;
    if crate::diag::read_words::<1>(registry)!=Some([base+0xcc21700]) {return None;}
    Some(registry)
}
fn menu_interfaces(base:u64)->Option<(u32,u64,u64)> {
    let registry=ui_registry(base)?;
    let (bank,service)=registry_entry(registry,0x0f695031)?;
    let strings=registry_entry(registry,0x0ebde53a).map(|v|v.1).unwrap_or(0);
    Some((bank,service,strings))
}
pub(crate) fn population_service(base:u64)->serde_json::Value{
    // This exact ID is consumed by the original MainMenuInitializeData
    // prepare callback5cabfd0. Only reuse the existing bounded, original
    // registry lookup; do not invoke the provider or other interfaces.
    let value=ui_registry(base).and_then(|registry|registry_entry(registry,0x254ef20a));
    match value{
        Some((bank,service))=>{
            let method=|slot|serde_json::from_str::<serde_json::Value>(&method_rva(service,base,slot)).unwrap_or_default();
            serde_json::json!({"version_verified":true,"interface_id":"0x254ef20a","bank":bank,
            "provider":serde_json::from_str::<serde_json::Value>(&object(service,base)).unwrap_or_default(),
            "populate_method_30":method(0x30),"method_50":method(0x50),"method_58":method(0x58),
            "limits":"Read-only original registry/provider identity at the original callback. Method addresses alone do not establish implementation behavior. No calls or data edits."})
        },
        None=>serde_json::json!({"version_verified":false}),
    }
}
fn ambassador_configuration(base:u64)->String {
    // Exact original constructor stores this configuration at frontend+2520.
    // The registry may expose a secondary interface, so use its independently
    // constructor-verified frontend global and validate both config vtables.
    if crate::diag::read_words::<1>(base+0x2d2d6f0)!=Some([0xf91292f8aa1703e0]) ||
       crate::diag::read_words::<3>(base+0xc704460)!=Some([base+0x305d640,base+0x305d7a0,base+0x305d810]) {return r#"{"stage":"version_guard_failed"}"#.into();}
    let Some([frontend])=crate::diag::read_words::<1>(base+0xdc557e8)else{return r#"{"stage":"frontend_global_unreadable"}"#.into();};
    if frontend==0 {return r#"{"stage":"frontend_not_initialized"}"#.into();}
    let Some([configuration])=crate::diag::read_words::<1>(frontend+0x2520)else{return r#"{"stage":"configuration_pointer_unreadable"}"#.into();};
    if crate::diag::read_words::<2>(configuration)!=Some([base+0xc704460,base+0xc704490]) {return format!(r#"{{"stage":"configuration_identity_failed","object":{}}}"#,object(configuration,base));}
    let mut metadata=[0u8;0xf8];if !copy(configuration,&mut metadata) {return r#"{"stage":"configuration_copy_failed"}"#.into();}
    let selected=|at|ea_identifier(configuration+at).filter(|s|!s.to_ascii_lowercase().contains("http")).map(|s|super::config_inspection::quote_ascii(&s)).unwrap_or("null".into());
    format!(r#"{{"stage":"verified_original_configuration","object":{},"background_layer_name":{},"filtered_string_48":{},"source_filename":{},"filtered_string_78":{},"http_configuration_timeout_ms":{},"http_kit_metadata_timeout_ms":{},"limits":"FCHUBAmbassadorService original configuration; names and HTTP timeouts do not establish a scene playback rate or active media type"}}"#,
        object(configuration,base),selected(0x10),selected(0x48),selected(0x60),selected(0x78),integer(&metadata,0xac).unwrap(),integer(&metadata,0xb0).unwrap())
}
fn menu_service(base:u64)->String {
    let Some((bank,service,locations))=menu_interfaces(base) else{return r#"{"version_verified":false}"#.into();};
    let name=|id|response_string(locations,base,id).map(|v|super::config_inspection::quote_ascii(&v)).unwrap_or("null".into());
    format!(r#"{{"version_verified":true,"registry_bank":{bank},"interface":{},"bool_response_method_90_rva":{},"string_catalog_service":{},"string_lookup_method_38_rva":{},"response_string_53":{},"response_string_54":{},"frontend":{}}}"#,
        object(service,base),method_rva(service,base,0x90),object(locations,base),method_rva(locations,base,0x38),name(53),name(54),menu_component(service,base))
}
fn inspect(output:&Path) {
    let Some(base)=version_base() else {let _=std::fs::write(output,r#"{"version_verified":false,"objects":[]}"#);return;};
    let started=Instant::now();let space=crate::kernel::aspace();let first=space.heap();let bound=space.stack();let copy_bound=4096u64<<20;
    const CHUNK:usize=1<<20;const TAIL:usize=0x1000;
    let mut cursor=first;let mut scanned=0u64;let mut failures=0;let mut rows=Vec::new();let mut primary_candidates=0;let mut cinematic_primary_candidates=0;let mut cinematic_count=0;let mut slideshow_count=0;let mut zero_samples=0;let mut bytes=vec![0;CHUNK];let mut paths=Vec::new();let mut seen=std::collections::HashSet::new();
    let slideshow_verified=slideshow_version(base);let mut slideshow_tracks=0;let mut slideshow_assets=0;let mut shared_types=crate::shared_type_diagnostics::Observation::new(output);let mut menu_config=crate::menu_config_diagnostics::Observation::new(output);
    let mut alias_scanned=0u64;let mut alias_paths=Vec::new();let mut alias_seen=std::collections::HashSet::new();
    let still_version=crate::menu_still_frame_diagnostics::version(base);let mut still_count=0u32;let mut still_rows=Vec::new();
    let mut runtime_candidates=[0u64;4];
    let mut cinematic_candidates=crate::menu_cinematic_diagnostics::Observation::new(base);
    while cursor<bound && scanned<copy_bound && started.elapsed()<Duration::from_secs(15) {
        let (start,size,block)=space.map.lock().unwrap().query(cursor);
        let Some(end)=start.checked_add(size).map(|v|v.min(bound)).filter(|v|*v>cursor) else {break;};
        if !matches!(block.state,crate::mem::state::NORMAL|crate::mem::state::ALIAS) || block.perm&crate::mem::perm::R==0 {cursor=end;continue;}
        let step=(end-cursor).min((CHUNK-TAIL)as u64).min(copy_bound-scanned)as usize;let amount=(end-cursor).min((step+TAIL)as u64)as usize;
        if copy(cursor,&mut bytes[..amount]) {
            scanned+=step as u64;
            shared_types.observe(&bytes[..amount],cursor,step,bound);
            menu_config.observe(&bytes[..amount],cursor,step);
            if cursor>=space.alias() {alias_scanned+=step as u64;collect_paths(&bytes[..amount],cursor,&mut alias_paths,&mut alias_seen);}
            else{collect_paths(&bytes[..amount],cursor,&mut paths,&mut seen);}
            for at in (0..step.saturating_sub(7)).step_by(8) {
                let b=&bytes[at..amount];
                let address=cursor+at as u64;
                let runtime_candidate=crate::menu_still_frame_diagnostics::runtime_candidate(b,base);
                if still_version {if let Some(index)=runtime_candidate{runtime_candidates[index]+=1;}}
                if runtime_candidate==Some(2){cinematic_candidates.observe(b,address);}
                if still_version {if let Some(row)=crate::menu_still_frame_diagnostics::row(b,address,base){still_count+=1;if still_rows.len()<32{still_rows.push(row);}continue;}}
                if slideshow_verified && word(b,0)==Some(base+0xcd62a98) && b.len()>=0x68 && b[0x5c]<=1 && b[0x64]<=1 {
                    slideshow_tracks+=1;
                    if rows.len()<64 && slideshow_tracks<=16 {
                        let data=word(b,0x18).unwrap()&!4;
                        let mut metadata=[0u8;0x48];let valid=copy(data,&mut metadata)&&slideshow_data(&metadata,base);
                        let time=f32::from_bits(integer(b,0x60).unwrap()as u32);
                        rows.push(format!(r#"{{"class":"ClientSlideShowTrack","address":"{address:#x}","data":"{data:#x}","data_verified":{valid},"handle_58":{},"flag_5c":{},"flag_64":{},"last_time_60":{},"times":{}}}"#,
                            integer(b,0x58).unwrap(),b[0x5c],b[0x64],if time.is_finite(){time.to_string()}else{"null".into()},if valid{keyframe_times(word(&metadata,0x38).unwrap())}else{"null".into()}));
                    }
                    continue;
                }
                if slideshow_verified && slideshow_data(b,base) {
                    slideshow_assets+=1;
                    if rows.len()<64 && slideshow_assets<=4 {rows.push(format!(r#"{{"class":"SlideShowTrackData","address":"{address:#x}","asset_path":{},"times":{}}}"#,selected_path(word(b,0x18).unwrap()),keyframe_times(word(b,0x38).unwrap())));}
                    continue;
                }
                if word(b,0)==Some(base+0xc886208) {cinematic_primary_candidates+=1;}
                if cinematic_identity(b,base) {
                    cinematic_count+=1;if b[0xe4]!=0 {slideshow_count+=1;}
                    if rows.len()<48 && (b[0xe4]!=0 || zero_samples<4) {
                    if b[0xe4]==0 {zero_samples+=1;}
                    rows.push(format!(r#"{{"class":"FifaCinematicInstance","address":"{address:#x}","asset_path":{},"IsSlideshow":{},"Is30FPSLODgrp":{},"FullSuspendBCEngineWhileRunning":{},"DefaultHomeTeamId":{},"DefaultAwayTeamId":{}}}"#,
                        selected_path(word(b,0x18).unwrap()),b[0xe4],b[0xe8],b[0xe2],integer(b,0xcc).unwrap(),integer(b,0xd4).unwrap()));
                    }
                    continue;
                }
                if word(b,0)==Some(base+VTABLE) {primary_candidates+=1;}
                if !identity(b,base) || rows.len()>=64 {continue;}
                rows.push(format!(r#"{{"class":"MainMenuMoviePlayerViewModel","address":"{address:#x}","request_120":{},"stream_200":{},"content_map_count_198":{},"content_map_count_1d8":{},"content_type_1f8":{},"content_index_208":{},"flags_20c_20e":[{},{},{}]}}"#,
                    object(word(b,0x120).unwrap(),base),object(word(b,0x200).unwrap(),base),integer(b,0x198).unwrap(),integer(b,0x1d8).unwrap(),integer(b,0x1f8).unwrap(),integer(b,0x208).unwrap(),b[0x20c],b[0x20d],b[0x20e]));
            }
        }else{failures+=1;}
        cursor+=step as u64;
    }
    let player=crate::diag::read_words::<1>(base+0xd9192d8).map(|v|v[0]).unwrap_or(0);
    let mut flags=[0u8;2];let flags_ok=copy(base+0xd9124b8,&mut flags[..1]) && copy(base+0xd9124cc,&mut flags[1..]);
    let report=format!(r#"{{"version_verified":true,"seconds":{},"bytes_scanned":{scanned},"failed_chunks":{failures},"range_complete":{},"scan_first":"{first:#x}","scan_bound":"{bound:#x}","movie_primary_candidates":{primary_candidates},"cinematic_primary_candidates":{cinematic_primary_candidates},"cinematic_instance_count":{cinematic_count},"cinematic_slideshow_count":{slideshow_count},"slideshow_version_verified":{slideshow_verified},"slideshow_track_count":{slideshow_tracks},"slideshow_asset_count":{slideshow_assets},"alias_bytes_scanned":{alias_scanned},"menu_query_response_bridge":{},"ambassador_configuration":{},"alias_resource_paths":[{}],"player_global_d9192d8":{},"global_flags_4b8_4cc":{},"objects":[{}],"resource_paths":[{}],"limits":"Read-only entire data reservation heap through alias, max4096MiB copied from NORMAL/ALIAS readable pages,15sec,full counting with at most64 retained object rows and96selected credential-filtered local paths per heap/alias range; exact original primary/secondary vtables, constructor/update and cinematic bool-field metadata. No content, URLs, raw heap, guest writes, or virtual calls. Loaded cinematic assets alone do not establish which instance is active."}}"#,
        started.elapsed().as_secs_f64(),cursor>=bound,menu_service(base),ambassador_configuration(base),alias_paths.join(","),object(player,base),if flags_ok {format!("[{},{}]",flags[0],flags[1])}else{"null".into()},rows.join(","),paths.join(","));
    let report=report.replacen("\"version_verified\":true,",&format!("\"version_verified\":true,\"shared_type_descriptors\":{},",shared_types.report()),1);
    let report=report.replacen("\"version_verified\":true,",&format!("\"version_verified\":true,\"menu_configuration_metadata\":{},",menu_config.report()),1);
    let report=report.replacen("\"version_verified\":true,",&format!(r#""version_verified":true,"still_frame_selection":{{"version_verified":{still_version},"objects_count":{still_count},"objects":[{}],"limits":"Read-only exact original reflected fields, typed vtables and constructor guards; max32 retained rows, full counts. Loaded containers do not alone establish active selection. No writes or virtual calls."}},"#,still_rows.join(",")),1);
    let candidates=crate::menu_still_frame_diagnostics::RUNTIME_CLASSES.iter().zip(runtime_candidates).map(|((name,_),count)|format!("\"{name}\":{count}")).collect::<Vec<_>>().join(",");
    let report=report.replacen("\"version_verified\":true,",&format!(r#""version_verified":true,"runtime_entity_vtable_candidates":{{"version_verified":{still_version},"counts":{{{candidates}}},"limits":"All matching aligned words in the complete observed readable NORMAL/ALIAS reservation; no data-container filtering. Vtable matches may be stale or non-instance references and do not establish an active entity. This is independent of typed still_frame_selection rows."}},"#),1);
    let report=report.replacen("\"version_verified\":true,",&format!(r#""version_verified":true,"named_cinematic_candidates":{},"#,cinematic_candidates.report()),1);
    let _=std::fs::write(output,report);eprintln!("[menu-playback] objects={} bytes={scanned}",rows.len());
}
pub(super) fn poll(control:&Path) {
    static ENABLED:LazyLock<bool>=LazyLock::new(||std::env::var("NX_FC27_VIDEO_TRACE").is_ok_and(|v|v=="1"));
    if !*ENABLED {return;}
    let request=control.with_extension("menu-playback");let Some(text)=super::read(&request) else {return;};let _=std::fs::remove_file(request);
    let Some(sequence)=text.trim().parse::<u64>().ok().filter(|v|*v>0) else {return;};
    static READS:AtomicU32=AtomicU32::new(0);if READS.fetch_add(1,Relaxed)>=4 {return;}
    let output=control.with_extension(format!("menu-playback-{sequence}.json"));
    let _=std::thread::Builder::new().name("menu-playback-read".into()).spawn(move||inspect(&output));
}
#[cfg(test)]mod tests {
    use super::*;
    #[test]fn movie_object_requires_both_original_vtables_and_complete_copy() {
        let base=0x7100000000;let mut b=[0u8;SIZE];b[..8].copy_from_slice(&(base+VTABLE).to_le_bytes());b[0x118..0x120].copy_from_slice(&(base+VTABLE+0xa8).to_le_bytes());
        assert!(identity(&b,base));assert!(!identity(&b[..SIZE-1],base));assert!(!identity(&b,base+4096));
        b[0x118]^=1;assert!(!identity(&b,base));b[0x118]^=1;b[0]^=1;assert!(!identity(&b,base));
    }
    #[test]fn cinematic_observation_requires_original_type_and_valid_boolean_fields() {
        let base=0x7100000000;let mut b=[0u8;0xf0];b[..8].copy_from_slice(&(base+0xc886208u64).to_le_bytes());b[8..16].copy_from_slice(&(base+0xd6d9dc8u64).to_le_bytes());
        b[0xe4]=1;assert!(cinematic_identity(&b,base));assert!(!cinematic_identity(&b[..0xef],base));assert!(!cinematic_identity(&b,base+4096));
        b[0xe4]=2;assert!(!cinematic_identity(&b,base));b[0xe4]=1;b[8]^=1;assert!(!cinematic_identity(&b,base));
    }
    #[test]fn menu_path_observation_excludes_urls_credentials_and_unrelated_data() {
        for line in ["ui/movies/mainmenu/intro.webm","ui/main_menu/background","mainmenuconfig.json"] {assert!(resource_path(line.as_bytes()));}
        for line in ["https://example.com/mainmenu.webm","ui/mainmenu/token_value.webm","ui/mainmenu/prod.keys","ui/mainmenu/user.json","unrelated/file.dat","private_password_mainmenu/file.bin"] {assert!(!resource_path(line.as_bytes()));}
        assert!(!resource_path(&vec![b'a';257]));
        let mut paths=Vec::new();let mut seen=std::collections::HashSet::new();
        collect_paths(b"ui/main_menu/background\0https://example.com/mainmenu.webm\0ui/main_menu/background\0",100,&mut paths,&mut seen);assert_eq!(paths.len(),1);
    }
}
