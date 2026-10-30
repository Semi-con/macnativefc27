use super::{Stage,Translated,PRELUDE};

pub struct DynamicVertexSources {
    pub library:String,
    pub wrapper:String,
    pub entry:String,
}

fn structure<'a>(source:&'a str,name:&str)->Option<&'a str> {
    let start=source.find(&format!("struct {name} {{"))?;
    let end=start+source[start..].find("};")?+2;
    Some(&source[start..end])
}

fn parameters(source:&str)->Option<Vec<&str>> {
    let mut depth=[0i32;3];let mut start=0;let mut result=Vec::new();
    for (at,c) in source.char_indices() {
        match c {'<'=>depth[0]+=1,'>'=>depth[0]-=1,'('=>depth[1]+=1,')'=>depth[1]-=1,'['=>depth[2]+=1,']'=>depth[2]-=1,
            ',' if depth==[0;3]=>{result.push(source[start..at].trim());start=at+1;},_=>{}}
        if depth.iter().any(|v|*v<0) {return None;}
    }
    if depth!=[0;3] {return None;}
    result.push(source[start..].trim());
    Some(result)
}

impl Translated {
    // Only the vertex entry is split. Fragment discard has a demonstrated
    // different result across a dynamic call on the target GPU; fragment and
    // compute programs must retain their original compilation paths.
    pub fn dynamic_vertex_sources(&self,suffix:&str)->Option<DynamicVertexSources> {
        if self.stage!=Stage::Vertex || self.msl.contains("discard_fragment();") {return None;}
        let (source,entry)=self.named(suffix);let output=format!("VOut_{suffix}");
        let signature=format!("vertex {output} {entry}(");let start=source.find(&signature)?;
        let arguments_start=start+signature.len();let arguments_end=arguments_start+source[arguments_start..].find(") {")?;
        let arguments=&source[arguments_start..arguments_end];let body=&source[arguments_end+2..];
        if !body.starts_with('{') || !body.trim_end().ends_with('}') {return None;}
        let mut declarations=Vec::new();let mut names=Vec::new();
        for parameter in parameters(arguments)? {
            let bare=parameter.split("[[").next()?.trim();
            let name=bare.split_whitespace().next_back()?;
            if name.is_empty() || !name.bytes().all(|b|b.is_ascii_alphanumeric()||b==b'_') {return None;}
            declarations.push(bare);names.push(name);
        }
        let helper=format!("{entry}_body");let declarations=declarations.join(", ");
        let library=format!("{}{output} {helper}({declarations}) {body}",&source[..start]);
        // Copy the actual ABI declarations, excluding unused helpers and
        // embedded constant arrays from the small executable entry library.
        let mut types=String::from("#include <metal_stdlib>\nusing namespace metal;\n#pragma METAL fp contract(off)\n");
        for name in ["Res","DynamicTextures"] {types.push_str(structure(PRELUDE,name)?);types.push('\n');}
        types.push_str(structure(&source,&output)?);types.push('\n');
        let static_name=format!("StaticTextures_{suffix}");
        if arguments.contains(&static_name) {types.push_str(structure(&source,&static_name)?);types.push('\n');}
        let wrapper=format!("{types}extern {output} {helper}({declarations});\nvertex {output} {entry}({arguments}) {{return {helper}({});}}\n",names.join(", "));
        Some(DynamicVertexSources{library,wrapper,entry})
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn vertex_split_keeps_body_bindings_varyings_and_parameter_order() {
        let tail="struct VOut {float4 pos [[position,invariant]];float4 v0 [[user(locn0)]];};\nvertex VOut vmain(uint vid [[vertex_id]], uint iid [[instance_id]], constant Res& res [[buffer(0)]], texture2d<uint,access::read_write> im0 [[texture(32)]]) {VOut out;out.pos=float4(vid,iid,res.flags,1);return out;}\n";
        let t=Translated{stage:Stage::Vertex,msl:format!("{PRELUDE}{tail}"),entry:"vmain".into(),interface:Default::default(),grid_write_footprint:None,descriptor_write_footprint:None,write_window_footprint:None};
        let s=t.dynamic_vertex_sources("abcd").unwrap();
        assert!(s.library.contains("VOut_abcd vmain_abcd_body(uint vid, uint iid, constant Res& res, texture2d<uint,access::read_write> im0) {VOut_abcd out;out.pos=float4(vid,iid,res.flags,1);return out;}"));
        assert!(s.wrapper.contains("[[vertex_id]]"));assert!(s.wrapper.contains("[[buffer(0)]]"));assert!(s.wrapper.contains("[[texture(32)]]"));assert!(s.wrapper.contains("vmain_abcd_body(vid, iid, res, im0)"));
        assert!(!s.wrapper.contains("out.pos="));assert!(!s.wrapper.contains("nx_mufu_rcp"));
        let mut rejected=t;rejected.stage=Stage::Fragment;assert!(rejected.dynamic_vertex_sources("abcd").is_none());
        rejected.stage=Stage::Vertex;rejected.msl.push_str("discard_fragment();");assert!(rejected.dynamic_vertex_sources("abcd").is_none());
    }
}
