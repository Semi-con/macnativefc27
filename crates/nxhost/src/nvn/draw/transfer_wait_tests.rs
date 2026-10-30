use super::*;

#[test]
#[ignore="requires a real Metal GPU; run the shared NVN fixture serially"]
fn gpu_known_transfers_defer_only_unrelated_reads_and_preserve_partial_alias_uploads() {
    objc2::rc::autoreleasepool(|_|unsafe {
        crate::kernel::KERNEL.aspace.get_or_init(||crate::mem::AddressSpace::reserve().unwrap());
        let page=crate::mem::PAGE;let cpu=crate::kernel::aspace().alias()+0x6800000;
        crate::kernel::aspace().set(cpu,page*4,crate::mem::Block{state:crate::mem::state::NORMAL,perm:crate::mem::perm::RW,attr:0}).unwrap();
        let physical=cpu+4096;let output=cpu+2*page;let va=0x28_0000_0000;let vb=0x29_0000_0000;let vc=0x2a_0000_0000;
        std::ptr::write_bytes(physical as *mut u8,0,8192);
        let mut pools=gfx::pool_memory(&[(0xcf00,physical,8192),(0xcf01,output,page)]).unwrap();
        // vb maps only the upper half of the original 32-byte write. Checking
        // just the start address would miss this alias entirely.
        gfx::add_pool_aliases(&mut pools,&[(va,physical,8192),(vb,physical+256,4096),(vc,physical,4096),(vc+4096,physical+4096,4096)]).unwrap();
        let mut e=Engine::default();let cb=gfx::gpu().queue.0.commandBuffer().unwrap();
        let mut r=Replay{e:&mut e,cb,pass:None,heap:None,pools,passes:0,copies:HashMap::new(),fault:None,pool_table:None,resource_buffers:Default::default()};
        r.clear_buffer(va+240,32,0x12345678).unwrap();
        let unsubmitted=Retained::as_ptr(&r.cb);r.run(&[Cmd::Barrier(0x10)],0);
        if cpu_events::transfers_enabled() {
            assert_eq!(Retained::as_ptr(&r.cb),unsubmitted,"known transfer should not commit for an unrelated barrier");
            r.prepare_cpu_read(vb+16,4);
            assert_eq!(Retained::as_ptr(&r.cb),unsubmitted,"the endpoint immediately after the write is unrelated");
            assert_eq!(((physical+256) as *const u32).read(),0,"unsubmitted GPU transfer unexpectedly changed CPU storage");
        }
        assert_eq!(r.cpu_bytes(vb,16).unwrap(),0x12345678u32.to_le_bytes().repeat(4),"host read through partial alias missed the write");
        r.clear_buffer(va+240,32,0x11223344).unwrap();r.run(&[Cmd::Barrier(0x10)],0);
        r.update_uniform(physical+256,16,4,&42u32.to_le_bytes());
        let overlay=r.e.overlays.get(&(physical+256)).unwrap();
        assert_eq!(std::slice::from_raw_parts(overlay.2 as *const u32,4),[0x11223344,42,0x11223344,0x11223344],"uniform snapshot must see preceding GPU output");
        r.e.overlays.clear();

        // Copy a GPU-produced value into registered texture storage and read
        // the resulting actual Metal texture back. No event-only shortcut.
        let mut builder=[0u8;0x68];builder[8..16].copy_from_slice(&0xcf00u64.to_le_bytes());builder[16..24].copy_from_slice(&512u64.to_le_bytes());
        for (at,value) in [(0x1c,1u32),(0x20,1),(0x24,1),(0x28,1),(0x2c,1),(0x30,11)] {builder[at..at+4].copy_from_slice(&value.to_le_bytes());}
        {let mut state=STATE.lock().unwrap();state.pools.insert(0xcf00,super::super::objects::MemoryPool{cpu:physical,size:8192,flags:0x22});state.registered_textures.insert((0xcf80,0x100),builder);super::super::objects::TEXTURE_DESCRIPTOR_EPOCH.fetch_add(1,Relaxed);}
        r.clear_buffer(va+1024,16,0xabcdef01).unwrap();r.copy_buffer(va+1024,va+512,16).unwrap();r.run(&[Cmd::Barrier(0x10)],0);
        let mut state=GState::default();state.texture_pool=0xcf80;r.prepare_stage_cpu_reads(&state,true);
        let (_,texture)=r.texture(&builder,false).unwrap();
        copy::copy_to_buffer(&r.cb,&texture.0,(0,0),[0,0,0,1,1,1],11,output,&r.pools,(0,0)).unwrap();r.commit_boundary(true);
        assert_eq!((output as *const u32).read(),0xabcdef01,"texture hash/upload read stale transfer output");

        // A texture and a GPU-only binding can occupy different allocations
        // in the same pool. Neither descriptor registration nor binding the
        // buffer payload consumes that unrelated payload on the CPU.
        r.clear_buffer(va+2048,16,0xfeed1234).unwrap();let before=Retained::as_ptr(&r.cb);
        r.run(&[Cmd::Barrier(0x10)],0);
        state.storage[2][0]=(va+2048,16);state.ubos[2][0]=(va+2048,16);
        r.prepare_stage_cpu_reads(&state,true);
        if cpu_events::transfers_enabled() {assert_eq!(Retained::as_ptr(&r.cb),before,"GPU-only binding or unrelated texture in the same pool caused a host wait");}
        r.copy_buffer(va+2048,output+128,16).unwrap();r.commit_boundary(true);
        assert_eq!(std::slice::from_raw_parts((output+128) as *const u32,4),[0xfeed1234;4],"deferred producer/consumer GPU chain was reordered");

        // The complete texture allocation still includes its block-linear
        // padding, which the existing CPU hash can inspect.
        r.clear_buffer(va+1020,4,0x99887766).unwrap();r.run(&[Cmd::Barrier(0x10)],0);
        r.prepare_stage_cpu_reads(&state,true);r.texture(&builder,false).unwrap();
        assert_eq!(((physical+1020) as *const u32).read(),0x99887766,"texture allocation padding was omitted from CPU read guards");

        // An unselected descriptor can overlap a pending transfer without a
        // host observation. A cached render texture also stays on the GPU.
        let mut private_builder=builder;private_builder[16..24].copy_from_slice(&3072u64.to_le_bytes());private_builder[0x30..0x34].copy_from_slice(&37u32.to_le_bytes());private_builder[0x20..0x24].copy_from_slice(&4u32.to_le_bytes());
        let (private_key,private)=r.texture(&private_builder,true).unwrap();
        {let mut registry=STATE.lock().unwrap();registry.registered_textures.insert((0xcf80,0x101),private_builder);super::super::objects::TEXTURE_DESCRIPTOR_EPOCH.fetch_add(1,Relaxed);}
        r.clear_buffer(va+3072,16,0x55667788).unwrap();let before=Retained::as_ptr(&r.cb);r.run(&[Cmd::Barrier(0x10)],0);
        r.prepare_stage_cpu_reads(&state,true);
        let (_,same)=r.texture_keyed(Some(private_key),&private_builder,false).unwrap();
        assert_eq!(Retained::as_ptr(&same.0),Retained::as_ptr(&private.0),"private render texture identity changed");
        if cpu_events::transfers_enabled() {assert_eq!(Retained::as_ptr(&r.cb),before,"unselected descriptor or private GPU-only texture caused a host wait");}
        r.commit_boundary(true);assert_eq!(((physical+3072) as *const u32).read(),0x55667788);

        // A selected CPU source can force a wait after a render pass has
        // already opened. The caller must reopen it without clearing away
        // earlier target contents, as the real draw path does.
        r.clear_buffer(va+512,16,0x11221122).unwrap();r.run(&[Cmd::Barrier(0x10)],0);
        r.run(&[Cmd::SetRenderTargets{colors:vec![Some(Box::new(private_builder))],depth:None},Cmd::ClearColor{index:0,rgba:[0.25,0.5,0.75,1.0]}],0);
        r.ensure_pass().unwrap();assert!(r.pass.is_some());
        r.texture(&builder,false).unwrap();
        if cpu_events::transfers_enabled() {assert!(r.pass.is_none(),"the actual CPU source read did not close and complete its preceding encoder");}
        r.ensure_pass().unwrap();r.end_pass();
        copy::copy_to_buffer(&r.cb,&private.0,(0,0),[0,0,0,1,1,1],37,output+256,&r.pools,(0,0)).unwrap();r.commit_boundary(true);
        assert_eq!(std::slice::from_raw_parts((output+256) as *const u8,4),[64,128,191,255],"reopened pass lost the preceding real GPU color clear");
        r.e.colors.clear();

        // Readback writes also have known destinations. Cross a virtual map
        // boundary, then consume through another partial CPU alias.
        std::ptr::write_bytes((physical+4088) as *mut u8,0,16);
        r.run(&[Cmd::CopyTextureToBuffer{src:Box::new(private_builder),region:[0,0,0,4,1,1],dst:vc+4088,strides:(0,0),flags:0},Cmd::Barrier(0x10)],0);
        let before=Retained::as_ptr(&r.cb);r.prepare_cpu_read(vb+3816,16);
        if cpu_events::transfers_enabled() {assert_eq!(r.cb.status(),MTLCommandBufferStatus::NotEnqueued);assert_eq!(Retained::as_ptr(&r.cb),before);}
        assert_eq!(r.cpu_bytes(vb+3832,16).unwrap(),[64u8,128,191,255].repeat(4),"split texture readback destination or partial alias tracking was incomplete");
        ((physical+8184) as *mut u64).write(0x12345678abcdef01);
        assert!(copy::copy_to_buffer(&r.cb,&private.0,(0,0),[0,0,0,4,1,1],37,vc+8184,&r.pools,(0,0)).is_err());
        assert_eq!(r.cb.status(),MTLCommandBufferStatus::NotEnqueued,"a rejected readback hole encoded a partial transfer");
        r.commit_boundary(true);assert_eq!(((physical+8184) as *const u64).read(),0x12345678abcdef01);

        // A known write survives a nonblocking commit, and an unknown writer
        // still forces the conservative barrier even with the gate enabled.
        r.clear_buffer(va+2048,16,0x76543210).unwrap();let prior=r.commit_boundary(false);r.run(&[Cmd::Barrier(0x10)],0);
        r.prepare_cpu_read(physical+2060,4);
        assert_eq!(prior.status(),MTLCommandBufferStatus::Completed);
        assert_eq!(((physical+2060) as *const u32).read(),0x76543210);
        r.e.gpu_buffer_dirty=true;let prior=Retained::as_ptr(&r.cb);r.run(&[Cmd::Barrier(0x10)],0);
        assert_ne!(Retained::as_ptr(&r.cb),prior,"unknown shader writes must not be deferred by the transfer gate");
        assert!(!r.e.gpu_buffer_dirty);
        {let mut state=STATE.lock().unwrap();state.pools.remove(&0xcf00);state.registered_textures.remove(&(0xcf80,0x100));state.registered_textures.remove(&(0xcf80,0x101));super::super::objects::TEXTURE_DESCRIPTOR_EPOCH.fetch_add(1,Relaxed);}
        gfx::forget_pool(0xcf00);gfx::forget_pool(0xcf01);
        eprintln!("[verify] real GPU transfer aliases, exact endpoint, uniform snapshot, selected texture upload/padding, unselected/private GPU-only texture, same-pool GPU-only chain, cross-submit readback and unknown-writer fallback passed");
    });
}
