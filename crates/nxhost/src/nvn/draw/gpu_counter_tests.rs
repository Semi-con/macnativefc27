use super::*;

#[test]
#[ignore="requires a real Metal GPU; shared NVN fixture must run serially"]
fn gpu_timestamp_reports_are_ordered_asynchronous_and_match_guest_clock_through_aliases() {
    objc2::rc::autoreleasepool(|_|unsafe {
        crate::kernel::KERNEL.aspace.get_or_init(||crate::mem::AddressSpace::reserve().unwrap());let page=crate::mem::PAGE;let cpu=crate::kernel::aspace().alias()+0x7400000;
        crate::kernel::aspace().set(cpu,page*4,crate::mem::Block{state:crate::mem::state::NORMAL,perm:crate::mem::perm::RW,attr:0}).unwrap();std::ptr::write_bytes(cpu as *mut u8,0,(page*4) as usize);
        let va=0x3000000000u64;let alias=0x3100000000u64;
        let mut pools=gfx::pool_memory(&[(0xcf70,cpu,page*4)]).unwrap();gfx::add_pool_aliases(&mut pools,&[(va,cpu,4096),(va+4096,cpu+4096,4096),(alias,cpu+4096,4096)]).unwrap();
        let mut e=Engine::default();let cb=gfx::gpu().queue.0.commandBuffer().unwrap();let mut r=Replay{e:&mut e,cb,pass:None,heap:None,pools,passes:0,copies:HashMap::new(),fault:None,pool_table:None,resource_buffers:Default::default()};
        if !gpu_counters::enabled() {
            r.run(&[Cmd::ReportCounter{counter:16,addr:cpu+4088},Cmd::ReportCounter{counter:0,addr:cpu+4160}],0);
            let top=((cpu+4096) as *const u64).read_unaligned();let bottom=((cpu+4168) as *const u64).read_unaligned();assert!(bottom>=top && top!=0);r.commit_boundary(true);gfx::forget_pool(0xcf70);return;
        }
        // Warm the tiny built-in pipelines, then block actual GPU work. Every
        // report below must return while this event still prevents execution.
        assert!(r.gpu_report_counter(16,cpu+512).unwrap());r.commit_boundary(true);
        std::ptr::write_bytes((cpu+4088) as *mut u8,0xa5,16);std::ptr::write_bytes((cpu+4160) as *mut u8,0xa5,16);
        let event=gfx::Shared(gfx::gpu().device.0.newSharedEvent().unwrap());r.cb.encodeWaitForEvent_value(ProtocolObject::from_ref(&*event.0),1);
        let signal=event.clone();let (send,receive)=std::sync::mpsc::channel();let release=std::thread::spawn(move ||{let timely=receive.recv_timeout(std::time::Duration::from_secs(1)).is_ok();signal.0.setSignaledValue(1);timely});
        let start=super::super::objects::timestamp_ns();
        assert!(r.gpu_report_counter(16,va+4088).unwrap());let first=r.cb.clone();
        assert!(r.gpu_report_counter(0,va+4160).unwrap());
        assert_ne!(first.status(),MTLCommandBufferStatus::Completed,"timestamp report waited on the CPU");
        assert_eq!(((cpu+4088) as *const u64).read_unaligned(),0xa5a5a5a5a5a5a5a5,"query was synthesized before GPU execution");
        r.copy_buffer(va+4160, cpu+8192,16).unwrap();assert_eq!(r.cb.status(),MTLCommandBufferStatus::NotEnqueued);
        send.send(()).unwrap();assert!(release.join().unwrap(),"GPU report blocked until the safety release deadline");
        // This host read intersects the second word of the split top report.
        // It must wait, and following GPU copies must see the published result.
        let _=r.cpu_bytes(alias,8).unwrap();let finish=super::super::objects::timestamp_ns();
        let top=((cpu+4096) as *const u64).read_unaligned();let bottom=((cpu+4168) as *const u64).read_unaligned();
        assert!(top>=start.saturating_sub(5_000_000) && bottom<=finish+5_000_000 && bottom>=top,"GPU counters do not match the guest nanosecond clock: {start},{top},{bottom},{finish}");
        assert_eq!(((cpu+4088) as *const u64).read_unaligned(),top);assert_eq!(((cpu+4160) as *const u64).read_unaligned(),bottom);
        assert_eq!(r.cpu_bytes(cpu+8192,16).unwrap(),r.cpu_bytes(alias+64,16).unwrap(),"GPU consumer did not observe the resolved timestamp");
        let before=Retained::as_ptr(&r.cb);assert!(r.gpu_report_counter(0,va+8184).is_err());assert_eq!(Retained::as_ptr(&r.cb),before,"mapping hole changed the command stream");
        r.commit_boundary(true);gfx::forget_pool(0xcf70);
        eprintln!("[verify] real GPU timestamp stage sampling/resolve/publication, no CPU wait while GPU blocked, split alias readback, following GPU consumer, guest-clock correlation and atomic hole rejection passed");
    });
}
