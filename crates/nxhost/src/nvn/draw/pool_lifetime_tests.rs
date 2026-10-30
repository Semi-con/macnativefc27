use super::*;
use crate::nvn::objects;
use crate::stubs::Args;
use std::sync::mpsc;

#[test]
#[ignore="requires a real Metal GPU; run the shared NVN fixture serially"]
fn gpu_pool_snapshot_pins_storage_before_concurrent_finalize_can_return() {
    objc2::rc::autoreleasepool(|_| unsafe {
        // Other GPU fixtures also use the global registry. The verification
        // runner uses --test-threads=1; restore every pre-existing entry.
        let previous=std::mem::take(&mut *STATE.lock().unwrap());
        let page=crate::mem::PAGE as usize;
        let allocation=libc::mmap(std::ptr::null_mut(),page,libc::PROT_READ|libc::PROT_WRITE,libc::MAP_PRIVATE|libc::MAP_ANON,-1,0);
        assert_ne!(allocation,libc::MAP_FAILED);
        let cpu=allocation as u64;
        for i in 0..page/4 {(allocation as *mut u32).add(i).write(0xfc270000+i as u32);}
        let mut object=[0u64;32];let obj=object.as_mut_ptr() as u64;
        let builder=[0x22u64<<32,0,cpu,page as u64,0];
        let initialize=objects::host_impl("nvnMemoryPoolInitialize").unwrap();
        assert_eq!(initialize(&Args([obj,builder.as_ptr() as u64,0,0,0,0,0,0],[0;8])),1);
        let (begin_tx,begin_rx)=mpsc::channel();
        let (attempt_tx,attempt_rx)=mpsc::channel();
        let (finalized_tx,finalized_rx)=mpsc::channel();
        let (unmap_tx,unmap_rx)=mpsc::channel();
        let worker=std::thread::spawn(move || {
            begin_rx.recv().unwrap();attempt_tx.send(()).unwrap();
            objects::host_impl("nvnMemoryPoolFinalize").unwrap()(&Args([obj,0,0,0,0,0,0,0],[0;8]));
            finalized_tx.send(()).unwrap();
            // A conforming guest keeps storage until its in-flight GPU work
            // completes. Remove the CPU mapping as soon as that is allowed.
            unmap_rx.recv().unwrap();
            assert_eq!(libc::mmap(cpu as *mut _,page,libc::PROT_NONE,libc::MAP_PRIVATE|libc::MAP_ANON|libc::MAP_FIXED,-1,0) as u64,cpu);
        });
        let pools=mapped_pool_snapshot(|| {
            begin_tx.send(()).unwrap();attempt_rx.recv().unwrap();
            assert!(STATE.try_lock().is_err(),"pool registry must still be locked after copying numeric addresses");
            assert!(matches!(finalized_rx.try_recv(),Err(mpsc::TryRecvError::Empty)),"finalize returned while its uncached storage was still about to be wrapped");
        }).unwrap();
        finalized_rx.recv_timeout(std::time::Duration::from_secs(5)).unwrap();
        assert!(!STATE.lock().unwrap().pools.contains_key(&obj));
        let (chunk,offset)=gfx::pool_buffer_range(&pools,cpu,page as u64).unwrap();
        let output=gfx::gpu().device.0.newBufferWithLength_options(page,MTLResourceOptions::StorageModeShared).unwrap();
        let cb=gfx::gpu().queue.0.commandBuffer().unwrap();let blit=cb.blitCommandEncoder().unwrap();
        blit.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(&chunk.buffer.0,offset,&output,0,page);
        blit.endEncoding();cb.commit();cb.waitUntilCompleted();assert_eq!(cb.status(),MTLCommandBufferStatus::Completed);
        for (i,&word) in std::slice::from_raw_parts(output.contents().as_ptr().cast::<u32>(),page/4).iter().enumerate() {assert_eq!(word,0xfc270000+i as u32);}
        unmap_tx.send(()).unwrap();worker.join().unwrap();
        assert!(mapped_pool_snapshot(||{}).unwrap().is_empty(),"finalized and unmapped storage must not re-enter a later snapshot");
        // Reuse both the NVN object and CPU virtual address for different
        // physical pages. A resurrected cache entry would read the old data.
        assert_eq!(libc::mmap(cpu as *mut _,page,libc::PROT_READ|libc::PROT_WRITE,libc::MAP_PRIVATE|libc::MAP_ANON|libc::MAP_FIXED,-1,0) as u64,cpu);
        for i in 0..page/4 {(cpu as *mut u32).add(i).write(0x1bad0000+i as u32);}
        assert_eq!(initialize(&Args([obj,builder.as_ptr() as u64,0,0,0,0,0,0],[0;8])),1);
        let replacement=mapped_pool_snapshot(||{}).unwrap();
        let (chunk,offset)=gfx::pool_buffer_range(&replacement,cpu,page as u64).unwrap();
        let cb=gfx::gpu().queue.0.commandBuffer().unwrap();let blit=cb.blitCommandEncoder().unwrap();
        blit.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(&chunk.buffer.0,offset,&output,0,page);
        blit.endEncoding();cb.commit();cb.waitUntilCompleted();assert_eq!(cb.status(),MTLCommandBufferStatus::Completed);
        for (i,&word) in std::slice::from_raw_parts(output.contents().as_ptr().cast::<u32>(),page/4).iter().enumerate() {assert_eq!(word,0x1bad0000+i as u32,"reinitialized pool reused stale GPU storage");}
        objects::host_impl("nvnMemoryPoolFinalize").unwrap()(&Args([obj,0,0,0,0,0,0,0],[0;8]));
        drop(replacement);drop(pools);assert_eq!(libc::munmap(cpu as *mut _,page),0);
        *STATE.lock().unwrap()=previous;
        eprintln!("[verify] concurrent finalize blocked through NoCopy wrapping; GPU readbacks exact before and after same-object/same-VA physical remap; unmapped pool excluded");
    });
}
