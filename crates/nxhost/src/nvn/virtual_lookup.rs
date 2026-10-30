// map_virtual keeps these intervals sorted and disjoint after every transaction.
// Lookups still reject holes and retain the original physical-pool resolution.
use std::sync::{LazyLock, atomic::{AtomicBool, Ordering::Relaxed}};

type Mapping = (u64, u64, u64, u64);
static INDEXED: LazyLock<bool> = LazyLock::new(|| std::env::var_os("NX_INDEXED_VIRTUAL_LOOKUP").is_some());
static TEST_LINEAR: AtomicBool = AtomicBool::new(false);

fn indexed(maps: &[Mapping], offset: u64) -> Option<&Mapping> {
    let at = maps.partition_point(|m| m.0 <= offset).checked_sub(1)?;
    let map = &maps[at];
    (offset - map.0 < map.1).then_some(map)
}

fn linear(maps: &[Mapping], offset: u64) -> Option<&Mapping> {
    maps.iter().find(|m| offset >= m.0 && offset - m.0 < m.1)
}

pub(super) fn find(maps: &[Mapping], offset: u64) -> Option<&Mapping> {
    if *INDEXED && !TEST_LINEAR.load(Relaxed) { indexed(maps, offset) } else { linear(maps, offset) }
}

// Same-process timing control for explicitly enabled local test sessions.
// Only the equivalent CPU lookup algorithm changes; GPU work is untouched.
pub(super) fn poll_test_mode(submit: u32) {
    static LIVE: LazyLock<bool> = LazyLock::new(|| std::env::var_os("NX_TEST_CONTROL").is_some());
    if !*INDEXED || !*LIVE || submit % 32 != 0 { return; }
    let Ok(mode) = std::fs::read_to_string("work/virtual-lookup-mode") else { return; };
    let _ = std::fs::remove_file("work/virtual-lookup-mode");
    let linear = match mode.trim() {
        "linear" => true,
        "indexed" => false,
        _ => { eprintln!("[virtual-lookup] rejected unsupported test mode"); return; }
    };
    TEST_LINEAR.store(linear, Relaxed);
    eprintln!("[virtual-lookup] submit={submit} mode={}", if linear { "linear" } else { "indexed" });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn indexed_lookup_matches_original_at_every_boundary_and_hole() {
        let maps = (0..2048).map(|i| (4096 + i * 12288, 4096 + (i % 2) * 4096, i + 10, i * 4096)).collect::<Vec<_>>();
        for &(start, len, _, _) in &maps {
            for offset in [start - 1, start, start + 1, start + len - 1, start + len, start + len + 1] {
                assert_eq!(indexed(&maps, offset), linear(&maps, offset), "offset={offset:#x}");
            }
        }
        for offset in [0, 4095, u64::MAX] { assert_eq!(indexed(&maps, offset), linear(&maps, offset)); }
        assert_eq!(indexed(&[], 0), None);
        let end = [(u64::MAX - 4095, 4096, 7, 0)];
        assert_eq!(indexed(&end, u64::MAX), Some(&end[0]));
    }

}
