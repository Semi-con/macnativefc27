# Third-party notices

This repository is a modified source distribution. Original file notices are retained. A license on compatibility software does not license any game data processed by it.

| Component | Source | License / treatment |
|---|---|---|
| Rust compatibility layer, file tooling, offline rewriter and shader translator | [c22dev/mush8](https://github.com/c22dev/mush8), base commit `c27579db76efc0288891a02589bf02f6e1c775ee` | GPL-3.0; root LICENSE and existing Constantin Clerc notices retained; FC27 modifications dated 2026-10-30 |
| `crates/nxhost/src/motion.rs` | yuzu MotionInput, ported by Mush8 | File declares GPL-2.0-or-later; retained attribution, used under GPL-3.0 in this combined work |
| `launcher/game_slot.py` | [MacNativeGames](https://github.com/ly05010419/MacNativeGames), commit `5281186af56bdcce6ae9b3bb9daa9791bc02994e` | MIT; full notice in `licenses/MacNativeGames-MIT.txt` |
| `tools/cas` dependency `oozextract` | [oozextract](https://github.com/lvlvllvlvllvlvl/oozextract) 0.5.7 | MIT; fetched as a Cargo dependency, not vendored here |
| libopus | [Xiph.Org](https://opus-codec.org/) | BSD-3-Clause; linked from the user's locally installed archive, no binary is distributed here |
| Cargo dependencies, including SDL, objc2, RustCrypto, image and parser libraries | Resolved by Cargo.lock from their respective upstreams | Their respective licenses apply; sources and binaries are not vendored in this repository |

The optional `unicorn-engine` / `unicorn-engine-sys` combination is not included in the public export while its exact GPL-version compatibility is unresolved. The related offline verification branch explicitly fails; no substitute success result is returned.

New FC27 compatibility modifications are provided under the root GPL-3.0 license. Source publication must retain notices and modification information. A later binary distribution needs its own exact corresponding-source and third-party-notice review; this source snapshot is not approval for publishing a locally built game App.

Pure format analysis in the private development tree also consulted [FrostyToolsuite](https://github.com/FrostyToolsuite/FrostyToolsuite/tree/ee2a587af394fc784b309bb7990dab051065330f), GPL-3.0. Those private Python analysis scripts and extracted assets are not part of this export.
