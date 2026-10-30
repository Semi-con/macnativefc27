// Mush8 - lib.rs
// Constantin Clerc (c22dev) - 0.1
pub mod corpus;
pub mod inst;
pub mod msl;
pub mod program;
pub mod sph;
pub mod table;

pub use msl::{FRAGMENT_DUAL_SOURCE,FRAGMENT_ALPHA_TEST,FRAGMENT_EARLY_TESTS,can_force_early_fragment_tests};
pub use msl::{Interface, ImageBinding, TexKind, Translated, prelude, translate, translate_compute,translate_compute_with_memory,translate_compute_with_constants, translate_with,translate_with_constants};
pub use msl::translate_graphics_forward;
pub use msl::translate_graphics_regions;
pub use msl::translate_graphics_structured;

pub use msl::{GraphicsDispatch,TranslationConfig,translate_configured};

pub use msl::{GridWriteFootprint,DescriptorWriteFootprint,WriteWindowFootprint};
