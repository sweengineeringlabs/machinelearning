//! E2E tests for OptProfile definition.

use tensor_engine::OptProfile;

#[test]
fn test_opt_profile_variants_differ() {
    assert_ne!(OptProfile::Optimized, OptProfile::Baseline);
    assert_ne!(OptProfile::Baseline, OptProfile::Aggressive);
}
