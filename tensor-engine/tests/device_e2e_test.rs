//! E2E tests for the Device type.

use tensor_engine::Device;

/// @covers: Device (Default)
#[test]
fn test_device_default_is_cpu() {
    let d = Device::default();
    assert_eq!(d, Device::Cpu);
}

/// @covers: Device (Display)
#[test]
fn test_device_display_shows_lowercase() {
    assert_eq!(format!("{}", Device::Cpu), "cpu");
}

/// @covers: Device (Clone)
#[test]
fn test_device_clone_preserves_equality() {
    let d = Device::Cpu;
    assert_eq!(d, d.clone());
}
