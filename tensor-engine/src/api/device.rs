/// Device type for tensor computations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Device {
    #[default]
    Cpu,
}

impl Device {
    /// Returns the string name of the device.
    pub fn name(&self) -> &'static str {
        match self {
            Device::Cpu => "cpu",
        }
    }
}

impl std::fmt::Display for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// @covers: name
    #[test]
    fn test_device_name_returns_cpu() {
        assert_eq!(Device::Cpu.name(), "cpu");
    }

    /// @covers: name
    #[test]
    fn test_device_default_is_cpu() {
        assert_eq!(Device::default(), Device::Cpu);
        assert_eq!(Device::default().name(), "cpu");
    }

    /// @covers: name
    #[test]
    fn test_device_display_cpu_shows_lowercase() {
        assert_eq!(format!("{}", Device::Cpu), "cpu");
    }

    /// @covers: name
    #[test]
    fn test_device_debug_cpu_shows_variant_name() {
        assert_eq!(format!("{:?}", Device::Cpu), "Cpu");
    }

    /// @covers: name
    #[test]
    fn test_device_clone_preserves_equality() {
        let d = Device::Cpu;
        assert_eq!(d, d.clone());
    }
}
