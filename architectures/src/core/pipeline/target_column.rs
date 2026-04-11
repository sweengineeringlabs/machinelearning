/// Specifies which column(s) form the prediction target.
#[derive(Debug, Clone)]
pub enum TargetColumn {
    /// A single named column (e.g., "open").
    Single(String),
    /// Multiple named columns.
    Multi(Vec<String>),
    /// Shorthand for the "close" column (default).
    Close,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_target_column_close_variant() {
        let tc = TargetColumn::Close;
        assert!(matches!(tc, TargetColumn::Close));
    }

    #[test]
    fn test_target_column_single_variant() {
        let tc = TargetColumn::Single("open".to_string());
        match tc {
            TargetColumn::Single(name) => assert_eq!(name, "open"),
            _ => panic!("Expected Single variant"),
        }
    }

    #[test]
    fn test_target_column_multi_variant() {
        let tc = TargetColumn::Multi(vec!["open".to_string(), "close".to_string()]);
        match tc {
            TargetColumn::Multi(cols) => assert_eq!(cols.len(), 2),
            _ => panic!("Expected Multi variant"),
        }
    }
}
