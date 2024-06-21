pub fn round_to_decimal_places(value: f64, decimal_places: u32) -> f64 {
    let multiplier = 10_f64.powi(decimal_places as i32);
    (value * multiplier).round() / multiplier
}

pub fn round2(value: f64) -> f64 {
    round_to_decimal_places(value, 2)
}

/// gives the last bit of the path as the label e.g. "path/to/file.png" -> "file"
pub fn get_label_from_path(path: String) -> Option<String> {
    let path = std::path::Path::new(&path);
    let file_name = path.file_stem()?.to_str()?.to_string();
    Some(file_name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_label() {
        let path = "path/to/file.png".to_string();
        let label = get_label_from_path(path).unwrap();
        assert_eq!(label, "file");
    }
}
