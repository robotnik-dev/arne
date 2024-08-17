use std::path::PathBuf;

pub fn round_to_decimal_places(value: f32, decimal_places: u32) -> f32 {
    let multiplier = 10_f32.powi(decimal_places as i32);
    (value * multiplier).round() / multiplier
}

pub fn round2(value: f32) -> f32 {
    round_to_decimal_places(value, 2)
}

/// gives the last bit of the path as the label e.g. "path/to/file.png" -> "file"
pub fn get_label_from_path(path: PathBuf) -> Option<String> {
    let path = std::path::Path::new(&path);
    let file_name = path.file_stem()?.to_str()?.to_string();
    Some(file_name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_label() {
        let path = PathBuf::from("path/to/file.png");
        let label = get_label_from_path(path).unwrap();
        assert_eq!(label, "file");
    }
}
