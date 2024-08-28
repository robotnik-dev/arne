use std::path::PathBuf;

use crate::{image::Image, Error, CONFIG};

pub fn round_to_decimal_places(value: f32, decimal_places: u32) -> f32 {
    let multiplier = 10_f32.powi(decimal_places as i32);
    (value * multiplier).round() / multiplier
}

pub fn round2(value: f32) -> f32 {
    round_to_decimal_places(value, 2)
}

pub fn round3(value: f32) -> f32 {
    round_to_decimal_places(value, 3)
}

/// gives the last bit of the path as the label e.g. "path/to/file.png" -> "file"
#[allow(dead_code)]
pub fn get_label_from_path(path: PathBuf) -> Option<String> {
    let path = std::path::Path::new(&path);
    let file_name = path.file_stem()?.to_str()?.to_string();
    Some(file_name)
}

#[allow(dead_code)]
pub fn binarize_image(image: Image) -> std::result::Result<Image, Error> {
    let mut image = image.clone();
    image
        .resize_all(
            CONFIG.image_processing.goal_image_width as u32,
            CONFIG.image_processing.goal_image_height as u32,
        )
        .map(|i| i.clone())?
        .edged(Some(CONFIG.image_processing.sobel_threshold as f32))
        .map(|i| i.clone())?
        .erode(
            imageproc::distance_transform::Norm::L1,
            CONFIG.image_processing.erode_pixels as u8,
        )
        .map(|i| i.clone())?;
    Ok(image)
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
