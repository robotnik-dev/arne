use image::buffer::ConvertBuffer;
use image::imageops::resize;
use image::{DynamicImage, GenericImage, GenericImageView, GrayAlphaImage, ImageBuffer, LumaA, Rgba, RgbaImage};
use imageproc::drawing::{draw_filled_circle_mut, draw_hollow_rect_mut, draw_line_segment_mut};
use nalgebra::clamp;
use rayon::vec;
use std::ops::{Add, Sub};
use std::{fmt::Debug, ops::AddAssign};

use crate::{Error, Result, CONFIG};

use crate::utils::get_label_from_path;

pub enum ImageType {
    Rgba,
    Grey,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ImageLabel(pub String);

impl std::fmt::Display for ImageLabel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

pub struct ImageReader {
    images: Vec<(ImageLabel, Image)>,
}

impl ImageReader {
    /// reads a directory and returns a list of all images in it
    pub fn from_path(path: String) -> std::result::Result<Self, Error> {
        let mut images = vec![];
        for entry in std::fs::read_dir(path)? {
            let path = entry?.path();
            if let Some(to_str) = path.to_str() {
                let path_str = to_str.to_string();
                let image = Image::from_path(path_str.clone())?;
                if let Some(label) = get_label_from_path(path_str) {
                    images.push((ImageLabel(label), image));
                } else {
                    return Err("ValueError: could not get label from path".into());
                }
            } else {
                return Err("ValueError: could not convert path to string".into());
            }
        }
        Ok(ImageReader { images })
    }

    pub fn get_image(&self, index: usize) -> std::result::Result<&(ImageLabel, Image), Error> {
        if index >= self.images.len() {
            return Err("IndexError: index out of bounds".into());
        }
        Ok(&self.images[index])
    }

    pub fn images(&self) -> &Vec<(ImageLabel, Image)> {
        &self.images
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Position {
    x: i32,
    y: i32,
}

impl Position {
    pub fn new(x: i32, y: i32) -> Self {
        Position { x, y }
    }

    pub fn len(&self) -> f32 {
        ((self.x.pow(2) + self.y.pow(2)) as f32).sqrt()
    }

    /// returns the normalized vector
    pub fn normalized(&self) -> Position {
        Position {
            x: (self.x as f32 / self.len()) as i32,
            y: (self.y as f32 / self.len()) as i32,
        }
    }

    pub fn x(&self) -> i32 {
        self.x
    }

    pub fn y(&self) -> i32 {
        self.y
    }
}

impl Add for Position {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }
}

impl Sub for Position {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self {
            x: self.x - other.x,
            y: self.y - other.y,
        }
    }
}

impl AddAssign for Position {
    fn add_assign(&mut self, other: Self) {
        self.x += other.x;
        self.y += other.y;
    }
}

#[derive(Debug, Clone)]
pub struct Image {
    /// generic container for the image data
    rgba: RgbaImage,
    grey: GrayAlphaImage,
    binarized_grey: GrayAlphaImage,
    width: u32,
    height: u32,
    binarized_white: u8,
    binarized_black: u8,
    /// used to visualize the retina movement on an upscaled image
    retina_positions: Vec<Position>,
}

impl Image {
    /// Empty image with the default size
    pub fn empty() -> Self {
        Image {
            rgba: RgbaImage::new(
                CONFIG.image_processing.real_image_width as u32,
                CONFIG.image_processing.real_image_height as u32,
            ),
            grey: GrayAlphaImage::new(
                CONFIG.image_processing.real_image_width as u32,
                CONFIG.image_processing.real_image_height as u32,
            ),
            binarized_grey: GrayAlphaImage::new(
                CONFIG.image_processing.real_image_width as u32,
                CONFIG.image_processing.real_image_height as u32,
            ),
            width: CONFIG.image_processing.real_image_width as u32,
            height: CONFIG.image_processing.real_image_height as u32,
            binarized_white: 255,
            binarized_black: 0,
            retina_positions: vec![],
        }
    }

    /// load from a path and return an image with color data
    pub fn from_path(path: String) -> std::result::Result<Self, Error> {
        let rgba = image::io::Reader::open(path.clone())?.decode()?.into_rgba8();
        let grey = image::io::Reader::open(path)?.decode()?.into_luma_alpha8();
        let width = rgba.width();
        let height = rgba.height();
        Ok(Image {
            rgba,
            grey,
            width,
            height,
            binarized_grey: GrayAlphaImage::new(
                width,
                height,
            ),
            binarized_white: 255,
            binarized_black: 0,
            retina_positions: vec![],
        })
    }

    pub fn rgba(&self) -> &RgbaImage {
        &self.rgba
    }

    pub fn grey(&self) -> &GrayAlphaImage {
        &self.grey
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }
    
    /// get grey value normalized between 0 and 1
    pub fn get_pixel(&self, x: u32, y: u32) -> f32 {
        self.grey.get_pixel(x, y).0[0] as f32 / 255.0
    }

    /// resizes all internal datastructures to new size
    pub fn resize_all(&mut self, width: u32, height: u32) -> std::result::Result<&mut Self, Error> {
        let new_rgba = resize(
            &self.rgba,
            width,
            height,
            image::imageops::FilterType::Nearest,
        );
        let new_grey = resize(
            &self.grey,
            width,
            height,
            image::imageops::FilterType::Nearest,
        );
        let new_bin_grey = resize(
            &self.binarized_grey,
            width,
            height,
            image::imageops::FilterType::Nearest,
        );
        self.rgba = new_rgba;
        self.grey = new_grey;
        self.binarized_grey = new_bin_grey;

        Ok(self)
    }

    // binarizes the grey image
    pub fn binarize(&mut self, relative: bool) -> std::result::Result<&mut Self, Error> {
        // 1. calculated the mean color of the image
        // 2. collect dark and white pixels in seperate vector to cacluate the mean of the white and dark pixels respectively
        // 3. use the mean of the dark pixels to set every dark pixel of the real image to that mean value
        // 4. do it wth the white pixels the same way
        let mean_color =
            self.grey().pixels().map(|p| p.0[0] as f32).sum::<f32>() / self.grey().pixels().count() as f32;

        let (dark_pixels, white_pixels): (Vec<f32>, Vec<f32>) = self
            .grey()
            .pixels()
            .map(|pixel| pixel.0[0] as f32)
            .partition(|pixel| *pixel < mean_color);

        let black_binary = dark_pixels.iter().sum::<f32>() / dark_pixels.len() as f32;
        let white_binary = white_pixels.iter().sum::<f32>() / white_pixels.len() as f32;

        // save the binarized values
        self.binarized_black = black_binary as u8;
        self.binarized_white = white_binary as u8;
        let black = if relative {black_binary as u8} else {0};
        let white = if relative {white_binary as u8} else {255};
        for (x, y, pixel) in self.grey.enumerate_pixels() {
            if (pixel.0[0] as f32) < black_binary {
                self.binarized_grey.put_pixel(x, y, LumaA([black, 255]));
            } else {
                self.binarized_grey.put_pixel(x, y, LumaA([white, 255]));
            }
        }

        Ok(self)
    }

    /// blurs the grey image with the given sigma
    pub fn blur(&mut self, sigma: f32) -> std::result::Result<&mut Self, Error> {
        let blurred = image::imageops::blur(&self.grey, sigma);
        self.grey = blurred;
        Ok(self)
    }

    /// create a subview into the image with the given position with size
    /// if the ends of the retina would be outside the image, an index error is returned
    pub fn create_retina_at(
        &self,
        position: Position,
        size: usize,
    ) -> std::result::Result<Retina, Error> {
        // make sure size is odd number
        if size % 2 == 0 {
            return Err("ValueError: size must be an odd number".into());
        }
        let mut data = vec![];
        let offset = size as i32 / 2 + 1;
        for i in 0..size as i32 {
            for j in 0..size as i32 {
                // when going negative with this operation it means that we try to access a pixel that is outside of the image
                // so we give back an error
                if position.x >= self.width() as i32 + offset
                    || position.y >= self.height() as i32 + offset
                    || position.x < offset
                    || position.y < offset
                {
                    return Err("IndexError: position is out of bounds".into());
                }
                let x = position.x - offset + j;
                let y = position.y - offset + i;
                data.push(self.get_pixel(x as u32, y as u32));
            }
        }
        Ok(Retina {
            data,
            size,
            center_position: position,
            delta_position: Position::new(0, 0),
            last_delta_position: Position::new(0, 0),
            binarized_white_normalized: self.binarized_white as f32 / 255.0,
            binarized_black_normalized: self.binarized_black as f32 / 255.0,
        })
    }

    pub fn update_retina_movement(&mut self, retina: &Retina) {
        self.retina_positions.push(retina.get_center_position());
    }

    pub fn save_rgba(&self, path: String) -> Result {
        self.rgba.save(path)?;
        Ok(())
    }

    pub fn save_grey(&self, path: String) -> Result {
        self.grey.save(path)?;
        Ok(())
    }

    pub fn save_binarized_grey(&self, path: String) -> Result {
        self.binarized_grey.save(path)?;
        Ok(())
    }

    /// on the rgba version of the image
    pub fn save_with_retina(&self, path: String) -> Result {
        let size = CONFIG.image_processing.retina_size as f32;
        let circle_radius = CONFIG.image_processing.retina_circle_radius as f32;

        let scaling_factor_x = self.width() as f32 / CONFIG.image_processing.image_width as f32;
        let scaling_factor_y = self.height() as f32 / CONFIG.image_processing.image_height as f32;
        let scaled_size = size * scaling_factor_x;

        let mut canvas = self.rgba.clone();
        for (index, retina_position) in self.retina_positions.iter().enumerate() {
            let scaled_x = (retina_position.x as f32 - 0.5) * scaling_factor_x;
            let scaled_y = (retina_position.y as f32 - 0.5) * scaling_factor_y;

            // draw border of the retina
            draw_hollow_rect_mut(
                &mut canvas,
                imageproc::rect::Rect::at(
                    scaled_x as i32 - scaled_size as i32 / 2,
                    scaled_y as i32 - scaled_size as i32 / 2,
                )
                .of_size(scaled_size as u32, scaled_size as u32),
                Rgba([255, 0, 0, 255]),
            );

            if index == 0 {
                continue;
            }
            // draw a line from the last retina position to the current retina position
            let line_begin = &self.retina_positions[index - 1];
            let line_end = retina_position;
            draw_line_segment_mut(
                &mut canvas,
                (
                    (line_begin.x as f32 - 0.5) * scaling_factor_x,
                    (line_begin.y as f32 - 0.5) * scaling_factor_y,
                ),
                (
                    (line_end.x as f32 - 0.5) * scaling_factor_x,
                    (line_end.y as f32 - 0.5) * scaling_factor_y,
                ),
                Rgba([127, 127, 127, 255]),
            );

            // draw at the end of the linesegment a circle
            draw_filled_circle_mut(
                &mut canvas,
                (
                    ((line_end.x as f32 - 0.5) * scaling_factor_x) as i32,
                    ((line_end.y as f32 - 0.5) * scaling_factor_y) as i32,
                ),
                circle_radius as i32,
                Rgba([0, 255, 0, 255]),
            );
        }
        canvas.save(path)?;
 
        Ok(())
    }
    
}

pub struct Retina {
    // color data stored in a vector
    data: Vec<f32>,
    size: usize,
    delta_position: Position,
    last_delta_position: Position,
    // this is only for visualization purpose, the Rnn does not know this information
    center_position: Position,
    binarized_white_normalized: f32,
    binarized_black_normalized: f32,
}

impl Retina {
    /// counting from 0
    pub fn get_value(&self, x: usize, y: usize) -> f32 {
        self.data[y * self.size + x]
    }

    pub fn get_center_value(&self) -> f32 {
        self.get_value(self.size / 2, self.size / 2)
    }

    pub fn get_data(&self) -> &Vec<f32> {
        &self.data
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn binarized_white(&self) -> f32 {
        self.binarized_white_normalized
    }

    pub fn binarized_black(&self) -> f32 {
        self.binarized_black_normalized
    }

    pub fn get_center_position(&self) -> Position {
        self.center_position.clone()
    }

    pub fn get_delta_position(&self) -> Position {
        self.delta_position.clone()
    }

    pub fn create_png_at(&self, path: String) -> Result {
        let mut imgbuf = ImageBuffer::new(self.size as u32, self.size as u32);
        for (x, y, pixel) in imgbuf.enumerate_pixels_mut() {
            *pixel = LumaA([(self.get_value(x as usize, y as usize) * 255.0) as u8, 255]);
        }
        imgbuf.save(path)?;
        Ok(())
    }

    /// moves the retina by the given delta position and clamps it to the borders of the image
    /// when the retina would move outside the image
    pub fn move_mut(&mut self, delta: &Position, image: &Image) {
        let offset = self.size as i32 / 2 + 1;
        // calculate the difference of the amount of pixels that the retina might move outside the image
        let mut highest_diff_x = 0i32;
        let mut highest_diff_y = 0i32;
        for i in 0..self.size as i32 {
            for j in 0..self.size as i32 {
                let x = self.center_position.x - offset + i + delta.x;
                let diff_x = clamp(x, 0, CONFIG.image_processing.image_width as u32 as i32 - 1) - x;
                if diff_x.abs() > highest_diff_x.abs() {
                    highest_diff_x = diff_x;
                }
                let y = self.center_position.y - offset + j + delta.y;
                let diff_y =
                    clamp(y, 0, CONFIG.image_processing.image_height as u32 as i32 - 1) - y;
                if diff_y.abs() > highest_diff_y.abs() {
                    highest_diff_y = diff_y;
                }
            }
        }
        // move the retina to the new position and add the highest difference to the delta position
        self.delta_position = delta.clone() + Position::new(highest_diff_x, highest_diff_y);
        self.center_position += self.delta_position.clone();

        // update the data vector with the new values
        let mut new_data = vec![];
        let offset = self.size() as i32 / 2 + 1;
        for i in 0..self.size() as i32 {
            for j in 0..self.size() as i32 {
                let x = self.center_position.x - offset + j;
                let y = self.center_position.y - offset + i;
                new_data.push(image.get_pixel(x as u32, y as u32));
            }
        }
        self.data = new_data;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_retina_out_of_bounds() {
        let image = Image::from_path("images/artificial/checkboard.png".to_string()).unwrap();
        // getting the first pixel in the top left corner should give an error
        let retina = image.create_retina_at(
            Position::new(1, 1),
            CONFIG.image_processing.retina_size as usize,
        );
        assert!(retina.is_err());
    }

    #[test]
    fn test_invalid_retina_movement_to_the_right() {
        let image = Image::from_path("images/artificial/checkboard.png".to_string()).unwrap();
        let mut retina = image
            .create_retina_at(
                Position::new(20, 13),
                CONFIG.image_processing.retina_size as usize,
            )
            .unwrap();

        retina.move_mut(&Position::new(50, 0), &image);

        assert_eq!(
            retina.get_center_position().x,
            (CONFIG.image_processing.image_width as u32 as i32 - 1)
                - (CONFIG.image_processing.retina_size as usize as i32 / 2)
                + 1
        );
    }

    #[test]
    fn test_invalid_retina_movement_to_the_left() {
        let image = Image::from_path("images/artificial/checkboard.png".to_string()).unwrap();
        let mut retina = image
            .create_retina_at(
                Position::new(5, 13),
                CONFIG.image_processing.retina_size as usize,
            )
            .unwrap();

        retina.move_mut(&Position::new(-50, 0), &image);

        assert_eq!(
            retina.get_center_position().x,
            (CONFIG.image_processing.retina_size as usize as i32 / 2) + 1
        );
    }

    #[test]
    fn test_invalid_retina_movement_to_the_top() {
        let image = Image::from_path("images/artificial/checkboard.png".to_string()).unwrap();
        let mut retina = image
            .create_retina_at(
                Position::new(5, 13),
                CONFIG.image_processing.retina_size as usize,
            )
            .unwrap();

        retina.move_mut(&Position::new(0, -60), &image);
        assert_eq!(
            retina.get_center_position().y,
            (CONFIG.image_processing.retina_size as usize as i32 / 2) + 1
        );
    }

    #[test]
    fn test_invalid_retina_movement_to_the_bottom() {
        let image = Image::from_path("images/artificial/checkboard.png".to_string()).unwrap();
        let mut retina = image
            .create_retina_at(
                Position::new(5, 13),
                CONFIG.image_processing.retina_size as usize,
            )
            .unwrap();

        retina.move_mut(&Position::new(0, 80), &image);
        assert_eq!(
            retina.get_center_position().y,
            CONFIG.image_processing.image_height as u32 as i32
                - 1
                - (CONFIG.image_processing.retina_size as usize as i32 / 2)
                + 1
        );
    }

    #[test]
    fn test_retina_movement() {
        let image = Image::from_path("images/artificial/checkboard.png".to_string()).unwrap();
        let mut retina = image
            .create_retina_at(
                Position::new(5, 5),
                CONFIG.image_processing.retina_size as usize,
            )
            .unwrap();
        retina.move_mut(&Position::new(1, 1), &image);
        assert_eq!(retina.get_center_position().x, 6);
        assert_eq!(retina.get_center_position().y, 6);
        retina.move_mut(&Position::new(1, 1), &image);
        assert_eq!(retina.get_center_position().x, 7);
        assert_eq!(retina.get_center_position().y, 7);
        retina.move_mut(&Position::new(-1, -1), &image);
        assert_eq!(retina.get_center_position().x, 6);
        assert_eq!(retina.get_center_position().y, 6);
    }

    #[test]
    fn test_load_real_image() {
        let mut image = Image::from_path("images/test/03.png".to_string()).unwrap();
        image.blur(3.0).unwrap();
        image.resize_all(360, 277);
        image.binarize(false).unwrap();
        image.save_binarized_grey("test/images/02_resized.png".to_string()).unwrap();

        image.resize_all(33, 25);
        image.save_binarized_grey("test/images/02_binarized_resized.png".to_string()).unwrap();
        

        // image.save_grey("test/images/01_grey.png".to_string()).unwrap();
        // image.save_rgba("test/images/01_rgba.png".to_string()).unwrap();
        // image.resize(33, 25).unwrap();
        // image.save_grey("test/images/01_grey_resized.png".to_string()).unwrap();
        // image.save_rgba("test/images/01_rgba_resized.png".to_string()).unwrap();
        // image.save_with_retina("test/images/01_orig.png".to_string()).unwrap();
        // println!("{:?}", image.grey().pixels().min_by(|a, b| a.0[0].partial_cmp(&b.0[0]).unwrap()));
        // println!("{:?}", image.grey().pixels().max_by(|a, b| a.0[0].partial_cmp(&b.0[0]).unwrap()));

        // image.to_grey().unwrap();
        // image.save_with_retina("test/images/01_grey.png".to_string()).unwrap();
        

        // image
        //     .save("test/images/real_img.png".to_string())
        //     .unwrap();
    }
}
