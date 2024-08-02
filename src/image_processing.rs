use image::imageops::resize;
use image::{GrayImage, ImageBuffer, Luma, Rgba, RgbaImage};
use imageproc::drawing::{draw_filled_circle_mut, draw_hollow_rect_mut, draw_text_mut};
use log::debug;
use nalgebra::clamp;
use rand::prelude::*;
use rusttype::{Font, Scale};
use serde::Deserialize;
use std::ops::{Add, Sub};
use std::{fmt::Debug, ops::AddAssign};

use crate::utils::get_label_from_path;
use crate::{Error, Result, CONFIG};
use skeletonize::edge_detection::sobel4;
use skeletonize::foreground;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ImageLabel(pub String);

impl std::fmt::Display for ImageLabel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Deserialize, Clone, Debug)]
pub struct ImageDescription {
    pub components: Components,
    pub nodes: Nodes,
}

#[derive(Deserialize, Clone, Debug)]
pub struct Components {
    pub resistor: Option<u32>,
    pub capacitor: Option<u32>,
    pub source_dc: Option<u32>,
}

#[derive(Deserialize, Clone, Debug)]
pub struct Nodes {
    pub resistor: Option<Vec<Vec<u32>>>,
    pub capacitor: Option<Vec<Vec<u32>>>,
    pub source_dc: Option<Vec<Vec<u32>>>,
}

pub struct ImageReader {
    images: Vec<(ImageLabel, Image, ImageDescription)>,
}

impl ImageReader {
    /// reads a directory and returns a list of all images in it (preprocessed)
    /// also loads the image descriptions with it
    pub fn from_path(path: String, description_path: String) -> std::result::Result<Self, Error> {
        let mut images = vec![];

        for entry in std::fs::read_dir(path)? {
            let path = entry?.path();
            if let Some(to_str) = path.to_str() {
                let path_str = to_str.to_string();
                let image = Image::from_path(path_str.clone())?;
                if let Some(label) = get_label_from_path(path_str) {
                    // load description
                    let desc_entry = std::fs::read_dir(description_path.clone())?
                        .last()
                        .ok_or("ValueError: could not get description path")??;
                    let desc_str = std::fs::read_to_string(desc_entry.path())?;
                    let description: ImageDescription = toml::from_str(&desc_str)?;
                    debug!(
                        "loaded image: label: {:?}, description: {:?}",
                        label.clone(),
                        description.clone()
                    );
                    images.push((ImageLabel(label), image.clone(), description));
                } else {
                    return Err("ValueError: could not get label from path".into());
                }
            } else {
                return Err("ValueError: could not convert path to string".into());
            }
        }
        Ok(ImageReader { images })
    }

    pub fn get_image(
        &self,
        index: usize,
    ) -> std::result::Result<&(ImageLabel, Image, ImageDescription), Error> {
        if index >= self.images.len() {
            return Err("IndexError: index out of bounds".into());
        }
        Ok(&self.images[index])
    }

    pub fn images(&self) -> &Vec<(ImageLabel, Image, ImageDescription)> {
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

    /// length of the vector normalized between 0 and 1
    pub fn normalized_len(&self) -> f32 {
        let move_speed = CONFIG.neural_network.retina_movement_speed as i32;
        let max_len = ((move_speed.pow(2) + move_speed.pow(2)) as f32).sqrt();
        self.len() / max_len
    }

    pub fn x(&self) -> i32 {
        self.x
    }

    pub fn y(&self) -> i32 {
        self.y
    }

    pub fn invert(&self) -> Position {
        Position {
            x: -self.x,
            y: -self.y,
        }
    }

    pub fn random(rng: &mut dyn RngCore, top_left: Position, bottom_right: Position) -> Position {
        let x = rng.gen_range(top_left.x..bottom_right.x);
        let y = rng.gen_range(top_left.y..bottom_right.y);
        Position { x, y }
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
    grey: GrayImage,
    width: u32,
    height: u32,
    /// used to visualize the retina movement on an upscaled image (Position, size of retina, label)
    retina_positions: Vec<(Position, usize, String)>,
}

impl Image {
    /// Empty image with the default size
    pub fn empty() -> Self {
        Image {
            rgba: RgbaImage::new(
                CONFIG.image_processing.input_image_width as u32,
                CONFIG.image_processing.input_image_height as u32,
            ),
            grey: GrayImage::new(
                CONFIG.image_processing.input_image_width as u32,
                CONFIG.image_processing.input_image_height as u32,
            ),
            width: CONFIG.image_processing.input_image_width as u32,
            height: CONFIG.image_processing.input_image_height as u32,
            retina_positions: vec![],
        }
    }

    /// create an image from a vector of f32 values. Must be a square length
    pub fn from_vec(vec: Vec<f32>) -> std::result::Result<Self, Error> {
        // figure out if the vector length can be squared
        if (vec.len() as f32).sqrt() % 1.0 != 0.0 {
            return Err("ValueError: vector length must be a square number".into());
        }

        let width = f32::sqrt(vec.len() as f32) as u32;
        let height = f32::sqrt(vec.len() as f32) as u32;
        let mut rgba = RgbaImage::new(width, height);
        let mut grey = GrayImage::new(width, height);
        for i in 0..height {
            for j in 0..width {
                let value = (vec[(i * width + j) as usize]) as u8;
                rgba.put_pixel(j, i, Rgba([value, value, value, 255]));
                grey.put_pixel(j, i, Luma([value]));
            }
        }
        Ok(Image {
            rgba,
            grey,
            width,
            height,
            retina_positions: vec![],
        })
    }

    /// load from a path and return an image(preprocessed)
    pub fn from_path(path: String) -> std::result::Result<Self, Error> {
        let rgba = image::io::Reader::open(path.clone())?
            .decode()?
            .into_rgba8();
        let grey = image::io::Reader::open(path)?.decode()?.into_luma8();
        let width = rgba.width();
        let height = rgba.height();
        let mut image = Image {
            rgba,
            grey,
            width,
            height,
            retina_positions: vec![],
        };

        // preprocess the image
        image
            .resize_all(
                CONFIG.image_processing.goal_image_width as u32,
                CONFIG.image_processing.goal_image_height as u32,
            )?
            .edged(Some(CONFIG.image_processing.sobel_threshold as f32))?
            .erode(
                imageproc::distance_transform::Norm::L1,
                CONFIG.image_processing.erode_pixels as u8,
            )?;

        Ok(image)
    }

    pub fn rgba(&self) -> &RgbaImage {
        &self.rgba
    }

    pub fn grey(&self) -> &GrayImage {
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
            image::imageops::FilterType::Lanczos3,
        );
        self.rgba = new_rgba;
        self.grey = new_grey;
        self.width = width;
        self.height = height;

        Ok(self)
    }

    pub fn edged(&mut self, threshold: Option<f32>) -> std::result::Result<&mut Self, Error> {
        let img = image::DynamicImage::ImageLuma8(self.grey.clone());
        self.grey = sobel4::<foreground::Black>(&img, threshold)?.into_luma8();
        // thin_image_edges::<foreground::White>(&mut filtered, method, None)?;
        Ok(self)
    }

    pub fn erode(
        &mut self,
        norm: imageproc::distance_transform::Norm,
        k: u8,
    ) -> std::result::Result<&mut Self, Error> {
        imageproc::morphology::erode_mut(&mut self.grey, norm, k);
        Ok(self)
    }

    /// create a subview into the image with the given position with size
    /// if the ends of the retina would be outside the image, an index error is returned
    pub fn create_retina_at(
        &self,
        position: Position,
        size: usize,
        label: String,
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
            label,
            center_position: position,
            current_delta_position: Position::new(0, 0),
            last_delta_position: Position::new(0, 0),
        })
    }

    pub fn update_retina_movement(&mut self, retina: &Retina) {
        self.retina_positions.push((
            retina.get_center_position(),
            retina.size(),
            retina.label.clone(),
        ));
    }

    pub fn save_rgba(&mut self, path: String) -> std::result::Result<&mut Self, Error> {
        self.rgba.save(path)?;
        Ok(self)
    }

    pub fn save_grey(&mut self, path: String) -> std::result::Result<&mut Self, Error> {
        self.grey.save(path)?;
        Ok(self)
    }

    pub fn save_with_retina(&self, path: String) -> Result {
        let mut canvas = self.grey().clone();
        for (index, (retina_position, retina_size, label)) in
            self.retina_positions.iter().enumerate()
        {
            let scaled_size = *retina_size as f32;
            let x = retina_position.x as f32 - 0.5;
            let y = retina_position.y as f32 - 0.5;

            // draw border of the retina
            draw_hollow_rect_mut(
                &mut canvas,
                imageproc::rect::Rect::at(
                    x as i32 - scaled_size as i32 / 2,
                    y as i32 - scaled_size as i32 / 2,
                )
                .of_size(scaled_size as u32, scaled_size as u32),
                Luma([0]),
            );

            if index == 0 {
                continue;
            }
            // // draw a line from the last retina position to the current retina position
            // let (line_begin, _) = &self.retina_positions[index - 1];
            // let line_end = retina_position;
            // draw_line_segment_mut(
            //     &mut canvas,
            //     ((line_begin.x as f32 - 0.5), (line_begin.y as f32 - 0.5)),
            //     ((line_end.x as f32 - 0.5), (line_end.y as f32 - 0.5)),
            //     Luma([127]),
            // );

            // draw in the middle a circle
            draw_filled_circle_mut(&mut canvas, (x as i32, y as i32), 1_i32, Luma([0]));

            // add a label to the retina
            let font_data = include_bytes!("../assets/Roboto-Regular.ttf");
            let Some(font) = Font::try_from_bytes(font_data) else {
                return Err("Could not load font".into());
            };
            let scale = Scale {
                x: CONFIG.image_processing.retina_label_scale as f32,
                y: CONFIG.image_processing.retina_label_scale as f32,
            };
            let color = Luma([0]);
            draw_text_mut(&mut canvas, color, x as i32, y as i32, scale, &font, label);
        }
        canvas.save(path)?;

        Ok(())
    }

    /// on the rgba version of the image upscaled
    pub fn save_with_retina_upscaled(&self, path: String) -> Result {
        let circle_radius = CONFIG.image_processing.retina_circle_radius as f32;
        let upscaled_width = CONFIG.image_processing.goal_image_width as u32;
        let upscaled_height = CONFIG.image_processing.goal_image_height as u32;

        let scaling_factor_x = upscaled_width as f32 / self.width() as f32;
        let scaling_factor_y = upscaled_height as f32 / self.height() as f32;

        let mut canvas = resize(
            &self.rgba().clone(),
            upscaled_width,
            upscaled_height,
            image::imageops::FilterType::Nearest,
        );
        for (index, (retina_position, retina_size, label)) in
            self.retina_positions.iter().enumerate()
        {
            let scaled_size = *retina_size as f32 * scaling_factor_x;
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
            // // draw a line from the last retina position to the current retina position
            // let (line_begin, _) = &self.retina_positions[index - 1];
            // let line_end = retina_position;
            // draw_line_segment_mut(
            //     &mut canvas,
            //     (
            //         (line_begin.x as f32 - 0.5) * scaling_factor_x,
            //         (line_begin.y as f32 - 0.5) * scaling_factor_y,
            //     ),
            //     (
            //         (line_end.x as f32 - 0.5) * scaling_factor_x,
            //         (line_end.y as f32 - 0.5) * scaling_factor_y,
            //     ),
            //     Rgba([127, 127, 127, 255]),
            // );

            // draw at the middle of the retina a circle
            draw_filled_circle_mut(
                &mut canvas,
                ((scaled_x) as i32, (scaled_y) as i32),
                circle_radius as i32,
                Rgba([0, 255, 0, 255]),
            );

            // add a label to the retina
            let font_data: &[u8] = include_bytes!("../assets/Roboto-Regular.ttf");
            let Some(font) = Font::try_from_bytes(font_data) else {
                return Err("Could not load font".into());
            };
            let scale = Scale {
                x: CONFIG.image_processing.retina_label_scale as f32,
                y: CONFIG.image_processing.retina_label_scale as f32,
            };
            let color = Rgba([0, 0, 0, 255]);
            draw_text_mut(
                &mut canvas,
                color,
                scaled_x as i32,
                scaled_y as i32,
                scale,
                &font,
                label,
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
    label: String,
    last_delta_position: Position,
    current_delta_position: Position,
    /// this is only for visualization purpose, the Rnn does not know this information
    center_position: Position,
}

impl Retina {
    /// counting from 0
    pub fn get_value(&self, x: usize, y: usize) -> f32 {
        self.get_data()[y * self.size() + x]
    }

    pub fn get_center_value(&self) -> f32 {
        self.get_value(self.size() / 2, self.size() / 2)
    }

    pub fn get_data(&self) -> &Vec<f32> {
        &self.data
    }

    pub fn set_data(&mut self, data: Vec<f32>) {
        self.data = data;
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn set_size(&mut self, size: usize, image: &Image) -> Result {
        if CONFIG.image_processing.min_retina_size as usize > size {
            return Err("ValueError: size is smaller than the minimum retina size".into());
        }

        if size % 2 == 0 {
            return Err("ValueError: size must be an odd number".into());
        }

        if self.size == size {
            return Err("ValueError: cannot set size, it's the same".into());
        }

        // make sure that the new size does not exceed the image boundaries
        let offset = size as i32 / 2 + 1;
        let x = self.get_center_position().x;
        let y = self.get_center_position().y;
        if x - offset < 0
            || y - offset < 0
            || x + offset >= image.width() as i32
            || y + offset >= image.height() as i32
        {
            return Err("IndexError: new size exceeds image boundaries".into());
        }

        // update the data vector with the new values
        let mut new_data = vec![];
        for i in 0..size as i32 {
            for j in 0..size as i32 {
                let x = x - offset + j;
                let y = y - offset + i;
                new_data.push(image.get_pixel(x as u32, y as u32));
            }
        }
        self.size = size;
        self.set_data(new_data);
        Ok(())
    }

    /// same as set_size but ignores all bounds. Can panic
    pub fn set_size_override(&mut self, size: usize, image: &Image) {
        // make sure that the new size does not exceed the image boundaries
        let offset = size as i32 / 2 + 1;
        let x = self.get_center_position().x;
        let y = self.get_center_position().y;

        // update the data vector with the new values
        let mut new_data = vec![];
        for i in 0..size as i32 {
            for j in 0..size as i32 {
                let x = x - offset + j;
                let y = y - offset + i;
                new_data.push(image.get_pixel(x as u32, y as u32));
            }
        }
        self.size = size;
        self.set_data(new_data);
    }

    pub fn get_center_position(&self) -> Position {
        self.center_position.clone()
    }

    pub fn get_last_delta_position(&self) -> Position {
        self.last_delta_position.clone()
    }

    pub fn get_current_delta_position(&self) -> Position {
        self.current_delta_position.clone()
    }

    pub fn create_png_at(&self, path: String) -> Result {
        let mut imgbuf = ImageBuffer::new(self.size() as u32, self.size() as u32);
        for (x, y, pixel) in imgbuf.enumerate_pixels_mut() {
            *pixel = Luma([(self.get_value(x as usize, y as usize) * 255.0) as u8]);
        }
        imgbuf.save(path)?;
        Ok(())
    }

    /// moves the retina by the given delta position and clamps it to the borders of the image
    /// when the retina would move outside the image
    pub fn move_mut(&mut self, delta: &Position, image: &Image) {
        let offset = self.size() as i32 / 2 + 1;
        // calculate the difference of the amount of pixels that the retina might move outside the image
        let mut highest_diff_x = 0i32;
        let mut highest_diff_y = 0i32;
        for i in 0..self.size() as i32 {
            for j in 0..self.size() as i32 {
                let x = self.get_center_position().x - offset + i + delta.x;
                let diff_x = clamp(x, 0, image.width() as i32 - 1) - x;
                if diff_x.abs() > highest_diff_x.abs() {
                    highest_diff_x = diff_x;
                }
                let y = self.get_center_position().y - offset + j + delta.y;
                let diff_y = clamp(y, 0, image.height() as i32 - 1) - y;
                if diff_y.abs() > highest_diff_y.abs() {
                    highest_diff_y = diff_y;
                }
            }
        }
        // move the retina to the new position and add the highest difference to the delta position
        self.last_delta_position = self.get_current_delta_position();
        self.current_delta_position = delta.clone() + Position::new(highest_diff_x, highest_diff_y);
        self.center_position += self.get_current_delta_position();

        // update the data vector with the new values
        let mut new_data = vec![];
        let offset = self.size() as i32 / 2 + 1;
        for i in 0..self.size() as i32 {
            for j in 0..self.size() as i32 {
                let x = self.get_center_position().x - offset + j;
                let y = self.get_center_position().y - offset + i;
                new_data.push(image.get_pixel(x as u32, y as u32));
            }
        }
        self.set_data(new_data);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_retina_out_of_bounds() {
        let image = Image::from_vec(vec![0.0; 33 * 33]).unwrap();
        // getting the first pixel in the top left corner should give an error
        let retina = image.create_retina_at(Position::new(1, 1), 5 as usize, "test".to_string());
        assert!(retina.is_err());
    }

    #[test]
    fn test_invalid_retina_movement_to_the_right() {
        let image = Image::from_vec(vec![0.0; 33 * 33]).unwrap();
        let mut retina = image
            .create_retina_at(Position::new(5, 13), 5 as usize, "test".to_string())
            .unwrap();

        retina.move_mut(&Position::new(80, 0), &image);
        assert_eq!(retina.get_center_position().x, 31);
    }

    #[test]
    fn test_invalid_retina_movement_to_the_left() {
        let image = Image::from_vec(vec![0.0; 33 * 33]).unwrap();
        let mut retina = image
            .create_retina_at(Position::new(5, 13), 5 as usize, "test".to_string())
            .unwrap();

        retina.move_mut(&Position::new(-80, 0), &image);
        assert_eq!(retina.get_center_position().x, 3);
    }

    #[test]
    fn test_invalid_retina_movement_to_the_top() {
        let image = Image::from_vec(vec![0.0; 33 * 33]).unwrap();
        let mut retina = image
            .create_retina_at(Position::new(5, 13), 5 as usize, "test".to_string())
            .unwrap();

        retina.move_mut(&Position::new(0, -80), &image);
        assert_eq!(retina.get_center_position().y, 3);
    }

    #[test]
    fn test_invalid_retina_movement_to_the_bottom() {
        let image = Image::from_vec(vec![0.0; 33 * 33]).unwrap();
        let mut retina = image
            .create_retina_at(Position::new(5, 13), 5 as usize, "test".to_string())
            .unwrap();

        retina.move_mut(&Position::new(0, 80), &image);
        assert_eq!(retina.get_center_position().y, 31);
    }

    #[test]
    fn test_retina_movement() {
        let mut image = Image::from_vec(vec![0.0; 33 * 33]).unwrap();
        let mut retina = image
            .create_retina_at(Position::new(5, 5), 5, "test".to_string())
            .unwrap();
        image.update_retina_movement(&retina);
        retina.move_mut(&Position::new(1, 1), &image);
        image.update_retina_movement(&retina);
        assert_eq!(retina.get_center_position().x, 6);
        assert_eq!(retina.get_center_position().y, 6);

        retina.move_mut(&Position::new(10, 6), &image);
        image.update_retina_movement(&retina);
        let _ = retina.set_size(9, &image);

        retina.move_mut(&Position::new(-1, -1), &image);
        image.update_retina_movement(&retina);
        assert_eq!(retina.get_center_position().x, 15);
        assert_eq!(retina.get_center_position().y, 11);
    }

    #[test]
    #[should_panic]
    fn test_invalid_retina_size() {
        let image = Image::from_vec(vec![0.0; 9 * 9]).unwrap();
        let mut retina = image
            .create_retina_at(Position::new(5, 5), 5, "test".to_string())
            .unwrap();
        retina.set_size(10, &image).unwrap();
    }
}
