use image::imageops::resize;
use image::{GrayImage, ImageBuffer, Luma, Rgba, RgbaImage};
use imageproc::drawing::{draw_filled_circle_mut, draw_hollow_rect_mut, draw_line_segment_mut};
use nalgebra::clamp;
use rand::prelude::*;
use std::ops::{Add, Div, Sub};
use std::path::PathBuf;
use std::{fmt::Debug, ops::AddAssign};

use crate::annotations::Bndbox;
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

#[derive(Debug, Clone)]
pub enum TrainingStage {
    Artificial {
        stage: u8,
    },
    #[allow(dead_code)]
    RealBinarized,
    #[allow(dead_code)]
    Real,
}

/// Counted with one more than image idx. Image index 0 -> Position index 1.
#[derive(Debug, Clone, Eq, PartialEq, PartialOrd)]
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

impl Div<i32> for Position {
    type Output = Self;

    fn div(self, divisor: i32) -> Self::Output {
        if divisor == 0 {
            Self {
                x: self.x,
                y: self.y,
            }
        } else {
            let x = if self.x % 2 == 0 {
                self.x / divisor
            } else {
                self.x / divisor + 1
            };
            let y = if self.y % 2 == 0 {
                self.y / divisor
            } else {
                self.y / divisor + 1
            };
            Self { x, y }
        }
    }
}

#[derive(Debug, Clone)]
pub enum ImageFormat {
    Landscape,
    Portrait,
}

/// generic container for the image data
#[derive(Debug, Clone)]
pub struct Image {
    rgba: RgbaImage,
    grey: GrayImage,
    width: u32,
    height: u32,
    pub format: ImageFormat,
    /// used to visualize the retina movement on an upscaled image (Position, size of retina, label)
    retina_positions: Vec<(Position, usize, String)>,
    dark_pixel_positions: Vec<Position>,
}

impl Image {
    /// Empty image with the default size
    pub fn empty() -> Self {
        Image {
            rgba: RgbaImage::new(
                CONFIG.image_processing.goal_image_width as u32,
                CONFIG.image_processing.goal_image_height as u32,
            ),
            grey: GrayImage::new(
                CONFIG.image_processing.goal_image_width as u32,
                CONFIG.image_processing.goal_image_height as u32,
            ),
            width: CONFIG.image_processing.goal_image_width as u32,
            height: CONFIG.image_processing.goal_image_height as u32,
            format: ImageFormat::Landscape,
            retina_positions: vec![],
            dark_pixel_positions: vec![],
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
        let mut image = Image {
            rgba,
            grey,
            width,
            height,
            format: ImageFormat::Landscape,
            retina_positions: vec![],
            dark_pixel_positions: vec![],
        };

        // TODO: threshold
        image.generate_dark_pixel_positions(0.5)?;
        Ok(image)
    }

    /// load from a path and return an image(preprocessed)
    pub fn from_path(path: PathBuf) -> std::result::Result<Self, Error> {
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
            format: if width >= height {
                ImageFormat::Landscape
            } else {
                ImageFormat::Portrait
            },
            retina_positions: vec![],
            dark_pixel_positions: vec![],
        };

        // preprocess the image
        image.preprocess()?;

        // TODO: threshold
        image.generate_dark_pixel_positions(0.5)?;
        Ok(image)
    }

    /// load from a path and return an image without preprocessing
    pub fn from_path_raw(path: PathBuf) -> std::result::Result<Self, Error> {
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
            format: if width >= height {
                ImageFormat::Landscape
            } else {
                ImageFormat::Portrait
            },
            retina_positions: vec![],
            dark_pixel_positions: vec![],
        };
        // TODO: threshold
        image.generate_dark_pixel_positions(0.5)?;
        Ok(image)
    }

    /// resizes, find edges and binarizes it
    pub fn preprocess(&mut self) -> Result {
        let (width, height) = match self.format {
            ImageFormat::Landscape => (
                CONFIG.image_processing.goal_image_width as u32,
                CONFIG.image_processing.goal_image_height as u32,
            ),
            ImageFormat::Portrait => (
                CONFIG.image_processing.goal_image_height as u32,
                CONFIG.image_processing.goal_image_width as u32,
            ),
        };
        self.resize_all(width, height)?
            .edged(Some(CONFIG.image_processing.sobel_threshold as f32))?
            .erode(
                imageproc::distance_transform::Norm::L1,
                CONFIG.image_processing.erode_pixels as u8,
            )?;
        Ok(())
    }

    pub fn binarize(&mut self) -> Result {
        todo!("real binarization and save the lower und upper binarization value in the image")
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

    pub fn dilate(
        &mut self,
        norm: imageproc::distance_transform::Norm,
        k: u8,
    ) -> std::result::Result<&mut Self, Error> {
        imageproc::morphology::dilate_mut(&mut self.grey, norm, k);
        Ok(self)
    }

    /// all positions of darker pixels. starting with Position top left x: 1, y: 1
    /// because of the Position conversion from Position to image indices!!
    pub fn dark_pixel_positions(&self) -> &Vec<Position> {
        &self.dark_pixel_positions
    }

    /// saves all the darker pixel in the GREY image, determined by the threshold between 0 and 1
    fn generate_dark_pixel_positions(&mut self, threshold: f32) -> Result {
        if !(0.0..=1.0).contains(&threshold) {
            return Err("Threshold must be between 0 and 1".into());
        };

        let mut positions = vec![];
        self.grey.enumerate_pixels().for_each(|(x, y, p)| {
            if p.0[0] as f32 / 255.0 <= threshold {
                positions.push(Position::new((x + 1) as i32, (y + 1) as i32));
            }
        });

        self.dark_pixel_positions = positions;

        Ok(())
    }

    /// create a subview into the image with the given position with size
    /// if the ends of the retina would be outside the image, an index error is returned
    pub fn create_retina_at(
        &self,
        position: Position,
        size: usize,
        superpixel_size: usize,
        label: String,
    ) -> std::result::Result<Retina, Error> {
        if size % 2 == 0 {
            return Err(format!("ValueError: retina size: {} must be odd", size).into());
        }

        if size % superpixel_size != 0 {
            return Err(format!(
                "ValueError: retina size: {} must be dividable by the superpixel size: {}",
                size, superpixel_size
            )
            .into());
        }

        let mut data = vec![];
        let offset = size as i32 / 2 + 1;
        for col in 0..size as i32 {
            for row in 0..size as i32 {
                // when going negative with this operation it means that we try to access a pixel that is outside of the image
                // so we give back an error
                if position.x >= self.width() as i32 + offset
                    || position.y >= self.height() as i32 + offset
                    || position.x < offset
                    || position.y < offset
                {
                    return Err("IndexError: position is out of bounds".into());
                }
                let x = position.x - offset + row;
                let y = position.y - offset + col;
                data.push(self.get_pixel(x as u32, y as u32));
            }
        }

        // create superpixels
        let mut superpixels = vec![];
        // assuming size is 35 and superpixle size is either 5 or 7
        let superpixel_amount = (size / superpixel_size).pow(2);
        let mut row = superpixel_size / 2;
        for i in 0..superpixel_amount {
            if i % (size / superpixel_size) == 0 && i != 0 {
                row += superpixel_size;
            }

            let row_idx = i % (size / superpixel_size);
            let center_idx = if i == 0 {
                (row * size) + superpixel_size / 2
            } else {
                (row * size) + superpixel_size / 2 + superpixel_size * row_idx
            };

            let mut superpixel_value = 0f32;
            // collect the pixels data and average it with threshold
            for row in 0..superpixel_size {
                for col in 0..superpixel_size {
                    let offset = superpixel_size / 2;
                    let data_idx = center_idx - (size * offset) - offset + col + (row * size);
                    superpixel_value += data[data_idx];
                }
            }
            superpixel_value /= superpixel_size.pow(2) as f32;
            // TODO: threshold should be from a binarazation algorithm and not hardcoded
            superpixel_value = if superpixel_value >= 0.5 { 1.0 } else { 0.0 };
            superpixels.push(Superpixel::new(superpixel_value));
        }

        Ok(Retina {
            data,
            size,
            label,
            superpixels,
            superpixel_size,
            center_position: position,
            current_delta_position: Position::new(0, 0),
            last_delta_position: Position::new(0, 0),
            dark_pixel_positions: self.dark_pixel_positions().clone(),
            dark_pixel_positions_visited: vec![],
        })
    }

    pub fn update_retina_movement(&mut self, retina: &Retina) {
        self.retina_positions.push((
            retina.get_center_position(),
            retina.size(),
            retina.label.clone(),
        ));
    }

    pub fn save_rgba(&mut self, path: PathBuf) -> std::result::Result<&mut Self, Error> {
        self.rgba.save(path)?;
        Ok(self)
    }

    pub fn save_grey(&mut self, path: PathBuf) -> std::result::Result<&mut Self, Error> {
        self.grey.save(path)?;
        Ok(self)
    }

    pub fn save_with_retina(&self, path: PathBuf) -> Result {
        let mut canvas = self.grey().clone();
        for (index, (retina_position, retina_size, _)) in self.retina_positions.iter().enumerate() {
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
            // draw a line from the last retina position to the current retina position
            let (line_begin, _, _) = &self.retina_positions[index - 1];
            let line_end = retina_position;
            draw_line_segment_mut(
                &mut canvas,
                ((line_begin.x as f32 - 0.5), (line_begin.y as f32 - 0.5)),
                ((line_end.x as f32 - 0.5), (line_end.y as f32 - 0.5)),
                Luma([127]),
            );

            // draw in the middle a circle
            draw_filled_circle_mut(&mut canvas, (x as i32, y as i32), 1_i32, Luma([0]));

            // // add a label to the retina
            // let font_data = include_bytes!("../assets/Roboto-Regular.ttf");
            // let Some(font) = Font::try_from_bytes(font_data) else {
            //     return Err("Could not load font".into());
            // };
            // let scale = Scale {
            //     x: CONFIG.image_processing.retina_label_scale as f32,
            //     y: CONFIG.image_processing.retina_label_scale as f32,
            // };
            // let color = Luma([0]);
            // draw_text_mut(&mut canvas, color, x as i32, y as i32, scale, &font, label);
        }
        canvas.save(path)?;

        Ok(())
    }

    /// on the rgba version of the image upscaled
    pub fn save_with_retina_upscaled(&self, path: PathBuf) -> Result {
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
        for (index, (retina_position, retina_size, _)) in self.retina_positions.iter().enumerate() {
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
            // draw a line from the last retina position to the current retina position
            let (line_begin, _, _) = &self.retina_positions[index - 1];
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

            // draw at the middle of the retina a circle
            draw_filled_circle_mut(
                &mut canvas,
                ((scaled_x) as i32, (scaled_y) as i32),
                circle_radius as i32,
                Rgba([0, 255, 0, 255]),
            );

            // // add a label to the retina
            // let font_data: &[u8] = include_bytes!("../assets/Roboto-Regular.ttf");
            // let Some(font) = Font::try_from_bytes(font_data) else {
            //     return Err("Could not load font".into());
            // };
            // let scale = Scale {
            //     x: CONFIG.image_processing.retina_label_scale as f32,
            //     y: CONFIG.image_processing.retina_label_scale as f32,
            // };
            // let color = Rgba([0, 0, 0, 255]);
            // draw_text_mut(
            //     &mut canvas,
            //     color,
            //     scaled_x as i32,
            //     scaled_y as i32,
            //     scale,
            //     &font,
            //     label,
            // );
        }
        canvas.save(path)?;

        Ok(())
    }

    /// checks if the bndbox is fully wrapped from the retina rectangle (inclusive edged)
    pub fn wraps_bndbox(&self, bndbox: &Bndbox, retina: &Retina) -> bool {
        let top_left_bndbox = bndbox.top_left();
        let bottom_right_bndbox = bndbox.bottom_rigt();
        let top_left_retina = retina.get_top_left_position();
        let bottom_right_retina = retina.get_bottom_right_position();
        top_left_bndbox >= top_left_retina && bottom_right_bndbox <= bottom_right_retina
    }
}

#[derive(Clone)]
pub struct Retina {
    // color data stored in a vector
    data: Vec<f32>,
    size: usize,
    label: String,
    superpixels: Vec<Superpixel>,
    superpixel_size: usize,
    last_delta_position: Position,
    current_delta_position: Position,
    center_position: Position,
    dark_pixel_positions: Vec<Position>,
    dark_pixel_positions_visited: Vec<Position>,
}

impl Retina {
    /// access the raw data the retina 'sees'. counting from 0.
    pub fn get_value(&self, x: usize, y: usize) -> f32 {
        self.get_data()[y * self.size() + x]
    }

    /// acces the superpixel value at given position, counting from 0.
    pub fn get_superpixel_value(&self, x: usize, y: usize) -> f32 {
        self.superpixels[y * self.superpixel_rows_or_col() + x].value
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

    pub fn superpixel_size(&self) -> usize {
        self.superpixel_size
    }

    /// amount of the rows and the coloumns of superpixels
    pub fn superpixel_rows_or_col(&self) -> usize {
        self.size() / self.superpixel_size()
    }

    pub fn dark_pixel_positions_visited(&self) -> &Vec<Position> {
        &self.dark_pixel_positions_visited
    }

    pub fn dark_pixel_positions_visited_mut(&mut self) -> &mut Vec<Position> {
        &mut self.dark_pixel_positions_visited
    }

    pub fn dark_pixel_positions(&self) -> &Vec<Position> {
        &self.dark_pixel_positions
    }

    /// collects all dark pixel the retina currently sees and updates its list of
    /// all dark pixel visited
    pub fn update_positions_visited(&mut self) {
        // TODO: threshold
        let mut new_pixels = vec![];
        self.dark_pixel_positions_in_frame(0.5)
            .into_iter()
            .filter(|pos| !self.dark_pixel_positions_visited().contains(pos))
            .for_each(|new_pos| {
                new_pixels.push(new_pos);
            });
        new_pixels.iter().for_each(|new_pos| {
            self.dark_pixel_positions_visited_mut()
                .push(new_pos.clone())
        });
    }

    /// calculates the percentage of all darker pixels the retina already visited in the image.
    /// Normalized between 0 and 1
    pub fn percentage_visited(&self) -> f32 {
        let percentage = self.dark_pixel_positions_visited().len() as f32
            / self.dark_pixel_positions().len() as f32;
        assert!(percentage <= 1.0);
        percentage
    }

    /// calculates all positions of dark pixels the retina currently sees
    pub fn dark_pixel_positions_in_frame(&self, threshold: f32) -> Vec<Position> {
        let mut positions: Vec<Position> = vec![];
        self.data.iter().enumerate().for_each(|(idx, p)| {
            if p <= &threshold {
                // calculate the index of the pixel in the real image global system
                let local_x = idx % self.size();
                let local_y = idx / self.size();
                let local_position = Position::new((local_x) as i32, (local_y) as i32);
                // offset needed because of position and image indices difference
                positions.push(self.get_top_left_position() + local_position);
            }
        });
        positions
    }

    pub fn get_center_position(&self) -> Position {
        self.center_position.clone()
    }

    /// global position inside the image
    pub fn get_top_left_position(&self) -> Position {
        Position::new(
            self.center_position.x - (self.size / 2) as i32,
            self.center_position.y - (self.size / 2) as i32,
        )
    }

    /// global position inside the image
    pub fn get_bottom_right_position(&self) -> Position {
        Position::new(
            self.center_position.x + (self.size / 2) as i32,
            self.center_position.y + (self.size / 2) as i32,
        )
    }
    pub fn get_last_delta_position(&self) -> Position {
        // FIXME: why clone here
        self.last_delta_position.clone()
    }

    pub fn get_current_delta_position(&self) -> Position {
        // FIXME: why clone here
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

#[derive(Debug, Clone)]
struct Superpixel {
    value: f32,
}

impl Superpixel {
    pub fn new(value: f32) -> Self {
        Superpixel { value }
    }
}

#[cfg(test)]
mod tests {
    use rand_chacha::ChaCha8Rng;

    use super::*;

    fn get_test_dir() -> String {
        let dir = "tests/images".to_string();
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn get_test_image() -> Image {
        Image::from_vec(vec![0.0; 101 * 101]).unwrap()
    }

    #[test]
    fn test_get_retina_out_of_bounds() {
        let image = get_test_image();
        // getting the first pixel in the top left corner should give an error
        let retina = image.create_retina_at(Position::new(1, 1), 35, 7, "test".to_string());
        assert!(retina.is_err());
    }

    #[test]
    fn test_invalid_retina_movement_to_the_right() {
        let image = get_test_image();
        let mut retina = image
            .create_retina_at(Position::new(50, 50), 35, 7, "test".to_string())
            .unwrap();

        retina.move_mut(&Position::new(80, 0), &image);
        assert_eq!(retina.get_center_position().x, 84);
    }

    #[test]
    fn test_invalid_retina_movement_to_the_left() {
        let image = get_test_image();
        let mut retina = image
            .create_retina_at(Position::new(50, 50), 35, 7, "test".to_string())
            .unwrap();

        retina.move_mut(&Position::new(-80, 0), &image);
        assert_eq!(retina.get_center_position().x, 18);
    }

    #[test]
    fn test_invalid_retina_movement_to_the_top() {
        let image = get_test_image();
        let mut retina = image
            .create_retina_at(Position::new(50, 50), 35, 7, "test".to_string())
            .unwrap();

        retina.move_mut(&Position::new(0, -80), &image);
        assert_eq!(retina.get_center_position().y, 18);
    }

    #[test]
    fn test_invalid_retina_movement_to_the_bottom() {
        let image = get_test_image();
        let mut retina = image
            .create_retina_at(Position::new(50, 50), 35, 7, "test".to_string())
            .unwrap();

        retina.move_mut(&Position::new(0, 80), &image);
        assert_eq!(retina.get_center_position().y, 84);
    }

    #[test]
    fn test_retina_movement() {
        let mut image = get_test_image();
        let mut retina = image
            .create_retina_at(Position::new(50, 50), 35, 7, "test".to_string())
            .unwrap();
        image.update_retina_movement(&retina);
        retina.move_mut(&Position::new(1, 1), &image);
        image.update_retina_movement(&retina);
        assert_eq!(retina.get_center_position().x, 51);
        assert_eq!(retina.get_center_position().y, 51);

        retina.move_mut(&Position::new(10, 6), &image);
        image.update_retina_movement(&retina);

        retina.move_mut(&Position::new(-1, -1), &image);
        image.update_retina_movement(&retina);
        assert_eq!(retina.get_center_position().x, 60);
        assert_eq!(retina.get_center_position().y, 56);
    }

    #[test]
    fn superpixel_retina() {
        let mut rng = ChaCha8Rng::seed_from_u64(2);
        let mut pixels = vec![];
        for _ in 0..100 * 100 {
            pixels.push(if rng.gen_bool(0.5) { 0f32 } else { 255f32 });
        }
        let image = Image::from_vec(pixels).unwrap();
        let retina = image
            .create_retina_at(Position { x: 18, y: 18 }, 35, 7, String::from("1"))
            .unwrap();

        let dir = get_test_dir();
        let file = String::from("superpixel.png");
        image
            .save_with_retina(PathBuf::from(format!("{}/{}", dir, file)))
            .unwrap();

        // real value is 0.44 but thresholded to 0.0
        assert_eq!(retina.superpixels[0].value, 0.0);
    }

    #[test]
    fn dark_pixel_positions_in_frame() {
        let mut image =
            Image::from_path_raw(PathBuf::from("images/training-stage-artificial/t1-01.png"))
                .unwrap();

        let mut retina = image
            .create_retina_at(Position { x: 35, y: 35 }, 35, 7, "1".to_string())
            .unwrap();

        image.update_retina_movement(&retina);
        retina.update_positions_visited();

        assert_eq!(image.dark_pixel_positions(), retina.dark_pixel_positions());

        let real_pos = image.dark_pixel_positions();
        let retina_pos = retina.dark_pixel_positions_in_frame(0.5);

        assert!(retina_pos.iter().all(|pos| { real_pos.contains(pos) }))
    }

    #[test]
    fn get_top_left_position() {
        let image = get_test_image();
        let retina = image
            .create_retina_at(Position { x: 35, y: 35 }, 35, 5, "1".to_string())
            .unwrap();

        assert_eq!(retina.get_top_left_position(), Position::new(18, 18))
    }

    #[test]
    fn dark_pixel_positions_visited() {
        let mut image =
            Image::from_path_raw(PathBuf::from("images/unit_tests/dark-pixel-test.png")).unwrap();

        let mut retina = image
            .create_retina_at(Position { x: 20, y: 20 }, 35, 5, "1".to_string())
            .unwrap();
        image.update_retina_movement(&retina);
        retina.update_positions_visited();
        assert_eq!(retina.dark_pixel_positions_visited().len(), 4);

        retina.move_mut(&Position { x: 30, y: 30 }, &image);
        image.update_retina_movement(&retina);
        retina.update_positions_visited();
        assert_eq!(retina.dark_pixel_positions_visited().len(), 13);

        retina.move_mut(&Position { x: -30, y: -30 }, &image);
        image.update_retina_movement(&retina);
        retina.update_positions_visited();
        assert_eq!(retina.dark_pixel_positions_visited().len(), 13);
    }

    #[test]
    fn dark_pixel_positions() {
        let mut image =
            Image::from_path_raw(PathBuf::from("images/training-stage-artificial/t1-01.png"))
                .unwrap();

        let mut retina = image
            .create_retina_at(Position { x: 55, y: 180 }, 35, 7, "1".to_string())
            .unwrap();

        let speed = 31;

        // move up
        image.update_retina_movement(&retina);
        retina.update_positions_visited();

        assert_eq!(retina.dark_pixel_positions(), image.dark_pixel_positions());
        assert!(retina
            .dark_pixel_positions_in_frame(0.5)
            .iter()
            .all(|pos| { image.dark_pixel_positions().contains(pos) }));

        retina.move_mut(&Position::new(0, -speed), &image);
        image.update_retina_movement(&retina);
        retina.update_positions_visited();

        assert!(retina
            .dark_pixel_positions_in_frame(0.5)
            .iter()
            .all(|pos| { image.dark_pixel_positions().contains(pos) }));

        retina.move_mut(&Position::new(0, -speed), &image);
        image.update_retina_movement(&retina);
        retina.update_positions_visited();

        assert!(retina
            .dark_pixel_positions_in_frame(0.5)
            .iter()
            .all(|pos| { image.dark_pixel_positions().contains(pos) }));

        retina.move_mut(&Position::new(0, -speed), &image);
        image.update_retina_movement(&retina);
        retina.update_positions_visited();

        assert!(retina
            .dark_pixel_positions_in_frame(0.5)
            .iter()
            .all(|pos| { image.dark_pixel_positions().contains(pos) }));

        // move right
        retina.move_mut(&Position::new(speed, 0), &image);
        image.update_retina_movement(&retina);
        retina.update_positions_visited();

        assert!(retina
            .dark_pixel_positions_in_frame(0.5)
            .iter()
            .all(|pos| { image.dark_pixel_positions().contains(pos) }));

        //move down
        retina.move_mut(&Position::new(0, speed), &image);
        image.update_retina_movement(&retina);
        retina.update_positions_visited();

        assert!(retina
            .dark_pixel_positions_in_frame(0.5)
            .iter()
            .all(|pos| { image.dark_pixel_positions().contains(pos) }));

        retina.move_mut(&Position::new(0, speed), &image);
        image.update_retina_movement(&retina);
        retina.update_positions_visited();

        assert!(retina
            .dark_pixel_positions_in_frame(0.5)
            .iter()
            .all(|pos| { image.dark_pixel_positions().contains(pos) }));

        retina.move_mut(&Position::new(0, speed), &image);
        image.update_retina_movement(&retina);
        retina.update_positions_visited();

        assert!(retina
            .dark_pixel_positions_in_frame(0.5)
            .iter()
            .all(|pos| { image.dark_pixel_positions().contains(pos) }));

        // move up
        image.update_retina_movement(&retina);
        retina.update_positions_visited();

        assert!(retina
            .dark_pixel_positions_in_frame(0.5)
            .iter()
            .all(|pos| { image.dark_pixel_positions().contains(pos) }));

        retina.move_mut(&Position::new(0, -speed), &image);
        image.update_retina_movement(&retina);
        retina.update_positions_visited();

        assert!(retina
            .dark_pixel_positions_in_frame(0.5)
            .iter()
            .all(|pos| { image.dark_pixel_positions().contains(pos) }));

        retina.move_mut(&Position::new(0, -speed), &image);
        image.update_retina_movement(&retina);
        retina.update_positions_visited();

        assert!(retina
            .dark_pixel_positions_in_frame(0.5)
            .iter()
            .all(|pos| { image.dark_pixel_positions().contains(pos) }));

        retina.move_mut(&Position::new(0, -speed), &image);
        image.update_retina_movement(&retina);
        retina.update_positions_visited();

        assert!(retina
            .dark_pixel_positions_in_frame(0.5)
            .iter()
            .all(|pos| { image.dark_pixel_positions().contains(pos) }));

        // move right
        retina.move_mut(&Position::new(speed, 0), &image);
        image.update_retina_movement(&retina);
        retina.update_positions_visited();

        assert!(retina
            .dark_pixel_positions_in_frame(0.5)
            .iter()
            .all(|pos| { image.dark_pixel_positions().contains(pos) }));

        retina.move_mut(&Position::new(speed, 0), &image);
        image.update_retina_movement(&retina);
        retina.update_positions_visited();

        assert!(retina
            .dark_pixel_positions_in_frame(0.5)
            .iter()
            .all(|pos| { image.dark_pixel_positions().contains(pos) }));

        retina.move_mut(&Position::new(speed, 0), &image);
        image.update_retina_movement(&retina);
        retina.update_positions_visited();

        assert!(retina
            .dark_pixel_positions_in_frame(0.5)
            .iter()
            .all(|pos| { image.dark_pixel_positions().contains(pos) }));

        retina.move_mut(&Position::new(speed, 0), &image);
        image.update_retina_movement(&retina);
        retina.update_positions_visited();

        assert!(retina
            .dark_pixel_positions_in_frame(0.5)
            .iter()
            .all(|pos| { image.dark_pixel_positions().contains(pos) }));

        retina.move_mut(&Position::new(speed, 0), &image);
        image.update_retina_movement(&retina);
        retina.update_positions_visited();

        assert!(retina
            .dark_pixel_positions_in_frame(0.5)
            .iter()
            .all(|pos| { image.dark_pixel_positions().contains(pos) }));

        // move down
        retina.move_mut(&Position::new(0, speed), &image);
        image.update_retina_movement(&retina);
        retina.update_positions_visited();

        assert!(retina
            .dark_pixel_positions_in_frame(0.5)
            .iter()
            .all(|pos| { image.dark_pixel_positions().contains(pos) }));

        retina.move_mut(&Position::new(0, speed), &image);
        image.update_retina_movement(&retina);
        retina.update_positions_visited();

        assert!(retina
            .dark_pixel_positions_in_frame(0.5)
            .iter()
            .all(|pos| { image.dark_pixel_positions().contains(pos) }));

        retina.move_mut(&Position::new(0, speed), &image);
        image.update_retina_movement(&retina);
        retina.update_positions_visited();

        assert!(retina
            .dark_pixel_positions_in_frame(0.5)
            .iter()
            .all(|pos| { image.dark_pixel_positions().contains(pos) }));

        image
            .save_with_retina(PathBuf::from("tests/images/t.png"))
            .unwrap();
    }

    #[test]
    fn wraps_bndbox() {
        let image = get_test_image();
        let retina = image
            .create_retina_at(Position::new(50, 50), 35, 7, "".to_string())
            .unwrap();
        let bndbox = Bndbox {
            xmin: String::from("45"),
            xmax: String::from("65"),
            ymin: String::from("45"),
            ymax: String::from("65"),
        };
        assert!(image.wraps_bndbox(&bndbox, &retina));
    }
    #[test]
    fn doesnt_wraps_bndbox() {
        let image = get_test_image();
        let retina = image
            .create_retina_at(Position::new(50, 50), 35, 7, "".to_string())
            .unwrap();
        let bndbox = Bndbox {
            xmin: String::from("75"),
            xmax: String::from("95"),
            ymin: String::from("75"),
            ymax: String::from("95"),
        };
        assert!(!image.wraps_bndbox(&bndbox, &retina));
    }
}
