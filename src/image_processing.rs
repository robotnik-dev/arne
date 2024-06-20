use env_logger::fmt::style::Color;
use image::math::Rect;
use image::{ImageBuffer, Luma, LumaA};
use imageproc::drawing::{draw_cross_mut, draw_filled_circle, draw_filled_circle_mut, draw_filled_rect_mut, draw_hollow_rect_mut, draw_line_segment_mut, draw_polygon_mut};
use nalgebra::clamp;
use plotters::style::full_palette::BLACK;
use std::ops::{Add, Sub};
use std::{fmt::Debug, ops::AddAssign};
use image::imageops::resize;

use crate::{Error, Result, CONFIG};

pub struct ImageReader {
    images: Vec<Image>,
}

impl ImageReader {
    /// reads a directory and returns a list of all images in it
    pub fn from_path(path: String) -> std::result::Result<Self, Error> {
        let mut images = vec![];
        for entry in std::fs::read_dir(path)? {
            let path = entry?.path();
            let path_str = path.to_str().unwrap().to_string();
            let image = Image::from_path(path_str)?;
            images.push(image);
        }
        Ok(ImageReader { images })
    }

    pub fn get_image(&self, index: usize) -> std::result::Result<&Image, Error> {
        self.images
            .get(index)
            .ok_or("IndexError: index out of bounds".into())
    }

    pub fn images(&self) -> &Vec<Image> {
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

// /// Used to transform between small to real image dimensions
// #[derive(Debug, Clone)]
// struct UpScaledImage {
//     data: ImageBuffer<LumaA<u8>, Vec<u8>>,
// }

// impl UpScaledImage {
//     pub fn empty() -> Self {
//         UpScaledImage {
//             data: ImageBuffer::new(
//                 CONFIG.image_processing.real_image_width as u32,
//                 CONFIG.image_processing.real_image_height as u32,
//             ),
//         }
//     }
    
//     pub fn save(&self, path: String) -> Result {
//         self.data.save(path)?;
//         Ok(())
//     }

//     pub fn data(&self) -> &ImageBuffer<LumaA<u8>, Vec<u8>> {
//         &self.data
//     }
// }

// impl From<&Image> for UpScaledImage {
//     fn from(image: &Image) -> Self {
//         // scale all the data in the small 33x25 image to the real image dimensions 4032x3024
//         // create an image buffer with the real image dimensions
        
//         let data = resize(
//             &image.data,
//             CONFIG.image_processing.real_image_width as u32,
//             CONFIG.image_processing.real_image_height as u32,
//             image::imageops::FilterType::Nearest
//         );
//         UpScaledImage { data }
//     }
// }

#[derive(Debug, Clone)]
pub struct Image {
    data: ImageBuffer<LumaA<u8>, Vec<u8>>,
    /// normalized AND binarized data in this vector
    normalized_data: Vec<f32>,
    binarized_white: f32,
    binarized_black: f32,
    /// used to visualize the retina movement on an upscaled image
    retina_positions: Vec<Position>,
}

impl Image {
    pub fn empty() -> Self {
        Image {
            data: ImageBuffer::new(
                CONFIG.image_processing.image_width as u32,
                CONFIG.image_processing.image_height as u32,
            ),
            normalized_data: vec![
                0.0;
                CONFIG.image_processing.image_width as usize
                    * CONFIG.image_processing.image_height as usize
            ],
            binarized_white: 1.0,
            binarized_black: 0.0,
            retina_positions: vec![],
        }
    }

    // creates a new image, normalizes the data and binarizes it
    pub fn from_path(path: String) -> std::result::Result<Self, Error> {
        let data = image::io::Reader::open(path)?.decode()?.into_luma_alpha8();
        let mut normalized_data = vec![];
        for pixel in data.pixels() {
            normalized_data.push(pixel.0[0] as f32 / 255.0);
        }
        let mut image = Image {
            data,
            normalized_data,
            binarized_white: 1.0,
            binarized_black: 0.0,
            retina_positions: vec![],
        };
        image.binarize();
        Ok(image)
    }

    pub fn data(&self) -> &Vec<f32> {
        &self.normalized_data
    }

    /// wrapper to get only the normalized value of a pixel
    pub fn get_pixel(&self, x: u32, y: u32) -> f32 {
        self.normalized_data[y as usize * self.data.width() as usize + x as usize]
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
                if position.x >= CONFIG.image_processing.image_width as u32 as i32 + offset
                    || position.y >= CONFIG.image_processing.image_height as u32 as i32 + offset
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
            binarized_white: self.binarized_white,
            binarized_black: self.binarized_black,
        })
    }

    pub fn push_new_retina_movement(&mut self, retina: &Retina) {
        self.retina_positions.push(retina.get_center_position());
    }

    pub fn update_retina_movement_mut(&mut self, retina: &Retina) {
        let mut image = self.data.clone();

        // change the center pixels alha value to 127
        image
            .get_pixel_mut(
                (retina.get_center_position().x - 1) as u32,
                (retina.get_center_position().y - 1) as u32,
            )
            .0[1] = CONFIG.image_processing.retina_highlight_alpha as u8;

        let offset = retina.size() as i32 / 2 + 1;
        // changing the alpha value to 127 for all pixel that touches the border of the retina
        for i in 0..retina.size() as i32 {
            for j in 0..retina.size() as i32 {
                // clamp here because when going negative it turn into a buffer overflow
                let x = clamp(
                    retina.get_center_position().x - offset + i,
                    0,
                    CONFIG.image_processing.image_width as u32 as i32 - 1,
                );
                let y = clamp(
                    retina.get_center_position().y - offset + j,
                    0,
                    CONFIG.image_processing.image_height as u32 as i32 - 1,
                );
                if i == 0 || i == retina.size() as i32 - 1 || j == 0 || j == retina.size() as i32 - 1 {
                    image.get_pixel_mut(x as u32, y as u32).0[1] =
                        CONFIG.image_processing.retina_highlight_alpha as u8;
                }
            }
        }
        self.data = image;
    }

    /// TODO: delete this fn
    /// highliting the pixels in the original image that overlap with the border of the retina
    /// and writes it to the image buffer and then saves it to the path
    pub fn save_with_retina_mut(&mut self, retina: &Retina, path: String) -> Result {
        let mut image = self.data.clone();

        // change the center pixels alha value to 127
        image
            .get_pixel_mut(
                (retina.center_position.x - 1) as u32,
                (retina.center_position.y - 1) as u32,
            )
            .0[1] = CONFIG.image_processing.retina_highlight_alpha as u8;

        let offset = retina.size as i32 / 2 + 1;
        // changing the alpha value to 127 for all pixel that touches the border of the retina
        for i in 0..retina.size as i32 {
            for j in 0..retina.size as i32 {
                // clamp here because when going negative it turn into a buffer overflow
                let x = clamp(
                    retina.center_position.x - offset + i,
                    0,
                    CONFIG.image_processing.image_width as u32 as i32 - 1,
                );
                let y = clamp(
                    retina.center_position.y - offset + j,
                    0,
                    CONFIG.image_processing.image_height as u32 as i32 - 1,
                );
                if i == 0 || i == retina.size as i32 - 1 || j == 0 || j == retina.size as i32 - 1 {
                    image.get_pixel_mut(x as u32, y as u32).0[1] =
                        CONFIG.image_processing.retina_highlight_alpha as u8;
                }
            }
        }
        // draw a line from the center of the retina to the new position
        // if retina.get_delta_position() != Position::new(0, 0) {
        //     draw_antialiased_line_segment_mut(
        //         &mut image,
        //         (retina.center_position.x - retina.delta_position.x - 1, retina.center_position.y - retina.delta_position.y - 1),
        //         (retina.center_position.x - 1, retina.center_position.y - 1),
        //         LumaA([0, 127]),
        //         |left, right, left_weight| interpolate(left, right, left_weight))
        // }
        image.save(&path)?;
        self.data = image;
        Ok(())
    }

    /// saves the image to the path but does not change the internal data.
    /// Good for showing only the latest retina position
    pub fn save_with_retina(&self, retina: &Retina, path: String) -> Result {
        let mut image = self.data.clone();
        // change the center pixels alha value to 127
        image
            .get_pixel_mut(
                (retina.center_position.x - 1) as u32,
                (retina.center_position.y - 1) as u32,
            )
            .0[1] = CONFIG.image_processing.retina_highlight_alpha as u8;

        let offset = retina.size as i32 / 2 + 1;
        // changing the alpha value to 127 for all pixel that touches the border of the retina
        for i in 0..retina.size as i32 {
            for j in 0..retina.size as i32 {
                // clamp here because when going negative it turn into a buffer overflow
                let x = clamp(
                    retina.center_position.x - offset + i,
                    0,
                    CONFIG.image_processing.image_width as u32 as i32 - 1,
                );
                let y = clamp(
                    retina.center_position.y - offset + j,
                    0,
                    CONFIG.image_processing.image_height as u32 as i32 - 1,
                );
                if i == 0 || i == retina.size as i32 - 1 || j == 0 || j == retina.size as i32 - 1 {
                    image.get_pixel_mut(x as u32, y as u32).0[1] =
                        CONFIG.image_processing.retina_highlight_alpha as u8;
                }
            }
        }
        image.save(path)?;
        Ok(())
    }

    pub fn save(&self, path: String) -> Result {
        self.data.save(path)?;
        Ok(())
    }

    pub fn save_upscaled(&self, path: String) -> Result {
        let width = CONFIG.image_processing.real_image_width as u32;
        let height = CONFIG.image_processing.real_image_height as u32;
        let size = CONFIG.image_processing.retina_size as f32;
        let circle_radius = CONFIG.image_processing.retina_circle_radius as f32;
        let mut canvas = resize(
            &self.data,
            width,
            height,
            image::imageops::FilterType::Nearest,
        );

        let scaling_factor_x = width as f32
            / CONFIG.image_processing.image_width as f32;
        let scaling_factor_y = height as f32
            / CONFIG.image_processing.image_height as f32;
        let scaled_size = size * scaling_factor_x;

        for (index, retina_position) in self.retina_positions.iter().enumerate() {
            let scaled_x = (retina_position.x as f32 - 0.5) * scaling_factor_x;
            let scaled_y = (retina_position.y as f32 - 0.5) * scaling_factor_y;

            // draw border of the retina
            draw_hollow_rect_mut(
                &mut canvas,
                imageproc::rect::Rect::at(
                    scaled_x as i32 - scaled_size as i32 / 2,
                    scaled_y as i32 - scaled_size as i32 / 2 ,
                ).of_size(
                    scaled_size as u32,
                    scaled_size as u32),
                    LumaA([127, 255])
            );

            if index == 0 {
                continue;
            }
            // draw an arrow from the last retina position to the current retina position
            let arrow_begin = &self.retina_positions[index - 1];
            let arrow_end = retina_position;
            draw_line_segment_mut(
                &mut canvas,
                ((arrow_begin.x as f32 - 0.5) * scaling_factor_x, (arrow_begin.y as f32 - 0.5) * scaling_factor_y),
                ((arrow_end.x as f32 - 0.5) * scaling_factor_x, (arrow_end.y as f32 - 0.5) * scaling_factor_y),
                LumaA([127, 255])
            );

            // draw at the end of the linesegment a circle
            draw_filled_circle_mut(
                &mut canvas,
                (
                        ((arrow_end.x as f32 - 0.5) * scaling_factor_x) as i32,
                        ((arrow_end.y as f32 - 0.5) * scaling_factor_y) as i32,
                    ),
                    circle_radius as i32,
                LumaA([127, 255])
            );
        }

        canvas.save(path)?;
        Ok(())
    }

    /// saves the image to the path as greyscale image
    pub fn save_greyscale(&mut self, path: String) -> Result {
        // for each pixel in normalized data vector, transform it back to LumA<u8> and save it to the path
        let mut buf = self
            .normalized_data
            .iter()
            .map(|pixel| LumaA([(*pixel * 255.0) as u8, 255]))
            .collect::<Vec<LumaA<u8>>>();
        let mut image: ImageBuffer<LumaA<u8>, Vec<u8>> = ImageBuffer::new(
            CONFIG.image_processing.image_width as u32,
            CONFIG.image_processing.image_height as u32,
        );

        buf.reverse();
        for pixel in image.pixels_mut() {
            if let Some(p) = buf.pop() {
                *pixel = p;
            } else {
                return Err("IndexError: not enough pixels in buffer".into());
            }
        }
        image.save(path)?;
        Ok(())
    }

    // binarizes the image and stores it internally in the normalized_data vector
    fn binarize(&mut self) {
        // 1. calculated the mean color of the image
        // 2. collect dark and white pixels in seperate vector to cacluate the mean of the white and dark pixels respectively
        // 3. use the mean of the dark pixels to set every dark pixel of the real image to that mean value
        // 4. do it wth the white pixels the same way
        let mean_color =
            self.normalized_data.iter().sum::<f32>() / self.normalized_data.len() as f32;
        let (dark_pixels, white_pixels): (Vec<f32>, Vec<f32>) = self
            .normalized_data
            .iter()
            .partition(|pixel| **pixel < mean_color);
        let dark_binary = dark_pixels.iter().sum::<f32>() / dark_pixels.len() as f32;
        let white_binary = white_pixels.iter().sum::<f32>() / white_pixels.len() as f32;

        // save the binarized values
        self.binarized_black = dark_binary;
        self.binarized_white = white_binary;

        for pixel in self.normalized_data.iter_mut() {
            if *pixel <= mean_color {
                *pixel = dark_binary;
            } else {
                *pixel = white_binary;
            }
        }
    }
}

pub struct Retina {
    // color data stored in a vector
    data: Vec<f32>,
    size: usize,
    delta_position: Position,
    // this is only for visualization purpose, the Rnn does not know this information
    center_position: Position,
    binarized_white: f32,
    binarized_black: f32,
}

impl Retina {
    /// counting from 0
    /// TODO: return Result
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
        self.binarized_white
    }

    pub fn binarized_black(&self) -> f32 {
        self.binarized_black
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
    pub fn move_mut(&mut self, delta: &Position) {
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
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::round2;
    use crate::Rnn;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    

    #[test]
    fn test_load_image() {
        // using a image size of 33x25 px (nearly 4:3)
        let image: ImageBuffer<LumaA<u8>, Vec<u8>> =
            image::io::Reader::open("images/artificial/checkboard.png")
                .unwrap()
                .decode()
                .unwrap()
                .into_luma_alpha8();
        // white
        assert_eq!(*image.get_pixel(0, 0), LumaA([255, 255]));
        // black
        assert_eq!(*image.get_pixel(3, 0), LumaA([0, 255]));
        // black
        assert_eq!(*image.get_pixel(32, 24), LumaA([0, 255]));
    }

    #[test]
    #[should_panic]
    fn test_invalid_load_image() {
        let image: ImageBuffer<LumaA<u8>, Vec<u8>> =
            image::io::Reader::open("images/artificial/checkboard.png")
                .unwrap()
                .decode()
                .unwrap()
                .into_luma_alpha8();
        // using a image size of 33x25 px (nearly 4:3) so this should panic
        let _ = *image.get_pixel(33, 25);
    }

    #[test]
    fn test_get_retina() {
        let image = Image::from_path("images/artificial/checkboard.png".to_string()).unwrap();

        // using a image size of 33x25 px that the center pixel is at position 16, 12 (countning from 1 not 0)
        let retina = image
            .create_retina_at(
                Position::new(5, 5),
                CONFIG.image_processing.retina_size as usize,
            )
            .unwrap();

        retina
            .create_png_at("test/images/only_retina.png".to_string())
            .unwrap();
        image
            .save_with_retina(&retina, "test/images/with_retina_movement.png".to_string())
            .unwrap();

        assert_eq!(retina.data[12], retina.get_value(2, 2));
        assert_eq!(retina.data[0], retina.get_value(0, 0));
        assert_eq!(retina.data[4], retina.get_value(4, 0));

        // all corners are white
        assert_eq!(retina.get_value(0, 0), 1.);
        assert_eq!(retina.get_value(0, 4), 1.);
        assert_eq!(retina.get_value(4, 0), 1.);
        assert_eq!(retina.get_value(4, 4), 1.);
    }

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

        retina.move_mut(&Position::new(50, 0));

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

        retina.move_mut(&Position::new(-50, 0));

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

        retina.move_mut(&Position::new(0, -60));
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

        retina.move_mut(&Position::new(0, 80));
        assert_eq!(
            retina.get_center_position().y,
            CONFIG.image_processing.image_height as u32 as i32
                - 1
                - (CONFIG.image_processing.retina_size as usize as i32 / 2)
                + 1
        );
    }

    #[test]
    fn test_update_from_retina_inputs() {
        let mut rng = ChaCha8Rng::seed_from_u64(2);
        let mut rnn = Rnn::new(&mut rng, 3);
        let image = Image::from_path("images/artificial/checkboard.png".to_string()).unwrap();
        let retina: Retina = image
            .create_retina_at(
                Position::new(10, 10),
                CONFIG.image_processing.retina_size as usize,
            )
            .unwrap();
        image
            .save_with_retina(&retina, "test/images/with_retina_movement.png".to_string())
            .unwrap();
        retina
            .create_png_at("test/images/retina.png".to_string())
            .unwrap();

        rnn.update_inputs_from_retina(&retina);

        assert_eq!(round2(rnn.neurons()[0].retina_inputs()[0] as f64), 1.0);
        assert_eq!(round2(rnn.neurons()[0].retina_inputs()[4] as f64), 0.0);
        assert_eq!(round2(rnn.neurons()[0].retina_inputs()[24] as f64), 1.0);
    }

    #[test]
    fn test_binarize_image() {
        let image = Image::from_path("images/artificial/gradient.png".to_string()).unwrap();
        image.save("test/images/binarized.png".to_string()).unwrap();
    }

    #[test]
    fn test_load_images() {
        let images = ImageReader::from_path("images/training".to_string()).unwrap();
        // let image = images.get_image(0).unwrap();
        // let image2 = images.get_image(1).unwrap();
        // assert_eq!(image.get_pixel(0, 0), 1.0);
        // assert_eq!(image.get_pixel(3, 0), 0.0);
        // assert_eq!(round2(image2.get_pixel(0, 0).into()), 0.18);

        // assert_eq!(images.images.len(), 3);
        println!("{:?}", images.images);
    }

    #[test]
    #[should_panic]
    fn test_invalid_load_images() {
        let images = ImageReader::from_path("images/artificial".to_string()).unwrap();
        images.get_image(3).unwrap();
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
        retina.move_mut(&Position::new(1, 1));
        assert_eq!(retina.get_center_position().x, 6);
        assert_eq!(retina.get_center_position().y, 6);
        retina.move_mut(&Position::new(1, 1));
        assert_eq!(retina.get_center_position().x, 7);
        assert_eq!(retina.get_center_position().y, 7);
        retina.move_mut(&Position::new(-1, -1));
        assert_eq!(retina.get_center_position().x, 6);
        assert_eq!(retina.get_center_position().y, 6);
    }

    #[test]
    fn test_display_retina_movement() {
        let mut image = Image::from_path("images/artificial/checkboard.png".to_string()).unwrap();
        let mut retina = image
            .create_retina_at(
                Position::new(5, 5),
                CONFIG.image_processing.retina_size as usize,
            )
            .unwrap();
        image
            .save_with_retina_mut(&retina, "test/images/retina_movement.png".to_string())
            .unwrap();
        retina.move_mut(&Position::new(10, 1));
        image
            .save_with_retina_mut(&retina, "test/images/retina_movement.png".to_string())
            .unwrap();
    }

    #[test]
    fn test_save_with_retina_movement() {
        std::fs::create_dir_all("test/saves/retina").unwrap();
        let mut image = Image::from_path("images/artificial/checkboard.png".to_string()).unwrap();
        let mut retina = image
            .create_retina_at(
                Position::new(5, 5),
                CONFIG.image_processing.retina_size as usize,
            )
            .unwrap();
        image.update_retina_movement_mut(&retina);
        retina.move_mut(&Position::new(10, 1));
        image.update_retina_movement_mut(&retina);
        retina.move_mut(&Position::new(1, 4));
        image.update_retina_movement_mut(&retina);

        image
            .save("test/saves/retina/movement.png".to_string())
            .unwrap();
    }

    #[test]
    fn test_empty_to_image() {
        std::fs::create_dir_all("test/saves/images").unwrap();
        let mut empty_image = Image::empty();
        let mut real_image =
            Image::from_path("images/artificial/checkboard.png".to_string()).unwrap();
        let mut retina = real_image
            .create_retina_at(
                Position::new(5, 5),
                CONFIG.image_processing.retina_size as usize,
            )
            .unwrap();
        retina.move_mut(&Position::new(10, 1));
        real_image.update_retina_movement_mut(&retina);
        retina.move_mut(&Position::new(1, 4));
        real_image.update_retina_movement_mut(&retina);

        empty_image = real_image.clone();

        empty_image
            .save("test/saves/images/from_empty.png".to_string())
            .unwrap();
    }

    #[test]
    fn test_scale_image_up() {
        let image = Image::from_path("images/artificial/resistor.png".to_string()).unwrap();
        image.save_upscaled("test/images/upscaled_resistor.png".to_string()).unwrap();
    }

    #[test]
    fn test_scale_image_up_with_retina_movement() {
        let mut image1 = Image::from_path("images/artificial/resistor.png".to_string()).unwrap();
        let mut image2 = Image::from_path("images/artificial/resistor.png".to_string()).unwrap();

        let mut retina1 = image1
            .create_retina_at(
                Position::new(7, 7),
                CONFIG.image_processing.retina_size as usize,
            )
            .unwrap();
        let mut retina2 = image2
            .create_retina_at(
                Position::new(7, 7),
                CONFIG.image_processing.retina_size as usize,
            )
            .unwrap();

        image1.update_retina_movement_mut(&retina1);
        image2.retina_positions.push(retina2.get_center_position());

        retina1.move_mut(&Position::new(21, 0));
        retina2.move_mut(&Position::new(21, 0));

        image1.update_retina_movement_mut(&retina1);
        image2.retina_positions.push(retina2.get_center_position());

        retina1.move_mut(&Position::new(0, 6));
        retina2.move_mut(&Position::new(0, 6));

        image1.update_retina_movement_mut(&retina1);
        image2.retina_positions.push(retina2.get_center_position());

        image1.save("test/images/resistor.png".to_string()).unwrap();
        image2.save_upscaled("test/images/resistor_upscaled.png".to_string()).unwrap();
    }
}
