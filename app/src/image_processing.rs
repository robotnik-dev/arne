use std::{fmt::Debug, ops::AddAssign};
use image::{ImageBuffer, LumaA};

use crate::Result;
use crate::Error;

pub const IMAGE_DIMENSIONS: (u32, u32) = (33, 25);
pub const RETINA_SIZE: usize = 5;

#[derive(Debug, Clone)]
pub struct Position {
    x: i32,
    y: i32,
}

impl Position {
    pub fn new(x: i32, y: i32) -> Self {
        Position {
            x,
            y,
        }
    }
}

impl AddAssign for Position {
    fn add_assign(&mut self, other: Self) {
        self.x += other.x;
        self.y += other.y;
    }
}

pub struct Image {
    data: ImageBuffer<LumaA<u8>, Vec<u8>>,
    normalized_data: Vec<f32>
}

impl Image {
    pub fn from_path(path: String) -> std::result::Result<Self, Error> {
        let data = image::io::Reader::open(path)?.decode()?.into_luma_alpha8();
        let mut normalized_data = vec![];
        for pixel in data.pixels() {
            normalized_data.push(pixel.0[0] as f32 / 255.0);
        }
        Ok(
            Image {
                data,
                normalized_data,
            }
        )
    }

    pub fn data(&self) -> &Vec<f32> {
        &self.normalized_data
    }

    /// wrapper to get only the normalized value of a pixel
    pub fn get_pixel(&self, x: u32, y: u32) -> f32 {
        self.normalized_data[y as usize * self.data.width() as usize + x as usize]
    }

    // binarizes the image and stores it internally in the normalized_data vector
    fn binarize(&mut self) {
        // 1. calculated the mean color of the image
        // 2. collect dark and white pixels in seperate vector to cacluate the mean of the white and dark pixels respectively
        // 3. use the mean of the dark pixels to set every dark pixel of the real image to that mean value
        // 4. do it wth the white pixels the same way
        let mean_color = self.normalized_data
            .iter()
            .sum::<f32>() / self.normalized_data.len() as f32;
        let (dark_pixels, white_pixels): (Vec<f32>, Vec<f32>) = self.normalized_data
            .iter()
            .partition(|pixel| **pixel < mean_color);
        let dark_binary = dark_pixels
            .iter()
            .sum::<f32>() / dark_pixels.len() as f32;
        let white_binary = white_pixels
            .iter()
            .sum::<f32>() / white_pixels.len() as f32;

        for pixel in self.normalized_data.iter_mut() {
            if *pixel > mean_color {
                *pixel = dark_binary;
            } else {
                *pixel = white_binary;
            }
        }
    }

    /// create a subview into the image with the given position with size
    /// if the ends of the retina would be outside the image, an index error is returned
    pub fn create_retina_at(&self, position: Position, size: usize) -> std::result::Result<Retina, Error> {
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
                // TODO: "3" shoudl be calcualted from the x and y size
                if position.x >= IMAGE_DIMENSIONS.0 as i32 + offset || position.y >= IMAGE_DIMENSIONS.1 as i32 + offset || position.x < offset || position.y < offset {
                    return Err("IndexError: position is out of bounds".into());
                }
                let x = position.x - offset + j;
                let y = position.y - offset + i;
                data.push(self.get_pixel(x as u32, y as u32));
            }
        }
        Ok(
            Retina {
                data,
                size,
                center_position: position,
                delta_position: Position::new(0, 0),
            }
        )
    }

    /// highliting the pixels in the original image that overlap with the border of the retina
    /// and writes it to the image buffer and then saves it to the path
    pub fn show_with_retina_movement_mut(&mut self, retina: &Retina, path: String) -> Result {
        let mut image = self.data.clone();
        // change the center pixels alha value to 127
        image.get_pixel_mut((retina.center_position.x - 1) as u32, (retina.center_position.y - 1) as u32).0[1] = 127;

        let offset = retina.size as i32 / 2 + 1;
        // changing the alpha value to 127 for all pixel that touches the border of the retina
        for i in 0..retina.size as i32 {
            for j in 0..retina.size as i32 {
                let x = retina.center_position.x - offset + i;
                let y = retina.center_position.y - offset + j;
                if i == 0 || i == retina.size as i32 - 1 || j == 0 || j == retina.size as i32 - 1 {
                    image.get_pixel_mut(x as u32, y as u32).0[1] = 127;
                }
            }
        }
        image.save(path)?;
        self.data = image;
        Ok(())
    }

    pub fn save_binarized(&mut self, path: String) -> Result {
        self.binarize();
        // for each pixel in normalized data vector, transform it back to LumA<u8> and save it to the path
        let mut buf = self.normalized_data
            .iter()
            .map(|pixel| LumaA([(*pixel * 255.0) as u8, 255]))
            .collect::<Vec<LumaA<u8>>>();

        let mut image: ImageBuffer<LumaA<u8>, Vec<u8>> = ImageBuffer::new(IMAGE_DIMENSIONS.0, IMAGE_DIMENSIONS.1);

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
}

pub struct Retina {
    // color data stored in a vector
    data: Vec<f32>,
    size: usize,
    delta_position: Position,
    // this is only for visualization purpose, the Rnn does not know this information
    center_position: Position,
}

impl Retina {
    /// counting from 0
    /// TODO: return Result
    pub fn get_value(&self, x: usize, y: usize) -> f32 {
        self.data[y * self.size + x]
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
            *pixel = LumaA([self.get_value(x as usize, y as usize) as u8, 255]);
        }
        imgbuf.save(path)?;
        Ok(())
    }

    pub fn move_mut(&mut self, delta_x: i32, delta_y: i32) -> Result {
        let offset = self.size as i32 / 2 + 1;
        // check if any of the retina pixels would be outside the image
        for i in 0..self.size as i32 {
            for j in 0..self.size as i32 {
                let x = self.center_position.x - offset + i + delta_x;
                let y = self.center_position.y - offset + j + delta_y;
                if x < 0 || y < 0 || x >= IMAGE_DIMENSIONS.0 as i32 || y >= IMAGE_DIMENSIONS.1 as i32 {
                    return Err("IndexError: cant move retina out of the image".into());
                }
            }
        }
        self.delta_position = Position::new(delta_x, delta_y);
        self.center_position += self.delta_position.clone();
        Ok(())
    }
}
