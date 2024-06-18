use image::{ImageBuffer, LumaA, Rgba};
use nalgebra::clamp;
use std::ops::Add;
use std::{fmt::Debug, ops::AddAssign};

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

impl AddAssign for Position {
    fn add_assign(&mut self, other: Self) {
        self.x += other.x;
        self.y += other.y;
    }
}

#[derive(Debug, Clone)]
pub struct Image {
    data: ImageBuffer<LumaA<u8>, Vec<u8>>,
    normalized_data: Vec<f32>,
    binarized_white: f32,
    binarized_black: f32,
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

    fn to_rgba(&self) -> std::result::Result<ImageBuffer<Rgba<u8>, Vec<u8>>, Error> {
        let mut buf = self
            .normalized_data
            .iter()
            .map(|pixel| {
                Rgba([
                    (*pixel * 255.0) as u8,
                    (*pixel * 255.0) as u8,
                    (*pixel * 255.0) as u8,
                    255,
                ])
            })
            .collect::<Vec<Rgba<u8>>>();
        let mut image: ImageBuffer<Rgba<u8>, Vec<u8>> = ImageBuffer::new(
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
        Ok(image)
    }

    fn to_lumaa(&self) -> std::result::Result<ImageBuffer<LumaA<u8>, Vec<u8>>, Error> {
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
        Ok(image)
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

    pub fn update_retina_movement_mut(&mut self, retina: &Retina) {
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
    use rand_distr::num_traits::Inv;

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
        let mut image = Image::from_path("images/artificial/checkboard.png".to_string()).unwrap();

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
        let mut image = Image::from_path("images/artificial/checkboard.png".to_string()).unwrap();
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
        let mut image = Image::from_path("images/artificial/gradient.png".to_string()).unwrap();
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
        let mut image = Image::from_path("images/artificial/checkboard.png".to_string()).unwrap();
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
}
