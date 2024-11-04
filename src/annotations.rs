use bevy::{prelude::*, utils::info};
use serde_json::Value;
use std::{io::Read, path::PathBuf};
use xml2json_rs::JsonBuilder;

use crate::{
    image::{Image, ImageFormat, Position},
    netlist::{ComponentBuilder, Netlist},
    AdaptiveConfig, Error,
};

#[derive(Resource, Default)]
pub struct XMLParser {
    pub data: Vec<(Annotation, Image, Netlist)>,
    pub loaded: usize,
}

impl XMLParser {
    /// loads the images from the specified folder
    pub fn load(
        &mut self,
        drafter_path: PathBuf,
        all: bool,
        amount: usize,
    ) -> std::result::Result<&mut Self, Error> {
        let mut count = 0usize;
        let path = drafter_path.clone().to_string_lossy().into_owned();
        'outer: for subdir in std::fs::read_dir(path)? {
            let subdir_entry = subdir?;
            let folder_name = subdir_entry
                .file_name()
                .into_string()
                .map_err(|_| "Could not convert to String".to_string())?;
            if folder_name == *"annotations" {
                for annotation_file in std::fs::read_dir(subdir_entry.path())? {
                    // check stop condition
                    if count == amount && !all {
                        break 'outer;
                    };
                    let mut annotation = Annotation::from_path(annotation_file?.path())?;
                    let folder_name = "resized";
                    let path = PathBuf::from(format!(
                        "{}/{}/{}",
                        drafter_path.clone().to_string_lossy().into_owned(),
                        folder_name,
                        annotation.filename.clone()
                    ));
                    // skip all annotations that have not a segmented images
                    if let Ok(mut image) = Image::from_path_raw(path) {
                        // generate once the optimal netlist for this image
                        let mut netlist = Netlist::new();
                        let mut r_idx = 0;
                        let mut c_idx = 0;
                        let mut v_idx = 0;
                        annotation.objects.iter().for_each(|object| {
                            let full_component = object.name.clone();
                            let component = full_component.split(".").take(1).collect::<String>();
                            // TODO(maybe): adding correct nodes to components
                            if component == *"resistor" {
                                netlist
                                    .add_component(
                                        ComponentBuilder::new(
                                            crate::netlist::ComponentType::Resistor,
                                            r_idx.to_string(),
                                        )
                                        .build(),
                                        format!("r{}", r_idx),
                                    )
                                    .unwrap();
                                r_idx += 1;
                            }
                            if component == *"capacitor" {
                                netlist
                                    .add_component(
                                        ComponentBuilder::new(
                                            crate::netlist::ComponentType::Capacitor,
                                            c_idx.to_string(),
                                        )
                                        .build(),
                                        format!("c{}", c_idx),
                                    )
                                    .unwrap();
                                c_idx += 1;
                            }
                            if component == *"voltage" {
                                netlist
                                    .add_component(
                                        ComponentBuilder::new(
                                            crate::netlist::ComponentType::VoltageSourceDc,
                                            v_idx.to_string(),
                                        )
                                        .build(),
                                        format!("v{}", v_idx),
                                    )
                                    .unwrap();
                                v_idx += 1;
                            }
                        });

                        // rotate the image and annotations
                        if image.format == ImageFormat::Portrait {
                            image.rotate90();
                            image.width = image.grey.width();
                            image.height = image.grey.height();
                            image.format = ImageFormat::Landscape;
                            annotation.rotate90();
                        };

                        self.data.push((annotation, image, netlist));
                        self.loaded += 1;
                        count += 1;
                    }
                }
            };
        }
        Ok(self)
    }

    pub fn resize_segmented_images(folder: PathBuf, adaptive_config: &Res<AdaptiveConfig>) {
        let path = folder.clone().to_string_lossy().into_owned();

        let resized_path = format!("{}/resized", path);
        std::fs::create_dir_all(PathBuf::from(resized_path.clone())).unwrap();
        for entry in std::fs::read_dir(path.clone()).unwrap() {
            let entry = entry.unwrap();
            let folder_name = entry.file_name().to_string_lossy().into_owned();

            if folder_name == *"segmentation" {
                for image_entry in std::fs::read_dir(entry.path()).unwrap() {
                    let image_entry = image_entry.unwrap();
                    let filename = image_entry.file_name().to_string_lossy().into_owned();
                    let mut image = Image::from_path_raw(image_entry.path()).unwrap();

                    // resize in regards of correct aspect ratio
                    let (width, height) = match image.format {
                        ImageFormat::Landscape => (
                            adaptive_config.goal_image_width as u32,
                            adaptive_config.goal_image_height as u32,
                        ),
                        ImageFormat::Portrait => (
                            adaptive_config.goal_image_height as u32,
                            adaptive_config.goal_image_width as u32,
                        ),
                    };
                    image.resize_all(width, height).unwrap();

                    image
                        .save_grey(PathBuf::from(format!(
                            "{}/{}",
                            resized_path.clone(),
                            filename
                        )))
                        .unwrap();
                }
            }
        }
    }
}

#[derive(Debug, PartialEq, Hash, Eq, Clone)]
pub struct Database {
    pub value: String,
}

#[derive(Debug, PartialEq, Hash, Eq, Clone)]
pub struct Size {
    pub width: String,
    pub height: String,
    pub depth: String,
}

#[derive(Debug, PartialEq, Hash, Eq, Clone, Default)]
pub struct Bndbox {
    pub xmin: String,
    pub ymin: String,
    pub xmax: String,
    pub ymax: String,
}

impl Bndbox {
    pub fn new() -> Self {
        Bndbox::default()
    }

    pub fn top_left(&self) -> Position {
        let x = self.xmin.parse::<i32>().unwrap();
        let y = self.ymin.parse::<i32>().unwrap();
        Position::new(x, y)
    }

    pub fn bottom_right(&self) -> Position {
        let x = self.xmax.parse::<i32>().unwrap();
        let y = self.ymax.parse::<i32>().unwrap();
        Position::new(x, y)
    }

    pub fn size(&self) -> (u32, u32) {
        let top_left = self.top_left();
        let bottom_right = self.bottom_right();
        let width = bottom_right.x() - top_left.x();
        let height = bottom_right.y() - top_left.y();
        (width as u32, height as u32)
    }
}

#[derive(Debug, PartialEq, Hash, Eq, Clone)]
pub struct Object {
    pub name: String,
    pub pose: String,
    pub truncated: String,
    pub difficult: String,
    pub bndbox: Bndbox,
    // #[serde(skip_serializing_if = "Option::is_none", default)]
    pub text: String,
}

#[derive(Debug, PartialEq, Clone, Component, Hash, Eq)]
pub struct Annotation {
    pub folder: String,
    pub filename: String,
    pub path: String,
    pub source: Database,
    pub size: Size,
    pub segmented: String,
    pub objects: Vec<Object>,
}

impl Annotation {
    pub fn from_path(path: PathBuf) -> std::result::Result<Self, Error> {
        let mut buf = String::new();
        std::fs::OpenOptions::new()
            .read(true)
            .open(path)?
            .read_to_string(&mut buf)?;
        let json_builder = JsonBuilder::default();
        let json = json_builder.build_from_xml(&buf).unwrap();
        let annotation = Annotation::from(json);
        Ok(annotation)
    }

    /// Rotates all bndboxes to the correct places
    pub fn rotate90(&mut self) {
        self.objects.iter_mut().for_each(|obj| {
            // position
            let bndbox = obj.bndbox.clone();
            let (h, w) = (
                self.size.height.parse::<i32>().unwrap(),
                self.size.width.parse::<i32>().unwrap(),
            );

            let mut out_bndbox = Bndbox::new();
            let top_left = bndbox.top_left();
            let bot_right = bndbox.bottom_right();
            for y in 0..h {
                for x in 0..w {
                    let pos = Position::new(x, y);
                    let rotated = Position::new(h - y, x);
                    if pos == top_left {
                        out_bndbox.xmax = rotated.x().to_string();
                        out_bndbox.ymin = rotated.y().to_string();
                    } else if pos == bot_right {
                        out_bndbox.xmin = rotated.x().to_string();
                        out_bndbox.ymax = rotated.y().to_string();
                    }
                }
            }

            // info(format!(
            //     "path: {:?} - image name: {:?} - img size: {:?} - name: {} - old bndbox: {:?} - new bndbox: {:?}",
            //     self.path.clone(),
            //     self.filename.clone(),
            //     self.size.clone(),
            //     obj.name.clone(),
            //     bndbox.clone(),
            //     out_bndbox.clone()
            // ));

            // rotation

            obj.bndbox = out_bndbox;
        });
    }
}

impl From<Value> for Annotation {
    fn from(value: Value) -> Self {
        let annotation_obj = value["annotation"].clone();
        let folder = String::from(annotation_obj["folder"][0].as_str().unwrap_or_default());
        let filename = String::from(annotation_obj["filename"][0].as_str().unwrap_or_default());
        let path = String::from(annotation_obj["path"][0].as_str().unwrap_or_default());
        let source = Database {
            value: String::from(
                annotation_obj["source"][0]["database"][0]
                    .as_str()
                    .unwrap_or_default(),
            ),
        };
        let size = Size {
            width: String::from(
                annotation_obj["size"][0]["width"][0]
                    .as_str()
                    .unwrap_or_default(),
            ),
            height: String::from(
                annotation_obj["size"][0]["height"][0]
                    .as_str()
                    .unwrap_or_default(),
            ),
            depth: String::from(
                annotation_obj["size"][0]["depth"][0]
                    .as_str()
                    .unwrap_or_default(),
            ),
        };
        let segmented = String::from(annotation_obj["segmented"][0].as_str().unwrap_or_default());
        let obj_arr = if let Some(values) = annotation_obj["object"].clone().as_array() {
            values.clone()
        } else {
            Vec::new()
        };
        let mut objects = vec![];
        for obj in obj_arr {
            let name = String::from(obj["name"][0].as_str().unwrap_or_default());
            let pose = String::from(obj["pose"][0].as_str().unwrap_or_default());
            let truncated = String::from(obj["truncated"][0].as_str().unwrap_or_default());
            let difficult = String::from(obj["difficult"][0].as_str().unwrap_or_default());
            let bndbox_obj = obj["bndbox"][0].clone();
            let xmin = String::from(bndbox_obj["xmin"][0].as_str().unwrap_or_default());
            let xmax = String::from(bndbox_obj["xmax"][0].as_str().unwrap_or_default());
            let ymin = String::from(bndbox_obj["ymin"][0].as_str().unwrap_or_default());
            let ymax = String::from(bndbox_obj["ymax"][0].as_str().unwrap_or_default());
            let bndbox = Bndbox {
                xmin,
                xmax,
                ymax,
                ymin,
            };
            let text = if let Some(text) = obj.get("text") {
                String::from(text[0].as_str().unwrap_or_default())
            } else {
                "".to_string()
            };

            let obejct = Object {
                name,
                pose,
                truncated,
                difficult,
                bndbox,
                text,
            };
            objects.push(obejct);
        }

        Annotation {
            folder,
            filename,
            path,
            source,
            size,
            segmented,
            objects,
        }
    }
}
