use bevy::prelude::*;
use indicatif::ProgressBar;
use serde_json::Value;
use std::{io::Read, path::PathBuf};
use xml2json_rs::JsonBuilder;
// use quick_xml::{de::from_str, se::to_string};

use crate::{
    image::{Image, ImageFormat, Position},
    netlist::{ComponentBuilder, Generate, Netlist},
    Error, Result,
};

pub enum LoadFolder {
    #[allow(dead_code)]
    Segmentation,
    Resized,
    #[allow(dead_code)]
    Images,
}

#[derive(Resource)]
pub struct XMLParser {
    /// String: optimal netlist for this image, once generated when loaded
    pub data: Vec<(Annotation, Image, String)>,
    pub loaded: usize,
}

impl Default for XMLParser {
    fn default() -> Self {
        Self {
            data: vec![],
            loaded: 0,
        }
    }
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
                    let annotation = Annotation::from_path(annotation_file?.path())?;
                    let folder_name = "resized";
                    let path = PathBuf::from(format!(
                        "{}/{}/{}",
                        drafter_path.clone().to_string_lossy().into_owned(),
                        folder_name,
                        annotation.filename.clone()
                    ));
                    // skip all annotations that have not a segmented images
                    if let Ok(image) = Image::from_path_raw(path) {
                        // generate once the optimal netlist for this image
                        let mut netlist = Netlist::new();
                        let mut r_idx = 0;
                        let mut c_idx = 0;
                        let mut v_idx = 0;
                        annotation.objects.iter().for_each(|object| {
                            let full_component = object.name.clone();
                            let component = full_component.split(".").take(1).collect::<String>();
                            // TODO(maybe): adding correct nodes to components
                            if component == "resistor".to_string() {
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
                            if component == "capacitor".to_string() {
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
                            if component == "voltage".to_string() {
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
                        let optimal_netlist = netlist.generate();
                        self.data.push((annotation, image, optimal_netlist));
                        self.loaded += 1;
                        count += 1;
                    }
                }
            };
        }
        Ok(self)
    }

    // #[allow(dead_code)]
    // pub fn resize_segmented_images(folder: PathBuf) -> Result {
    //     let path = folder.clone().to_string_lossy().into_owned();

    //     // debug("resizing images in folder: {:?}", path.clone());
    //     let resized_path = format!("{}/resized", path);
    //     std::fs::create_dir_all(PathBuf::from(resized_path.clone()))?;
    //     for entry in std::fs::read_dir(path.clone())? {
    //         let entry = entry?;
    //         let folder_name = entry.file_name().to_string_lossy().into_owned();

    //         if folder_name == *"segmentation" {
    //             let progress = ProgressBar::new(std::fs::read_dir(entry.path())?.count() as u64);
    //             for image_entry in std::fs::read_dir(entry.path())? {
    //                 progress.inc(1);
    //                 let image_entry = image_entry?;
    //                 let filename = image_entry.file_name().to_string_lossy().into_owned();
    //                 let mut image = Image::from_path_raw(image_entry.path())?;

    //                 // resize in regards of correct aspect ratio
    //                 let (width, height) = match image.format {
    //                     ImageFormat::Landscape => (
    //                         CONFIG.image_processing.goal_image_width as u32,
    //                         CONFIG.image_processing.goal_image_height as u32,
    //                     ),
    //                     ImageFormat::Portrait => (
    //                         CONFIG.image_processing.goal_image_height as u32,
    //                         CONFIG.image_processing.goal_image_width as u32,
    //                     ),
    //                 };
    //                 image.resize_all(width, height)?;

    //                 // rotate all the images that are in Portait format to get all images in the landscape format
    //                 if image.format == ImageFormat::Portrait {
    //                     image.rotate90();
    //                     image.format = ImageFormat::Landscape;
    //                 };

    //                 image.save_grey(PathBuf::from(format!(
    //                     "{}/{}",
    //                     resized_path.clone(),
    //                     filename
    //                 )))?;
    //             }
    //             progress.finish_and_clear();
    //         }
    //     }
    //     Ok(())
    // }
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

#[derive(Debug, PartialEq, Hash, Eq, Clone)]
pub struct Bndbox {
    pub xmin: String,
    pub ymin: String,
    pub xmax: String,
    pub ymax: String,
}

impl Bndbox {
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

// #[cfg(test)]
// mod tests {
//     use xml2json_rs::JsonBuilder;

//     use super::*;

//     fn test_annotation() -> &'static str {
//         r#"
// <annotation>
//     <folder>images</folder>
//     <filename>C-1_D1_P1.jpeg</filename>
//     <path>./drafter_-1/images/C-1_D1_P1.jpeg</path>
//     <source>
//         <database>CGHD</database>
//     </source>
//     <size>
//         <width>1000</width>
//         <height>1000</height>
//         <depth>3</depth>
//     </size>
//     <segmented>0</segmented>
//     <object>
//         <name>text</name>
//         <pose>Unspecified</pose>
//         <truncated>0</truncated>
//         <difficult>0</difficult>
//         <bndbox>
//             <xmin>410</xmin>
//             <ymin>504</ymin>
//             <xmax>460</xmax>
//             <ymax>558</ymax>
//         </bndbox>
//     </object>
//     <object>
//         <name>text</name>
//         <pose>Unspecified</pose>
//         <truncated>0</truncated>
//         <difficult>0</difficult>
//         <bndbox>
//             <xmin>418</xmin>
//             <ymin>749</ymin>
//             <xmax>462</xmax>
//             <ymax>803</ymax>
//         </bndbox>
//         <text>RE</text>
//     </object>
//     <object>
//         <name>text</name>
//         <pose>Unspecified</pose>
//         <truncated>0</truncated>
//         <difficult>0</difficult>
//         <bndbox>
//             <xmin>1048</xmin>
//             <ymin>472</ymin>
//             <xmax>1104</xmax>
//             <ymax>530</ymax>
//         </bndbox>
//         <text>R3</text>
//     </object>
// </annotation>
//         "#
//     }

//     #[test]
//     fn deserialize() {
//         let buf = test_annotation();
//         let json_builder = JsonBuilder::default();
//         let json = json_builder.build_from_xml(&buf).unwrap();
//         let should = Annotation::from(json);
//         assert_eq!(should.objects.len(), 3);
//         assert_eq!(should.objects[0].bndbox.xmin, String::from("410"));
//         assert_eq!(should.objects[1].bndbox.ymin, String::from("749"));
//     }

//     #[test]
//     fn from_path() {
//         let buf = test_annotation();
//         let json_builder = JsonBuilder::default();
//         let json = json_builder.build_from_xml(&buf).unwrap();
//         let should = Annotation::from(json);
//         let loaded =
//             Annotation::from_path(PathBuf::from("images/unit_tests/annotation.xml")).unwrap();
//         assert_eq!(should, loaded);
//     }

//     #[test]
//     fn parse_amount() {
//         let mut parser = XMLParser::new();
//         parser
//             .load(
//                 PathBuf::from(format!(
//                     "{}/drafter_1",
//                     CONFIG.image_processing.training.path as &str
//                 )),
//                 LoadFolder::Segmentation,
//                 false,
//                 1,
//             )
//             .unwrap();
//         assert_eq!(parser.data.len(), 1);
//     }
// }
