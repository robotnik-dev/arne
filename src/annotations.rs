use std::{io::Read, path::PathBuf};

use indicatif::ProgressBar;
use serde_json::Value;
use xml2json_rs::JsonBuilder;
// use quick_xml::{de::from_str, se::to_string};

use crate::{image_processing::Image, Error, Result, CONFIG};

pub struct XMLParser {
    pub data: Vec<(Annotation, Image)>,
}

impl XMLParser {
    pub fn new() -> Self {
        XMLParser { data: Vec::new() }
    }

    /// Gives an error when one of the following folders does not exist
    /// - annotations
    /// - preprocessed
    /// Failes also when one images was not yet preprocessed
    pub fn load_preprocessed(
        &mut self,
        drafter_path: PathBuf,
        all: bool,
        amount: usize,
    ) -> std::result::Result<&mut Self, Error> {
        let mut data = vec![];
        let mut count = 0usize;
        'outer: for subdir in std::fs::read_dir(drafter_path)? {
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
                    let path = CONFIG.image_processing.path_to_data as &str;
                    // HACK: changes the path so that only the preprocessed images are loaded
                    let preprocessed_path =
                        annotation.path.clone().replace("images", "preprocessed");
                    let local_path = PathBuf::from(preprocessed_path)
                        .strip_prefix(".")?
                        .to_string_lossy()
                        .into_owned();
                    let path = PathBuf::from(format!("{}/{}", path, local_path));
                    let image = Image::from_path_raw(path)?;
                    data.push((annotation, image));
                    count += 1;
                }
            };
        }
        self.data = data;
        Ok(self)
    }

    // /// loads the images from the segmentation folder
    // pub fn load_segmentation(
    //     &mut self,
    //     drafter_path: PathBuf,
    //     all: bool,
    //     amount: usize,
    // ) -> std::result::Result<&mut Self, Error> {
    //     let mut data = vec![];
    //     let mut count = 0usize;
    //     'outer: for subdir in std::fs::read_dir(drafter_path)? {
    //         let subdir_entry = subdir?;
    //         let folder_name = subdir_entry
    //             .file_name()
    //             .into_string()
    //             .map_err(|_| "Could not convert to String".to_string())?;
    //         if folder_name == *"annotations" {
    //             for annotation_file in std::fs::read_dir(subdir_entry.path())? {
    //                 // check stop condition
    //                 if count == amount && !all {
    //                     break 'outer;
    //                 };

    //                 let annotation = Annotation::from_path(annotation_file?.path())?;
    //                 let path = CONFIG.image_processing.path_to_data as &str;
    //                 let preprocessed_path =
    //                     annotation.path.clone().replace("images", "segmentation");
    //                 let local_path = PathBuf::from(preprocessed_path)
    //                     .strip_prefix(".")?
    //                     .to_string_lossy()
    //                     .into_owned();
    //                 let path = PathBuf::from(format!("{}/{}", path, local_path));
    //                 let image = Image::from_path_raw(path)?;
    //                 data.push((annotation, image));
    //                 count += 1;
    //             }
    //         };
    //     }
    //     self.data = data;
    //     Ok(self)
    // }

    /// This function is meant to run once and then never again. (Maybe when preprocessing changes)
    /// preprocess all images in the directory e.g. drafter_1 and save them in a new folder 'preprocessed'
    /// next to the other folder and lastly changes the filename in the annotation file
    pub fn preprocess_dir(dir_path: PathBuf, all: bool, amount: usize) -> Result {
        let path = dir_path.clone();
        log::debug!("preprocessing folder: {:?}", path.clone());
        
        let new_images_path = format!("{}/preprocessed", path.to_str().unwrap());
        let mut count = 0usize;
        'outer: for folder in std::fs::read_dir(path.clone())? {
            let folder_path = folder?;
            std::fs::create_dir_all(PathBuf::from(new_images_path.clone()))?;
            let folder_name = folder_path
            .file_name()
            .into_string()
            .map_err(|_| "Could not convert to String".to_string())?;
            
            if folder_name == *"annotations" {
                let progress = ProgressBar::new(if all {
                    std::fs::read_dir(folder_path.path())?.count() as u64
                } else {
                    amount as u64
                });
                for annotation_file in std::fs::read_dir(folder_path.path())? {
                    // check stop condition
                    if count == amount && !all {
                        break 'outer;
                    };
                    progress.inc(1);

                    let annotation = Annotation::from_path(annotation_file?.path())?;
                    let path = CONFIG.image_processing.path_to_data as &str;
                    let old_local_path = PathBuf::from(annotation.path.clone())
                        .strip_prefix(".")?
                        .to_string_lossy()
                        .into_owned();
                    let old_image_path = PathBuf::from(format!("{}/{}", path, old_local_path));

                    log::debug!("loading image");
                    let mut image = Image::from_path_raw(old_image_path.clone())?;
                    let new_local_image_path = old_image_path
                        .to_str()
                        .unwrap()
                        .replace("images", "preprocessed");
                    log::debug!("preprocessing image");
                    image.preprocess()?;
                    log::debug!("saving image");
                    image.save_grey(PathBuf::from(new_local_image_path))?;
                    count += 1;
                }
            }
        }
        Ok(())
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct Database {
    pub value: String,
}

#[derive(Debug, PartialEq, Clone)]
pub struct Size {
    pub width: String,
    pub height: String,
    pub depth: String,
}

#[derive(Debug, PartialEq, Clone)]
pub struct Bndbox {
    pub xmin: String,
    pub ymin: String,
    pub xmax: String,
    pub ymax: String,
}

#[derive(Debug, PartialEq, Clone)]
pub struct Object {
    pub name: String,
    pub pose: String,
    pub truncated: String,
    pub difficult: String,
    pub bndbox: Bndbox,
    // #[serde(skip_serializing_if = "Option::is_none", default)]
    pub text: String,
}

#[derive(Debug, PartialEq, Clone)]
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

#[cfg(test)]
mod tests {
    use xml2json_rs::JsonBuilder;

    use super::*;

    fn test_annotation() -> &'static str {
        r#"
<annotation>
    <folder>images</folder>
    <filename>C-1_D1_P1.jpeg</filename>
    <path>./drafter_-1/images/C-1_D1_P1.jpeg</path>
    <source>
        <database>CGHD</database>
    </source>
    <size>
        <width>1000</width>
        <height>1000</height>
        <depth>3</depth>
    </size>
    <segmented>0</segmented>
    <object>
        <name>text</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>410</xmin>
            <ymin>504</ymin>
            <xmax>460</xmax>
            <ymax>558</ymax>
        </bndbox>
    </object>
    <object>
        <name>text</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>418</xmin>
            <ymin>749</ymin>
            <xmax>462</xmax>
            <ymax>803</ymax>
        </bndbox>
        <text>RE</text>
    </object>
    <object>
        <name>text</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>1048</xmin>
            <ymin>472</ymin>
            <xmax>1104</xmax>
            <ymax>530</ymax>
        </bndbox>
        <text>R3</text>
    </object>
</annotation>
        "#
    }

    #[test]
    fn deserialize() {
        let buf = test_annotation();
        let json_builder = JsonBuilder::default();
        let json = json_builder.build_from_xml(&buf).unwrap();
        let should = Annotation::from(json);
        assert_eq!(should.objects.len(), 3);
        assert_eq!(should.objects[0].bndbox.xmin, String::from("410"));
        assert_eq!(should.objects[1].bndbox.ymin, String::from("749"));
    }

    #[test]
    fn from_path() {
        let buf = test_annotation();
        let json_builder = JsonBuilder::default();
        let json = json_builder.build_from_xml(&buf).unwrap();
        let should = Annotation::from(json);
        let loaded =
            Annotation::from_path(PathBuf::from("images/unit_tests/annotation.xml")).unwrap();
        assert_eq!(should, loaded);
    }

    #[test]
    fn parse_amount() {
        let mut parser = XMLParser::new();
        parser
            .load_preprocessed(PathBuf::from("data/drafter_-1"), false, 2)
            .unwrap();
        assert_eq!(parser.data.len(), 2);
    }
}
