use std::{
    fmt::format,
    io::{Read, Write},
    path::PathBuf,
};

use serde::{Deserialize, Serialize};
use quick_xml::{de::from_str, se::to_string};

use crate::{image_processing::Image, Error, CONFIG};

pub struct ImageLoader {
    count: usize,
    annotations: Vec<Annotation>,
}

impl ImageLoader {
    /// creates a Reader from a directory. Gives an error when one of the following folders does not exist
    /// - annotations
    /// - images
    pub fn build(path: PathBuf) -> std::result::Result<Self, Error> {
        let data_path = path.clone();
        let mut annotations = vec![];
        for drafter in std::fs::read_dir(data_path)? {
            let drafter_path = drafter.unwrap().path();
            for subdir in std::fs::read_dir(drafter_path)? {
                let subdir_entry = subdir?;
                let folder_name = subdir_entry
                    .file_name()
                    .into_string()
                    .map_err(|_| format!("Could not convert to String"))?;
                if folder_name == "annotations".to_string() {
                    for annotation_file in std::fs::read_dir(subdir_entry.path())? {
                        let file = annotation_file?;
                        let annotation_path = file.path();
                        let annotation = Annotation::from_path(annotation_path.clone())?;
                        annotations.push(annotation);
                    }
                };
            }
        }

        Ok(ImageLoader {
            count: 0,
            annotations,
        })
    }
}

impl Iterator for ImageLoader {
    type Item = (Annotation, Image);

    fn next(&mut self) -> Option<Self::Item> {
        let annotation = self.annotations.get(self.count)?;
        let path = CONFIG.image_processing.path_to_data as &str;
        let local_path = PathBuf::from(annotation.path.clone())
            .strip_prefix(".")
            .ok()?
            .to_string_lossy()
            .into_owned();
        let path = PathBuf::from(format!("{}/{}", path.to_string(), local_path));
        let image = Image::from_path_raw(path).ok()?;

        self.count += 1;
        Some((annotation.clone(), image))
    }
}

#[derive(Debug, Deserialize, Serialize, PartialEq, Clone)]
struct Database {
    #[serde(rename = "$value")]
    value: String,
}

#[derive(Debug, Deserialize, Serialize, PartialEq, Clone)]
struct Size {
    width: String,
    height: String,
    depth: String,
}

#[derive(Debug, Deserialize, Serialize, PartialEq, Clone)]
struct Bndbox {
    xmin: String,
    ymin: String,
    xmax: String,
    ymax: String,
}

#[derive(Debug, Deserialize, Serialize, PartialEq, Clone)]
struct Object {
    name: String,
    pose: String,
    truncated: String,
    difficult: String,
    bndbox: Bndbox,
    text: Option<String>,
}

#[derive(Debug, Deserialize, Serialize, PartialEq, Clone)]
#[serde(rename_all = "lowercase")]
pub struct Annotation {
    folder: String,
    filename: String,
    path: String,
    source: Database,
    size: Size,
    segmented: String,
    #[serde(rename = "object")]
    objects: Vec<Object>,
}

impl Annotation {
    pub fn from_path(path: PathBuf) -> std::result::Result<Self, Error> {
        let mut buf = String::new();
        std::fs::OpenOptions::new()
            .read(true)
            .open(path)?
            .read_to_string(&mut buf)?;

        let annotation = from_str(buf.as_str())?;
        Ok(annotation)
    }
}

#[cfg(test)]
mod tests {
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
                <text>RC</text>
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
        let annotation = test_annotation();
        let should: Annotation = from_str(&annotation).unwrap();
        assert_eq!(should.objects.len(), 3);
        assert_eq!(should.objects[0].bndbox.xmin, String::from("410"));
        assert_eq!(should.objects[1].bndbox.ymin, String::from("749"));
    }

    #[test]
    fn from_path() {
        let annotation = test_annotation();
        let should: Annotation = from_str(annotation).unwrap();
        let loaded =
            Annotation::from_path(PathBuf::from("images/unit_tests/annotation.xml")).unwrap();
        assert_eq!(should, loaded);
    }

    #[test]
    fn change_annotation() {
        std::fs::create_dir_all(String::from("tests/annotation")).unwrap();
        let mut annotation = Annotation::from_path(PathBuf::from("images/unit_tests/annotation.xml")).unwrap();
        annotation.path = String::from("test");
        let annotation_string = to_string(&annotation).unwrap();
        std::fs::write(PathBuf::from("tests/annotation/change_test.xml"), annotation_string).unwrap();
    }
}
