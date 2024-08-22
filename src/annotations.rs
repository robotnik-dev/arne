use std::{io::Read, path::PathBuf};

use serde_json::Value;
use xml2json_rs::JsonBuilder;
// use quick_xml::{de::from_str, se::to_string};

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
                        let annotation = Annotation::from_path(annotation_file?.path())?;
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

#[derive(Debug, PartialEq, Clone)]
struct Database {
    value: String,
}

#[derive(Debug, PartialEq, Clone)]
struct Size {
    width: String,
    height: String,
    depth: String,
}

#[derive(Debug, PartialEq, Clone)]
struct Bndbox {
    xmin: String,
    ymin: String,
    xmax: String,
    ymax: String,
}

#[derive(Debug, PartialEq, Clone)]
struct Object {
    name: String,
    pose: String,
    truncated: String,
    difficult: String,
    bndbox: Bndbox,
    // #[serde(skip_serializing_if = "Option::is_none", default)]
    text: String,
}

#[derive(Debug, PartialEq, Clone)]
pub struct Annotation {
    folder: String,
    filename: String,
    path: String,
    source: Database,
    size: Size,
    segmented: String,
    objects: Vec<Object>,
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
}
