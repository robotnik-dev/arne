use serde::Deserialize;
use serde_xml_rs::{from_str, to_string};

#[derive(Debug, Deserialize, PartialEq)]
struct Database {
    #[serde(rename = "$value")]
    value: String
}

#[derive(Debug, Deserialize, PartialEq)]
struct Size {
    width: String,
    height: String,
    depth: String,
}

#[derive(Debug, Deserialize, PartialEq)]
struct Bndbox {
    xmin: String,
    ymin: String,
    xmax: String,
    ymax: String,
}

#[derive(Debug, Deserialize, PartialEq)]
struct Object {
    name: String,
    pose: String,
    truncated: String,
    difficult: String,
    bndbox: Bndbox,
    text: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "lowercase")]
pub struct Annotation {
    folder: String,
    filename: String,
    path: String,
    source: Database,
    size: Size,
    segmented: String,
    #[serde(rename = "object")]
    objects: Vec<Object>
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
}