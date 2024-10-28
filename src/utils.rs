use crate::{annotations::Annotation, genetic_algorithm::Agent, image::Image, AdaptiveConfig};
use bevy::{prelude::*, utils::info};

pub fn round_to_decimal_places(value: f32, decimal_places: u32) -> f32 {
    let multiplier = 10_f32.powi(decimal_places as i32);
    (value * multiplier).round() / multiplier
}

pub fn round2(value: f32) -> f32 {
    round_to_decimal_places(value, 2)
}

pub fn round3(value: f32) -> f32 {
    round_to_decimal_places(value, 3)
}

pub fn netlist_empty(netlist: &str) -> bool {
    netlist.split('\n').count() <= 3
}

pub fn dot_product(v1: &[f32], v2: &[f32]) -> f32 {
    v1.iter().zip(v2.iter()).map(|(x, y)| x * y).sum()
}

pub fn amount_of_components(netlist: &str) -> usize {
    netlist.lines().count() - 3
}

pub fn recreate_retina_movement(
    adaptive_config: &Res<AdaptiveConfig>,
    annotation: &Annotation,
    agent: &Agent,
    orignal_image: &Image,
    save_path: &str,
) {
    let mut image = orignal_image.clone();
    let network = agent.genotype().control_network();
    image.retina_positions = network.retina_positions.clone();
    info(agent.genotype().found_components.clone());
    image
        .save_with_retina(format!("{}/retina_orig.png", save_path).into())
        .unwrap();

    image
        .save_with_retina_upscaled(
            format!("{}/retina_upscaled.png", save_path).into(),
            adaptive_config,
        )
        .unwrap();

    image
        .save_with_bndboxes_with_text(
            format!("{}/bndboxes_with_text.png", save_path).as_str(),
            &adaptive_config,
            annotation,
        )
        .unwrap();

    image
        .save_with_bndboxes(
            format!("{}/bndboxes.png", save_path).as_str(),
            &adaptive_config,
            annotation,
        )
        .unwrap();

    image
        .save_with_found_components(
            format!("{}/retina_and_components.png", save_path).as_str(),
            &adaptive_config,
            agent,
        )
        .unwrap();

    image
        .save_with_found_components_and_bndboxes(
            format!("{}/components_and_bndboxes.png", save_path).as_str(),
            &adaptive_config,
            annotation,
            agent,
        )
        .unwrap();

    image
        .save_with_all(
            format!("{}/all.png", save_path).as_str(),
            &adaptive_config,
            agent,
            annotation,
        )
        .unwrap();
}

#[cfg(test)]
mod tests {

    #[test]
    fn netlist_empty() {
        let netlist = String::from(".SUBCKT main\n.ENDS\n.END");
        assert!(super::netlist_empty(&netlist));
    }

    #[test]
    fn netlist_not_empty() {
        let netlist = String::from(".SUBCKT main\nr1 0 0 3.3k\n.ENDS\n.END");
        assert!(!super::netlist_empty(&netlist));
        let netlist = String::from(".SUBCKT main\nr1 0 0 3.3k\nr2 2 0 500\n.ENDS\n.END");
        assert!(!super::netlist_empty(&netlist));
    }
}
