use crate::{image::Image, AdaptiveConfig, Rnn};
use bevy::prelude::*;

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
    network: &Rnn,
    orignal_image: &Image,
    save_path: &str,
) {
    let mut image = orignal_image.clone();
    image.retina_positions = network.retina_positions.clone();
    image
        .save_with_retina(format!("{}/retina_orig.png", save_path).into())
        .unwrap();
    image
        .save_with_retina_upscaled(
            format!("{}/retina_upscaled.png", save_path).into(),
            adaptive_config,
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
