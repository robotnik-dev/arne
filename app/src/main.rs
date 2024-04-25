use bevy::prelude::*;
use genetic_algorithm::GeneticAlgorithmPlugin;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(GeneticAlgorithmPlugin)
        .run();
}
