use annotations::XMLParser;
use bevy::{
    log::{Level, LogPlugin},
    prelude::*,
};
use image::TrainingStage;
use rand::SeedableRng;
use rand::{Rng, RngCore};
pub use rand_chacha::ChaCha8Rng;
use serde::{ser::StdError, Deserialize, Serialize};
use serde_json::from_str;
use static_toml::static_toml;
use std::fmt::Display;
use std::fs::read_to_string;
pub use std::time::{Duration, Instant};
mod utils;
pub use utils::{
    amount_of_components, dot_product, netlist_empty, round2, round3, round_to_decimal_places,
};

mod image;
pub use image::Retina;

mod neural_network;
pub use neural_network::{Rnn, ShortTermMemory, SnapShot};

mod genetic_algorithm;
pub use genetic_algorithm::{Agent, AgentEvaluation, Population, SelectionMethod};

mod annotations;
mod netlist;
mod plotting;

mod training;

type Error = Box<dyn std::error::Error>;
type Result = std::result::Result<(), Error>;

static_toml! {
    pub static CONFIG = include_toml!("config.toml");
}

fn main() {
    App::new()
        .add_plugins((
            MinimalPlugins,
            LogPlugin {
                level: Level::DEBUG,
                filter: "wgpu=error,bevy_render=info,bevy_ecs=trace".to_string(),
                custom_layer: |_| None,
            },
        ))
        // .add_systems(Startup, preprocess)
        // .add_systems(Startup, test_configs)
        .add_systems(Startup, run_one_config)
        // .add_systems(Startup, test_agents)
        .run();
}

#[derive(Debug)]
pub struct LocalMaximumError;

impl Display for LocalMaximumError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Reached a local maximum, ending this cycle..")
    }
}

impl Into<Box<dyn StdError>> for LocalMaximumError {
    fn into(self) -> Box<dyn StdError> {
        "local maximum".into()
    }
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
pub struct AdaptiveConfig {
    pub number_of_network_updates: usize,
    pub neuron_lower: f32,
    pub neuron_upper: f32,
    pub retina_lower: f32,
    pub retina_upper: f32,
    pub init_non_zero_retina_weights: f32,
    pub initial_population_size: usize,
    pub max_generations: u64, // NOT configurable
    pub tournament_size: usize,
    pub variance: f32,
    pub variance_decay: f32,
    pub mean: f32,
    pub delete_neuron: f32,
    pub delete_weights: f32,
    pub delete_bias: f32,
    pub delete_self_activation: f32,
    pub mutate_neuron: f32,
    pub mutate_weights: f32,
    pub mutate_bias: f32,
    pub mutate_self_activation: f32,
}

impl AdaptiveConfig {
    #[allow(dead_code)]
    fn new() -> Self {
        AdaptiveConfig {
            number_of_network_updates: usize::default(),
            neuron_lower: f32::default(),
            neuron_upper: f32::default(),
            retina_lower: f32::default(),
            retina_upper: f32::default(),
            init_non_zero_retina_weights: f32::default(),
            initial_population_size: usize::default(),
            max_generations: u64::default(),
            tournament_size: usize::default(),
            variance: f32::default(),
            variance_decay: f32::default(),
            mean: f32::default(),
            delete_neuron: f32::default(),
            delete_weights: f32::default(),
            delete_bias: f32::default(),
            delete_self_activation: f32::default(),
            mutate_neuron: f32::default(),
            mutate_weights: f32::default(),
            mutate_bias: f32::default(),
            mutate_self_activation: f32::default(),
        }
    }

    #[allow(dead_code)]
    fn randomize(&mut self, rng: &mut dyn RngCore) {
        // change here the max generations for every iteration loop
        self.max_generations = 1000;
        self.number_of_network_updates = rng.gen_range(80..=120);
        self.neuron_lower = round2(rng.gen_range(-10.0..=-1.0));
        self.neuron_upper = round2(rng.gen_range(1.0..=10.0));
        self.retina_lower = round2(rng.gen_range(-10.0..=-1.0));
        self.retina_upper = round2(rng.gen_range(1.0..=10.0));
        self.init_non_zero_retina_weights = round2(rng.gen_range(0.3..=0.5));
        self.initial_population_size = rng.gen_range(50..=100);
        self.tournament_size = rng.gen_range(2..self.initial_population_size / 2);
        self.variance = round2(rng.gen_range(0.05..=0.5));
        self.variance_decay = round2(rng.gen_range(0.95..=0.99));
        self.mean = round2(rng.gen_range(0.0..=0.5));
        self.delete_neuron = round2(rng.gen_range(0.0..=0.9));
        self.delete_weights = round2(rng.gen_range(0.0..=0.9));
        self.delete_bias = round2(rng.gen_range(0.0..=0.9));
        self.delete_self_activation = round2(rng.gen_range(0.0..=0.9));
        self.mutate_neuron = round2(rng.gen_range(0.0..=0.9));
        self.mutate_weights = round2(rng.gen_range(0.0..=0.9));
        self.mutate_bias = round2(rng.gen_range(0.0..=0.9));
        self.mutate_self_activation = round2(rng.gen_range(0.0..=0.9));
    }
}

#[allow(dead_code)]
fn preprocess(mut exit: EventWriter<AppExit>) {
    // training folder
    for entry in std::fs::read_dir("data/training").unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();
        XMLParser::resize_segmented_images(path).unwrap();
    }
    // training folder
    for entry in std::fs::read_dir("data/testing").unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();
        XMLParser::resize_segmented_images(path).unwrap();
    }
    exit.send(AppExit::Success);
}

#[allow(dead_code)]
fn test_agents(mut exit: EventWriter<AppExit>) {
    training::test_agents(String::from("iterations/0/agents"), 100usize).unwrap();
    exit.send(AppExit::Success);
}

#[allow(dead_code)]
fn run_one_config(mut exit: EventWriter<AppExit>) {
    let filepath = String::from("current_config.json");
    let adaptive_config: AdaptiveConfig =
        from_str(read_to_string(filepath).unwrap().as_str()).unwrap();
    let iteration = 0;
    training::train_agents(
        TrainingStage::Artificial { stage: 0 },
        None,
        format!("iterations/{}/agents", iteration),
        iteration,
        &adaptive_config,
        false,
        false,
    )
    .unwrap();
    exit.send(AppExit::Success);
}

#[allow(dead_code)]
fn test_configs(mut exit: EventWriter<AppExit>) {
    let max_iterations = 100;
    let mut rng = ChaCha8Rng::from_entropy();
    let _ = std::fs::remove_dir_all("iterations");
    let _ = std::fs::remove_file("iteration_results.txt");
    let mut iteration = 0usize;
    let mut adaptive_config = AdaptiveConfig::new();
    adaptive_config.randomize(&mut rng);
    loop {
        training::train_agents(
            TrainingStage::Artificial { stage: 0 },
            None,
            format!("iterations/{}/agents", iteration),
            iteration,
            &adaptive_config,
            false,
            true,
        )
        .unwrap();

        if iteration >= max_iterations {
            break;
        }

        // tweak configuration
        adaptive_config.randomize(&mut rng);

        iteration += 1;
    }
    exit.send(AppExit::Success);
}

#[cfg(test)]
mod tests {
    use std::fs::write;

    use serde_json::to_string_pretty;

    use super::*;

    #[test]
    fn test_load_config() {
        let mut rng = ChaCha8Rng::from_entropy();
        let mut config = AdaptiveConfig::new();
        config.randomize(&mut rng);
        let buf = to_string_pretty(&config).unwrap();
        write("tests/config.json", buf).unwrap();
        let loaded_str = read_to_string("tests/config.json").unwrap();
        let loaded: AdaptiveConfig = from_str(&loaded_str).unwrap();
        assert_eq!(config, loaded);
    }
}
