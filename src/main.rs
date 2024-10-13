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
        .insert_state(AppState::Setup)
        .insert_resource(AdaptiveConfig::default())
        // .add_systems(Startup, preprocess)
        // .add_systems(Startup, test_configs)
        .add_systems(Startup, (load_config))
        // .add_systems(Startup, test_agents)
        .run();
}

#[derive(States, Debug, PartialEq, Hash, Eq, Clone)]
enum AppState {
    Setup,
    Run,
    Cleanup,
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Resource)]
pub struct AdaptiveConfig {
    pub number_of_network_updates: usize,
    pub neuron_lower: f32,
    pub neuron_upper: f32,
    pub retina_lower: f32,
    pub retina_upper: f32,
    pub init_non_zero_retina_weights: f32,
    pub initial_population_size: usize,
    pub max_generations: u64,
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

impl Default for AdaptiveConfig {
    fn default() -> Self {
        Self {
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
}

// Components we have
// - Agent
// - Genotype
// - Image

// Logic is we have X generations to process, and they built on top of each other so it have to run in serial.
// Each process of a generation starts with the same population used in the generation before, but adapted.
// (Optional) We can run multiple processes in parallel meaning multiple populations that evolve next to each other, but do not know each other.

// Agent
// we have X Agents per Population. The Entities stay the same the whole app run through so we just can manipulate the components directly
//

fn load_config(mut adaptive_config: ResMut<AdaptiveConfig>, mut next: ResMut<NextState<AppState>>) {
    let filepath = String::from("current_config.json");
    let loaded_adaptive_config: AdaptiveConfig =
        from_str(read_to_string(filepath).unwrap().as_str()).unwrap();
    adaptive_config.number_of_network_updates = loaded_adaptive_config.number_of_network_updates;
    adaptive_config.neuron_lower = loaded_adaptive_config.neuron_lower;
    adaptive_config.neuron_upper = loaded_adaptive_config.neuron_upper;
    adaptive_config.retina_lower = loaded_adaptive_config.retina_lower;
    adaptive_config.retina_upper = loaded_adaptive_config.retina_upper;
    adaptive_config.init_non_zero_retina_weights =
        loaded_adaptive_config.init_non_zero_retina_weights;
    adaptive_config.max_generations = loaded_adaptive_config.max_generations;
    adaptive_config.tournament_size = loaded_adaptive_config.tournament_size;
    adaptive_config.variance = loaded_adaptive_config.variance;
    adaptive_config.variance_decay = loaded_adaptive_config.variance_decay;
    adaptive_config.mean = loaded_adaptive_config.mean;
    adaptive_config.delete_neuron = loaded_adaptive_config.delete_neuron;
    adaptive_config.delete_weights = loaded_adaptive_config.delete_weights;
    adaptive_config.delete_self_activation = loaded_adaptive_config.delete_self_activation;
    adaptive_config.mutate_neuron = loaded_adaptive_config.mutate_neuron;
    adaptive_config.mutate_weights = loaded_adaptive_config.mutate_weights;
    adaptive_config.mutate_bias = loaded_adaptive_config.mutate_bias;
    adaptive_config.mutate_self_activation = loaded_adaptive_config.mutate_self_activation;
    next.set(AppState::Run);
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

// #[allow(dead_code)]
// fn test_configs(mut exit: EventWriter<AppExit>) {
//     let max_iterations = 100;
//     let mut rng = ChaCha8Rng::from_entropy();
//     let _ = std::fs::remove_dir_all("iterations");
//     let _ = std::fs::remove_file("iteration_results.txt");
//     let mut iteration = 0usize;
//     let mut adaptive_config = AdaptiveConfig::new();
//     adaptive_config.randomize(&mut rng);
//     loop {
//         training::train_agents(
//             TrainingStage::Artificial { stage: 0 },
//             None,
//             format!("iterations/{}/agents", iteration),
//             iteration,
//             &adaptive_config,
//             false,
//             true,
//         )
//         .unwrap();

//         if iteration >= max_iterations {
//             break;
//         }

//         // tweak configuration
//         adaptive_config.randomize(&mut rng);

//         iteration += 1;
//     }
//     exit.send(AppExit::Success);
// }
