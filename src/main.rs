use bevy::{log::LogPlugin, prelude::*, utils::info};
use clap::Parser;
use image::TrainingStage;
use rand::{Rng, RngCore, SeedableRng};
pub use rand_chacha::ChaCha8Rng;
use serde::{ser::StdError, Deserialize, Serialize};
use static_toml::static_toml;
pub use std::time::{Duration, Instant};
use std::{
    fmt::Display,
    io::{self, Write},
};

mod utils;
pub use utils::{round2, round3, round_to_decimal_places};

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

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// number of times the training should run, 0 for initialize agents
    #[arg(short, long, default_value_t = 0)]
    count: u8,
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_systems(Startup, run)
        .run();

    // env_logger::init();

    // let args = Args::parse();

    // // HACK: just run this once and delete when all are preprocessed
    // if args.count == 99 {
    //     // XMLParser::resize_segmented_images(PathBuf::from("data/training/drafter_1"))?;
    //     // XMLParser::resize_segmented_images(PathBuf::from("data/training/drafter_2"))?;
    //     // XMLParser::resize_segmented_images(PathBuf::from("data/training/drafter_3"))?;
    //     return;
    // }

    // if args.count == 0 {
    //     training::train_agents(
    //         TrainingStage::Artificial { stage: 0 },
    //         None,
    //         "agents".to_string(),
    //     )
    //     .unwrap();
    //     // train first iteration of agents that every other tage builds upon
    // } else {
    //     std::fs::remove_dir_all("agents_trained".to_string()).unwrap_or_default();
    //     for i in 0..args.count {
    //         let load_path = if i == 0 {
    //             "agents".to_string()
    //         } else {
    //             format!("agents_trained/agents_stage_{}", i - 1)
    //         };
    //         let save_path = format!("agents_trained/agents_stage_{}", i);
    //         training::train_agents(
    //             TrainingStage::Artificial { stage: 0 },
    //             Some(load_path),
    //             save_path,
    //         )
    //         .unwrap();
    //     }
    // }
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

#[derive(Serialize, Deserialize, Debug)]
pub struct AdaptiveConfig {
    pub number_of_network_updates: usize,
    // not configurable
    pub neuron_lower: f64,
    // not configurable
    pub neuron_upper: f64,
    // not configurable
    pub retina_lower: f64,
    // not configurable
    pub retina_upper: f64,
    // not configurable
    pub init_zero_retina_weights: f32,
    pub initial_population_size: usize,
    pub max_generations: u64,
    // not configurable
    pub tournament_size: usize,
    pub variance: f64,
    pub variance_decay: f64,
    pub mean: f64,
    pub delete_neuron: f64,
    pub delete_weights: f64,
    pub delete_bias: f64,
    pub delete_self_activation: f64,
    pub mutate_neuron: f64,
    pub mutate_weights: f64,
    pub mutate_bias: f64,
    pub mutate_self_activation: f64,
}

impl AdaptiveConfig {
    fn new() -> Self {
        AdaptiveConfig {
            number_of_network_updates: CONFIG.neural_network.number_of_network_updates as usize,
            neuron_lower: CONFIG.neural_network.weight_bounds.neuron_lower as f64,
            neuron_upper: CONFIG.neural_network.weight_bounds.neuron_upper as f64,
            retina_lower: CONFIG.neural_network.weight_bounds.retina_lower as f64,
            retina_upper: CONFIG.neural_network.weight_bounds.retina_upper as f64,
            init_zero_retina_weights: CONFIG.neural_network.weight_bounds.init_zero_retina_weights
                as f32,
            initial_population_size: CONFIG.genetic_algorithm.initial_population_size as usize,
            max_generations: CONFIG.genetic_algorithm.max_generations as u64,
            tournament_size: CONFIG.genetic_algorithm.tournament_size as usize,
            variance: CONFIG.genetic_algorithm.mutation_rates.variance as f64,
            variance_decay: CONFIG.genetic_algorithm.mutation_rates.variance_decay as f64,
            mean: CONFIG.genetic_algorithm.mutation_rates.mean as f64,
            delete_neuron: CONFIG.genetic_algorithm.mutation_rates.delete_neuron as f64,
            delete_weights: CONFIG.genetic_algorithm.mutation_rates.delete_weights as f64,
            delete_bias: CONFIG.genetic_algorithm.mutation_rates.delete_bias as f64,
            delete_self_activation: CONFIG
                .genetic_algorithm
                .mutation_rates
                .delete_self_activation as f64,
            mutate_neuron: CONFIG.genetic_algorithm.mutation_rates.mutate_neuron as f64,
            mutate_weights: CONFIG.genetic_algorithm.mutation_rates.mutate_weights as f64,
            mutate_bias: CONFIG.genetic_algorithm.mutation_rates.mutate_bias as f64,
            mutate_self_activation: CONFIG
                .genetic_algorithm
                .mutation_rates
                .mutate_self_activation as f64,
        }
    }

    fn randomize(&mut self, rng: &mut dyn RngCore) {
        self.number_of_network_updates = rng.gen_range(20..=200);
        self.initial_population_size = rng.gen_range(50..=500);
        self.variance = rng.gen_range(0.0..=1.0);
        self.variance_decay = rng.gen_range(0.95..=0.99);
        self.mean = rng.gen_range(0.0..=1.0);
        self.delete_neuron = rng.gen_range(0.0..=1.0);
        self.delete_weights = rng.gen_range(0.0..=1.0);
        self.delete_bias = rng.gen_range(0.0..=1.0);
        self.delete_self_activation = rng.gen_range(0.0..=1.0);
        self.mutate_neuron = rng.gen_range(0.0..=1.0);
        self.mutate_weights = rng.gen_range(0.0..=1.0);
        self.mutate_bias = rng.gen_range(0.0..=1.0);
        self.mutate_self_activation = rng.gen_range(0.0..=1.0);
        self.tournament_size = rng.gen_range(2..self.initial_population_size);
        self.init_zero_retina_weights = rng.gen_range(0.1..=0.9);
    }
}

fn run(mut exit: EventWriter<AppExit>) {
    let max_iterations = 10000;
    let mut rng = ChaCha8Rng::from_entropy();
    let _ = std::fs::remove_dir_all("iterations");
    let _ = std::fs::remove_file("iteration_results.txt");
    let mut iteration = 0usize;
    let mut adaptive_config = AdaptiveConfig::new();
    while let Err(e) = training::train_agents(
        TrainingStage::Artificial { stage: 0 },
        None,
        "agents".to_string(),
        iteration,
        &adaptive_config,
        true,
    ) {
        if iteration >= max_iterations {
            break;
        }
        // tweak configuration
        adaptive_config.randomize(&mut rng);

        std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open("iteration_results.txt")
            .unwrap()
            .write_fmt(format_args!("iteration: {}\n", iteration))
            .unwrap();

        info(format!("iteration: {}", iteration));

        iteration += 1;
    }
    exit.send(AppExit::Success);
}
