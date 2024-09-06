use bevy::prelude::*;
use clap::Parser;
use image::TrainingStage;
pub use rand_chacha::ChaCha8Rng;
use static_toml::static_toml;
pub use std::time::{Duration, Instant};

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
        .add_plugins(MinimalPlugins)
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

fn run(mut exit: EventWriter<AppExit>) {
    training::train_agents(
        TrainingStage::Artificial { stage: 0 },
        None,
        "agents".to_string(),
    )
    .unwrap();
    exit.send(AppExit::Success);
}
