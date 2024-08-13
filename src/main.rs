use clap::Parser;
use image_processing::TrainingStage;
pub use rand_chacha::ChaCha8Rng;
use static_toml::static_toml;

mod utils;
pub use utils::{round2, round_to_decimal_places};

mod image_processing;
pub use image_processing::{ImageReader, Retina};

mod neural_network;
pub use neural_network::{Rnn, ShortTermMemory, SnapShot};

mod genetic_algorithm;
pub use genetic_algorithm::{Agent, AgentEvaluation, Population, SelectionMethod};

mod netlist;
mod training;

type Error = Box<dyn std::error::Error>;
type Result = std::result::Result<(), Error>;

static_toml! {
    pub static CONFIG = include_toml!("config.toml");
}

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// number of times the training should run, 0 for initilize agents_init
    #[arg(short, long, default_value_t = 0)]
    count: u8,
}

fn main() -> Result {
    env_logger::init();

    let args = Args::parse();

    if args.count == 0 {
        // train first iteration of agents that every other tage builds upon
        training::train_agents(
            TrainingStage::Artificial { stage: 0 },
            None,
            format!("agents_init"),
        )?;
    } else {
        std::fs::remove_dir_all(format!("agents_trained")).unwrap_or_default();
        for i in 0..args.count {
            let load_path = if i == 0 {
                format!("agents_init")
            } else {
                format!("agents_trained/agents_stage_{}", i - 1)
            };
            let save_path = format!("agents_trained/agents_stage_{}", i);
            training::train_agents(
                TrainingStage::Artificial { stage: 1 },
                Some(load_path),
                save_path,
            )?;
        }
    }
    Ok(())
}
