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

fn main() -> Result {
    env_logger::init();

    // train first iteration of agents that stay on top of black pixels
    training::train_agents(
        TrainingStage::Artificial{stage: 0},
        None,
        format!("agents"),
    )?;

    let agent_path = CONFIG.image_processing.path_to_agents_dir as &str;
    for i in 1..10 {
        let load_path = if i == 1 { agent_path.to_string() } else { format!("agents_stage_{}", i-1) };
        let save_path = format!("agents_stage_{}", i);
        // deleting folder at save_path
        std::fs::remove_dir_all(save_path.clone())?;
        training::train_agents(
            TrainingStage::Artificial{stage: 0},
            Some(load_path),
            save_path,
        )?;
    }

    Ok(())
}
