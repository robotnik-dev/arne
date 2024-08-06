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

    let agent_path = CONFIG.image_processing.path_to_agents_dir as &str;

    training::train_agents(
        TrainingStage::Artificial,
        Some(agent_path.to_string()),
        String::from("agents_after_phase_1"),
    )?;

    Ok(())
}
