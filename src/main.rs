use static_toml::static_toml;
pub use rand_chacha::ChaCha8Rng;

mod utils;
pub use utils::round2;

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

    training::train_agents()?;

    Ok(())
}
