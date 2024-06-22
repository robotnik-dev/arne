use indicatif::ProgressBar;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use static_toml::static_toml;

mod utils;
pub use utils::round2;

mod image_processing;
pub use image_processing::{ImageReader, Retina};

mod neural_network;
pub use neural_network::{Rnn, ShortTermMemory, SnapShot};

mod genetic_algorithm;
pub use genetic_algorithm::{Agent, AgentEvaluation, Population, SelectionMethod};

type Error = Box<dyn std::error::Error>;
type Result = std::result::Result<(), Error>;

static_toml! {
    pub static CONFIG = include_toml!("config.toml");
}

fn main() -> Result {
    env_logger::init();

    log::info!("loading config variables");

    let max_generations = CONFIG.genetic_algorithm.max_generations as u64;
    let seed = CONFIG.genetic_algorithm.seed as u64;
    let with_seed = CONFIG.genetic_algorithm.with_seed;
    let path_to_training_data = CONFIG.image_processing.path_to_training_data as &str;
    let neurons_per_network = CONFIG.neural_network.neurons_per_network as usize;
    let population_size = CONFIG.genetic_algorithm.population_size as usize;
    let number_of_network_updates = CONFIG.neural_network.number_of_network_updates as usize;
    let take_agents = CONFIG.genetic_algorithm.take_agents as usize;
    let path_to_agents_dir = CONFIG.image_processing.path_to_agents_dir as &str;

    log::info!("setting up rng");

    let mut rng = ChaCha8Rng::from_entropy();

    if with_seed {
        log::info!("using seed: {}", seed);
        rng = ChaCha8Rng::seed_from_u64(seed);
    }

    log::info!("initializing population...");

    // intialize population
    let mut population = Population::new(&mut rng, population_size, neurons_per_network);

    log::info!("loading training dataset...");

    // create a reader to buffer training dataset
    let image_reader = ImageReader::from_path(path_to_training_data.to_string())?;

    let algorithm_bar = ProgressBar::new(max_generations);

    // loop until stop criterial is met
    log::info!("starting genetic algorithm");
    for _ in 0..max_generations {
        algorithm_bar.inc(1);
        // for each image in the dataset
        for index in 0..image_reader.images().len() {
            // load image
            let (label, image) = image_reader.get_image(index)?;

            // evaluate the fitness of each individual of the population
            population.agents_mut().par_iter_mut().for_each(|agent| {
                let fitness = agent
                    .evaluate(label.clone(), &mut image.clone(), number_of_network_updates)
                    .unwrap();
                agent.set_fitness(fitness);
            });
        }
        // sort the population by fitness
        population
            .agents_mut()
            .sort_by(|a, b| b.fitness().partial_cmp(&a.fitness()).unwrap());

        // select, crossover and mutate
        let new_agents = (0..population.agents().len())
            .map(|_| {
                let (parent1, parent2) =
                    population.select(&mut rng, SelectionMethod::Tournament);
                let mut offspring = parent1.crossover(&mut rng, parent2);
                offspring.mutate(&mut rng);
                offspring
            })
            .collect::<Vec<Agent>>();

        // evolve the population
        population.evolve(new_agents);
    }
    algorithm_bar.finish();
    log::info!("genetic algorithm finished");
    log::info!("stopped after {} generations", population.generation());
    log::info!("generating files for the best {} agents...", take_agents);

    // remove 'best_agents' directory if it exists
    std::fs::remove_dir_all(path_to_agents_dir).unwrap_or_default();

    let generating_files_bar = ProgressBar::new(take_agents as u64);
    population
        .agents_mut()
        .par_iter_mut()
        .enumerate()
        .inspect(|(index, agent)| {
            log::debug!("agent {} fitness: {}", index, agent.fitness());
        })
        .take(take_agents)
        .for_each(|(index, agent)| {
            agent
                .statistics_mut()
                .par_iter_mut()
                .for_each(|(label, (image, stm, rnn))| {
                    std::fs::create_dir_all(format!("{}/{}/{}", path_to_agents_dir, index, label))
                        .unwrap();
                    // image
                    //     .save_upscaled(format!(
                    //         "{}/{}/{}/retina.png",
                    //         path_to_agents_dir, index, label
                    //     ))
                    //     .unwrap();
                    stm.visualize(format!(
                        "{}/{}/{}/memory.png",
                        path_to_agents_dir, index, label
                    ))
                    .unwrap();
                    rnn.to_json(format!(
                        "{}/{}/{}/rnn.json",
                        path_to_agents_dir, index, label
                    ))
                    .unwrap();
                    rnn.to_dot(format!(
                        "{}/{}/{}/rnn.dot",
                        path_to_agents_dir, index, label
                    ))
                    .unwrap();
                });
            generating_files_bar.inc(1);
        });
    generating_files_bar.finish();

    Ok(())
}
