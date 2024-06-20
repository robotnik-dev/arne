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

// Stuff to change and experiment with:
// - number of generations
// - TODO: change global variance over time (decrease)
// - crossover method is uniform, try other methods
// - selection method is roulette wheel, try other methods
// - mutation chances
// - number of neurons in the RNN
// - Population size

fn main() -> Result {
    env_logger::init();

    let mut rng = ChaCha8Rng::from_entropy();

    if CONFIG.genetic_algorithm.with_seed {
        log::info!("using seed: {}", CONFIG.genetic_algorithm.seed);
        rng = ChaCha8Rng::seed_from_u64(CONFIG.genetic_algorithm.seed as u64);
    }

    // intialize population
    let mut population = Population::new(
        &mut rng,
        CONFIG.genetic_algorithm.population_size as usize,
        CONFIG.neural_network.neurons_per_network as usize,
    );
    // create a reader to buffer training dataset
    let image_reader =
        ImageReader::from_path(CONFIG.image_processing.path_to_training_data.to_string())?;

    let algorithm_bar = ProgressBar::new(CONFIG.genetic_algorithm.max_generations as u64);

    // loop until stop criterial is met
    log::info!("starting genetic algorithm");
    loop {
        // for each image in the dataset
        for index in 0..image_reader.images().len() {
            // load image
            let image = image_reader.get_image(index)?;
            // evaluate the fitness of each individual of the population
            population.agents_mut().par_iter_mut().for_each(|agent| {
                let fitness = agent
                    .evaluate(
                        &mut image.clone(),
                        CONFIG.neural_network.number_of_network_updates as usize,
                    )
                    .unwrap();
                agent.set_fitness(fitness);
            });
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

        // check stop criteria
        // max generations
        if population.generation() >= CONFIG.genetic_algorithm.max_generations as u32 {
            break;
        }
        algorithm_bar.inc(1);
    }
    algorithm_bar.finish();
    log::info!("genetic algorithm finished");
    log::info!("stopped after {} generations", population.generation());
    log::info!("generating files for agents...");

    // the best 30 agents should be saved with:
    // - their fitness
    // - the rnn as json file
    // - the rnn as png image
    // - the rnn as dot file
    // - the movement of the retina over time

    // remove 'best_agents' directory if it exists
    std::fs::remove_dir_all(CONFIG.image_processing.path_to_agents_dir).unwrap_or_default();

    let generating_files_bar = ProgressBar::new(CONFIG.genetic_algorithm.take_agents as u64);
    population
        .agents_mut()
        .par_iter_mut()
        .enumerate()
        .inspect(|(index, agent)| {
            log::debug!("agent {} fitness: {}", index, agent.fitness());
        })
        .take(CONFIG.genetic_algorithm.take_agents as usize)
        .for_each(|(index, agent)| {
            // create a new directory for every agent with the index as name if the directory does not exist
            std::fs::create_dir_all(format!(
                "{}/{}",
                CONFIG.image_processing.path_to_agents_dir, index
            ))
            .unwrap();

            // save statistics per generation
            agent
                .statistics_mut()
                .par_iter_mut()
                .enumerate()
                .take(
                    CONFIG
                        .genetic_algorithm
                        .generate_images_for_first_generations as usize,
                )
                .for_each(|(i, (image, memory, rnn))| {
                    std::fs::create_dir_all(format!(
                        "{}/{}/gen_{}",
                        CONFIG.image_processing.path_to_agents_dir, index, i
                    ))
                    .unwrap();

                    memory
                        .visualize(format!(
                            "{}/{}/gen_{}/memory.png",
                            CONFIG.image_processing.path_to_agents_dir, index, i
                        ))
                        .unwrap();
                    image
                        .save_upscaled(format!(
                            "{}/{}/gen_{}/retina.png",
                            CONFIG.image_processing.path_to_agents_dir, index, i
                        ))
                        .unwrap();
                    rnn.to_json(Some(&format!(
                        "{}/{}/gen_{}/rnn.json",
                        CONFIG.image_processing.path_to_agents_dir, index, i
                    )))
                    .unwrap();
                    rnn.to_dot(format!(
                        "{}/{}/gen_{}/rnn.dot",
                        CONFIG.image_processing.path_to_agents_dir, index, i
                    ))
                    .unwrap();
                });

            // save the last image and memory of the final generation
            if let Some((image, memory, rnn)) = agent.statistics_mut().iter_mut().last() {
                std::fs::create_dir_all(format!(
                    "{}/{}/final",
                    CONFIG.image_processing.path_to_agents_dir, index
                ))
                .unwrap();
                memory
                    .visualize(format!(
                        "{}/{}/final/memory.png",
                        CONFIG.image_processing.path_to_agents_dir, index
                    ))
                    .unwrap();
                image
                    .save_upscaled(format!(
                        "{}/{}/final/retina.png",
                        CONFIG.image_processing.path_to_agents_dir, index
                    ))
                    .unwrap();
                rnn.to_json(Some(&format!(
                    "{}/{}/final/rnn.json",
                    CONFIG.image_processing.path_to_agents_dir, index
                )))
                .unwrap();
                rnn.to_dot(format!(
                    "{}/{}/final/rnn.dot",
                    CONFIG.image_processing.path_to_agents_dir, index
                ))
                .unwrap();
            };

            generating_files_bar.inc(1);
        });
    generating_files_bar.finish();

    Ok(())
}
