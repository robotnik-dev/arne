use indicatif::ProgressBar;
use petgraph::{dot::Dot, Graph};
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use static_toml::static_toml;
use std::{fs::OpenOptions, io::Write};

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

    // creating all necessary directories
    std::fs::create_dir_all("test/best_agents")?;

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
            std::fs::create_dir_all(format!("test/best_agents/{}", index)).unwrap();

            agent
                .genotype()
                .short_term_memory()
                .visualize(format!("test/best_agents/{}/rnn_updates.png", index))
                .unwrap();
            agent
                .genotype_mut()
                .to_json(Some(&format!("test/best_agents/{}/rnn.json", index)))
                .unwrap();
            agent
                .image
                .save_upscaled(format!("test/best_agents/{}/retina_movement.png", index))
                .unwrap();

            let graph = Graph::from(agent.genotype().clone());
            let dot = Dot::new(&graph);
            OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open(format!("test/best_agents/{}/viz.dot", index))
                .unwrap()
                .write_fmt(format_args!("{:?}\n", dot))
                .unwrap();
            generating_files_bar.inc(1);
        });
    generating_files_bar.finish();

    Ok(())
}
