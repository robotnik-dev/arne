use std::{fs::OpenOptions, io::Write};
use rayon::prelude::*;
use rand::prelude::*;
use petgraph::{dot::Dot, Graph};
use rand_chacha::ChaCha8Rng;
use static_toml::static_toml;

mod utils;
pub use utils::round2;

mod image_processing;
pub use image_processing::{Retina, ImageReader};

mod neural_network;
pub use neural_network::{Rnn, SnapShot, ShortTermMemory};

mod genetic_algorithm;
pub use genetic_algorithm::{Agent, Population, AgentEvaluation, SelectionMethod};

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

    // creating all necessary directories
    std::fs::create_dir_all("test/best_agents")?;

    // let mut rng = ChaCha8Rng::seed_from_u64(4);
    let mut rng = ChaCha8Rng::from_entropy();

    // intialize population
    let mut population = Population::new(
        &mut rng,
        CONFIG.genetic_algorithm.population_size as usize,
        CONFIG.neural_network.neurons_per_network as usize);
    // create an reader to buffer training dataset
    let image_reader = ImageReader::from_path("images/training".to_string())?;
    // loop until stop criterial is met
    loop {
        // for each image in the dataset
        for index in 0..image_reader.images().len() {
            // load image
            let image = image_reader.get_image(index)?;
            // evaluate the fitness of each individual of the population
            population
                .agents_mut()
                .par_iter_mut()
                .for_each(|agent| {
                    let fitness = agent
                        .evaluate(&mut image.clone(), CONFIG.neural_network.number_of_network_updates as usize)
                        .unwrap();
                    agent.set_fitness(fitness);
                });
            // sort the population by fitness
            population.agents_mut().sort_by(|a, b|b.fitness().partial_cmp(&a.fitness()).unwrap());
            
            // select, crossover and mutate
            let new_agents = (0..population.agents().len())
                    .map(|_| {
                        let (parent1, parent2) = population.select(&mut rng, SelectionMethod::Tournament);
                        let mut offspring = parent1.crossover(&mut rng, parent2);
                        offspring.mutate(&mut rng);
                        offspring
                    })
                    .collect::<Vec<Agent>>();
            
            // evolve the population
            population.evolve(new_agents);
        }
                
        // check stop criteria
        if population.generation() >= CONFIG.genetic_algorithm.max_generations as u32
        {
            break;
        }
    }
    // sort the population by fitness
    // population.agents_mut().sort_by(|a, b|a.fitness().partial_cmp(&b.fitness()).unwrap());

    // the best 30 agents should be saved with:
    // - their fitness
    // - the rnn as json file
    // - the rnn as png image
    // - the rnn as dot file
    // - the movement of the retina over time TODO
    population
        .agents_mut()
        .par_iter_mut()
        .enumerate()
        .inspect(|(index, agent)| {
            println!("agent {} fitness: {}", index, agent.fitness());
        })
        .take(30)
        .for_each(|(index, agent)| {
            println!("generating files for agent {} ...", index);
            // create a new directory for every agent with the index as name if the directory does not exist
            std::fs::create_dir_all(format!("test/best_agents/{}", index)).unwrap();
            
            agent.genotype().short_term_memory().visualize(format!("test/best_agents/{}/rnn_updates.png", index)).unwrap();
            agent.genotype_mut().to_json(Some(&format!("test/best_agents/{}/rnn.json", index))).unwrap();
            agent.image.save(format!("test/best_agents/{}/retina_movement.png", index)).unwrap();

            let graph = Graph::from(agent.genotype().clone());
            let dot = Dot::new(&graph);
            OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open(format!("test/best_agents/{}/viz.dot", index)).unwrap()
                .write_fmt(format_args!("{:?}\n", dot)).unwrap();
        });
    Ok(())
}
