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
pub use genetic_algorithm::{Agent, Population, AgentEvaluation};

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
    let mut rng = ChaCha8Rng::seed_from_u64(2);

    // intialize population
    let mut population = Population::new(
        &mut rng,
        CONFIG.genetic_algorithm.population_size as usize,
        CONFIG.neural_network.neurons_per_network as usize);
    // create an reader to buffer training dataset
    let image_reader = ImageReader::from_path(CONFIG.image_processing.path_to_training_data.to_string())?;

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
                        let parent1 = population.select_weighted(&mut rng);
                        let parent2 = population.select_weighted(&mut rng);
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

    println!("Stopped at generation {}", population.generation());
    
    // visualize the best agent as png image
    let best_agent = population.agents_mut().first_mut().unwrap();
    best_agent.genotype().short_term_memory().visualize("test/images/agents/best_agent.png".into())?;

    // save visualization as .dot file
    let graph = Graph::from(best_agent.genotype().clone());
    let dot = Dot::new(&graph);
    OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open("test/images/agents/best_agent.dot")?
        .write_fmt(format_args!("{:?}\n", dot))?;

    // save as json file in "saves/rnn/"
    best_agent.genotype_mut().to_json(Some(&"test/saves/rnn/best_agent.json".to_string()))?;

    Ok(())
}
