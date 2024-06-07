use std::{fs::OpenOptions, io::Write};
use rayon::prelude::*;
use rand::prelude::*;
use petgraph::{dot::Dot, Graph};
use rand_chacha::ChaCha8Rng;

mod utils;
pub use utils::round2;

mod image_processing;
pub use image_processing::{Retina, ImageReader};

mod neural_network;
pub use neural_network::{Rnn, SnapShot, ShortTermMemory, NEURONS_PER_RNN, NUMBER_OF_RNN_UPDATES};

mod genetic_algorithm;
pub use genetic_algorithm::{Agent, Population, POPULATION_SIZE, MAX_GENERATIONS, AgentEvaluation, FollowLine};

type Error = Box<dyn std::error::Error>;
type Result = std::result::Result<(), Error>;

const PATH_TO_TRAINING_DATASET: &str = "test/images/dataset";

// Stuff to change and experiment with:
// - crossover method is uniform, try other methods
// - selection method is roulette wheel, try other methods
// - mutation chances
// - number of neurons in the RNN
// - Population size

fn main() -> Result {
    let mut rng = ChaCha8Rng::seed_from_u64(2);

    // intialize population
    let mut population = Population::new(&mut rng, POPULATION_SIZE, NEURONS_PER_RNN);
    
    // create an reader to buffer training dataset
    let image_reader = ImageReader::from_path(PATH_TO_TRAINING_DATASET.to_string())?;

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
                    let fitness = agent.evaluate(&mut image.clone(), NUMBER_OF_RNN_UPDATES);
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
        if population.generation() >= MAX_GENERATIONS
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
