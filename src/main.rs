use std::{fs::OpenOptions, io::Write};
use approx::AbsDiffEq;
use rayon::prelude::*;
use rand::prelude::*;
use petgraph::{dot::Dot, Graph};
use rand_chacha::ChaCha8Rng;

mod utils;
pub use utils::round2;

mod image_processing;
pub use image_processing::Retina;

mod neural_network;
pub use neural_network::{Rnn, SnapShot, ShortTermMemory, NEURONS_PER_RNN, NUMBER_OF_RNN_UPDATES};

mod genetic_algorithm;
pub use genetic_algorithm::{Agent, Population, POPULATION_SIZE, MAX_GENERATIONS, AgentEvaluation, SimpleGrayscale, GREYSCALE_TO_MATCH};

type Error = Box<dyn std::error::Error>;
type Result = std::result::Result<(), Error>;


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

    // loop until stop criterial is met
    loop {
        // evaluate the fitness of each individual of the population
        population
            .agents_mut()
            .par_iter_mut()
            .for_each(|agent| {
                let fitness = agent.evaluate(GREYSCALE_TO_MATCH, NUMBER_OF_RNN_UPDATES);
                agent.set_fitness(fitness);
            });
        
        // sort the population by fitness
        population.agents_mut().sort_by(|a, b|b.fitness().partial_cmp(&a.fitness()).unwrap());
        
        // check stop criteria
        if population.generation() >= MAX_GENERATIONS || population.agents().iter().any(|agent| agent.fitness().abs_diff_eq(&1.0, 0.01) )
        {
            break;
        }

        let new_agents = (0..population.agents().len())
            .map(|_| {
                let parent1 = population.select_weighted(&mut rng);
                let parent2 = population.select_weighted(&mut rng);
                let mut offspring = parent1.crossover(&mut rng, parent2);
                offspring.mutate(&mut rng);
                offspring
            })
            .collect::<Vec<Agent>>();
            
        population.evolve(new_agents);
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
        .open("test/images/agents/best_agent.dot")?
        .write_fmt(format_args!("{:?}\n", dot))?;

    // save as json file in "saves/rnn/"
    best_agent.genotype_mut().to_json(None)?;

    Ok(())
}
