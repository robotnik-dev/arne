use std::{collections::HashMap, fmt::Debug};
use rand::prelude::*;
use lazy_static::lazy_static;

use crate::neural_network::Rnn;

pub const POPULATION_SIZE: usize = 100;
pub const MAX_GENERATIONS: u32 = 10;
pub const GREYSCALE_TO_MATCH: SimpleGrayscale = SimpleGrayscale(255);
lazy_static! {
    pub static ref MUTATION_PROBABILITIES: HashMap<String, f32> = {
        let mut m = HashMap::new();
        // global mutation rate which can be changed later
        m.insert("global_variance".to_string(), 0.2);
        // setting all incoming weights, self activation and bias to 0
        m.insert("delete_neuron".to_string(), 0.1);
        // setting all incoming weights to 0
        m.insert("delete_weights".to_string(), 0.1);
        // setting bias to 0
        m.insert("delete_bias".to_string(), 0.1);
        // setting self activation to 0
        m.insert("delete_self_activation".to_string(), 0.1);
        // randomize the weights self activation and bias
        m.insert("mutate_neuron".to_string(), 0.2);
        // randomize all incoming weights
        m.insert("mutate_weights".to_string(), 0.2);
        // randomize the bias
        m.insert("mutate_bias".to_string(), 0.1);
        // randomize the self activation
        m.insert("mutate_self_activation".to_string(), 0.1);
        m
    };
}


trait Phenotype {}

#[derive(Debug, Clone)]
pub struct SimpleGrayscale(u8);

impl Phenotype for SimpleGrayscale {}

impl GenotypePhenotypeMapping<SimpleGrayscale> for Rnn {
    fn map_to_phenotype(&self) -> SimpleGrayscale {
        let num_neurons = 3;
        let greyscale = self
            .neurons()
            .iter()
            .skip(self.neurons().len() - num_neurons)
            // output of 0 should be 0 and output of 1 should be 255
            .map(|neuron| (neuron.output() * 255.0) as f64)
            // take the avarage of the outcome
            .sum::<f64>() / num_neurons as f64;
        SimpleGrayscale(greyscale.round() as u8)
    }
}

/// This phenotype/solution to the problem is a line follower.
/// If the agent/retina(TBD) can stay in each iteration step on the line(the center pixel of the image)
/// the higher the fitness value will be.
#[derive(Debug, Clone)]
struct FollowLine {
    // store only the difference between the current position and the position
    // of the last iteration
    delta_position: nalgebra::Point2<f32>,
}

impl Phenotype for FollowLine {}

trait GenotypePhenotypeMapping<P: Phenotype> {
    fn map_to_phenotype(&self) -> P;
}

pub trait AgentEvaluation<T> {
    /// normalized fitness value between 0 and 1
    fn calculate_fitness(&self, data: T) -> f64;

    /// evaluate the agent with the given preferred output
    fn evaluate(&mut self, data: T, number_of_updates: usize) -> f64;
}


pub struct Population {
    agents: Vec<Agent>,
    generation: u32,
}

impl Population {
    pub fn new(rng: &mut dyn RngCore, size: usize, neurons_per_rnn: usize) -> Self {
        let agents = (0..size)
            .map(|_| Agent::new(rng, neurons_per_rnn))
            .collect();
        Population {
            agents,
            generation: 0,
        }
    }

    pub fn agents(&self) -> &Vec<Agent> {
        &self.agents
    }

    pub fn agents_mut(&mut self) -> &mut Vec<Agent> {
        &mut self.agents
    }

    pub fn generation(&self) -> u32 {
        self.generation
    }

    pub fn evolve(&mut self, new_agents: Vec<Agent>) {
        self.agents = new_agents;
        self.generation += 1;
    }

    /// roulette wheel selection
    pub fn select_weighted(&self, rng: &mut dyn RngCore) -> &Agent {
        self.agents.choose_weighted(rng, |agent| agent.fitness.max(0.000001)).unwrap()
    }

    /// tournament selection
    fn select_tournament(&self, rng: &mut dyn RngCore) -> &Agent {
        todo!()
    }
}


pub struct Agent {
    fitness: f64,
    genotype: Rnn,
}

impl Clone for Agent {
    fn clone(&self) -> Self {
        Agent {
            fitness: self.fitness,
            genotype: self.genotype.clone(),
        }
    }
}

impl AgentEvaluation<SimpleGrayscale> for Agent {
    fn calculate_fitness(&self, data: SimpleGrayscale) -> f64 {
        let correct_greyscale = data.0 as f64;
        let phenotype = self.genotype.map_to_phenotype().0 as f64;
        1.0 - (correct_greyscale - phenotype).abs() / 255.0
    }

    fn evaluate(&mut self, data: SimpleGrayscale, number_of_updates: usize) -> f64 {
        let mut local_fitness = 0.0;
        self.genotype_mut().short_term_memory_mut().clear();
        for i in 0..number_of_updates {
            // get the current position of the network (encoded in the output of some neurons TBD)
            // getting the current Retina from the image at teh position provided
            // update all input connections to the retina from each neuron

            // do one update step
            self.genotype.update();
            // creating snapshot of the network at the current time step
            let outputs = self.genotype.neurons().iter().map(|neuron| neuron.output()).collect::<Vec<f64>>();
            self.genotype_mut().add_snapshot(outputs, (i + 1) as u32);
            // CLONING here is okay because its only a u8, but for other implementations it might be a problem
            local_fitness += self.calculate_fitness(data.clone());
        }
        local_fitness / number_of_updates as f64
    }
}

impl Agent {
    pub fn new(rng: &mut dyn RngCore, number_of_neurons: usize) -> Self {
        Agent {
            fitness: 0.0,
            genotype: Rnn::new(rng, number_of_neurons),
        }
    }

    pub fn fitness(&self) -> f64 {
        self.fitness
    }

    pub fn set_fitness(&mut self, fitness: f64) {
        self.fitness = fitness;
    }

    pub fn genotype(&self) -> &Rnn {
        &self.genotype
    }

    pub fn genotype_mut(&mut self) -> &mut Rnn {
        &mut self.genotype
    }

    pub fn crossover(&self, rng: &mut dyn RngCore, with: &Agent) -> Agent {
        let offspring = self.genotype.crossover_uniform(rng, &with.genotype);
        Agent::from(offspring)
    }

    pub fn mutate(&mut self, rng: &mut dyn RngCore) {
        self.genotype.mutate(rng);
    }
}

impl From<Rnn> for Agent {
    fn from(rnn: Rnn) -> Self {
        Agent {
            fitness: 0.0,
            genotype: rnn,
        }
    }
}


#[cfg(test)]
mod tests {
    use petgraph::{dot::Dot, Graph};
    use rand_chacha::ChaCha8Rng;

    use super::*;

    #[test]
    fn test_crossover_uniform() {
        let mut rng = ChaCha8Rng::seed_from_u64(2);
        let mut agent = Agent::new(&mut rng, 10);
        let mut agent2 = Agent::new(&mut rng, 10);

        agent
            .genotype_mut()
            .neurons_mut()
            .iter_mut()
            .for_each(|neuron|{
                neuron
                    .input_connections_mut()
                    .iter_mut()
                    .for_each(|(_, weight)| *weight = 0.5);
                neuron.set_self_activation(0.1);
                neuron.set_bias(1.);
            });
        
        agent2
            .genotype_mut()
            .neurons_mut()
            .iter_mut()
            .for_each(|neuron|{
                neuron
                    .input_connections_mut()
                    .iter_mut()
                    .for_each(|(_, weight)| *weight = -0.5);
                neuron.set_self_activation(-0.1);
                neuron.set_bias(-1.);
            });

        let offspring = agent.crossover(&mut rng, &agent2);
        
        // check if the offspring is different from the parents
        assert_ne!(agent.genotype(), offspring.genotype());
        assert_ne!(agent2.genotype(), offspring.genotype());

        // print the parents and then the offspring as graph
        let parent1_graph = Graph::from(agent.genotype.clone());
        let dot1 = Dot::new(&parent1_graph);
        let parent2_graph = Graph::from(agent2.genotype.clone());
        let dot2 = Dot::new(&parent2_graph);
        let offspring_graph = Graph::from(offspring.genotype.clone());
        let dot3 = Dot::new(&offspring_graph);
        println!("Parent1 \n {:?}", dot1);
        println!("Parent2 \n {:?}", dot2);
        println!("Offspring \n {:?}", dot3);

        // check if the number count of all negative numbers in the offsrping are approximately the saame as the psotive numbers
        let negative_count = offspring
            .genotype()
            .neurons()
            .iter()
            .map(|neuron| neuron.input_connections().iter().filter(|(_, weight)| *weight < 0.0).count())
            .sum::<usize>();
        let positive_count = offspring
            .genotype()
            .neurons()
            .iter()
            .map(|neuron| neuron.input_connections().iter().filter(|(_, weight)| *weight > 0.0).count())
            .sum::<usize>();
        
        assert_eq!(positive_count, 43);
        assert_eq!(negative_count, 47);
    }

    #[test]
    fn test_delete_neuron() {
        let mut rng = ChaCha8Rng::seed_from_u64(2);
        let mut rnn = Rnn::new(&mut rng, 3);

        rnn.delete_neuron(&mut rng);

        assert_eq!(rnn.neurons()[0].input_connections().iter().map(|(_, weight)| *weight).sum::<f64>(), 0.0);
        assert_eq!(rnn.neurons()[0].self_activation(), 0.0);
        assert_eq!(rnn.neurons()[0].bias(), 0.0);
    }

    #[test]
    fn test_delete_weights() {
        let mut rng = ChaCha8Rng::seed_from_u64(2);
        let mut rnn = Rnn::new(&mut rng, 3);

        rnn.delete_weights(&mut rng);

        assert_eq!(rnn.neurons()[0].input_connections().iter().map(|(_, weight)| *weight).sum::<f64>(), 0.0);
    }

    #[test]
    fn test_delete_bias() {
        let mut rng = ChaCha8Rng::seed_from_u64(2);
        let mut rnn = Rnn::new(&mut rng, 3);

        rnn.delete_bias(&mut rng);

        assert_eq!(rnn.neurons()[0].bias(), 0.0);
    }

    #[test]
    fn test_delete_self_activation() {
        let mut rng = ChaCha8Rng::seed_from_u64(2);
        let mut rnn = Rnn::new(&mut rng, 3);

        rnn.delete_self_activation(&mut rng);

        assert_eq!(rnn.neurons()[0].self_activation(), 0.0);
    }

    #[test]
    fn test_mutate_neuron() {
        let mut rng = ChaCha8Rng::seed_from_u64(2);
        let mut rnn = Rnn::new(&mut rng, 3);
        let bias = rnn.neurons()[0].bias();
        let self_activation = rnn.neurons()[0].self_activation();
        let weights = rnn.neurons()[0].input_connections().iter().map(|(_, weight)| *weight).collect::<Vec<f64>>();

        rnn.mutate_neuron(&mut rng);

        // Check that the properties of the neuron have been changed.
        assert_ne!(bias, rnn.neurons()[0].bias());
        assert_ne!(self_activation, rnn.neurons()[0].self_activation());
        assert_ne!(weights, rnn.neurons()[0].input_connections().iter().map(|(_, weight)| *weight).collect::<Vec<f64>>());
    }

    #[test]
    fn test_mutate_weights() {
        let mut rng = ChaCha8Rng::seed_from_u64(2);
        let mut rnn = Rnn::new(&mut rng, 3);
        let weights = rnn.neurons()[0].input_connections().iter().map(|(_, weight)| *weight).collect::<Vec<f64>>();

        rnn.mutate_weights(&mut rng);

        // Check that the properties of the neuron have been changed.
        assert_ne!(weights, rnn.neurons()[0].input_connections().iter().map(|(_, weight)| *weight).collect::<Vec<f64>>());
    }

    #[test]
    fn test_mutate_bias() {
        let mut rng = ChaCha8Rng::seed_from_u64(2);
        let mut rnn = Rnn::new(&mut rng, 3);
        let bias = rnn.neurons()[0].bias();

        rnn.mutate_bias(&mut rng);

        // Check that the properties of the neuron have been changed.
        assert_ne!(bias, rnn.neurons()[0].bias());
    }

    #[test]
    fn test_mutate_self_activation() {
        let mut rng = ChaCha8Rng::seed_from_u64(2);
        let mut rnn = Rnn::new(&mut rng, 3);
        let self_activation = rnn.neurons()[0].self_activation();

        rnn.mutate_self_activation(&mut rng);

        // Check that the properties of the neuron have been changed.
        assert_ne!(self_activation, rnn.neurons()[0].self_activation());
    }
}