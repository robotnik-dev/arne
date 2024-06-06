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
