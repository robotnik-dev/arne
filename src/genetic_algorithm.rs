use std::collections::HashMap;
use std::os::unix::net;

use rand::prelude::*;
use serde::{Deserialize, Serialize};

use crate::config::image_processing::initial_retina_size;
use crate::image_processing::{Image, ImageLabel, Position};
use crate::neural_network::Rnn;
use crate::{Error, Retina, ShortTermMemory, CONFIG};

/// statisteics per Agent to store some data relevant for human statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Statistics {
    pub deleted_neurons: u32,
    pub deleted_weights: u32,
    pub deleted_biases: u32,
    pub deleted_self_activations: u32,
    pub mutated_neurons: u32,
    pub mutated_weights: u32,
    pub mutated_biases: u32,
    pub mutated_self_activations: u32,
    pub fitness: f32,
}

impl Statistics {
    pub fn new() -> Self {
        Statistics {
            deleted_neurons: 0,
            deleted_weights: 0,
            deleted_biases: 0,
            deleted_self_activations: 0,
            mutated_neurons: 0,
            mutated_weights: 0,
            mutated_biases: 0,
            mutated_self_activations: 0,
            fitness: 0.0,
        }
    }
}

pub enum SelectionMethod {
    Tournament,
    Weighted,
}

/// The actual mapping between the genotype and the phenotype.
/// Need to be implemented for each invdividual marker type like FollowLine
trait FitnessCalculation {
    fn calculate_fitness(&self) -> f32;
}

pub trait AgentEvaluation {
    fn evaluate(
        &mut self,
        rng: &mut dyn RngCore,
        label: ImageLabel,
        image: &mut Image,
        number_of_updates: usize,
    ) -> std::result::Result<f32, Error>;
}

/// Marker type. This phenotype/solution to the problem is a line follower.
/// If the agent/retina(TBD) can stay in each iteration step on the line(the center pixel of the image)
/// the higher the fitness value will be.
// struct FollowLine;

impl FitnessCalculation for Agent {
    fn calculate_fitness(&self) -> f32 {
        // all networks are evaluated at the same time

        // just an example hardcoded for now ..
        // fitness is high when
        // only one network regisers voltage source
        // and exactly two networks register a resistor
        // and every oher network registers nothing
        // voltage source is registered when neurons 4s output is 1.0 and neuron 5 is 0.0
        // resistor is registered when neurons 5s output is 1.0 and neuron 4 is 0.0
        // as mentioned is is a harcoded example, i need to find a way to pass this data to the agent

        let one_voltage_source = self.genotype().networks().iter().filter(|network| {
            network.neurons()[3].output() == 1.0
                && network.neurons()[4].output() == 0.0
        }).count() == 1;

        let two_resistors = self.genotype().networks().iter().filter(|network| {
            network.neurons()[4].output() == 1.0
                && network.neurons()[3].output() == 0.0
        }).count() == 2;

        if one_voltage_source && two_resistors {
            1.0
        } else if one_voltage_source || two_resistors {
            0.5
        } else {
            0.0
        }

        // fitness is high when:
        // let fitness_vec = [
        //     // - the neuron 4 has a high activation -> recognized as square (normalized to 0-1)
        //     (1.0 + self.genotype().neurons()[3].output()) / 2.0,
        //     // - lots of black pixels
        //     retina.get_data().iter().filter(|p| **p == 0.0).count() as f32
        //     / (retina.size() * retina.size()) as f32,
        //     // - the retina moved in the last time step
        //     retina.get_current_delta_position().normalized_len(),
        //     // - the retina is small
        //     1.0 - (retina.size() as f32 / CONFIG.image_processing.max_retina_size as f32),
        // ];
        // fitness_vec.iter().sum::<f32>() / fitness_vec.len() as f32
    }
}

impl AgentEvaluation for Agent {
    fn evaluate(
        &mut self,
        rng: &mut dyn RngCore,
        label: ImageLabel,
        image: &mut Image,
        number_of_updates: usize,
    ) -> std::result::Result<f32, Error> {
        let mut retinas = vec![];
        // initialize networks with retinas
        for _ in 0..self.genotype_mut().networks_mut().len() {
            let initial_retina_size = CONFIG.image_processing.initial_retina_size as usize;
            // create a retina at a random position
            let low_x = 0 + initial_retina_size as i32;
            let high_x = image.width() as i32 - initial_retina_size as i32;
            let low_y = 0 + initial_retina_size as i32;
            let high_y = image.height() as i32 - initial_retina_size as i32;
            
            let retina = image.create_retina_at(
                self.get_starting_position(rng, low_x, high_x, low_y, high_y),
                initial_retina_size,
            )?;
            retinas.push(retina);
        }
        self.clear_short_term_memories();
        
        let mut local_fitness = 0.0;
        for i in 0..number_of_updates {
            // for each of the network in the genotype
            for (network, retina) in self.genotype_mut().networks_mut().iter_mut().zip(retinas.iter_mut()) {

                // first location of the retina
                image.update_retina_movement(&retina);

                // calculate the next delta position of the retina, encoded in the neurons
                let delta = network.next_delta_position();
                let new_size = (retina.size() as f32 + network.next_size_factor()) as usize;

                // move the retina to the next position and scale up or down after the movement
                retina.move_mut(&delta, image);
                if retina.set_size(new_size, image).is_ok() {
                    network.update_retina_size(new_size);
                }

                // update all input connections to the retina from each neuron
                network.update_inputs_from_retina(&retina);

                // do one update step
                network.update();

                // save retina movement in buffer
                // image.update_retina_movement_mut(&retina);
                image.update_retina_movement(&retina);

                // creating snapshot of the network at the current time step
                let outputs = network
                    .neurons()
                    .iter()
                    .map(|neuron| neuron.output())
                    .collect::<Vec<f32>>();
                let time_step = (i + 1) as u32;
                network.add_snapshot(outputs, time_step);
            }
            // calculate the fitness of the genotype (all networks)
            local_fitness += self.calculate_fitness();
            // TODO: for statistics
            // self.update_fitness(fitness);
        }
        // save the image in the hashmap of the agent with label
        let image = image.clone();
        let genotype = self.genotype().clone();
        self.statistics_mut()
            .insert(label.clone(), (image, genotype));
        
        let fitness = local_fitness / number_of_updates as f32;
        Ok(fitness)
    }
}

pub struct Population {
    agents: Vec<Agent>,
    generation: u32,
}

impl Population {
    pub fn new(rng: &mut dyn RngCore, size: usize, networks_per_agent: usize, neurons_per_rnn: usize) -> Self {
        let agents = (0..size)
            .map(|_| Agent::new(rng, networks_per_agent, neurons_per_rnn))
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

    pub fn select(&self, rng: &mut dyn RngCore, method: SelectionMethod) -> (&Agent, &Agent) {
        match method {
            SelectionMethod::Tournament => self.select_tournament(rng),
            SelectionMethod::Weighted => self.select_weighted(rng),
        }
    }

    fn select_tournament(&self, rng: &mut dyn RngCore) -> (&Agent, &Agent) {
        let tournament_size = CONFIG.genetic_algorithm.tournament_size as usize;
        let mut tournament = Vec::with_capacity(tournament_size);
        for _ in 0..tournament_size {
            tournament.push(self.agents.choose(rng).unwrap());
        }
        tournament.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());
        (tournament[0], tournament[1])
    }

    fn select_weighted(&self, rng: &mut dyn RngCore) -> (&Agent, &Agent) {
        (
            self.agents
                .choose_weighted(rng, |agent| agent.fitness.max(0.000001))
                .unwrap(),
            self.agents
                .choose_weighted(rng, |agent| agent.fitness.max(0.000001))
                .unwrap(),
        )
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Genotype {
    networks: Vec<Rnn>,
}

impl Genotype {
    pub fn new(rng: &mut dyn RngCore, networks_per_agent: usize, number_of_neurons: usize) -> Self {
        let networks = (0..networks_per_agent)
            .map(|_| Rnn::new(rng, number_of_neurons))
            .collect();
        Genotype { networks }
    }

    pub fn networks(&self) -> &Vec<Rnn> {
        &self.networks
    }

    pub fn networks_mut(&mut self) -> &mut Vec<Rnn> {
        &mut self.networks
    }

    pub fn crossover_uniform(&self, rng: &mut dyn RngCore, with: &Genotype) -> Genotype {
        let networks = self
            .networks()
            .iter()
            .zip(with.networks())
            .map(|(a, b)| a.crossover_uniform(rng, b))
            .collect();
        Genotype { networks }
    }

    pub fn mutate(&mut self, rng: &mut dyn RngCore) {
        for network in self.networks_mut() {
            network.mutate(rng);
        }
    }

    pub fn clear_short_term_memories(&mut self) {
        for network in self.networks_mut() {
            network.short_term_memory_mut().clear();
        }
    }
}
/// Conductor to control all networks. The idea is to set off multiple RNNs per update step and to collect the results.
/// The maximum number of RNNs are set in the config file.
/// Each RNN has a set number of Neurons, namely 7. It can detect either a resitor, capacitor, voltage source or ground.
/// The fitness of the whole set of networks is determined instead of a single network.
pub struct Agent {
    fitness: f32,
    genotype: Genotype,
    // for statistics purposes, we store the final images with the retina movement and all the short term memories here
    pub statistics: HashMap<ImageLabel, (Image, Genotype)>,     // TODO: mutliple networks per agent
}

impl Clone for Agent {
    fn clone(&self) -> Self {
        Agent {
            fitness: self.fitness,
            genotype: self.genotype.clone(),
            statistics: self.statistics.clone(),
        }
    }
}

impl Agent {
    pub fn new(rng: &mut dyn RngCore, networks_per_agent: usize, number_of_neurons: usize) -> Self {
        Agent {
            fitness: 0.0,
            genotype: Genotype::new(rng, networks_per_agent, number_of_neurons),
            statistics: HashMap::new(),
        }
    }

    pub fn fitness(&self) -> f32 {
        self.fitness
    }

    pub fn set_fitness(&mut self, fitness: f32) {
        self.fitness = fitness;
    }

    pub fn genotype(&self) -> &Genotype {
        &self.genotype
    }

    pub fn genotype_mut(&mut self) -> &mut Genotype {
        &mut self.genotype
    }

    pub fn statistics(&self) -> &HashMap<ImageLabel, (Image, Genotype)> {
        &self.statistics
    }

    pub fn statistics_mut(&mut self) -> &mut HashMap<ImageLabel, (Image, Genotype)> {
        &mut self.statistics
    }

    pub fn crossover(&self, rng: &mut dyn RngCore, with: &Agent) -> Agent {
        let offspring_genotype = self.genotype.crossover_uniform(rng, &with.genotype);
        let mut new_agent = self.clone();
        new_agent.genotype = offspring_genotype;
        new_agent
    }

    pub fn mutate(&mut self, rng: &mut dyn RngCore) {
        self.genotype.mutate(rng);
    }

    pub fn get_starting_position(&self, rng: &mut dyn RngCore, low_x: i32, high_x: i32, low_y: i32, high_y: i32) -> Position {
        let random_x = rng.gen_range(low_x..=high_x);
        let random_y = rng.gen_range(low_y..=high_y);
        Position::new(random_x, random_y)
    }

    pub fn clear_short_term_memories(&mut self) {
        self.genotype.clear_short_term_memories();
    }
}

#[cfg(test)]
mod tests {
    use rand_chacha::ChaCha8Rng;

    use super::*;

    #[test]
    fn test_delete_neuron() {
        let mut rng = ChaCha8Rng::seed_from_u64(2);
        let mut rnn = Rnn::new(&mut rng, 3);
        let rnn2 = rnn.clone();
        rnn.delete_neuron(&mut rng);

        assert_ne!(rnn, rnn2);
    }

    #[test]
    fn test_delete_weights() {
        let mut rng = ChaCha8Rng::seed_from_u64(2);
        let mut rnn = Rnn::new(&mut rng, 3);
        let rnn2 = rnn.clone();
        rnn.delete_weights(&mut rng);

        assert_ne!(rnn, rnn2);
    }

    #[test]
    fn test_delete_bias() {
        let mut rng = ChaCha8Rng::seed_from_u64(2);
        let mut rnn = Rnn::new(&mut rng, 3);
        let rnn2 = rnn.clone();
        rnn.delete_bias(&mut rng);

        assert_ne!(rnn, rnn2);
    }

    #[test]
    fn test_delete_self_activation() {
        let mut rng = ChaCha8Rng::seed_from_u64(2);
        let mut rnn = Rnn::new(&mut rng, 3);
        let rnn2 = rnn.clone();
        rnn.delete_self_activation(&mut rng);

        assert_ne!(rnn, rnn2);
    }

    #[test]
    fn test_mutate_neuron() {
        let mut rng = ChaCha8Rng::seed_from_u64(2);
        let mut rnn = Rnn::new(&mut rng, 3);
        let rnn2 = rnn.clone();

        rnn.mutate_neuron(&mut rng);

        // Check that the properties of the neuron have been changed.
        assert_ne!(rnn, rnn2);
    }

    #[test]
    fn test_mutate_weights() {
        let mut rng = ChaCha8Rng::seed_from_u64(2);
        let mut rnn = Rnn::new(&mut rng, 3);
        let rnn2 = rnn.clone();

        rnn.mutate_weights(&mut rng);

        // Check that the properties of the neuron have been changed.
        assert_ne!(rnn, rnn2);
    }

    #[test]
    fn test_mutate_bias() {
        let mut rng = ChaCha8Rng::seed_from_u64(2);
        let mut rnn = Rnn::new(&mut rng, 3);
        let rnn2 = rnn.clone();

        rnn.mutate_bias(&mut rng);

        // Check that the properties of the neuron have been changed.
        assert_ne!(rnn, rnn2);
    }

    #[test]
    fn test_mutate_self_activation() {
        let mut rng = ChaCha8Rng::seed_from_u64(2);
        let mut rnn = Rnn::new(&mut rng, 3);
        let rnn2 = rnn.clone();

        rnn.mutate_self_activation(&mut rng);

        // Check that the properties of the neuron have been changed.
        assert_ne!(rnn, rnn2);
    }
}
