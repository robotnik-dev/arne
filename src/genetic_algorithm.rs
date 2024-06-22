use std::collections::HashMap;

use rand::prelude::*;
use serde::{Deserialize, Serialize};

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
        }
    }
}

pub enum SelectionMethod {
    Tournament,
    Weighted,
}

/// The actual mapping between the genotype and the phenotype.
/// Need to be implemented for each invdividual marker type like FollowLine
trait FitnessCalculation<T> {
    fn calculate_fitness(&self, retina: &Retina) -> f32;
}

pub trait AgentEvaluation {
    fn evaluate(
        &mut self,
        label: ImageLabel,
        image: &mut Image,
        number_of_updates: usize,
    ) -> std::result::Result<f32, Error>;
}

/// Marker type. This phenotype/solution to the problem is a line follower.
/// If the agent/retina(TBD) can stay in each iteration step on the line(the center pixel of the image)
/// the higher the fitness value will be.
struct FollowLine;

impl FitnessCalculation<FollowLine> for Agent {
    fn calculate_fitness(&self, retina: &Retina) -> f32 {
        // if the neuron 3 has a high activation and the retina center value is a black pixel, than the fitness is high
        // and the retina moved in the last time step. and NOT going back in the same direction
        let neuron_3 = self.genotype().neurons()[2].output();
        let retina_center_value = retina.get_center_value();
        let moved = retina.get_delta_position().len() / CONFIG.neural_network.movement_scale as f32;
        
        (retina.binarized_white() - retina_center_value + neuron_3 + moved) / 3.0
    }
}

impl AgentEvaluation for Agent {
    fn evaluate(
        &mut self,
        label: ImageLabel,
        image: &mut Image,
        number_of_updates: usize,
    ) -> std::result::Result<f32, Error> {
        let mut local_fitness = 0.0;
        self.genotype_mut().short_term_memory_mut().clear();
        // create a retina at a specific position (top left of image)
        // let offset = CONFIG.image_processing.retina_size as i32 / 2 + 1;
        // let the agents start at the left side in the middle
        let offset_x = CONFIG.image_processing.retina_size as i32 / 2 + 1;
        let offset_y = CONFIG.image_processing.retina_size as i32 / 2
            + CONFIG.image_processing.image_height as i32 / 2
            - 1;

        let mut retina = image.create_retina_at(
            Position::new(offset_x, offset_y),
            CONFIG.image_processing.retina_size as usize,
        )?;

        // first location of the retina
        image.update_retina_movement(&retina);

        for i in 0..number_of_updates {
            // calculate the next delta position of the retina, encoded in the neurons
            let delta = self.genotype().next_delta_position();

            // move the retina to the next position
            retina.move_mut(&delta, &image);

            // update all input connections to the retina from each neuron
            self.genotype_mut().update_inputs_from_retina(&retina);

            // do one update step
            self.genotype.update();

            // save retina movement in buffer
            // image.update_retina_movement_mut(&retina);
            image.update_retina_movement(&retina);

            // creating snapshot of the network at the current time step
            let outputs = self
                .genotype
                .neurons()
                .iter()
                .map(|neuron| neuron.output())
                .collect::<Vec<f32>>();
            let time_step = (i + 1) as u32;
            self.genotype_mut().add_snapshot(outputs, time_step);

            // calculate the fitness of the agent
            local_fitness += self.calculate_fitness(&retina);
        }
        // save the image in the hashmap of the agent with label
        let image = image.clone();
        let stm = self.genotype().short_term_memory().clone();
        let rnn = self.genotype().clone();
        self.statistics_mut()
            .insert(label.clone(), (image, stm, rnn));

        Ok(local_fitness / number_of_updates as f32)
    }
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

pub struct Agent {
    fitness: f32,
    genotype: Rnn,
    // for statistics purposes, we store the final images with the retina movement and all the short term memories here
    pub statistics: HashMap<ImageLabel, (Image, ShortTermMemory, Rnn)>,
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
    pub fn new(rng: &mut dyn RngCore, number_of_neurons: usize) -> Self {
        Agent {
            fitness: 0.0,
            genotype: Rnn::new(rng, number_of_neurons),
            statistics: HashMap::new(),
        }
    }

    pub fn fitness(&self) -> f32 {
        self.fitness
    }

    pub fn set_fitness(&mut self, fitness: f32) {
        self.fitness = fitness;
    }

    pub fn genotype(&self) -> &Rnn {
        &self.genotype
    }

    pub fn genotype_mut(&mut self) -> &mut Rnn {
        &mut self.genotype
    }

    pub fn statistics(&self) -> &HashMap<ImageLabel, (Image, ShortTermMemory, Rnn)> {
        &self.statistics
    }

    pub fn statistics_mut(&mut self) -> &mut HashMap<ImageLabel, (Image, ShortTermMemory, Rnn)> {
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
}

#[cfg(test)]
mod tests {
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
            .for_each(|neuron| {
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
            .for_each(|neuron| {
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

        // check if the number count of all negative numbers in the offsrping are approximately the saame as the psotive numbers
        let negative_count = offspring
            .genotype()
            .neurons()
            .iter()
            .map(|neuron| {
                neuron
                    .input_connections()
                    .iter()
                    .filter(|(_, weight)| *weight < 0.0)
                    .count()
            })
            .sum::<usize>();
        let positive_count = offspring
            .genotype()
            .neurons()
            .iter()
            .map(|neuron| {
                neuron
                    .input_connections()
                    .iter()
                    .filter(|(_, weight)| *weight > 0.0)
                    .count()
            })
            .sum::<usize>();

        assert_eq!(positive_count, 40);
        assert_eq!(negative_count, 50);
    }

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
