use rand::prelude::*;

use crate::{Error, Retina, CONFIG};
use crate::neural_network::Rnn;
use crate::image_processing::{Image, Position};


/// The actual mapping between the genotype and the phenotype.
/// Need to be implemented for each invdividual marker type like FollowLine
trait FitnessCalculation<T> {
    fn calculate_fitness(&self, retina: &Retina) -> f64;
}
    
pub trait AgentEvaluation {
    fn evaluate(&mut self, image: &mut Image, number_of_updates: usize) -> std::result::Result<f64, Error>;
}

/// Marker type. This phenotype/solution to the problem is a line follower.
/// If the agent/retina(TBD) can stay in each iteration step on the line(the center pixel of the image)
/// the higher the fitness value will be.
struct FollowLine;

impl FitnessCalculation<FollowLine> for Agent {
    fn calculate_fitness(&self, retina: &Retina) -> f64 {
        // the higher te difference between the center pixel to 1.0(white) the higher the fitness
        // means a dark pixel in the center is good
        1.0 - retina.get_value(2, 2) as f64
    }
}

impl AgentEvaluation for Agent {
    fn evaluate(&mut self, image: &mut Image, number_of_updates: usize) -> std::result::Result<f64, Error> {
        let mut local_fitness = 0.0;
        self.genotype_mut().short_term_memory_mut().clear();
        // create a retina at a specific position (top left of image)
        let mut retina = image.create_retina_at(Position::new(
            CONFIG.image_processing.retina_size as i32,
            CONFIG.image_processing.retina_size as i32 as i32),
            CONFIG.image_processing.retina_size as usize)?;

        for i in 0..number_of_updates {
            // calculate the next delta position of the retina, encoded in the neurons
            let delta = self.genotype().next_delta_position();

            // move the retina to the next position
            retina.move_mut(&delta);

            // update all input connections to the retina from each neuron
            self.genotype_mut().update_inputs_from_retina(&retina);

            // do one update step
            self.genotype.update();

            // save as png in a folder with retina movement. For each agent a new folder is created
            // TODO
            
            // creating snapshot of the network at the current time step
            let outputs = self.genotype.neurons().iter().map(|neuron| neuron.output()).collect::<Vec<f64>>();
            let time_step = (i + 1) as u32;
            self.genotype_mut().add_snapshot(outputs, time_step);

            // calculate the fitness of the agent
            local_fitness += self.calculate_fitness(&retina);
        }
        Ok(local_fitness / number_of_updates as f64)
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