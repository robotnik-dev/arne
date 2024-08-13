use approx::AbsDiffEq;
use indicatif::ProgressBar;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::ffi::OsStr;
use std::fs::{self, OpenOptions};
use std::io::Read;
use std::path::PathBuf;

use crate::image_processing::{Image, ImageDescription, ImageLabel, Position, Retina};
use crate::netlist::{ComponentBuilder, ComponentType, Generate, Netlist, Node, NodeType};
use crate::neural_network::Rnn;
use crate::{Error, CONFIG};

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
    pub variance: f32,
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
            variance: 0.0,
        }
    }
}

pub enum SelectionMethod {
    Tournament,
    Weighted,
}

pub trait AgentEvaluation {
    fn evaluate(
        &mut self,
        fitness_function: fn(
            agent: &mut Agent,
            description: ImageDescription,
            retinas: Vec<Retina>,
        ) -> f32,
        rng: &mut dyn RngCore,
        label: ImageLabel,
        image: &mut Image,
        desciption: ImageDescription,
        number_of_updates: usize,
    ) -> std::result::Result<f32, Error>;
}

impl AgentEvaluation for Agent {
    fn evaluate(
        &mut self,
        fitness_function: fn(
            agent: &mut Agent,
            description: ImageDescription,
            retinas: Vec<Retina>,
        ) -> f32,
        rng: &mut dyn RngCore,
        label: ImageLabel,
        image: &mut Image,
        desciption: ImageDescription,
        number_of_updates: usize,
    ) -> std::result::Result<f32, Error> {
        let mut retinas = vec![];
        // initialize networks with retinas
        for i in 0..self.genotype_mut().networks_mut().len() {
            let initial_retina_size = CONFIG.image_processing.initial_retina_size as usize;
            // create a retina at a random position
            let top_left = Position::new(initial_retina_size as i32, initial_retina_size as i32);
            let bottom_right = Position::new(
                image.width() as i32 - initial_retina_size as i32,
                image.height() as i32 - initial_retina_size as i32,
            );

            let _random_position = Position::random(rng, top_left, bottom_right);
            let _image_center_position =
                Position::new((image.width() / 2) as i32, (image.height() / 2) as i32);
            let retina = image.create_retina_at(
                _random_position,
                initial_retina_size,
                CONFIG.image_processing.superpixel_size as usize,
                i.to_string(),
            )?;
            retinas.push(retina);
        }
        self.clear_short_term_memories();

        let mut local_fitness = 0.0;
        for i in 0..number_of_updates {
            // for each of the network in the genotype
            for (network, retina) in self
                .genotype_mut()
                .networks_mut()
                .iter_mut()
                .zip(retinas.iter_mut())
            {
                // first location of the retina
                image.update_retina_movement(retina);

                // calculate the next delta position of the retina, encoded in the neurons
                let delta = network.next_delta_position();
                // let new_size = (retina.size() as f32 + network.next_size_factor()) as usize;

                // move the retina to the next position and scale up or down after the movement
                retina.move_mut(&delta, image);
                // if retina.set_size(new_size, image).is_ok() {
                //     network.update_retina_size(new_size);
                // }

                // update all input connections to the retina from each neuron
                network.update_inputs_from_retina(retina);

                // do one update step
                network.update();

                // save retina movement in buffer
                // image.update_retina_movement_mut(&retina);
                image.update_retina_movement(retina);

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
            local_fitness += fitness_function(self, desciption.clone(), retinas.clone());
        }
        // save the image in the hashmap of the agent with label
        let image = image.clone();
        let genotype = self.genotype().clone();
        self.statistics_mut()
            .insert(label.clone(), (image, genotype, String::new()));

        let fitness = local_fitness / number_of_updates as f32;
        Ok(fitness)
    }
}

pub struct Population {
    agents: Vec<Agent>,
    generation: u32,
}

impl Population {
    pub fn new(
        progress_bar: &ProgressBar,
        size: usize,
        networks_per_agent: usize,
        neurons_per_rnn: usize,
    ) -> Self {
        let agents = (0..size)
            .into_par_iter()
            .map(|_| {
                progress_bar.inc(1);
                Agent::new(networks_per_agent, neurons_per_rnn)
            })
            .collect();
        Population {
            agents,
            generation: 0,
        }
    }

    /// load agents from a path to work with these as the first generation
    pub fn from_path(path: String) -> std::result::Result<Self, Error> {
        let mut agents = vec![];
        fs::read_dir(path).unwrap().for_each(|entry| {
            // entry => the actual agent index 0, 1, etc
            fs::read_dir(entry.unwrap().path())
                .unwrap()
                .for_each(|img_folder| {
                    // the folder named "final" is the one we have to use
                    let path_buf = img_folder.unwrap().path();
                    if path_buf.file_name().unwrap().to_str().unwrap() == "final" {
                        let agent = Agent::from_path(path_buf).unwrap();
                        agents.push(agent);
                    }
                })
        });

        let population = Population {
            agents,
            generation: 0,
        };
        Ok(population)
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

impl Generate for Genotype {
    fn generate(&self) -> String {
        let resistor_neuron_idx = 4usize;
        let capacitor_neuron_idx = 5usize;
        let source_dc_neuron_idx = 3usize;
        let in_node_neuron_idx = 6usize;
        let out_node_neuron_idx = 6usize;

        let resistor_networks = self
            .networks()
            .iter()
            .filter(|&network| {
                network.neurons()[resistor_neuron_idx]
                    .output()
                    .abs_diff_eq(&1.0, 0.01)
                    && network.neurons()[capacitor_neuron_idx]
                        .output()
                        .abs_diff_eq(&0.0, 0.01)
                    && network.neurons()[source_dc_neuron_idx]
                        .output()
                        .abs_diff_eq(&0.0, 0.01)
            })
            .cloned()
            .collect::<Vec<Rnn>>();

        let capacitor_networks = self
            .networks()
            .iter()
            .filter(|&network| {
                network.neurons()[resistor_neuron_idx]
                    .output()
                    .abs_diff_eq(&0.0, 0.01)
                    && network.neurons()[capacitor_neuron_idx]
                        .output()
                        .abs_diff_eq(&1.0, 0.01)
                    && network.neurons()[source_dc_neuron_idx]
                        .output()
                        .abs_diff_eq(&0.0, 0.01)
            })
            .cloned()
            .collect::<Vec<Rnn>>();

        let source_dc_networks = self
            .networks()
            .iter()
            .filter(|&network| {
                network.neurons()[resistor_neuron_idx]
                    .output()
                    .abs_diff_eq(&0.0, 0.01)
                    && network.neurons()[capacitor_neuron_idx]
                        .output()
                        .abs_diff_eq(&0.0, 0.01)
                    && network.neurons()[source_dc_neuron_idx]
                        .output()
                        .abs_diff_eq(&1.0, 0.01)
            })
            .cloned()
            .collect::<Vec<Rnn>>();

        // remove identical networks (comparison of the nodes)
        // only keep the ones that have unique outputs of in_nodes and out_nodes
        let resistor_networks = resistor_networks
            .iter()
            .filter(|&network| {
                let in_node_output = network.neurons()[in_node_neuron_idx].output();
                let out_node_output = network.neurons()[out_node_neuron_idx].output();
                resistor_networks.iter().all(|other_network| {
                    let other_in_node_output = other_network.neurons()[in_node_neuron_idx].output();
                    let other_out_node_output =
                        other_network.neurons()[out_node_neuron_idx].output();
                    in_node_output.abs_diff_eq(&other_in_node_output, 0.01)
                        && out_node_output.abs_diff_eq(&other_out_node_output, 0.01)
                })
            })
            .cloned()
            .collect::<Vec<Rnn>>();

        let capacitor_networks = capacitor_networks
            .iter()
            .filter(|&network| {
                let in_node_output = network.neurons()[in_node_neuron_idx].output();
                let out_node_output = network.neurons()[out_node_neuron_idx].output();
                capacitor_networks.iter().all(|other_network| {
                    let other_in_node_output = other_network.neurons()[in_node_neuron_idx].output();
                    let other_out_node_output =
                        other_network.neurons()[out_node_neuron_idx].output();
                    in_node_output.abs_diff_eq(&other_in_node_output, 0.01)
                        && out_node_output.abs_diff_eq(&other_out_node_output, 0.01)
                })
            })
            .cloned()
            .collect::<Vec<Rnn>>();

        let source_dc_networks = source_dc_networks
            .iter()
            .filter(|&network| {
                let in_node_output = network.neurons()[in_node_neuron_idx].output();
                let out_node_output = network.neurons()[out_node_neuron_idx].output();
                source_dc_networks.iter().all(|other_network| {
                    let other_in_node_output = other_network.neurons()[in_node_neuron_idx].output();
                    let other_out_node_output =
                        other_network.neurons()[out_node_neuron_idx].output();
                    in_node_output.abs_diff_eq(&other_in_node_output, 0.01)
                        && out_node_output.abs_diff_eq(&other_out_node_output, 0.01)
                })
            })
            .cloned()
            .collect::<Vec<Rnn>>();

        // creating netlist
        let mut netlist = Netlist::new();

        // generate components
        resistor_networks
            .iter()
            .enumerate()
            .for_each(|(i, network)| {
                let mut component =
                    ComponentBuilder::new(ComponentType::Resistor, i.to_string()).build();
                let in_node = network.neurons()[in_node_neuron_idx].output().abs()
                    * CONFIG.genetic_algorithm.node_range as f32;
                let out_node = network.neurons()[out_node_neuron_idx].output().abs()
                    * CONFIG.genetic_algorithm.node_range as f32;
                component.add_node(Node(in_node as u32), NodeType::In);
                component.add_node(Node(out_node as u32), NodeType::Out);
                let _ = netlist.add_component(
                    component.clone(),
                    format!("{}{}", component.symbol, component.name),
                );
            });
        capacitor_networks
            .iter()
            .enumerate()
            .for_each(|(i, network)| {
                let mut component =
                    ComponentBuilder::new(ComponentType::Capacitor, i.to_string()).build();
                let in_node = network.neurons()[in_node_neuron_idx].output().abs()
                    * CONFIG.genetic_algorithm.node_range as f32;
                let out_node = network.neurons()[out_node_neuron_idx].output().abs()
                    * CONFIG.genetic_algorithm.node_range as f32;
                component.add_node(Node(in_node as u32), NodeType::In);
                component.add_node(Node(out_node as u32), NodeType::Out);
                let _ = netlist.add_component(
                    component.clone(),
                    format!("{}{}", component.symbol, component.name),
                );
            });
        source_dc_networks
            .iter()
            .enumerate()
            .for_each(|(i, network)| {
                let mut component =
                    ComponentBuilder::new(ComponentType::VoltageSourceDc, i.to_string()).build();
                // the in_node is always ground
                let in_node = 0.0f32;
                let out_node = network.neurons()[out_node_neuron_idx].output().abs()
                    * CONFIG.genetic_algorithm.node_range as f32;
                component.add_node(Node(in_node as u32), NodeType::In);
                component.add_node(Node(out_node as u32), NodeType::Out);
                let _ = netlist.add_component(
                    component.clone(),
                    format!("{}{}", component.symbol, component.name),
                );
            });

        netlist.generate()
    }
}

/// Conductor to control all networks. The idea is to set off multiple RNNs per update step and to collect the results.
/// The maximum number of RNNs are set in the config file.
/// Each RNN has a set number of Neurons. It can detect either a resitor, capacitor or voltage source for now.
/// The fitness of the whole set of networks is determined instead of a single network.
pub struct Agent {
    fitness: f32,
    genotype: Genotype,
    /// String -> netlist string
    pub statistics: HashMap<ImageLabel, (Image, Genotype, String)>,
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
    pub fn new(networks_per_agent: usize, number_of_neurons: usize) -> Self {
        let mut rng = ChaCha8Rng::from_entropy();
        Agent {
            fitness: 0.0,
            genotype: Genotype::new(&mut rng, networks_per_agent, number_of_neurons),
            statistics: HashMap::new(),
        }
    }

    /// builds a new Aegnt from a multiple json files located at 'path'
    pub fn from_path(path: PathBuf) -> std::result::Result<Self, Error> {
        let mut rnns = vec![];
        fs::read_dir(path.clone()).unwrap().for_each(|entry| {
            let network_path = entry.unwrap().path();
            if network_path.is_dir() {
                fs::read_dir(network_path.clone())
                    .unwrap()
                    .for_each(|file| {
                        let file_path = file.unwrap().path();
                        if Some(OsStr::new("network.json")) == file_path.file_name() {
                            let mut file = OpenOptions::new().read(true).open(file_path).unwrap();
                            let mut buffer = String::new();
                            file.read_to_string(&mut buffer).unwrap();
                            let rnn: Rnn = serde_json::from_str(buffer.trim()).unwrap();
                            rnns.push(rnn);
                        }
                    })
            }
        });
        let genotype = Genotype { networks: rnns };
        Ok(Agent {
            fitness: 0.0,
            genotype,
            statistics: HashMap::new(),
        })
    }

    /// adds the fitness to the current fitness of the agent
    pub fn add_to_fitness(&mut self, fitness: f32) {
        self.fitness += fitness;
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

    pub fn statistics(&self) -> &HashMap<ImageLabel, (Image, Genotype, String)> {
        &self.statistics
    }

    pub fn statistics_mut(&mut self) -> &mut HashMap<ImageLabel, (Image, Genotype, String)> {
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

    pub fn get_current_variance(&self) -> f32 {
        self.genotype
            .networks()
            .iter()
            .map(|network| network.variance())
            .sum::<f32>()
            / self.genotype.networks().len() as f32
    }

    pub fn update_variance(&mut self, variance: f32) {
        self.genotype_mut()
            .networks_mut()
            .iter_mut()
            .for_each(|network| {
                network.update_variance(variance);
            });
    }

    pub fn get_starting_position(
        &self,
        rng: &mut dyn RngCore,
        low_x: i32,
        high_x: i32,
        low_y: i32,
        high_y: i32,
    ) -> Position {
        let random_x = rng.gen_range(low_x..=high_x);
        let random_y = rng.gen_range(low_y..=high_y);
        Position::new(random_x, random_y)
    }

    pub fn clear_short_term_memories(&mut self) {
        self.genotype.clear_short_term_memories();
    }

    pub fn debug_get_neuron_outputs(&self) -> Vec<f32> {
        self.genotype
            .networks()
            .iter()
            .flat_map(|network| network.neurons().iter().map(|neuron| neuron.output()))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use rand_chacha::ChaCha8Rng;

    use super::*;

    #[test]
    fn test_delete_neuron() {
        let mut rng = ChaCha8Rng::from_entropy();
        let mut rnn = Rnn::new(&mut rng, 3);
        let rnn2 = rnn.clone();
        rnn.delete_neuron(&mut rng);

        assert_ne!(rnn, rnn2);
    }

    #[test]
    fn test_delete_weights() {
        let mut rng = ChaCha8Rng::from_entropy();
        let mut rnn = Rnn::new(&mut rng, 3);
        let rnn2 = rnn.clone();
        rnn.delete_weights(&mut rng);

        assert_ne!(rnn, rnn2);
    }

    #[test]
    fn test_delete_bias() {
        let mut rng = ChaCha8Rng::from_entropy();
        let mut rnn = Rnn::new(&mut rng, 3);
        let rnn2 = rnn.clone();
        rnn.delete_bias(&mut rng);

        assert_ne!(rnn, rnn2);
    }

    #[test]
    fn test_delete_self_activation() {
        let mut rng = ChaCha8Rng::from_entropy();
        let mut rnn = Rnn::new(&mut rng, 3);
        let rnn2 = rnn.clone();
        rnn.delete_self_activation(&mut rng);

        assert_ne!(rnn, rnn2);
    }

    #[test]
    fn test_mutate_neuron() {
        let mut rng = ChaCha8Rng::from_entropy();
        let mut rnn = Rnn::new(&mut rng, 3);
        let rnn2 = rnn.clone();

        rnn.mutate_neuron(&mut rng);

        // Check that the properties of the neuron have been changed.
        assert_ne!(rnn, rnn2);
    }

    #[test]
    fn test_mutate_weights() {
        let mut rng = ChaCha8Rng::from_entropy();
        let mut rnn = Rnn::new(&mut rng, 3);
        let rnn2 = rnn.clone();

        rnn.mutate_weights(&mut rng);

        // Check that the properties of the neuron have been changed.
        assert_ne!(rnn, rnn2);
    }

    #[test]
    fn test_mutate_bias() {
        let mut rng = ChaCha8Rng::from_entropy();
        let mut rnn = Rnn::new(&mut rng, 3);
        let rnn2 = rnn.clone();

        rnn.mutate_bias(&mut rng);

        // Check that the properties of the neuron have been changed.
        assert_ne!(rnn, rnn2);
    }

    #[test]
    fn test_mutate_self_activation() {
        let mut rng = ChaCha8Rng::from_entropy();
        let mut rnn = Rnn::new(&mut rng, 3);
        let rnn2 = rnn.clone();

        rnn.mutate_self_activation(&mut rng);

        // Check that the properties of the neuron have been changed.
        assert_ne!(rnn, rnn2);
    }
}
