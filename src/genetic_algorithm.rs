use crate::annotations::Annotation;
use crate::image::{Image, ImageLabel, Position, Retina};
use crate::netlist::{ComponentBuilder, ComponentType, Generate, Netlist};
use crate::neural_network::Rnn;
use crate::{training, AdaptiveConfig, Error};
use bevy::prelude::*;
use bevy_prng::WyRand;
use bevy_rand::prelude::EntropyComponent;
use bevy_rand::traits::{ForkableInnerRng, ForkableRng};
use itertools::Itertools;
use rand::seq::SliceRandom;
use rand::Rng;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::ffi::OsStr;
use std::fs::{self, OpenOptions};
use std::io::Read;
use std::path::PathBuf;

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

// impl AgentEvaluation for Agent {
//     fn evaluate(
//         &mut self,
//         fitness_function: fn(
//             agent: &mut Agent,
//             annotation: &Annotation,
//             retina: &Retina,
//             image: &Image,
//         ) -> f32,
//         rng: EntropyComponent<WyRand>,
//         image: &mut Image,
//         annotation: &Annotation,
//         number_of_updates: usize,
//     ) -> std::result::Result<f32, Error> {
//         // initialize retina
//         let retina_size = CONFIG.image_processing.retina_size as usize;
//         // create a retina at a random position
//         let top_left = Position::new(retina_size as i32, retina_size as i32);
//         let bottom_right = Position::new(
//             image.width() as i32 - retina_size as i32,
//             image.height() as i32 - retina_size as i32,
//         );

//         let _random_position = Position::random(rng, top_left.clone(), bottom_right);
//         let _image_center_position =
//             Position::new((image.width() / 2) as i32, (image.height() / 2) as i32);

//         let mut retina = image.create_retina_at(
//             self.retina_start_pos.clone(),
//             retina_size,
//             CONFIG.image_processing.superpixel_size as usize,
//             "".to_string(),
//         )?;

//         self.clear_short_term_memories();

//         // initial network update and snapshot
//         self.genotype_mut()
//             .networks_mut()
//             .iter_mut()
//             .for_each(|network| {
//                 network.update_inputs_from_retina(&retina);
//                 network.update();
//             });
//         let control_outputs = self
//             .genotype()
//             .control_network()
//             .neurons()
//             .iter()
//             .map(|neuron| neuron.output())
//             .collect::<Vec<f32>>();
//         let categorize_outputs = self
//             .genotype()
//             .categorize_network()
//             .neurons()
//             .iter()
//             .map(|neuron| neuron.output())
//             .collect::<Vec<f32>>();
//         let time_step = (0) as u32;
//         self.genotype_mut()
//             .control_network_mut()
//             .add_snapshot(control_outputs, time_step);
//         self.genotype_mut()
//             .categorize_network_mut()
//             .add_snapshot(categorize_outputs, time_step);
//         let mut local_fitness = 0.0;
//         for i in 0..number_of_updates {
//             // first location of the retina
//             image.update_retina_movement(&retina);

//             // update the list of visited dark pixels
//             retina.update_positions_visited();

//             // calculate the next delta position of the retina, encoded in the neurons
//             let delta = self.genotype().control_network().next_delta_position();

//             // move the retina to the next position
//             // here gets the data updated the retina sees
//             retina.move_mut(&delta, image);

//             self.genotype_mut()
//                 .networks_mut()
//                 .iter_mut()
//                 .for_each(|network| {
//                     // update all super pixel input connections to each neuron
//                     network.update_inputs_from_retina(&retina);
//                     // let sum = network
//                     //     .neurons()
//                     //     .iter()
//                     //     .fold(0.0, |acc, n| acc + n.output());
//                     // info(format!("sum before: {}", sum));
//                     // do one update step
//                     network.update();
//                     // let sum = network
//                     //     .neurons()
//                     //     .iter()
//                     //     .fold(0.0, |acc, n| acc + n.output());
//                     // info(format!("sum after: {}", sum));
//                 });

//             // self.genotype_mut()
//             //     .control_network_mut()
//             //     .update_inputs_from_retina(&retina);
//             // self.genotype_mut().control_network_mut().update();

//             // self.genotype_mut()
//             //     .categorize_network_mut()
//             //     .update_inputs_from_retina(&retina);
//             // self.genotype_mut().categorize_network_mut().update();

//             // save retina movement in buffer
//             image.update_retina_movement(&retina);

//             // creating snapshot of the network at the current time step
//             let control_outputs = self
//                 .genotype()
//                 .control_network()
//                 .neurons()
//                 .iter()
//                 .map(|neuron| neuron.output())
//                 .collect::<Vec<f32>>();
//             let categorize_outputs = self
//                 .genotype()
//                 .categorize_network()
//                 .neurons()
//                 .iter()
//                 .map(|neuron| neuron.output())
//                 .collect::<Vec<f32>>();
//             let time_step = (i + 1) as u32;
//             self.genotype_mut()
//                 .control_network_mut()
//                 .add_snapshot(control_outputs, time_step);
//             self.genotype_mut()
//                 .categorize_network_mut()
//                 .add_snapshot(categorize_outputs, time_step);

//             // calculate the fitness of the genotype
//             local_fitness += fitness_function(self, annotation, &retina, &image);
//         }
//         // save the image in the hashmap of the agent with label
//         let image = image.clone();
//         let genotype = self.genotype().clone();
//         self.statistics_mut().insert(
//             ImageLabel(annotation.filename.replace(".jpg", "").clone()),
//             (image, genotype, String::new()),
//         );

//         let fitness = local_fitness / number_of_updates as f32;
//         Ok(fitness)
//     }
// }

pub struct Population {
    agents: Vec<Agent>,
    generation: u32,
}

impl Population {
    // pub fn new(size: usize, adaptive_config: &Res<AdaptiveConfig>) -> Self {
    //     let agents = (0..size)
    //         .into_par_iter()
    //         .map(|_| Agent::new(adaptive_config))
    //         .collect();
    //     Population {
    //         agents,
    //         generation: 0,
    //     }
    // }

    /// load agents from a path to work with these as the first generation
    pub fn from_path(
        path: String,
        adaptive_config: &Res<AdaptiveConfig>,
    ) -> std::result::Result<Self, Error> {
        let mut agents = vec![];
        fs::read_dir(path).unwrap().for_each(|entry| {
            let path_buf = entry.unwrap().path();
            if !path_buf
                .file_name()
                .unwrap()
                .to_string_lossy()
                .starts_with("best_agent")
            {
                let agent = Agent::from_path(path_buf, adaptive_config).unwrap();
                agents.push(agent);
            }
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

    /// recombinate the population. Adds all new agents to the list and the drop the worst ones until population is the correct size again
    pub fn evolve(&mut self, new_agents: Vec<Agent>, population_size: usize) {
        let mut combined = self
            .agents
            .iter()
            .cloned()
            .chain(new_agents.iter().cloned())
            .collect::<Vec<Agent>>();
        combined.sort_by(|a, b| b.fitness().partial_cmp(&a.fitness()).unwrap());
        // combined.shuffle(rng);
        let new_population = combined
            .iter()
            .cloned()
            .take(population_size)
            // resets the fitness
            .map(|mut a| {
                a.set_fitness(0.0f32);
                a
            })
            .collect::<Vec<Agent>>();
        self.agents = new_population;
        self.generation += 1;
    }

    pub fn select(
        &self,
        rng: EntropyComponent<WyRand>,
        method: SelectionMethod,
        tournament_size: Option<usize>,
    ) -> (&Agent, &Agent) {
        match method {
            SelectionMethod::Tournament => self.select_tournament(rng, tournament_size),
            SelectionMethod::Weighted => self.select_weighted(rng),
        }
    }

    fn select_tournament(
        &self,
        mut rng: EntropyComponent<WyRand>,
        tournament_size: Option<usize>,
    ) -> (&Agent, &Agent) {
        let tournament_size = tournament_size.unwrap_or_default();
        let mut tournament = Vec::with_capacity(tournament_size);
        for _ in 0..tournament_size {
            tournament.push(self.agents.choose(&mut rng.fork_inner()).unwrap());
        }
        tournament.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());
        (tournament[0], tournament[1])
    }

    fn select_weighted(&self, mut rng: EntropyComponent<WyRand>) -> (&Agent, &Agent) {
        (
            self.agents
                .choose_weighted(&mut rng.fork_inner(), |agent| agent.fitness.max(0.000001))
                .unwrap(),
            self.agents
                .choose_weighted(&mut rng.fork_inner(), |agent| agent.fitness.max(0.000001))
                .unwrap(),
        )
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Genotype {
    /// the first is the control network and second the categorize network
    networks: Vec<Rnn>,
    #[serde(skip)]
    found_components: Vec<(Position, ComponentType)>,
}

impl Default for Genotype {
    fn default() -> Self {
        Self {
            networks: vec![],
            found_components: vec![],
        }
    }
}

impl Genotype {
    pub fn init(mut rng: EntropyComponent<WyRand>, adaptive_config: &Res<AdaptiveConfig>) -> Self {
        let control_network = Rnn::new(
            rng.fork_rng(),
            adaptive_config.control_network_neurons as usize,
            adaptive_config,
        );
        let categorize_network = Rnn::new(
            rng.fork_rng(),
            adaptive_config.categorize_network_neurons as usize,
            adaptive_config,
        );
        Genotype {
            networks: vec![control_network, categorize_network],
            found_components: vec![],
        }
    }

    pub fn networks(&self) -> &Vec<Rnn> {
        &self.networks
    }

    pub fn networks_mut(&mut self) -> &mut Vec<Rnn> {
        &mut self.networks
    }

    pub fn control_network(&self) -> &Rnn {
        &self.networks[0]
    }

    pub fn control_network_mut(&mut self) -> &mut Rnn {
        &mut self.networks[0]
    }

    pub fn categorize_network(&self) -> &Rnn {
        &self.networks[1]
    }

    pub fn categorize_network_mut(&mut self) -> &mut Rnn {
        &mut self.networks[1]
    }

    pub fn found_components(&self) -> &Vec<(Position, ComponentType)> {
        &self.found_components
    }

    pub fn add_found_component(&mut self, position: Position, component_type: ComponentType) {
        self.found_components.push((position, component_type));
    }

    pub fn crossover_uniform(
        &self,
        mut rng: EntropyComponent<WyRand>,
        with: &Genotype,
    ) -> Genotype {
        let control_network = self
            .control_network()
            .crossover_uniform(rng.fork_rng(), with.control_network());
        let categorize_network = self
            .categorize_network()
            .crossover_uniform(rng.fork_rng(), with.categorize_network());
        // take the longer one
        // let found_components =
        //     if self.found_components().iter().len() >= with.found_components().iter().len() {
        //         self.found_components().clone()
        //     } else {
        //         with.found_components().clone()
        //     };
        Genotype {
            networks: vec![control_network, categorize_network],
            found_components: vec![],
        }
    }

    pub fn mutate(&mut self, mut rng: EntropyComponent<WyRand>, adaptive_config: &AdaptiveConfig) {
        for network in self.networks_mut() {
            network.mutate(rng.fork_rng(), adaptive_config);
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
        // we have a list of found components with the boundingbox position saved (Position, ComponentType)
        // there a already only unique positions listed
        let mut netlist = Netlist::new();

        self.found_components
            .iter()
            .cloned()
            .unique()
            .enumerate()
            .for_each(|(idx, (_, component_type))| {
                let component = ComponentBuilder::new(component_type, idx.to_string()).build();
                // adding components
                let _ = netlist.add_component(
                    component.clone(),
                    format!("{}{}", component.symbol, component.name),
                );
            });

        netlist.generate()
        // capacitor_networks
        //     .iter()
        //     .enumerate()
        //     .for_each(|(i, network)| {
        //         let mut component =
        //             ComponentBuilder::new(ComponentType::Capacitor, i.to_string()).build();
        //         let in_node = network.neurons()[in_node_neuron_idx].output().abs()
        //             * CONFIG.genetic_algorithm.node_range as f32;
        //         let out_node = network.neurons()[out_node_neuron_idx].output().abs()
        //             * CONFIG.genetic_algorithm.node_range as f32;
        //         component.add_node(Node(in_node as u32), NodeType::In);
        //         component.add_node(Node(out_node as u32), NodeType::Out);
        //         let _ = netlist.add_component(
        //             component.clone(),
        //             format!("{}{}", component.symbol, component.name),
        //         );
        //     });
    }
}

/// Conductor to control all networks. The idea is to set off multiple RNNs per update step and to collect the results.
/// The maximum number of RNNs are set in the config file.
/// Each RNN has a set number of Neurons. It can detect either a resitor, capacitor or voltage source for now.
/// The fitness of the whole set of networks is determined instead of a single network.
#[derive(Debug, Component, Default)]
pub struct Agent {
    pub fitness: f32,
    pub genotype: Genotype,
    pub retina_start_pos: Position,
    // netlist: String,
    // String -> netlist string
    // pub image_buffer: HashMap<ImageLabel, (Image, Genotype, String)>,
}

impl Clone for Agent {
    fn clone(&self) -> Self {
        Agent {
            fitness: self.fitness,
            genotype: self.genotype.clone(),
            retina_start_pos: self.retina_start_pos.clone(),
            // netlist: self.netlist.clone(),
            // image_buffer: self.image_buffer.clone(),
        }
    }
}

// impl Default for Agent {
//     fn default() -> Self {
//         Self::new()
//     }
// }

impl Agent {
    // pub fn new(adaptive_config: &Res<AdaptiveConfig>) -> Self {
    //     let mut rng = ChaCha8Rng::from_entropy();
    //     let retina_size = adaptive_config.retina_size;
    //     Agent {
    //         fitness: 0.0,
    //         genotype: Genotype::new(&mut rng, adaptive_config),
    //         // top left
    //         retina_start_pos: Position::new(retina_size as i32, retina_size as i32) / 2,
    //         netlist: String::new(),
    //         statistics: HashMap::new(),
    //     }
    // }

    pub fn evaluate(
        &mut self,
        adaptive_config: &Res<AdaptiveConfig>,
        image: &mut Image,
        annotation: &Annotation,
    ) -> std::result::Result<f32, Error> {
        // initialize retina
        let retina_size = adaptive_config.retina_size as usize;
        // create a retina at a random position
        let top_left = Position::new(retina_size as i32, retina_size as i32);
        // let bottom_right = Position::new(
        //     image.width() as i32 - retina_size as i32,
        //     image.height() as i32 - retina_size as i32,
        // );

        // let _image_center_position =
        //     Position::new((image.width() / 2) as i32, (image.height() / 2) as i32);

        let mut retina = image.create_retina_at(
            top_left,
            retina_size,
            adaptive_config.superpixel_size as usize,
            "".to_string(),
        )?;

        self.clear_short_term_memories();

        // initial network update and snapshot
        self.genotype_mut()
            .networks_mut()
            .iter_mut()
            .for_each(|network| {
                network.update_inputs_from_retina(&retina);
                network.update();
            });
        let control_outputs = self
            .genotype()
            .control_network()
            .neurons()
            .iter()
            .map(|neuron| neuron.output())
            .collect::<Vec<f32>>();
        let categorize_outputs = self
            .genotype()
            .categorize_network()
            .neurons()
            .iter()
            .map(|neuron| neuron.output())
            .collect::<Vec<f32>>();
        let time_step = (0) as u32;
        self.genotype_mut()
            .control_network_mut()
            .add_snapshot(control_outputs, time_step);
        self.genotype_mut()
            .categorize_network_mut()
            .add_snapshot(categorize_outputs, time_step);
        let mut local_fitness = 0.0;
        for i in 0..adaptive_config.number_of_network_updates {
            // first location of the retina
            image.update_retina_movement(&retina);

            // update the list of visited dark pixels
            retina.update_positions_visited();

            // calculate the next delta position of the retina, encoded in the neurons
            let delta = self
                .genotype()
                .control_network()
                .next_delta_position(adaptive_config);

            // move the retina to the next position
            // here gets the data updated the retina sees
            retina.move_mut(&delta, image);

            self.genotype_mut()
                .networks_mut()
                .iter_mut()
                .for_each(|network| {
                    // update all super pixel input connections to each neuron
                    network.update_inputs_from_retina(&retina);
                    // let sum = network
                    //     .neurons()
                    //     .iter()
                    //     .fold(0.0, |acc, n| acc + n.output());
                    // info(format!("sum before: {}", sum));
                    // do one update step
                    network.update();
                    // let sum = network
                    //     .neurons()
                    //     .iter()
                    //     .fold(0.0, |acc, n| acc + n.output());
                    // info(format!("sum after: {}", sum));
                });

            // self.genotype_mut()
            //     .control_network_mut()
            //     .update_inputs_from_retina(&retina);
            // self.genotype_mut().control_network_mut().update();

            // self.genotype_mut()
            //     .categorize_network_mut()
            //     .update_inputs_from_retina(&retina);
            // self.genotype_mut().categorize_network_mut().update();

            // save retina movement in buffer
            image.update_retina_movement(&retina);

            // creating snapshot of the network at the current time step
            let control_outputs = self
                .genotype()
                .control_network()
                .neurons()
                .iter()
                .map(|neuron| neuron.output())
                .collect::<Vec<f32>>();
            let categorize_outputs = self
                .genotype()
                .categorize_network()
                .neurons()
                .iter()
                .map(|neuron| neuron.output())
                .collect::<Vec<f32>>();
            let time_step = (i + 1) as u32;
            self.genotype_mut()
                .control_network_mut()
                .add_snapshot(control_outputs, time_step);
            self.genotype_mut()
                .categorize_network_mut()
                .add_snapshot(categorize_outputs, time_step);

            // calculate the fitness of the genotype
            local_fitness += training::fitness(self, annotation, &retina, &image);
        }
        // save the image in the hashmap of the agent with label
        // let image = image.clone();
        // let genotype = self.genotype().clone();
        // self.statistics_mut().insert(
        //     ImageLabel(annotation.filename.replace(".jpg", "").clone()),
        //     (image, genotype, String::new()),
        // );

        let fitness = local_fitness / adaptive_config.number_of_network_updates as f32;
        Ok(fitness)
    }

    /// builds a new Aegnt from a multiple json files located at 'path'
    pub fn from_path(
        path: PathBuf,
        adaptive_config: &Res<AdaptiveConfig>,
    ) -> std::result::Result<Self, Error> {
        let mut networks = vec![];
        fs::read_dir(path.clone()).unwrap().for_each(|entry| {
            let network_path = entry.unwrap().path();
            if network_path.is_dir() {
                if network_path.file_name().unwrap().to_string_lossy() == *"control" {
                    fs::read_dir(network_path.clone())
                        .unwrap()
                        .for_each(|file| {
                            let file_path = file.unwrap().path();
                            if Some(OsStr::new("network.json")) == file_path.file_name() {
                                let mut file =
                                    OpenOptions::new().read(true).open(file_path).unwrap();
                                let mut buffer = String::new();
                                file.read_to_string(&mut buffer).unwrap();
                                let rnn: Rnn = serde_json::from_str(buffer.trim()).unwrap();
                                networks.insert(0, rnn);
                            }
                        })
                } else if network_path.file_name().unwrap().to_string_lossy() == *"categorize" {
                    fs::read_dir(network_path.clone())
                        .unwrap()
                        .for_each(|file| {
                            let file_path = file.unwrap().path();
                            if Some(OsStr::new("network.json")) == file_path.file_name() {
                                let mut file =
                                    OpenOptions::new().read(true).open(file_path).unwrap();
                                let mut buffer = String::new();
                                file.read_to_string(&mut buffer).unwrap();
                                let rnn: Rnn = serde_json::from_str(buffer.trim()).unwrap();
                                networks.insert(1, rnn);
                            }
                        })
                }
            }
        });
        let genotype = Genotype {
            networks: vec![networks[0].clone(), networks[1].clone()],
            found_components: vec![],
        };
        Ok(Agent {
            fitness: 0.0,
            genotype,
            // top left
            retina_start_pos: Position::new(
                adaptive_config.retina_size as i32,
                adaptive_config.retina_size as i32,
            ) / 2,
            // netlist: String::new(),
            // image_buffer: HashMap::new(),
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

    // pub fn statistics(&self) -> &HashMap<ImageLabel, (Image, Genotype, String)> {
    //     &self.statistics
    // }

    // pub fn statistics_mut(&mut self) -> &mut HashMap<ImageLabel, (Image, Genotype, String)> {
    //     &mut self.statistics
    // }

    pub fn crossover(&self, rng: EntropyComponent<WyRand>, with: &Agent) -> Agent {
        let mut new_agent = self.clone();
        let offspring_genotype = self.genotype.crossover_uniform(rng, &with.genotype);
        new_agent.genotype = offspring_genotype;
        new_agent
    }

    pub fn mutate(&mut self, rng: EntropyComponent<WyRand>, adaptive_config: &AdaptiveConfig) {
        self.genotype.mutate(rng, adaptive_config);
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
        mut rng: EntropyComponent<WyRand>,
        low_x: i32,
        high_x: i32,
        low_y: i32,
        high_y: i32,
    ) -> Position {
        let random_x = rng.gen_range(low_x..=high_x);
        let random_y = rng.gen_range(low_y..=high_y);
        Position::new(random_x, random_y)
    }

    pub fn clear_short_term_memories(&mut self) -> &mut Self {
        self.genotype.clear_short_term_memories();
        self
    }

    pub fn debug_get_neuron_outputs(&self) -> Vec<f32> {
        self.genotype
            .networks()
            .iter()
            .flat_map(|network| network.neurons().iter().map(|neuron| neuron.output()))
            .collect()
    }
}
