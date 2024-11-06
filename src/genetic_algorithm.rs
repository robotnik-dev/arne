use std::ffi::OsStr;
use std::fs::OpenOptions;
use std::io::Read;
use std::path::PathBuf;

use crate::annotations::Annotation;
use crate::image::{Image, Position};
use crate::netlist::{Build, ComponentBuilder, ComponentType, Netlist};
use crate::neural_network::Rnn;
use crate::{AdaptiveConfig, Error, Retina};
use bevy::prelude::*;
use bevy::utils::hashbrown::HashMap;
use bevy_prng::WyRand;
use bevy_rand::prelude::EntropyComponent;
use bevy_rand::traits::ForkableRng;

use rand::Rng;
use serde::{Deserialize, Serialize};

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

pub fn control_fitness(
    agent: &mut Agent,
    annotation: &Annotation,
    retina: &Retina,
    image: &Image,
) -> f32 {
    // the output of one neuron needs to exceed this value to count as active
    let active_threshold = 0.95;

    let source_dc_neuron_idx = 0usize;
    let resistor_neuron_idx = 1usize;
    let capacitor_neuron_idx = 2usize;

    annotation.objects.iter().for_each(|obj| {
        let bndbox = image.translate_bndbox_to_size(annotation, obj);
        // we need to check if any bndbox specified in the annotation is currently inside the retina rectangle
        if image.wraps_bndbox(&bndbox, retina) {
            // If anyone is, then we check we check what kind of component is specified in this object
            let full_component = obj.name.clone();
            let component = full_component.split(".").take(1).collect::<String>();

            // And lastly we check if the corresponding neuron is active for this component and every other neuron is inactive
            if component == "resistor"
                && agent.genotype().categorize_network().neurons()[resistor_neuron_idx].output()
                    >= active_threshold
                && agent.genotype().categorize_network().neurons()[source_dc_neuron_idx].output()
                    <= -active_threshold
                && agent.genotype().categorize_network().neurons()[capacitor_neuron_idx].output()
                    <= -active_threshold
            {
                // categorize_fitness = 1f32;
                agent.genotype_mut().add_found_component(
                    image.id,
                    retina.get_center_position(),
                    retina.size(),
                    Position::from(bndbox.clone()),
                    ComponentType::Resistor,
                );
            } else if component == "voltage"
                && agent.genotype().categorize_network().neurons()[source_dc_neuron_idx].output()
                    >= active_threshold
                && agent.genotype().categorize_network().neurons()[resistor_neuron_idx].output()
                    <= -active_threshold
                && agent.genotype().categorize_network().neurons()[capacitor_neuron_idx].output()
                    <= -active_threshold
            {
                // categorize_fitness = 1f32;
                agent.genotype_mut().add_found_component(
                    image.id,
                    retina.get_center_position(),
                    retina.size(),
                    Position::from(bndbox.clone()),
                    ComponentType::VoltageSourceDc,
                );
            } else if component == "capacitor"
                && agent.genotype().categorize_network().neurons()[capacitor_neuron_idx].output()
                    >= active_threshold
                && agent.genotype().categorize_network().neurons()[resistor_neuron_idx].output()
                    <= -active_threshold
                && agent.genotype().categorize_network().neurons()[source_dc_neuron_idx].output()
                    <= -active_threshold
            {
                // categorize_fitness = 1f32;
                agent.genotype_mut().add_found_component(
                    image.id,
                    retina.get_center_position(),
                    retina.size(),
                    Position::from(bndbox.clone()),
                    ComponentType::Capacitor,
                );
            }
        }
        // else {
        //     // nothing in the retina! They should gain fitness when every neuron is inactive
        //     if agent.genotype().categorize_network().neurons()[resistor_neuron_idx].output()
        //         <= -active_threshold
        //         && agent.genotype().categorize_network().neurons()[source_dc_neuron_idx]
        //             .output()
        //             <= -active_threshold
        //         && agent.genotype().categorize_network().neurons()[capacitor_neuron_idx]
        //             .output()
        //             <= -active_threshold
        //     {
        //         // half as much fitness as if there was something found
        //         categorize_fitness = 0.1f32;
        //     }
        // }
    });

    retina.percentage_visited()
}

pub fn categorize_fitness(agent: &Agent, image_id: u64, optimal_netlist: &Netlist) -> f32 {
    let netlist = agent.genotype().build(image_id);
    optimal_netlist.compare(&netlist)
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct Genotype {
    /// the first is the control network and second the categorize network
    networks: Vec<Rnn>,
    /// image_id hashmap : (Retina position, retina_size, bndbox position)
    #[serde(skip)]
    pub found_components: HashMap<u64, Vec<((Position, usize, Position), ComponentType)>>,
}

impl Genotype {
    pub fn init(mut rng: EntropyComponent<WyRand>, adaptive_config: &Res<AdaptiveConfig>) -> Self {
        let control_network = Rnn::new(
            rng.fork_rng(),
            adaptive_config.control_network_neurons,
            adaptive_config,
        );
        let categorize_network = Rnn::new(
            rng.fork_rng(),
            adaptive_config.categorize_network_neurons,
            adaptive_config,
        );
        Genotype {
            networks: vec![control_network, categorize_network],
            found_components: HashMap::new(),
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

    pub fn add_found_component(
        &mut self,
        image_id: u64,
        retina_position: Position,
        retina_size: usize,
        bndbox_position: Position,
        component_type: ComponentType,
    ) {
        if let Some(positions) = self.found_components.get_mut(&image_id) {
            if !positions
                .iter()
                .any(|((_, _, bndbox_pos), _)| bndbox_pos == &bndbox_position)
            {
                positions.push((
                    (retina_position, retina_size, bndbox_position),
                    component_type,
                ));
            }
        } else {
            // insert first entry
            self.found_components.insert(
                image_id,
                vec![(
                    (retina_position, retina_size, bndbox_position),
                    component_type,
                )],
            );
        }
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
        Genotype {
            networks: vec![control_network, categorize_network],
            found_components: HashMap::new(),
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

impl Build for Genotype {
    fn build(&self, image_id: u64) -> Netlist {
        // we have a list of found components with the boundingbox position saved (Position, ComponentType)
        // there a already only unique positions listed
        let mut netlist = Netlist::new();

        if let Some(components) = self.found_components.get(&image_id) {
            components
                .iter()
                .enumerate()
                .for_each(|(idx, (_, component_type))| {
                    let component =
                        ComponentBuilder::new(component_type.clone(), idx.to_string()).build();
                    // adding components
                    let _ = netlist.add_component(
                        component.clone(),
                        format!("{}{}", component.symbol, component.name),
                    );
                });
        }

        netlist
    }
}

#[derive(Debug, Component, Default, PartialEq, Clone)]
pub struct Agent {
    pub id: u64,
    pub fitness: f32,
    pub genotype: Genotype,
    pub retina_start_pos: Position,
    /// used to visualize the retina movement on an upscaled image (Position, size of retina, label)
    pub retina_positions: Vec<(Position, usize, String)>,
}

impl Agent {
    pub fn new(mut rng: EntropyComponent<WyRand>) -> Self {
        Self {
            id: rng.gen(),
            fitness: f32::default(),
            genotype: Genotype::default(),
            retina_start_pos: Position::default(),
            retina_positions: vec![],
        }
    }

    /// builds a new Agent from json files located at 'path'
    pub fn from_path(
        path: PathBuf,
        adaptive_config: &Res<AdaptiveConfig>,
    ) -> std::result::Result<Self, Error> {
        if !path.is_dir() {
            return Err("No agent dir".into());
        }
        let mut networks = vec![];
        let path_binding = path.clone();
        let agent_folder_name = path_binding.file_name().unwrap().to_str().unwrap();
        let agent_id_vec = agent_folder_name.split("_").collect::<Vec<&str>>();
        let agent_id_str = agent_id_vec[1];
        let agent_id = agent_id_str.parse::<u64>().unwrap();
        std::fs::read_dir(path.clone()).unwrap().for_each(|entry| {
            let network_path = entry.unwrap().path();
            if network_path.is_dir() {
                if network_path.file_name().unwrap().to_string_lossy() == *"control" {
                    std::fs::read_dir(network_path.clone())
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
                    std::fs::read_dir(network_path.clone())
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
            found_components: HashMap::new(),
        };
        Ok(Agent {
            id: agent_id,
            fitness: 0.0,
            genotype,
            // top left
            retina_start_pos: Position::new(
                adaptive_config.retina_size_medium as i32,
                adaptive_config.retina_size_medium as i32,
            ) / 2,
            retina_positions: vec![],
        })
    }

    pub fn evaluate(
        &mut self,
        adaptive_config: &Res<AdaptiveConfig>,
        image: &mut Image,
        annotation: &Annotation,
        optimal_netlist: &Netlist,
    ) -> std::result::Result<f32, Error> {
        let retina_size = adaptive_config.retina_size_medium;
        let top_left = Position::new(retina_size as i32, retina_size as i32) / 2;
        let mut retina =
            image.create_retina_at(top_left, retina_size, retina_size / 9, "".to_string())?;

        // clear data before a fresh start of evaulation
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
        let time_step = 0_u32;
        self.genotype_mut()
            .control_network_mut()
            .add_snapshot(control_outputs, time_step);
        self.genotype_mut()
            .categorize_network_mut()
            .add_snapshot(categorize_outputs, time_step);
        let mut control_fitness_v = 0.0;
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
                    network.update_inputs_from_retina(&retina);
                    network.update();
                });

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
            control_fitness_v += control_fitness(self, annotation, &retina, image);
        }
        // save all the positions the retina visited on the control network to save it in json format
        self.genotype_mut().control_network_mut().retina_positions = image.retina_positions.clone();

        // normalize control fitness over number of updates
        control_fitness_v /= adaptive_config.number_of_network_updates as f32;

        // calculate the categorize fitness
        let categorize_fitness = categorize_fitness(self, image.id, optimal_netlist);
        let fitness = (control_fitness_v + categorize_fitness) / 2.0f32;
        Ok(fitness)
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

    pub fn crossover(&self, rng: EntropyComponent<WyRand>, with: &Agent) -> Agent {
        let mut new_agent = self.clone();
        let offspring_genotype = self.genotype.crossover_uniform(rng, &with.genotype);
        new_agent.genotype = offspring_genotype;
        new_agent
    }

    pub fn mutate(&mut self, rng: EntropyComponent<WyRand>, adaptive_config: &AdaptiveConfig) {
        self.genotype.mutate(rng, adaptive_config);
    }

    pub fn clear_short_term_memories(&mut self) -> &mut Self {
        self.genotype.clear_short_term_memories();
        self
    }
}
