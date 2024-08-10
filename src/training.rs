use core::panic;
use std::u8;

use crate::image_processing::{ImageDescription, TrainingStage};
use crate::netlist::Generate;
use crate::{
    Agent, AgentEvaluation, ChaCha8Rng, ImageReader, Population, Result, Retina, Rnn, SelectionMethod, CONFIG
};
use indicatif::ProgressBar;
use rand::prelude::*;
use rayon::prelude::*;
use approx::AbsDiffEq;

fn fitness_pixel_follow(agent: &mut Agent, _: ImageDescription, retinas: Vec<Retina>) -> f32 {
    // fitness reward for less white pixels
    let max_pixel_count = agent.genotype().networks().iter().fold(0, |acc, network| {
        acc + network.neurons()[0].retina_inputs().len()
    });

    let white_pixel_count = agent.genotype().networks().iter().fold(0, |acc, network| {
        acc + network.neurons()[0]
            .retina_inputs()
            .iter()
            .filter(|&pixel| pixel > &0.5)
            .count()
    });

    // reward more movement of the retinas
    let movement_fitness = retinas.iter().fold(0.0, |acc, retina| {
        acc + retina.get_current_delta_position().normalized_len() as f32
    }) / retinas.len() as f32;
        // / CONFIG.neural_network.retina_movement_speed as f32;

    let follow_pixel_fitness = 1.0 - (white_pixel_count as f32 / max_pixel_count as f32);
        
    assert!(movement_fitness >= 0. && movement_fitness <= 1.);
    let fitness = (follow_pixel_fitness + movement_fitness) / 2.0;
    fitness
}


fn fitness_recognize_components(agent: &mut Agent, description: ImageDescription, _: Vec<Retina>) -> f32 {
    let resistors = description.components.resistor.unwrap_or_default();
    let resistor_nodes = description.nodes.resistor.unwrap_or_default();
    let capacitors = description.components.capacitor.unwrap_or_default();
    let capacitors_nodes = description.nodes.capacitor.unwrap_or_default();
    let sources_dc = description.components.source_dc.unwrap_or_default();
    let sources_dc_nodes = description.nodes.source_dc.unwrap_or_default();

    let source_dc_neuron_idx = 2usize;
    let resistor_neuron_idx = 3usize;
    let capacitor_neuron_idx = 4usize;
    let in_node_neuron_idx = 5usize;
    let out_node_neuron_idx = 6usize;

    // collect all networks that 'see' some component
    let resistor_networks = agent
        .genotype()
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

    let capacitor_networks = agent
        .genotype()
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

    let source_dc_networks = agent
        .genotype()
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

    // for each correct determined node in the corresponding network, it gets a point
    // then the points are accumulated, normalized and weighted to add to the fitness calculation
    let mut correct_nodes = 0u32;
    let maximum_nodes = resistor_nodes
        .iter()
        .chain(capacitors_nodes.iter())
        .chain(sources_dc_nodes.iter())
        .count() as u32
        * 2;
    resistor_networks.iter().for_each(|network| {
        let in_node = network.neurons()[in_node_neuron_idx].output().abs()
            * CONFIG.genetic_algorithm.node_range as f32;
        let out_node = network.neurons()[out_node_neuron_idx].output().abs()
            * CONFIG.genetic_algorithm.node_range as f32;
        resistor_nodes.iter().for_each(|pair| {
            if in_node as u32 == pair[0] {
                correct_nodes += 1
            };
            if out_node as u32 == pair[1] {
                correct_nodes += 1
            };
        })
    });
    capacitor_networks.iter().for_each(|network| {
        let in_node = network.neurons()[in_node_neuron_idx].output().abs()
            * CONFIG.genetic_algorithm.node_range as f32;
        let out_node = network.neurons()[out_node_neuron_idx].output().abs()
            * CONFIG.genetic_algorithm.node_range as f32;
        capacitors_nodes.iter().for_each(|pair| {
            if in_node as u32 == pair[0] {
                correct_nodes += 1
            };
            if out_node as u32 == pair[1] {
                correct_nodes += 1
            };
        })
    });
    source_dc_networks.iter().for_each(|network| {
        let in_node = network.neurons()[in_node_neuron_idx].output().abs()
            * CONFIG.genetic_algorithm.node_range as f32;
        let out_node = network.neurons()[out_node_neuron_idx].output().abs()
            * CONFIG.genetic_algorithm.node_range as f32;
        sources_dc_nodes.iter().for_each(|pair| {
            if in_node as u32 == pair[0] {
                correct_nodes += 1
            };
            if out_node as u32 == pair[1] {
                correct_nodes += 1
            };
        })
    });
    // fitness is high when the count of the networks are exactly the ImageDescription numbers
    // Additionally the fitness gets lower the more blank_networks exists in this time step

    // if there are no components of this kind, the fitness is not so much weighted
    let max_networks = CONFIG.neural_network.networks_per_agent as f32;
    let resistor_fitness = if resistors == 0 {
        if resistor_networks.len() == resistors as usize {
            0.25
        } else {
            0.0
        }
    } else if resistor_networks.len() == resistors as usize {
        1.0
    } else {
        1.0 - ((resistor_networks.len() as f32 - resistors as f32).abs() / max_networks)
    };
    let capacitor_fitness = if capacitors == 0 {
        if capacitor_networks.len() == capacitors as usize {
            0.25
        } else {
            0.0
        }
    } else if capacitor_networks.len() == capacitors as usize {
        1.0
    } else {
        1.0 - ((capacitor_networks.len() as f32 - capacitors as f32).abs() / max_networks)
    };
    let source_dc_fitness = if sources_dc == 0 {
        if source_dc_networks.len() == sources_dc as usize {
            0.25
        } else {
            0.0
        }
    } else if source_dc_networks.len() == sources_dc as usize {
        1.0
    } else {
        1.0 - ((source_dc_networks.len() as f32 - sources_dc as f32).abs() / max_networks)
    };
    let network_fitness = (resistor_fitness + capacitor_fitness + source_dc_fitness) / 3.0;
    let node_fitness = 1.0 - (correct_nodes as f32 / maximum_nodes as f32);

    let fitness = (network_fitness * 0.4 + node_fitness * 0.6) / 2.0;
    fitness
}

pub fn train_agents(stage: TrainingStage, load_path: Option<String>, save_path: String) -> Result {
    log::info!("starting training stage {:?}", stage);

    log::info!("loading training config variables");

    let max_generations = CONFIG.genetic_algorithm.max_generations as u64;
    let seed = CONFIG.genetic_algorithm.seed as u64;
    let with_seed = CONFIG.genetic_algorithm.with_seed;
    let path_to_image_descriptions = CONFIG.image_processing.path_to_image_descriptions as &str;
    let neurons_per_rnn = CONFIG.neural_network.neurons_per_network as usize;
    let population_size = CONFIG.genetic_algorithm.initial_population_size as usize;
    let number_of_network_updates = CONFIG.neural_network.number_of_network_updates as usize;
    let take_agents = CONFIG.genetic_algorithm.take_agents as usize;
    let networks_per_agent = CONFIG.neural_network.networks_per_agent as usize;
    let variance_decay = CONFIG.genetic_algorithm.mutation_rates.variance_decay as f32;
    let goal_fitness = CONFIG.genetic_algorithm.goal_fitness as f32;

    log::info!("setting up rng");

    let mut rng = ChaCha8Rng::from_entropy();

    if with_seed {
        log::info!("using seed: {}", seed);
        rng = ChaCha8Rng::seed_from_u64(seed);
    }

    log::info!("initializing population...");

    // intialize population
    let population_bar = ProgressBar::new(population_size as u64);
    let mut population = if load_path.is_some() {
        Population::from_path(load_path.unwrap())?
    } else {
        Population::new(
            &population_bar,
            population_size,
            networks_per_agent,
            neurons_per_rnn,
        )
    };
    population_bar.finish();

    log::info!("loading training dataset...");

    // create a reader to buffer training dataset
    let image_path = match stage {
        TrainingStage::Artificial{..} => CONFIG.image_processing.path_to_training_artificial as &str,
        TrainingStage::RealBinarized => CONFIG.image_processing.path_to_training_binarized as &str,
        TrainingStage::Real => CONFIG.image_processing.path_to_analysis_stage as &str,
    };
    let image_reader = ImageReader::from_path(
        image_path.to_string(),
        path_to_image_descriptions.to_string(),
        stage.clone(),
    )?;

    let fitness_function = match stage {
        TrainingStage::Artificial { stage: 0 } => fitness_pixel_follow,
        TrainingStage::Artificial { stage: 1 } => fitness_recognize_components,
        TrainingStage::Artificial { stage: 2..=u8::MAX } => panic!("No third stage defined"),
        TrainingStage::RealBinarized => todo!(),
        TrainingStage::Real => todo!(),
    };

    let algorithm_bar = ProgressBar::new(max_generations);
    // loop until stop criterial is met
    log::info!("training agents...");
    loop {
        algorithm_bar.inc(1);
        // for each image in the dataset
        for index in 0..image_reader.images().len() {
            // load image
            let (label, image, description) = image_reader.get_image(index)?;

            // evaluate the fitness of each individual of the population
            population
                .agents_mut()
                .par_iter_mut()
                .enumerate()
                .for_each(|(_, agent)| {
                    let fitness = agent
                        .evaluate(
                            fitness_function,
                            &mut rng.clone(),
                            label.clone(),
                            &mut image.clone(),
                            description.clone(),
                            number_of_network_updates,
                        )
                        .unwrap();
                    agent.add_to_fitness(fitness);
                });
            
            // after one image was processed, save a netlist per image
            population
                .agents_mut()
                .par_iter_mut()
                .for_each(|agent| {
                    let generated_netlist = agent.genotype().generate();
                    if let Some((_, _, netlist)) = agent.statistics_mut().get_mut(&label.clone()) {
                        *netlist = generated_netlist;
                    }
                })
        }

        // average each agents fitness over the number of images
        population
            .agents_mut()
            .par_iter_mut()
            .enumerate()
            .for_each(|(_, agent)| {
                agent.set_fitness(agent.fitness() / image_reader.images().len() as f32);
            });

        // sort the population by fitness
        population
            .agents_mut()
            .sort_by(|a, b| b.fitness().partial_cmp(&a.fitness()).unwrap());

        // check stop criteria:
        // - if any agent has a high enough fitness
        // - if the maximum number of generations has been reached
        if population
            .agents()
            .iter()
            .any(|agent| agent.fitness() >= goal_fitness)
            || population.generation() >= max_generations as u32
        {
            break;
        }

        // after 50 % of max generations, decrease the variance by 10 % each generation
        if population.generation() > max_generations as u32 / 2 {
            population.agents_mut().iter_mut().for_each(|agent| {
                agent.update_variance(agent.get_current_variance() * variance_decay);
            })
        }

        // select, crossover and mutate
        let new_agents = (0..population.agents().len())
            .map(|_| {
                let (parent1, parent2) = population.select(&mut rng, SelectionMethod::Tournament);
                let mut offspring = parent1.crossover(&mut rng, parent2);
                offspring.mutate(&mut rng);
                offspring
            })
            .collect::<Vec<Agent>>();

        // evolve the population
        population.evolve(new_agents);
    }
    algorithm_bar.finish();
    log::info!("training finished");
    log::info!("stopped after {} generations", population.generation());
    log::info!("generating files for the best {} agents...", take_agents);

    // remove 'agents' directory if it exists
    std::fs::remove_dir_all(save_path.clone()).unwrap_or_default();

    let generating_files_bar = ProgressBar::new(take_agents as u64);
    population
        .agents_mut()
        .par_iter_mut()
        .enumerate()
        .inspect(|(index, agent)| {
            log::debug!("agent {} fitness: {}", index, agent.fitness());
        })
        .take(take_agents)
        .for_each(|(index, agent)| {
            agent
                .statistics_mut()
                .par_iter_mut()
                .for_each(|(label, (image, genotype, netlist))| {
                    std::fs::create_dir_all(format!("{}/{}/{}", save_path, index, label)).unwrap();

                    image
                        .save_with_retina(format!("{}/{}/{}/retina.png", save_path, index, label))
                        .unwrap();
                    image
                        .save_with_retina_upscaled(format!(
                            "{}/{}/{}/retina_orig.png",
                            save_path, index, label
                        ))
                        .unwrap();

                    // save netlist
                    std::fs::write(
                        format!("{}/{}/{}/netlist.net", save_path, index, label),
                        netlist.clone(),
                    )
                    .unwrap();

                    genotype
                        .networks_mut()
                        .iter_mut()
                        .enumerate()
                        .for_each(|(i, network)| {
                            std::fs::create_dir_all(format!(
                                "{}/{}/{}/{}",
                                save_path, index, label, i
                            ))
                            .unwrap();
                            network
                                .short_term_memory()
                                .visualize(format!(
                                    "{}/{}/{}/{}/memory.png",
                                    save_path, index, label, i
                                ))
                                .unwrap();
                            network
                                .to_json(format!(
                                    "{}/{}/{}/{}/network.json",
                                    save_path, index, label, i
                                ))
                                .unwrap();
                            network
                                .to_dot(format!(
                                    "{}/{}/{}/{}/network.dot",
                                    save_path, index, label, i
                                ))
                                .unwrap();
                        });
                });
            
            // create a final version of the agent with the current network
            agent
                .genotype_mut()
                .networks_mut()
                .iter_mut()
                .enumerate()
                .for_each(|(i, network)| {
                    std::fs::create_dir_all(format!(
                        "{}/{}/final/{}",
                        save_path, index, i
                    ))
                    .unwrap();
                    network
                        .short_term_memory()
                        .visualize(format!(
                            "{}/{}/final/{}/memory.png",
                            save_path, index, i
                        ))
                        .unwrap();
                    network
                        .to_json(format!(
                            "{}/{}/final/{}/network.json",
                            save_path, index, i
                        ))
                        .unwrap();
                    network
                        .to_dot(format!(
                            "{}/{}/final/{}/network.dot",
                            save_path, index, i
                        ))
                        .unwrap();
                });

            generating_files_bar.inc(1);
        });
    generating_files_bar.finish();

    Ok(())
}
