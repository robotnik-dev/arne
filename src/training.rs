use core::panic;
use std::path::PathBuf;
use std::u8;

use crate::annotations::{Annotation, LoadFolder, XMLParser};
use crate::image::{ImageLabel, TrainingStage};
use crate::netlist::Generate;
use crate::{
    round2, round3, Agent, AgentEvaluation, ChaCha8Rng, Population, Result, Retina,
    SelectionMethod, CONFIG,
};
use indicatif::ProgressBar;
use rand::prelude::*;
use rayon::prelude::*;

fn fitness_pixel_follow(_agent: &mut Agent, _annotation: &Annotation, retina: &Retina) -> f32 {
    // TODO: the categorize network is ignored for now
    // log::debug!(
    //     "all: {:?}; visited: {}, current frame: {}",
    //     retina.dark_pixel_positions().len(),
    //     retina.dark_pixel_positions_visited().len(),
    //     retina.dark_pixel_positions_in_frame(0.5).len()
    // );
    retina.percentage_visited()
}

fn fitness_recognize_components(
    _agent: &mut Agent,
    _annotation: &Annotation,
    _retina: &Retina,
) -> f32 {
    unimplemented!()

    // let resistors = description.components.resistor.unwrap_or_default();
    // let resistor_nodes = description.nodes.resistor.unwrap_or_default();
    // let capacitors = description.components.capacitor.unwrap_or_default();
    // let capacitors_nodes = description.nodes.capacitor.unwrap_or_default();
    // let sources_dc = description.components.source_dc.unwrap_or_default();
    // let sources_dc_nodes = description.nodes.source_dc.unwrap_or_default();

    // let source_dc_neuron_idx = 2usize;
    // let resistor_neuron_idx = 3usize;
    // let capacitor_neuron_idx = 4usize;
    // let in_node_neuron_idx = 5usize;
    // let out_node_neuron_idx = 6usize;

    // // collect all networks that 'see' some component
    // let resistor_networks = agent
    //     .genotype()
    //     .networks()
    //     .iter()
    //     .filter(|&network| {
    //         network.neurons()[resistor_neuron_idx]
    //             .output()
    //             .abs_diff_eq(&1.0, 0.01)
    //             && network.neurons()[capacitor_neuron_idx]
    //                 .output()
    //                 .abs_diff_eq(&0.0, 0.01)
    //             && network.neurons()[source_dc_neuron_idx]
    //                 .output()
    //                 .abs_diff_eq(&0.0, 0.01)
    //     })
    //     .cloned()
    //     .collect::<Vec<Rnn>>();

    // let capacitor_networks = agent
    //     .genotype()
    //     .networks()
    //     .iter()
    //     .filter(|&network| {
    //         network.neurons()[resistor_neuron_idx]
    //             .output()
    //             .abs_diff_eq(&0.0, 0.01)
    //             && network.neurons()[capacitor_neuron_idx]
    //                 .output()
    //                 .abs_diff_eq(&1.0, 0.01)
    //             && network.neurons()[source_dc_neuron_idx]
    //                 .output()
    //                 .abs_diff_eq(&0.0, 0.01)
    //     })
    //     .cloned()
    //     .collect::<Vec<Rnn>>();

    // let source_dc_networks = agent
    //     .genotype()
    //     .networks()
    //     .iter()
    //     .filter(|&network| {
    //         network.neurons()[resistor_neuron_idx]
    //             .output()
    //             .abs_diff_eq(&0.0, 0.01)
    //             && network.neurons()[capacitor_neuron_idx]
    //                 .output()
    //                 .abs_diff_eq(&0.0, 0.01)
    //             && network.neurons()[source_dc_neuron_idx]
    //                 .output()
    //                 .abs_diff_eq(&1.0, 0.01)
    //     })
    //     .cloned()
    //     .collect::<Vec<Rnn>>();

    // // remove identical networks (comparison of the nodes)
    // // only keep the ones that have unique outputs of in_nodes and out_nodes
    // let resistor_networks = resistor_networks
    //     .iter()
    //     .filter(|&network| {
    //         let in_node_output = network.neurons()[in_node_neuron_idx].output();
    //         let out_node_output = network.neurons()[out_node_neuron_idx].output();
    //         resistor_networks.iter().all(|other_network| {
    //             let other_in_node_output = other_network.neurons()[in_node_neuron_idx].output();
    //             let other_out_node_output = other_network.neurons()[out_node_neuron_idx].output();
    //             in_node_output.abs_diff_eq(&other_in_node_output, 0.01)
    //                 && out_node_output.abs_diff_eq(&other_out_node_output, 0.01)
    //         })
    //     })
    //     .cloned()
    //     .collect::<Vec<Rnn>>();

    // let capacitor_networks = capacitor_networks
    //     .iter()
    //     .filter(|&network| {
    //         let in_node_output = network.neurons()[in_node_neuron_idx].output();
    //         let out_node_output = network.neurons()[out_node_neuron_idx].output();
    //         capacitor_networks.iter().all(|other_network| {
    //             let other_in_node_output = other_network.neurons()[in_node_neuron_idx].output();
    //             let other_out_node_output = other_network.neurons()[out_node_neuron_idx].output();
    //             in_node_output.abs_diff_eq(&other_in_node_output, 0.01)
    //                 && out_node_output.abs_diff_eq(&other_out_node_output, 0.01)
    //         })
    //     })
    //     .cloned()
    //     .collect::<Vec<Rnn>>();

    // let source_dc_networks = source_dc_networks
    //     .iter()
    //     .filter(|&network| {
    //         let in_node_output = network.neurons()[in_node_neuron_idx].output();
    //         let out_node_output = network.neurons()[out_node_neuron_idx].output();
    //         source_dc_networks.iter().all(|other_network| {
    //             let other_in_node_output = other_network.neurons()[in_node_neuron_idx].output();
    //             let other_out_node_output = other_network.neurons()[out_node_neuron_idx].output();
    //             in_node_output.abs_diff_eq(&other_in_node_output, 0.01)
    //                 && out_node_output.abs_diff_eq(&other_out_node_output, 0.01)
    //         })
    //     })
    //     .cloned()
    //     .collect::<Vec<Rnn>>();

    // // for each correct determined node in the corresponding network, it gets a point
    // // then the points are accumulated, normalized and weighted to add to the fitness calculation
    // let mut correct_nodes = 0u32;
    // let maximum_nodes = resistor_nodes
    //     .iter()
    //     .chain(capacitors_nodes.iter())
    //     .chain(sources_dc_nodes.iter())
    //     .count() as u32
    //     * 2;
    // resistor_networks.iter().for_each(|network| {
    //     let in_node = network.neurons()[in_node_neuron_idx].output().abs()
    //         * CONFIG.genetic_algorithm.node_range as f32;
    //     let out_node = network.neurons()[out_node_neuron_idx].output().abs()
    //         * CONFIG.genetic_algorithm.node_range as f32;
    //     resistor_nodes.iter().for_each(|pair| {
    //         if in_node as u32 == pair[0] {
    //             correct_nodes += 1
    //         };
    //         if out_node as u32 == pair[1] {
    //             correct_nodes += 1
    //         };
    //     })
    // });
    // capacitor_networks.iter().for_each(|network| {
    //     let in_node = network.neurons()[in_node_neuron_idx].output().abs()
    //         * CONFIG.genetic_algorithm.node_range as f32;
    //     let out_node = network.neurons()[out_node_neuron_idx].output().abs()
    //         * CONFIG.genetic_algorithm.node_range as f32;
    //     capacitors_nodes.iter().for_each(|pair| {
    //         if in_node as u32 == pair[0] {
    //             correct_nodes += 1
    //         };
    //         if out_node as u32 == pair[1] {
    //             correct_nodes += 1
    //         };
    //     })
    // });
    // source_dc_networks.iter().for_each(|network| {
    //     let in_node = network.neurons()[in_node_neuron_idx].output().abs()
    //         * CONFIG.genetic_algorithm.node_range as f32;
    //     let out_node = network.neurons()[out_node_neuron_idx].output().abs()
    //         * CONFIG.genetic_algorithm.node_range as f32;
    //     sources_dc_nodes.iter().for_each(|pair| {
    //         if in_node as u32 == pair[0] {
    //             correct_nodes += 1
    //         };
    //         if out_node as u32 == pair[1] {
    //             correct_nodes += 1
    //         };
    //     })
    // });
    // // fitness is high when the count of the networks are exactly the ImageDescription numbers
    // // Additionally the fitness gets lower the more blank_networks exists in this time step

    // // if there are no components of this kind, the fitness is not so much weighted
    // let max_networks = CONFIG.neural_network.networks_per_agent as f32;
    // let resistor_fitness = if resistors == 0 {
    //     if resistor_networks.len() == resistors as usize {
    //         0.25
    //     } else {
    //         0.0
    //     }
    // } else if resistor_networks.len() == resistors as usize {
    //     1.0
    // } else {
    //     1.0 - ((resistor_networks.len() as f32 - resistors as f32).abs() / max_networks)
    // };
    // let capacitor_fitness = if capacitors == 0 {
    //     if capacitor_networks.len() == capacitors as usize {
    //         0.25
    //     } else {
    //         0.0
    //     }
    // } else if capacitor_networks.len() == capacitors as usize {
    //     1.0
    // } else {
    //     1.0 - ((capacitor_networks.len() as f32 - capacitors as f32).abs() / max_networks)
    // };
    // let source_dc_fitness = if sources_dc == 0 {
    //     if source_dc_networks.len() == sources_dc as usize {
    //         0.25
    //     } else {
    //         0.0
    //     }
    // } else if source_dc_networks.len() == sources_dc as usize {
    //     1.0
    // } else {
    //     1.0 - ((source_dc_networks.len() as f32 - sources_dc as f32).abs() / max_networks)
    // };
    // let network_fitness = (resistor_fitness + capacitor_fitness + source_dc_fitness) / 3.0;
    // let node_fitness = 1.0 - (correct_nodes as f32 / maximum_nodes as f32);

    // let fitness = (network_fitness * 0.4 + node_fitness * 0.6) / 2.0;
    // fitness
}

#[allow(dead_code)]
pub fn test_agents() -> Result {
    todo!()
}

pub fn train_agents(stage: TrainingStage, load_path: Option<String>, save_path: String) -> Result {
    log::info!("starting training stage {:?}", stage);

    log::info!("loading training config variables");

    let max_generations = CONFIG.genetic_algorithm.max_generations as u64;
    let seed = CONFIG.genetic_algorithm.seed as u64;
    let with_seed = CONFIG.genetic_algorithm.with_seed;
    let population_size = CONFIG.genetic_algorithm.initial_population_size as usize;
    let number_of_network_updates = CONFIG.neural_network.number_of_network_updates as usize;
    let variance_decay = CONFIG.genetic_algorithm.mutation_rates.variance_decay as f32;
    let goal_fitness = CONFIG.genetic_algorithm.goal_fitness as f32;
    let data_path = CONFIG.image_processing.training.path as &str;

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
        Population::new(&population_bar, population_size)
    };
    population_bar.finish();

    log::info!("loading training dataset...");

    let mut parser = XMLParser::new();
    let dir = std::fs::read_dir(PathBuf::from(data_path))?;

    let mut idx = 0usize;
    for folder in dir {
        if idx == CONFIG.image_processing.training.load_amount as usize
            && !CONFIG.image_processing.training.load_all
        {
            break;
        };
        let drafter_path = folder?.path();
        parser.load(
            drafter_path,
            LoadFolder::Resized,
            CONFIG.image_processing.training.load_all as bool,
            CONFIG.image_processing.training.load_amount as usize,
        )?;
        idx += 1;
    }

    log::info!("loaded {} images", parser.loaded);

    let fitness_function = match stage {
        TrainingStage::Artificial { stage: 0 } => fitness_pixel_follow,
        TrainingStage::Artificial { stage: 1 } => fitness_recognize_components,
        TrainingStage::Artificial { stage: 2..=u8::MAX } => panic!("No third stage defined"),
        TrainingStage::RealBinarized => todo!(),
        TrainingStage::Real => todo!(),
    };

    let algorithm_bar = ProgressBar::new(max_generations);

    //increas number of updates over time
    let mut nr_updates = number_of_network_updates;
    // loop until stop criterial is met
    log::info!("training agents...");
    loop {
        algorithm_bar.inc(1);
        // for each image in the dataset
        for (annotation, image) in parser.data.iter() {
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
                            &mut image.clone(),
                            annotation,
                            nr_updates,
                        )
                        .unwrap();
                    agent.add_to_fitness(fitness);
                });

            // after one image was processed, save a netlist per image
            population.agents_mut().par_iter_mut().for_each(|agent| {
                let generated_netlist = agent.genotype().generate();
                if let Some((_, _, netlist)) = agent
                    .statistics_mut()
                    .get_mut(&ImageLabel(annotation.filename.clone()))
                {
                    *netlist = generated_netlist;
                }
            });
        }

        // average each agents fitness over the number of images
        population
            .agents_mut()
            .par_iter_mut()
            .enumerate()
            .for_each(|(_, agent)| {
                agent.set_fitness(agent.fitness() / parser.loaded as f32);
            });

        // sort the population by fitness
        population
            .agents_mut()
            .sort_by(|a, b| b.fitness().partial_cmp(&a.fitness()).unwrap());

        // printing the best agents fitnes per iteration
        let highest = population.agents()[0].fitness();
        let lowest = population
            .agents()
            .iter()
            .last()
            .map(|a| a.fitness())
            .unwrap();
        let average = population
            .agents()
            .iter()
            .fold(0f32, |acc, a| acc + a.fitness())
            / population.agents().len() as f32;
        algorithm_bar.println(format!(
            "highest: {}, lowest: {}, avarage: {}, nr_updates: {}",
            round3(highest),
            round3(lowest),
            round3(average),
            nr_updates
        ));

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
        population.evolve(new_agents, &mut rng);

        if population.generation() % CONFIG.neural_network.increase_every_generations as u32 == 0
            && CONFIG.neural_network.increase as bool
        {
            nr_updates += CONFIG.neural_network.by_amount as usize;
        }
    }
    algorithm_bar.finish();
    log::info!("training finished");
    log::info!("stopped after {} generations", population.generation());

    // remove 'agents' directory if it exists
    std::fs::remove_dir_all(save_path.clone()).unwrap_or_default();

    population
        .agents_mut()
        .par_iter_mut()
        .enumerate()
        .inspect(|(index, agent)| {
            log::debug!("agent {} fitness: {}", index, agent.fitness());
        })
        .for_each(|(index, agent)| {
            // saves the folder name with index + fitness round2
            let agent_folder = format!("{}_{}", round2(agent.fitness()), index);
            std::fs::create_dir_all(format!("{}/{}", save_path, agent_folder)).unwrap();

            // save control network
            let folder_name = "control";
            std::fs::create_dir_all(format!("{}/{}/{}", save_path, agent_folder, folder_name))
                .unwrap();
            agent
                .genotype()
                .control_network()
                .short_term_memory()
                .visualize(format!(
                    "{}/{}/{}/memory.png",
                    save_path, agent_folder, folder_name
                ))
                .unwrap();
            agent
                .genotype_mut()
                .control_network_mut()
                .to_json(format!(
                    "{}/{}/{}/network.json",
                    save_path, agent_folder, folder_name
                ))
                .unwrap();
            agent
                .genotype()
                .control_network()
                .to_dot(format!(
                    "{}/{}/{}/network.dot",
                    save_path, agent_folder, folder_name
                ))
                .unwrap();

            // save categorize network
            let folder_name = "categorize";
            std::fs::create_dir_all(format!("{}/{}/{}", save_path, agent_folder, folder_name))
                .unwrap();
            agent
                .genotype()
                .categorize_network()
                .short_term_memory()
                .visualize(format!(
                    "{}/{}/{}/memory.png",
                    save_path, agent_folder, folder_name
                ))
                .unwrap();
            agent
                .genotype_mut()
                .categorize_network_mut()
                .to_json(format!(
                    "{}/{}/{}/network.json",
                    save_path, agent_folder, folder_name
                ))
                .unwrap();
            agent
                .genotype()
                .categorize_network()
                .to_dot(format!(
                    "{}/{}/{}/network.dot",
                    save_path, agent_folder, folder_name
                ))
                .unwrap();
        });

    // for the best agent, create the images
    log::info!("generating images for the best agents ..");
    population.agents().iter().take(1).for_each(|agent| {
        agent
            .statistics()
            .par_iter()
            .for_each(|(label, (image, _, netlist))| {
                let folder_name = format!("best_agent_{}", round2(agent.fitness()));
                std::fs::create_dir_all(format!("{}/{}/{}", save_path, folder_name, label))
                    .unwrap();

                image
                    .save_with_retina(PathBuf::from(format!(
                        "{}/{}/{}/retina.png",
                        save_path, folder_name, label
                    )))
                    .unwrap();
                image
                    .save_with_retina_upscaled(PathBuf::from(format!(
                        "{}/{}/{}/retina_orig.png",
                        save_path, folder_name, label
                    )))
                    .unwrap();

                // save netlist
                std::fs::write(
                    format!("{}/{}/{}/netlist.net", save_path, folder_name, label),
                    netlist.clone(),
                )
                .unwrap();
            });
    });
    Ok(())
}
