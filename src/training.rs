use core::panic;
use std::fs::write;
use std::io::Write;
use std::path::PathBuf;
use std::u8;

use crate::annotations::{Annotation, LoadFolder, XMLParser};
use crate::image::{Image, ImageLabel, TrainingStage};
use crate::netlist::Generate;
use crate::{
    plotting, round2, round3, AdaptiveConfig, Agent, AgentEvaluation, ChaCha8Rng,
    LocalMaximumError, Population, Result, Retina, SelectionMethod, CONFIG,
};
use bevy::log::{debug, info};
use bevy::utils::info;
use indicatif::ProgressBar;
use rand::prelude::*;
use rayon::prelude::*;
use serde::Serialize;

fn fitness(agent: &mut Agent, annotation: &Annotation, retina: &Retina, image: &Image) -> f32 {
    let source_dc_neuron_idx = 0usize;
    let resistor_neuron_idx = 1usize;
    let capacitor_neuron_idx = 2usize;

    let mut categorize_fitness = 0f32;
    annotation.objects.iter().for_each(|obj| {
        // we need to check if any bndbox specified in the annotation is currently inside the retina rectangle
        if image.wraps_bndbox(&obj.bndbox, retina) {
            // If anyone is, then we check we check what kind of component is specified in this object
            let full_component = obj.name.clone();
            let component = full_component.split(".").take(1).collect::<String>();

            // And lastly we check if the corresponding neuron is active for this component and every other neuron is inactive
            if component == "resistor".to_string() {
                if agent.genotype().categorize_network().neurons()[resistor_neuron_idx].output()
                    >= 1.0
                    && agent.genotype().categorize_network().neurons()[source_dc_neuron_idx]
                        .output()
                        <= -1.0
                    && agent.genotype().categorize_network().neurons()[capacitor_neuron_idx]
                        .output()
                        <= -1.0
                {
                    categorize_fitness = 1f32;
                }
            } else if component == "voltage".to_string() {
                if agent.genotype().categorize_network().neurons()[resistor_neuron_idx].output()
                    <= -1.0
                    && agent.genotype().categorize_network().neurons()[source_dc_neuron_idx]
                        .output()
                        >= 1.0
                    && agent.genotype().categorize_network().neurons()[capacitor_neuron_idx]
                        .output()
                        <= -1.0
                {
                    categorize_fitness = 1f32;
                }
            } else if component == "capacitor".to_string() {
                if agent.genotype().categorize_network().neurons()[resistor_neuron_idx].output()
                    <= -1.0
                    && agent.genotype().categorize_network().neurons()[source_dc_neuron_idx]
                        .output()
                        <= -1.0
                    && agent.genotype().categorize_network().neurons()[capacitor_neuron_idx]
                        .output()
                        >= 1.0
                {
                    categorize_fitness = 1f32;
                }
            } else {
                // nothing in the retina should gain fitness when every neuron is inactive
                if agent.genotype().categorize_network().neurons()[resistor_neuron_idx].output()
                    <= -1.0
                    && agent.genotype().categorize_network().neurons()[source_dc_neuron_idx]
                        .output()
                        <= -1.0
                    && agent.genotype().categorize_network().neurons()[capacitor_neuron_idx]
                        .output()
                        <= -1.0
                {
                    categorize_fitness = 1f32;
                }
            }
        }
    });

    let control_fitness = retina.percentage_visited();
    (categorize_fitness + control_fitness) / 2.0f32
}

#[allow(dead_code)]
pub fn test_agents() -> Result {
    todo!()
}

pub fn train_agents(
    stage: TrainingStage,
    load_path: Option<String>,
    save_path: String,
    iteration: usize,
    adaptive_config: &AdaptiveConfig,
    stuck_and_stale_check: bool,
) -> Result {
    info(format!("{:?}", adaptive_config));

    let max_generations = adaptive_config.max_generations;
    let population_size = adaptive_config.initial_population_size;
    let number_of_network_updates = adaptive_config.number_of_network_updates;
    let variance_decay = adaptive_config.variance_decay;

    let seed = CONFIG.genetic_algorithm.seed as u64;
    let with_seed = CONFIG.genetic_algorithm.with_seed;
    let goal_fitness = CONFIG.genetic_algorithm.goal_fitness as f32;
    let data_path = CONFIG.image_processing.training.path as &str;

    let mut rng = ChaCha8Rng::from_entropy();

    if with_seed {
        rng = ChaCha8Rng::seed_from_u64(seed);
    }

    // intialize population
    let population_bar = ProgressBar::new(population_size as u64);
    let mut population = if load_path.is_some() {
        Population::from_path(load_path.unwrap())?
    } else {
        Population::new(&population_bar, population_size)
    };
    population_bar.finish();

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

    info(format!("loaded {} images", parser.loaded));

    let fitness_function = match stage {
        TrainingStage::Artificial { stage: 0 } => fitness,
        TrainingStage::Artificial { stage: 1 } => fitness,
        TrainingStage::Artificial { stage: 2..=u8::MAX } => panic!("No third stage defined"),
        TrainingStage::RealBinarized => todo!(),
        TrainingStage::Real => todo!(),
    };

    let algorithm_bar = ProgressBar::new(max_generations);

    let mut average_fitness_data = vec![];

    //increas number of updates over time
    let mut nr_updates = number_of_network_updates;
    // loop until stop criterial is met
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

        // save fitness data for plotting
        let average = population
            .agents()
            .iter()
            .fold(-1f32, |acc, a| acc + a.fitness())
            / population.agents().len() as f32;
        average_fitness_data.push(average);

        // if the minimum fitness and the max fitness is the same, local maximum reached, break.
        // or when the avarage fitness is stuck for over X generations
        if stuck_and_stale_check {
            let stale = round3(population.agents()[0].fitness())
                == round3(population.agents().iter().last().unwrap().fitness());
            let avrg_of_avrg =
                average_fitness_data.iter().sum::<f32>() / average_fitness_data.iter().len() as f32;
            let stuck = (population.generation() as f32 / adaptive_config.max_generations as f32)
                > avrg_of_avrg;
            if stale || stuck {
                if stale {
                    info("stale");
                }
                if stuck {
                    info("stuck");
                }
                std::fs::create_dir_all(format!("iterations/{}", iteration)).unwrap();
                plotting::update_image(
                    &average_fitness_data,
                    format!("iterations/{}/fitness.png", iteration).as_str(),
                );
                let out = serde_json::to_string_pretty(adaptive_config).unwrap();
                write(
                    format!("iterations/{}/config.json", iteration).as_str(),
                    out,
                )
                .unwrap();
                std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open("iteration_results.txt")
                    .unwrap()
                    .write_fmt(format_args!(
                        "generations survived: {} ",
                        population.generation()
                    ))
                    .unwrap();
                info(format!("generations survived {}", population.generation()));
                return Err(LocalMaximumError.into());
            }
        } else {
            // save average fitness nonetheless
            std::fs::create_dir_all(format!("best_iterations/{}", iteration)).unwrap();
            plotting::update_image(
                &average_fitness_data,
                format!("best_iterations/{}/fitness.png", iteration).as_str(),
            );
        }
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
                agent.update_variance(agent.get_current_variance() * variance_decay as f32);
            })
        }

        // select, crossover and mutate
        let new_agents = (0..population.agents().len())
            .map(|_| {
                let (parent1, parent2) = population.select(&mut rng, SelectionMethod::Tournament);
                let mut offspring = parent1.crossover(&mut rng, parent2);
                offspring.mutate(&mut rng, adaptive_config);
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

    std::fs::create_dir_all(format!("iterations/final")).unwrap();
    let out = serde_json::to_string_pretty(adaptive_config).unwrap();
    write("iterations/final/config.json", out).unwrap();
    plotting::update_image(&average_fitness_data, "iterations/final/fitness.png");

    // remove 'agents' directory if it exists
    std::fs::remove_dir_all(save_path.clone()).unwrap_or_default();

    population
        .agents_mut()
        .par_iter_mut()
        .enumerate()
        .inspect(|(index, agent)| {
            // debug("agent {} fitness: {}", index, agent.fitness());
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
