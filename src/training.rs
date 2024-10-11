use core::panic;
use std::fs::write;
use std::io::Write;
use std::path::PathBuf;
use std::u8;

use crate::annotations::{Annotation, LoadFolder, XMLParser};
use crate::image::{Image, ImageLabel, Position, TrainingStage};
use crate::netlist::{ComponentType, Generate};
use crate::{
    netlist_empty, plotting, round2, round3, AdaptiveConfig, Agent, AgentEvaluation, ChaCha8Rng,
    Population, Result, Retina, SelectionMethod, CONFIG,
};
use log::info;
use rand::prelude::*;
use rayon::prelude::*;

fn fitness(agent: &mut Agent, annotation: &Annotation, retina: &Retina, image: &Image) -> f32 {
    // the output of one neuron needs to exceed this value to count as active
    let active_threshold = 0.99;

    let source_dc_neuron_idx = 0usize;
    let resistor_neuron_idx = 1usize;
    let capacitor_neuron_idx = 2usize;

    let mut categorize_fitness = 0f32;
    annotation.objects.iter().for_each(|obj| {
        let bndbox = image.translate_bndbox_to_size(annotation, obj);
        // we need to check if any bndbox specified in the annotation is currently inside the retina rectangle
        if image.wraps_bndbox(&bndbox, retina) {
            // If anyone is, then we check we check what kind of component is specified in this object
            let full_component = obj.name.clone();
            let component = full_component.split(".").take(1).collect::<String>();
            // And lastly we check if the corresponding neuron is active for this component and every other neuron is inactive
            if component == "resistor".to_string() {
                if agent.genotype().categorize_network().neurons()[resistor_neuron_idx].output()
                    >= active_threshold
                // && agent.genotype().categorize_network().neurons()[source_dc_neuron_idx]
                //     .output()
                //     <= -active_threshold
                // && agent.genotype().categorize_network().neurons()[capacitor_neuron_idx]
                //     .output()
                //     <= -active_threshold
                {
                    categorize_fitness = 1f32;
                    agent.genotype_mut().add_found_component(
                        Position::from(bndbox.clone()),
                        ComponentType::Resistor,
                    );
                }
            } else if component == "voltage".to_string() {
                if agent.genotype().categorize_network().neurons()[source_dc_neuron_idx].output()
                    <= -active_threshold
                // && agent.genotype().categorize_network().neurons()[source_dc_neuron_idx]
                //     .output()
                //     >= active_threshold
                // && agent.genotype().categorize_network().neurons()[capacitor_neuron_idx]
                //     .output()
                //     <= -active_threshold
                {
                    categorize_fitness = 1f32;
                    agent.genotype_mut().add_found_component(
                        Position::from(bndbox.clone()),
                        ComponentType::VoltageSourceDc,
                    );
                }
            } else if component == "capacitor".to_string() {
                if agent.genotype().categorize_network().neurons()[capacitor_neuron_idx].output()
                    <= -active_threshold
                // && agent.genotype().categorize_network().neurons()[source_dc_neuron_idx]
                //     .output()
                //     <= -active_threshold
                // && agent.genotype().categorize_network().neurons()[capacitor_neuron_idx]
                //     .output()
                //     >= active_threshold
                {
                    categorize_fitness = 1f32;
                    agent.genotype_mut().add_found_component(
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
            //         categorize_fitness = 0.5f32;
            //     }
            // }
        }
    });
    let control_fitness = retina.percentage_visited();
    // (categorize_fitness + control_fitness) / 2.0f32
    control_fitness
}

pub fn test_agents(path: String, number_of_updates: usize) -> Result {
    let mut population = Population::from_path(path)?;

    let rng = ChaCha8Rng::from_entropy();
    let data_path = CONFIG.image_processing.testing.path as &str;
    let mut parser = XMLParser::new();
    let dir = std::fs::read_dir(PathBuf::from(data_path))?;

    for folder in dir {
        let drafter_path = folder?.path();
        parser.load(drafter_path, LoadFolder::Resized, true, 0)?;
    }

    for (annotation, image, optimal_netlist) in parser.data.iter() {
        // for each image, let the agents evaluate once
        population
            .agents_mut()
            .par_iter_mut()
            .enumerate()
            .for_each(|(_, agent)| {
                let fitness = agent
                    .evaluate(
                        fitness,
                        &mut rng.clone(),
                        &mut image.clone(),
                        annotation,
                        number_of_updates,
                    )
                    .unwrap();
                agent.add_to_fitness(fitness);
            });
        // generate a netlist for each agent
        population.agents_mut().iter_mut().for_each(|agent| {
            let generated_netlist = agent.genotype().generate();
            if let Some((_, _, netlist)) = agent
                .statistics_mut()
                .get_mut(&ImageLabel(annotation.filename.replace(".jpg", "").clone()))
            {
                *netlist = generated_netlist.clone();
            }
        });

        // compare each generated netlist with the optimal one and create a ranking for each image
        let mut ranking = vec![];
        population.agents().iter().for_each(|agent| {
            let mut final_points = 0;
            agent.statistics().iter().for_each(|(_, (_, _, netlist))| {
                // for each image the agent collects a point when a netlist was correct:
                // 1 point for each corrent number of components
                // 3 points when all components are correct
                let mut points = 0;
                // the lines which are already compared with from the optimal netlist are stored here
                let mut already_compared = vec![];
                netlist.lines().for_each(|line| {
                    if !line.starts_with(".") {
                        points += 1;
                        optimal_netlist
                            .lines()
                            .take_while(|&x| !already_compared.contains(&x.to_string()))
                            .for_each(|l| {
                                let component =
                                    line.split_whitespace().take(1).collect::<Vec<&str>>()[0];
                                let optimal_component =
                                    l.split_whitespace().take(1).collect::<Vec<&str>>()[0];
                                if component == optimal_component {
                                    points += 3;
                                }
                            });
                    }
                    already_compared.push(line.to_string());
                });
                final_points += points;
            });
            ranking.push((agent.clone(), final_points));
        });

        // store the ranking in a file
        ranking.sort_by_key(|(_, p)| p.clone());
        ranking.iter().for_each(|(agent, points)| {
            info!("points: {}", points);
        });
    }
    Ok(())
}

pub fn train_agents(
    stage: TrainingStage,
    load_path: Option<String>,
    save_path: String,
    iteration: usize,
    adaptive_config: &AdaptiveConfig,
    stuck_check: bool,
    stale_check: bool,
) -> Result {
    info!("{:?}", adaptive_config);

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
    let mut population = if load_path.is_some() {
        Population::from_path(load_path.unwrap())?
    } else {
        Population::new(population_size, adaptive_config)
    };

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

    let fitness_function = match stage {
        TrainingStage::Artificial { stage: 0 } => fitness,
        TrainingStage::Artificial { stage: 1 } => fitness,
        TrainingStage::Artificial { stage: 2..=u8::MAX } => panic!("No third stage defined"),
        TrainingStage::RealBinarized => todo!(),
        TrainingStage::Real => todo!(),
    };

    let mut average_fitness_data = vec![];
    let mut netlists_data = vec![];

    //increas number of updates over time
    let nr_updates = number_of_network_updates;
    // loop until stop criterial is met
    loop {
        info!(
            "{} / {}; images: {}",
            population.generation(),
            max_generations,
            parser.loaded
        );
        // for each image in the dataset
        for (annotation, image, _) in parser.data.iter() {
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

            // after one image was processed, save a netlist per image.
            // The netlist will be overwritten after one generation was processed, so it is always the newest list
            population.agents_mut().iter_mut().for_each(|agent| {
                let generated_netlist = agent.genotype().generate();
                if let Some((_, _, netlist)) = agent
                    .statistics_mut()
                    .get_mut(&ImageLabel(annotation.filename.replace(".jpg", "").clone()))
                {
                    *netlist = generated_netlist.clone();
                }
            });
        }

        // when all images where process, filter the agents which got a netlist generated and add them to the netlist count data
        let mut netlist_count = 0f32;
        population.agents().iter().for_each(|agent| {
            agent.statistics().iter().for_each(|(_, (_, _, netlist))| {
                if !netlist_empty(netlist) {
                    netlist_count += 1.;
                }
            });
        });
        netlists_data.push(netlist_count);

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
        let avrg_of_avrg =
            average_fitness_data.iter().sum::<f32>() / average_fitness_data.iter().len() as f32;

        if stale_check {
            if round3(population.agents()[0].fitness())
                == round3(population.agents().iter().last().unwrap().fitness())
            {
                // info("stale");

                std::fs::create_dir_all(format!("iterations/{}", iteration)).unwrap();
                plotting::update_image(
                    &average_fitness_data,
                    format!("iterations/{}/fitness.png", iteration).as_str(),
                    max_generations,
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
                // save netlists over time
                plotting::netlists_over_time(
                    &netlists_data,
                    format!("iterations/{}/netlist_plot.png", iteration).as_str(),
                    population_size,
                    max_generations,
                );
                // info(format!("generations survived {}", population.generation()));
                // break out of outer loop
                break;
            }
        }
        if stuck_check {
            if (population.generation() as f32 / adaptive_config.max_generations as f32) * 0.5
                > avrg_of_avrg
            {
                // info("stuck");

                std::fs::create_dir_all(format!("iterations/{}", iteration)).unwrap();
                plotting::update_image(
                    &average_fitness_data,
                    format!("iterations/{}/fitness.png", iteration).as_str(),
                    max_generations,
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
                        "iteration: {} - generations survived: {} \n",
                        iteration,
                        population.generation()
                    ))
                    .unwrap();
                // info(format!("generations survived {}", population.generation()));
                // save netlists over time
                plotting::netlists_over_time(
                    &netlists_data,
                    format!("iterations/{}/netlist_plot.png", iteration).as_str(),
                    population_size,
                    max_generations,
                );
                // break out of outer loop
                break;
            }
        }
        if !stale_check && !stuck_check {
            // save average fitness nonetheless
            std::fs::create_dir_all(format!("best_iterations/{}", iteration)).unwrap();
            plotting::update_image(
                &average_fitness_data,
                format!("best_iterations/{}/fitness.png", iteration).as_str(),
                max_generations,
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
                let (parent1, parent2) = population.select(
                    &mut rng,
                    SelectionMethod::Tournament,
                    Some(adaptive_config.tournament_size),
                );
                let mut offspring = parent1.crossover(&mut rng, parent2);
                offspring.mutate(&mut rng, adaptive_config);
                offspring
            })
            .collect::<Vec<Agent>>();

        // evolve the population
        population.evolve(new_agents, population_size);
    }

    // save fitness and configuration the same way as it where stuck or stale
    std::fs::create_dir_all(format!("iterations/{}", iteration)).unwrap();
    plotting::update_image(
        &average_fitness_data,
        format!("iterations/{}/fitness.png", iteration).as_str(),
        max_generations,
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

    // save netlists over time
    plotting::netlists_over_time(
        &netlists_data,
        format!("iterations/{}/netlist_plot.png", iteration).as_str(),
        population_size,
        max_generations,
    );

    // remove 'agents' directory if it exists
    std::fs::remove_dir_all(save_path.clone()).unwrap_or_default();

    population
        .agents_mut()
        .par_iter_mut()
        .enumerate()
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
                .visualize(
                    format!("{}/{}/{}/memory.png", save_path, agent_folder, folder_name),
                    adaptive_config.number_of_network_updates,
                )
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
                .visualize(
                    format!("{}/{}/{}/memory.png", save_path, agent_folder, folder_name,),
                    adaptive_config.number_of_network_updates,
                )
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
