use indicatif::ProgressBar;
use rand::prelude::*;
use rayon::prelude::*;
use crate::image_processing::TrainingStage;
use crate::{Result, CONFIG, Population, ImageReader, Agent, AgentEvaluation, SelectionMethod, ChaCha8Rng};
use crate::netlist::Generate;

pub fn train_agents(stage: TrainingStage) -> Result {
    log::info!("starting training stage {:?}", stage);

    log::info!("loading training config variables");

    let max_generations = CONFIG.genetic_algorithm.max_generations as u64;
    let seed = CONFIG.genetic_algorithm.seed as u64;
    let with_seed = CONFIG.genetic_algorithm.with_seed;
    let path_to_image_descriptions = CONFIG.image_processing.path_to_image_descriptions as &str;
    let neurons_per_rnn = CONFIG.neural_network.neurons_per_network as usize;
    let population_size = CONFIG.genetic_algorithm.population_size as usize;
    let number_of_network_updates = CONFIG.neural_network.number_of_network_updates as usize;
    let take_agents = CONFIG.genetic_algorithm.take_agents as usize;
    let path_to_agents_dir = CONFIG.image_processing.path_to_agents_dir as &str;
    let networks_per_agent = CONFIG.neural_network.networks_per_agent as usize;
    let variance_decay = CONFIG.genetic_algorithm.mutation_rates.variance_decay as f32;

    log::info!("setting up rng");

    let mut rng = ChaCha8Rng::from_entropy();

    if with_seed {
        log::info!("using seed: {}", seed);
        rng = ChaCha8Rng::seed_from_u64(seed);
    }

    log::info!("initializing population...");

    // intialize population
    let population_bar = ProgressBar::new(population_size as u64);
    let mut population = Population::new(
        &population_bar,
        population_size,
        networks_per_agent,
        neurons_per_rnn,
    );
    population_bar.finish();

    log::info!("loading training dataset...");

    // create a reader to buffer training dataset
    let image_path = match stage {
        TrainingStage::Artificial => CONFIG.image_processing.path_to_training_artificial as &str,
        TrainingStage::RealBinarized => CONFIG.image_processing.path_to_training_binarized as &str,
        TrainingStage::Real => CONFIG.image_processing.path_to_analysis_stage as &str,
    };
    let image_reader = ImageReader::from_path(
        image_path.to_string(),
        path_to_image_descriptions.to_string(),
        stage,
    )?;

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
            population.agents_mut().par_iter_mut().enumerate().for_each(|(_, agent)| {
                let fitness = agent
                    .evaluate(
                        &mut rng.clone(),
                        label.clone(),
                        &mut image.clone(),
                        description.clone(),
                        number_of_network_updates,
                    )
                    .unwrap();
                agent.add_to_fitness(fitness);
            });
        }

        // average each agents fitness over the number of images
        population.agents_mut().par_iter_mut().enumerate().for_each(|(_, agent)| {
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
            .any(|agent| agent.fitness() > 0.99)
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
    std::fs::remove_dir_all(path_to_agents_dir).unwrap_or_default();

    let generating_files_bar = ProgressBar::new(take_agents as u64);
    population
        .agents_mut()
        .par_iter_mut()
        .enumerate()
        .inspect(|(index, agent)| {
            log::debug!("agent {} fitness: {}", index, agent.fitness());
            log::debug!("agent {} netlist: {:?}", index, agent.generate());
        })
        .take(take_agents)
        .for_each(|(index, agent)| {
            let netlist = agent.generate();
            agent
                .statistics_mut()
                .par_iter_mut()
                .for_each(|(label, (image, genotype))| {
                    std::fs::create_dir_all(format!("{}/{}/{}", path_to_agents_dir, index, label))
                        .unwrap();
                    image
                        .save_with_retina(format!(
                            "{}/{}/{}/retina.png",
                            path_to_agents_dir, index, label
                        ))
                        .unwrap();
                    image
                        .save_with_retina_upscaled(format!(
                            "{}/{}/{}/retina_orig.png",
                            path_to_agents_dir, index, label
                        ))
                        .unwrap();

                    // save netlist
                    std::fs::write(
                        format!("{}/{}/{}/netlist.net", path_to_agents_dir, index, label),
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
                                path_to_agents_dir, index, label, i
                            ))
                            .unwrap();
                            network
                                .short_term_memory()
                                .visualize(format!(
                                    "{}/{}/{}/{}/memory.png",
                                    path_to_agents_dir, index, label, i
                                ))
                                .unwrap();
                            network
                                .to_json(format!(
                                    "{}/{}/{}/{}/network.json",
                                    path_to_agents_dir, index, label, i
                                ))
                                .unwrap();
                            network
                                .to_dot(format!(
                                    "{}/{}/{}/{}/network.dot",
                                    path_to_agents_dir, index, label, i
                                ))
                                .unwrap();
                        });
                });
            generating_files_bar.inc(1);
        });
    generating_files_bar.finish();

    Ok(())
}
