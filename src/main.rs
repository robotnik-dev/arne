use annotations::{Annotation, XMLParser};
use bevy::prelude::*;
use bevy::{
    log::{Level, LogPlugin},
    state::app::StatesPlugin,
    utils::info,
};
use bevy_prng::WyRand;
use bevy_rand::prelude::{EntropyComponent, EntropyPlugin};
use bevy_rand::traits::ForkableRng;
use genetic_algorithm::{Agent, Genotype};
use image::Image;
use itertools::Itertools;
use netlist::{Build, Generate, Netlist};
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use serde::{Deserialize, Serialize};
use serde_json::from_str;
use std::fs::{read_to_string, write};
use std::path::PathBuf;
pub use std::time::{Duration, Instant};
mod utils;
pub use utils::{
    amount_of_components, dot_product, netlist_empty, round2, round3, round_to_decimal_places,
};

mod image;
pub use image::Retina;

mod neural_network;
pub use neural_network::{Rnn, ShortTermMemory, SnapShot};

mod genetic_algorithm;
pub use genetic_algorithm::SelectionMethod;

mod annotations;
mod netlist;
mod plotting;

mod training;

type Error = Box<dyn std::error::Error>;
type Result = std::result::Result<(), Error>;

fn main() {
    App::new()
        .add_plugins((
            MinimalPlugins,
            StatesPlugin,
            LogPlugin {
                level: Level::INFO,
                filter: "wgpu=error,bevy_render=info,bevy_ecs=trace".to_string(),
                custom_layer: |_| None,
            },
        ))
        .add_plugins(EntropyPlugin::<WyRand>::default())
        .insert_state(AppState::default())
        .insert_resource(AdaptiveConfig::default())
        .insert_resource(XMLParser::default())
        .insert_resource(Generation::default())
        // first loading the config
        .add_systems(Startup, (load_config,))
        // .add_systems(PostStartup, decide_training_or_testing)
        // after this do the preprocessing of images if enabled in the config
        .add_systems(Update, (preprocess,).run_if(in_state(AppState::Preprocess)))
        // next load all images and setup the run
        .add_systems(
            Update,
            (
                setup_source,
                load_images,
                initialize_population,
                initialize_average_fitness,
                decide_which_algorithm,
            )
                .chain()
                .run_if(in_state(AppState::Setup)),
        )
        .add_systems(
            Update,
            (evaluate_agents_testing, create_ranking)
                .chain()
                .run_if(in_state(AppState::EvaluateGenerationTesting)),
        )
        .add_systems(
            Update,
            (evaluate_agents_training, genetic_algorithm_step)
                .chain()
                .run_if(in_state(AppState::EvaluateGenerationTraining)),
        )
        .add_systems(
            Update,
            (increase_generation_count, clear_stats, tick)
                .chain()
                .run_if(in_state(AppState::PrepareNewGenerationTraining)),
        )
        .add_systems(Update, (cleanup,).run_if(in_state(AppState::AlgorithmDone)))
        // .add_systems(Update, log_transitions::<AppState>)
        .run();
}

#[derive(Component)]
struct Source;

#[derive(States, Debug, PartialEq, Hash, Eq, Clone, Default)]
enum AppState {
    #[default]
    Preprocess,
    Setup,
    EvaluateGenerationTraining,
    PrepareNewGenerationTraining,
    EvaluateGenerationTesting,
    AlgorithmDone,
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Resource, Default)]
pub struct AdaptiveConfig {
    pub number_of_populations: usize,
    pub number_of_network_updates: usize,
    pub neuron_lower: f32,
    pub neuron_upper: f32,
    pub retina_lower: f32,
    pub retina_upper: f32,
    pub init_non_zero_retina_weights: f32,
    pub population_size: usize,
    pub max_generations: u64,
    pub tournament_size: usize,
    pub variance: f32,
    pub variance_decay: f32,
    pub mean: f32,
    pub delete_neuron: f32,
    pub delete_weights: f32,
    pub delete_bias: f32,
    pub delete_self_activation: f32,
    pub mutate_neuron: f32,
    pub mutate_weights: f32,
    pub mutate_bias: f32,
    pub mutate_self_activation: f32,
    pub goal_image_width: usize,
    pub goal_image_height: usize,
    pub retina_size: usize,
    pub superpixel_size: usize,
    pub retina_circle_radius: f32,
    pub retina_label_scale: f32,
    pub sobel_threshold: f32,
    pub erode_pixels: usize,
    pub preprocess: bool,
    pub training_path: String,
    pub training_load_all: bool,
    pub training_load_amount: usize,
    pub testing_path: String,
    pub testing_load_all: bool,
    pub testing_load_amount: usize,
    pub control_network_neurons: usize,
    pub categorize_network_neurons: usize,
    pub retina_movement_speed: f32,
    pub with_seed: bool,
    pub seed: usize,
    pub goal_fitness: f32,
    pub agent_save_path: String,
    pub tested_agent_save_path: String,
    pub testing_stage: bool,
}

#[derive(Resource, Default)]
struct Generation(u64);

#[derive(Component, Debug, Clone)]
struct Stats {
    original_agent: Agent,
    evaluated_agent: Agent,
    evaluated_image: Image,
    annotation: Annotation,
    netlist: Netlist,
    optimal_netlist: Netlist,
}

#[derive(Component, Default)]
struct AverageFitness(Vec<f32>);

fn setup_source(mut commands: Commands, adaptive_config: Res<AdaptiveConfig>) {
    if adaptive_config.with_seed {
        commands.spawn((
            Source,
            EntropyComponent::<WyRand>::from_seed(adaptive_config.seed.to_ne_bytes()),
        ));
    } else {
        commands.spawn((Source, EntropyComponent::<WyRand>::default()));
    }
}

/// preprocesses images, only once needed for each image (just enable this to create more images to be used by the algorithm)
/// and put beforehand all the images you need into the data/training ord data/testing folder
fn preprocess(adaptive_config: Res<AdaptiveConfig>, mut next: ResMut<NextState<AppState>>) {
    if adaptive_config.preprocess {
        info("preprocessing images");
        // training folder
        for entry in std::fs::read_dir("data/training").unwrap() {
            let entry = entry.unwrap();
            let path = entry.path();
            XMLParser::resize_segmented_images(path, &adaptive_config);
        }
        // testing folder
        for entry in std::fs::read_dir("data/testing").unwrap() {
            let entry = entry.unwrap();
            let path = entry.path();
            XMLParser::resize_segmented_images(path, &adaptive_config);
        }
    }

    next.set(AppState::Setup);
}

fn load_config(mut adaptive_config: ResMut<AdaptiveConfig>) {
    let filepath = String::from("bevy_port_config.json");
    let loaded_adaptive_config: AdaptiveConfig =
        from_str(read_to_string(filepath).unwrap().as_str()).unwrap();
    adaptive_config.number_of_populations = loaded_adaptive_config.number_of_populations;
    adaptive_config.number_of_network_updates = loaded_adaptive_config.number_of_network_updates;
    adaptive_config.neuron_lower = loaded_adaptive_config.neuron_lower;
    adaptive_config.neuron_upper = loaded_adaptive_config.neuron_upper;
    adaptive_config.retina_lower = loaded_adaptive_config.retina_lower;
    adaptive_config.retina_upper = loaded_adaptive_config.retina_upper;
    adaptive_config.init_non_zero_retina_weights =
        loaded_adaptive_config.init_non_zero_retina_weights;
    adaptive_config.population_size = loaded_adaptive_config.population_size;
    adaptive_config.max_generations = loaded_adaptive_config.max_generations;
    adaptive_config.tournament_size = loaded_adaptive_config.tournament_size;
    adaptive_config.variance = loaded_adaptive_config.variance;
    adaptive_config.variance_decay = loaded_adaptive_config.variance_decay;
    adaptive_config.mean = loaded_adaptive_config.mean;
    adaptive_config.delete_neuron = loaded_adaptive_config.delete_neuron;
    adaptive_config.delete_weights = loaded_adaptive_config.delete_weights;
    adaptive_config.delete_self_activation = loaded_adaptive_config.delete_self_activation;
    adaptive_config.mutate_neuron = loaded_adaptive_config.mutate_neuron;
    adaptive_config.mutate_weights = loaded_adaptive_config.mutate_weights;
    adaptive_config.mutate_bias = loaded_adaptive_config.mutate_bias;
    adaptive_config.mutate_self_activation = loaded_adaptive_config.mutate_self_activation;
    adaptive_config.goal_image_width = loaded_adaptive_config.goal_image_width;
    adaptive_config.goal_image_height = loaded_adaptive_config.goal_image_height;
    adaptive_config.retina_size = loaded_adaptive_config.retina_size;
    adaptive_config.superpixel_size = loaded_adaptive_config.superpixel_size;
    adaptive_config.retina_circle_radius = loaded_adaptive_config.retina_circle_radius;
    adaptive_config.retina_label_scale = loaded_adaptive_config.retina_label_scale;
    adaptive_config.sobel_threshold = loaded_adaptive_config.sobel_threshold;
    adaptive_config.erode_pixels = loaded_adaptive_config.erode_pixels;
    adaptive_config.preprocess = loaded_adaptive_config.preprocess;
    adaptive_config.training_path = loaded_adaptive_config.training_path;
    adaptive_config.training_load_all = loaded_adaptive_config.training_load_all;
    adaptive_config.training_load_amount = loaded_adaptive_config.training_load_amount;
    adaptive_config.testing_path = loaded_adaptive_config.testing_path;
    adaptive_config.testing_load_all = loaded_adaptive_config.testing_load_all;
    adaptive_config.testing_load_amount = loaded_adaptive_config.testing_load_amount;
    adaptive_config.control_network_neurons = loaded_adaptive_config.control_network_neurons;
    adaptive_config.categorize_network_neurons = loaded_adaptive_config.categorize_network_neurons;
    adaptive_config.retina_movement_speed = loaded_adaptive_config.retina_movement_speed;
    adaptive_config.with_seed = loaded_adaptive_config.with_seed;
    adaptive_config.seed = loaded_adaptive_config.seed;
    adaptive_config.goal_fitness = loaded_adaptive_config.goal_fitness;
    adaptive_config.agent_save_path = loaded_adaptive_config.agent_save_path;
    adaptive_config.tested_agent_save_path = loaded_adaptive_config.tested_agent_save_path;
    adaptive_config.testing_stage = loaded_adaptive_config.testing_stage;
    println!("{:#?}", adaptive_config);
}

fn load_images(
    mut xml_parser: ResMut<XMLParser>,
    mut commands: Commands,
    adaptive_config: Res<AdaptiveConfig>,
    mut q_source: Query<&mut EntropyComponent<WyRand>, With<Source>>,
) {
    let (data_path, load_amount, load_all) = match adaptive_config.testing_stage {
        true => (
            adaptive_config.testing_path.to_string(),
            adaptive_config.testing_load_amount,
            adaptive_config.testing_load_all,
        ),
        false => (
            adaptive_config.training_path.to_string(),
            adaptive_config.training_load_amount,
            adaptive_config.training_load_all,
        ),
    };
    let dir = std::fs::read_dir(PathBuf::from(data_path)).unwrap();
    for (idx, folder) in dir.enumerate() {
        if idx == load_amount && !load_all {
            break;
        };
        let drafter_path = folder.unwrap().path();
        xml_parser
            .load(drafter_path, load_all, load_amount)
            .unwrap();
    }

    let mut rng = q_source.single_mut().fork_rng();
    for (annotation, image, netlist) in xml_parser.data.iter() {
        commands
            .spawn(Image {
                id: rng.gen(),
                rgba: image.rgba.clone(),
                grey: image.grey.clone(),
                width: image.width,
                height: image.height,
                format: image.format.clone(),
                retina_positions: image.retina_positions.clone(),
                dark_pixel_positions: image.dark_pixel_positions.clone(),
            })
            .insert(Annotation {
                folder: annotation.folder.clone(),
                filename: annotation.filename.clone(),
                path: annotation.path.clone(),
                source: annotation.source.clone(),
                size: annotation.size.clone(),
                segmented: annotation.segmented.clone(),
                objects: annotation.objects.clone(),
            })
            .insert(netlist.clone());
    }
}

fn initialize_population(
    adaptive_config: Res<AdaptiveConfig>,
    mut commands: Commands,
    mut q_source: Query<&mut EntropyComponent<WyRand>, With<Source>>,
) {
    let mut source = q_source.single_mut();
    let mut agents = vec![];
    if adaptive_config.testing_stage {
        // load agents from path
        let path = adaptive_config.agent_save_path.to_string();
        std::fs::read_dir(path).unwrap().for_each(|entry| {
            let path_buf = entry.unwrap().path();
            if !path_buf
                .file_name()
                .unwrap()
                .to_string_lossy()
                .starts_with("best_agent")
            {
                let agent = Agent::from_path(path_buf, &adaptive_config).unwrap();
                agents.push(agent);
            }
        });
    } else {
        // training stage -> generate new agents
        for _ in 0..adaptive_config.population_size {
            // spawn a new Agent
            let mut rng = source.fork_rng();
            let mut agent = Agent::new(rng.fork_rng());
            agent.genotype = Genotype::init(rng.fork_rng(), &adaptive_config);
            agents.push(agent);
        }
    }
    for agent in agents {
        commands
            .spawn(agent)
            // rng component forked
            .insert(source.fork_rng());
    }
}

fn tick(time: Res<Time<Real>>) {
    info(format!("elapsed time: {:?}", time.elapsed()));
}

fn initialize_average_fitness(mut commands: Commands) {
    commands.spawn(AverageFitness::default());
}

fn decide_which_algorithm(
    adaptive_config: Res<AdaptiveConfig>,
    mut next: ResMut<NextState<AppState>>,
) {
    if adaptive_config.testing_stage {
        next.set(AppState::EvaluateGenerationTesting);
    } else {
        next.set(AppState::EvaluateGenerationTraining);
    }
}

/// Test agents. Do all evaluation steps for all agents over all images
fn evaluate_agents_testing(
    q_images: Query<(&Image, &Annotation, &Netlist)>,
    adaptive_config: Res<AdaptiveConfig>,
    q_agents: Query<(Entity, &Agent, &EntropyComponent<WyRand>)>,
    par_commands: ParallelCommands,
) {
    q_agents.par_iter().for_each(|(_, agent, rng)| {
        q_images
            .par_iter()
            .for_each(|(image, annotation, optimal_netlist)| {
                par_commands.command_scope(|mut commands| {
                    let mut temp_agent = agent.clone();
                    let mut temp_image = image.clone();
                    let new_rng = rng.clone().fork_rng();
                    let fitness = temp_agent
                        .evaluate(&adaptive_config, &mut temp_image, annotation)
                        .unwrap();
                    assert_eq!(temp_agent.fitness(), 0.);
                    assert!(fitness <= 1.0);
                    temp_agent.add_to_fitness(fitness);

                    // generate a netlist for this image and this agent and append it
                    let netlist = temp_agent.genotype().build();
                    commands
                        .spawn(Stats {
                            // this is the original agent before evaluation and not changed snapshots
                            original_agent: agent.clone(),
                            // this is the changed agent with snapshots
                            evaluated_agent: temp_agent.clone(),
                            evaluated_image: temp_image.clone(),
                            annotation: annotation.clone(),
                            netlist,
                            optimal_netlist: optimal_netlist.clone(),
                        })
                        .insert(new_rng);
                });
            });
    });
}

fn create_ranking(mut next: ResMut<NextState<AppState>>) {
    // TODO: ranking

    next.set(AppState::AlgorithmDone);
}

/// Do all update steps for one generation
fn evaluate_agents_training(
    q_images: Query<(&Image, &Annotation, &Netlist)>,
    adaptive_config: Res<AdaptiveConfig>,
    q_agents: Query<(Entity, &Agent, &EntropyComponent<WyRand>)>,
    par_commands: ParallelCommands,
) {
    q_agents.par_iter().for_each(|(_, agent, rng)| {
        q_images
            .par_iter()
            .for_each(|(image, annotation, optimal_netlist)| {
                par_commands.command_scope(|mut commands| {
                    let mut temp_agent = agent.clone();
                    let mut temp_image = image.clone();
                    let new_rng = rng.clone().fork_rng();
                    let fitness = temp_agent
                        .evaluate(&adaptive_config, &mut temp_image, annotation)
                        .unwrap();
                    assert_eq!(temp_agent.fitness(), 0.);
                    assert!(fitness <= 1.0);
                    temp_agent.add_to_fitness(fitness);

                    // generate a netlist for this image and this agent and append it
                    let netlist = temp_agent.genotype().build();
                    commands
                        .spawn(Stats {
                            // this is the original agent before evaluation and not changed snapshots
                            original_agent: agent.clone(),
                            // this is the changed agent with snapshots
                            evaluated_agent: temp_agent.clone(),
                            evaluated_image: temp_image.clone(),
                            annotation: annotation.clone(),
                            netlist,
                            optimal_netlist: optimal_netlist.clone(),
                        })
                        .insert(new_rng);
                });
            });
    });
}

#[allow(clippy::too_many_arguments)]
fn genetic_algorithm_step(
    q_stats: Query<(&mut Stats, &EntropyComponent<WyRand>), Without<Source>>,
    par_commands: ParallelCommands,
    mut commands: Commands,
    adaptive_config: Res<AdaptiveConfig>,
    q_agents: Query<Entity, With<Agent>>,
    mut q_source: Query<&mut EntropyComponent<WyRand>, With<Source>>,
    mut q_average_fitness: Query<&mut AverageFitness>,
    mut next: ResMut<NextState<AppState>>,
    generation: ResMut<Generation>,
    xml_parser: Res<XMLParser>,
) {
    let mut source = q_source.single_mut();
    assert_eq!(
        q_stats.iter().len() / xml_parser.loaded,
        adaptive_config.population_size
    );

    let mut stats = q_stats
        .iter()
        .unique_by(|(stat, _)| stat.original_agent.id)
        .map(|(s, _)| {
            (
                s.evaluated_agent.clone(),
                s.evaluated_image.clone(),
                s.annotation.clone(),
            )
        })
        .collect::<Vec<(Agent, Image, Annotation)>>();
    assert_eq!(stats.iter().len(), adaptive_config.population_size);

    // take the best {population_size} agents for further processing
    stats.sort_by(|a, b| b.0.fitness().partial_cmp(&a.0.fitness()).unwrap());

    // save fitness data for plotting over generations
    let mut fitness_comp = q_average_fitness.single_mut();
    let average_fitness =
        stats.iter().fold(0f32, |acc, a| acc + a.0.fitness()) / stats.iter().len() as f32;
    fitness_comp.0.push(average_fitness);

    // TODO: after 50 % of max generations, decrease the variance by 10 % each generation

    // return and skip the next steps when max generations reached
    if generation.0 >= adaptive_config.max_generations {
        next.set(AppState::AlgorithmDone);
        return;
    }

    // select, crossover and mutate
    let new_agents = stats
        .iter()
        .map(|(_, image, annotation)| {
            // select
            let mut tournament = Vec::with_capacity(adaptive_config.tournament_size);
            for _ in 0..adaptive_config.tournament_size {
                tournament.push(stats.choose(&mut source.fork_rng()).unwrap());
            }
            tournament.sort_by(|a, b| b.0.fitness().partial_cmp(&a.0.fitness()).unwrap());
            // crossover
            let mut offspring = tournament[0]
                .0
                .crossover(source.fork_rng(), &tournament[1].0);
            // mutate
            offspring.mutate(source.fork_rng(), &adaptive_config);
            (offspring, image.clone(), annotation.clone())
        })
        .collect::<Vec<(Agent, Image, Annotation)>>();

    // evolve the population
    let mut combined = stats
        .iter()
        .cloned()
        .chain(new_agents.iter().cloned())
        .collect::<Vec<(Agent, Image, Annotation)>>();

    combined.sort_by(|a, b| b.0.fitness().partial_cmp(&a.0.fitness()).unwrap());

    let new_population = combined
        .iter()
        .take(adaptive_config.population_size)
        .cloned()
        // resets the fitness
        .map(|mut a| {
            a.0.set_fitness(0.0f32);
            a.0
        })
        .collect::<Vec<Agent>>();

    assert_eq!(new_population.len(), adaptive_config.population_size);

    // despawn all current Agents
    q_agents.par_iter().for_each(|entity| {
        par_commands.command_scope(|mut commands| {
            commands.entity(entity).despawn();
        })
    });
    // spawn new ones from the select, crossover and mutate step

    for agent in new_population.iter() {
        let mut new_agent = agent.clone();
        // generate new random id for agent
        let mut rng = source.fork_rng();
        new_agent.id = rng.gen();
        commands.spawn(new_agent).insert(rng);
    }

    next.set(AppState::PrepareNewGenerationTraining);
}

fn increase_generation_count(
    mut generation: ResMut<Generation>,
    q_images: Query<(&Image, &Annotation)>,
    adaptive_config: Res<AdaptiveConfig>,
) {
    let image_number = q_images.iter().len();
    generation.0 += 1;
    info(format!(
        "processed generation {} / {}, with {} images",
        generation.0, adaptive_config.max_generations, image_number
    ));
}

fn clear_stats(
    q_stats: Query<(Entity, &mut Stats)>,
    par_commands: ParallelCommands,
    mut next: ResMut<NextState<AppState>>,
) {
    // despawn every stat entity to process the next generation
    q_stats.par_iter().for_each(|(entity, _)| {
        par_commands.command_scope(|mut commands| {
            commands.entity(entity).despawn();
        })
    });

    next.set(AppState::EvaluateGenerationTraining);
}

fn cleanup(
    mut exit: EventWriter<AppExit>,
    adaptive_config: Res<AdaptiveConfig>,
    q_average_fitness: Query<&AverageFitness>,
    q_stats: Query<(&mut Stats, &EntropyComponent<WyRand>), Without<Source>>,
    xml_parser: Res<XMLParser>,
) {
    let save_path = match adaptive_config.testing_stage {
        true => adaptive_config.tested_agent_save_path.clone(),
        false => adaptive_config.agent_save_path.clone(),
    };

    assert_eq!(
        q_stats.iter().len() / xml_parser.loaded,
        adaptive_config.population_size
    );
    let mut stats = q_stats
        .iter()
        .unique_by(|(stat, _)| stat.original_agent.id)
        .map(|(s, _)| {
            (
                s.evaluated_agent.clone(),
                s.evaluated_image.clone(),
                s.annotation.clone(),
                s.original_agent.id,
            )
        })
        .collect::<Vec<(Agent, Image, Annotation, u64)>>();
    assert_eq!(stats.iter().len(), adaptive_config.population_size);

    // take the best agent and put all images in the list
    // 1. find out the agent id of the best agent
    stats.sort_by(|a, b| b.0.fitness().partial_cmp(&a.0.fitness()).unwrap());
    // let mut files_creation = vec![];
    let mut images_creation = vec![];
    for (s, _) in q_stats.iter() {
        // 2. collect all evaluated images of this agent id in a list
        if s.original_agent.id == stats[0].3 {
            images_creation.push((
                s.evaluated_agent.clone(),
                s.evaluated_image.clone(),
                s.annotation.clone(),
                s.netlist.clone(),
                s.optimal_netlist.clone(),
            ));
        }
    }

    assert_eq!(images_creation.iter().len(), xml_parser.loaded);

    // remove 'agents' directory if it exists
    std::fs::remove_dir_all(&save_path).unwrap_or_default();

    // save fitness data
    let average_fitness_data = q_average_fitness.single();
    std::fs::create_dir_all(&save_path).unwrap();
    plotting::plot_average_fitness(
        &average_fitness_data.0,
        format!("{}/fitness.png", save_path).as_str(),
        adaptive_config.max_generations,
    );

    let out = format!("{:#?}", adaptive_config);
    write(format!("{}/config.txt", save_path).as_str(), out).unwrap();

    // // save netlists over time
    // plotting::netlists_over_time(
    //     &netlists_data,
    //     format!("iterations/{}/netlist_plot.png", iteration).as_str(),
    //     population_size,
    //     max_generations,
    // );

    info("creating files");
    stats.par_iter().for_each(|(a, _, _, _)| {
        let mut agent = a.clone();

        // saves the folder name with fitness + entity_index
        let agent_folder = format!("{}_{}", round2(agent.fitness()), agent.id);
        std::fs::create_dir_all(format!("{}/{}", save_path, agent_folder)).unwrap();
        // save control network
        let folder_name = "control";
        std::fs::create_dir_all(format!("{}/{}/{}", save_path, agent_folder, folder_name)).unwrap();
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
        std::fs::create_dir_all(format!("{}/{}/{}", save_path, agent_folder, folder_name)).unwrap();
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

    info("creating images");
    images_creation
        .iter()
        .for_each(|(agent, image, annotation, netlist, optimal_netlist)| {
            let folder_name = format!("best_agent_{}", agent.id);
            std::fs::create_dir_all(format!(
                "{}/{}/{}",
                save_path, folder_name, annotation.filename
            ))
            .unwrap();

            image
                .save_with_retina(PathBuf::from(format!(
                    "{}/{}/{}/retina.png",
                    save_path, folder_name, annotation.filename
                )))
                .unwrap();
            image
                .save_with_retina_upscaled(
                    PathBuf::from(format!(
                        "{}/{}/{}/retina_orig.png",
                        save_path, folder_name, annotation.filename
                    )),
                    &adaptive_config,
                )
                .unwrap();

            // save optimal netlist
            std::fs::write(
                format!(
                    "{}/{}/{}/optimal_netlist.net",
                    save_path, folder_name, annotation.filename
                ),
                optimal_netlist.generate(),
            )
            .unwrap();

            // save netlist
            std::fs::write(
                format!(
                    "{}/{}/{}/netlist.net",
                    save_path, folder_name, annotation.filename
                ),
                netlist.generate(),
            )
            .unwrap();
        });
    info("Finished");
    exit.send(AppExit::Success);
}
