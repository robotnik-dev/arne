use annotations::{Annotation, XMLParser};
use bevy::dev_tools::states::log_transitions;
use bevy::prelude::*;
use bevy::utils::hashbrown::HashMap;
use bevy::{
    log::{Level, LogPlugin},
    state::app::StatesPlugin,
    utils::info,
};
use bevy_prng::{ChaCha8Rng, WyRand};
use bevy_rand::prelude::{EntropyComponent, EntropyPlugin};
use bevy_rand::traits::{ForkableInnerRng, ForkableRng};
use genetic_algorithm::{Agent, Genotype};
use image::{Image, Position};
use itertools::Itertools;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rayon::iter::{IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator};
use serde::{Deserialize, Serialize};
use serde_json::from_str;
use std::default;
use std::fmt::Display;
use std::fs::read_to_string;
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
                level: Level::DEBUG,
                filter: "wgpu=error,bevy_render=info,bevy_ecs=trace".to_string(),
                custom_layer: |_| None,
            },
        ))
        .add_plugins(EntropyPlugin::<WyRand>::default())
        .insert_state(AppState::default())
        .insert_resource(AdaptiveConfig::default())
        .insert_resource(XMLParser::default())
        .insert_resource(Populations::default())
        .insert_resource(Generation::default())
        .add_systems(PreStartup, (load_config, load_images).chain())
        .add_systems(Startup, (setup_source, initialize_populations).chain())
        .add_systems(
            Update,
            (intialize_genotypes_for_agents,)
                .chain()
                .run_if(in_state(AppState::InitializeNewGeneration)),
        )
        .add_systems(
            Update,
            (
                evaluate_agents,
                genetic_algorithm_step,
                clear_stats,
                increase_generation_count,
            )
                .chain()
                .run_if(in_state(AppState::EvaluateGenerations)),
        )
        .add_systems(Update, (cleanup,).run_if(in_state(AppState::Cleanup)))
        .add_systems(Update, log_transitions::<AppState>)
        .run();
}

#[derive(Component)]
struct Source;

#[derive(States, Debug, PartialEq, Hash, Eq, Clone, Default)]
enum AppState {
    #[default]
    Setup,
    InitializeNewGeneration,
    EvaluateGenerations,
    Cleanup,
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Resource)]
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
}

impl Default for AdaptiveConfig {
    fn default() -> Self {
        Self {
            number_of_populations: usize::default(),
            number_of_network_updates: usize::default(),
            neuron_lower: f32::default(),
            neuron_upper: f32::default(),
            retina_lower: f32::default(),
            retina_upper: f32::default(),
            init_non_zero_retina_weights: f32::default(),
            population_size: usize::default(),
            max_generations: u64::default(),
            tournament_size: usize::default(),
            variance: f32::default(),
            variance_decay: f32::default(),
            mean: f32::default(),
            delete_neuron: f32::default(),
            delete_weights: f32::default(),
            delete_bias: f32::default(),
            delete_self_activation: f32::default(),
            mutate_neuron: f32::default(),
            mutate_weights: f32::default(),
            mutate_bias: f32::default(),
            mutate_self_activation: f32::default(),
            goal_image_width: usize::default(),
            goal_image_height: usize::default(),
            retina_size: usize::default(),
            superpixel_size: usize::default(),
            retina_circle_radius: f32::default(),
            retina_label_scale: f32::default(),
            sobel_threshold: f32::default(),
            erode_pixels: usize::default(),
            training_path: String::default(),
            training_load_all: bool::default(),
            training_load_amount: usize::default(),
            testing_path: String::default(),
            testing_load_all: bool::default(),
            testing_load_amount: usize::default(),
            control_network_neurons: usize::default(),
            categorize_network_neurons: usize::default(),
            retina_movement_speed: f32::default(),
            with_seed: bool::default(),
            seed: usize::default(),
            goal_fitness: f32::default(),
        }
    }
}

// Components we have
// - Agent
// - Genotype
// - Image

// Logic is we have X generations to process, and they built on top of each other so it have to run in serial.
// Each process of a generation starts with the same population used in the generation before, but adapted.
// (Optional) We can run multiple processes in parallel meaning multiple populations that evolve next to each other, but do not know each other.

// Agent
// we have X Agents per Population. The Entities stay the same the whole app run through so we just can manipulate the components directly

#[derive(Resource)]
struct Populations(HashMap<usize, Vec<Entity>>);

impl Default for Populations {
    fn default() -> Self {
        Self(HashMap::default())
    }
}

#[derive(Bundle, Default)]
struct AgentBundle {
    agent: Agent,
}

#[derive(Resource, Default)]
struct Generation(u64);

#[derive(Component, Clone)]
struct BufferMemory {
    images: Vec<(Image, Annotation)>,
}

#[derive(Component, Debug)]
struct Stats {
    agent: Agent,
    image: Image,
}

// #[derive(Component, Default)]
// struct Agent {
//     genotype: Genotype,
//     retina_start_pos: Position,
// }

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
    println!("{:#?}", adaptive_config);
}

fn load_images(
    mut xml_parser: ResMut<XMLParser>,
    mut commands: Commands,
    adaptive_config: Res<AdaptiveConfig>,
) {
    let data_path = adaptive_config.training_path.to_string();
    let dir = std::fs::read_dir(PathBuf::from(data_path)).unwrap();
    let mut idx = 0usize;
    for folder in dir {
        if idx == adaptive_config.training_load_amount as usize
            && !adaptive_config.training_load_all
        {
            break;
        };
        let drafter_path = folder.unwrap().path();
        xml_parser
            .load(
                drafter_path,
                adaptive_config.training_load_all as bool,
                adaptive_config.training_load_amount as usize,
            )
            .unwrap();
        idx += 1;
    }

    for (annotation, image, netlist) in xml_parser.data.iter() {
        commands
            .spawn(Image {
                rgba: image.rgba.clone(),
                grey: image.grey.clone(),
                width: image.width.clone(),
                height: image.height.clone(),
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
            });
        // TODO: insert netlist to component (but with real netlist, not String)
    }
}

fn initialize_populations(
    mut populations: ResMut<Populations>,
    adaptive_config: Res<AdaptiveConfig>,
    mut commands: Commands,
    mut q_source: Query<&mut EntropyComponent<WyRand>, With<Source>>,
    mut next: ResMut<NextState<AppState>>,
) {
    let mut source = q_source.single_mut();
    for p in 0..adaptive_config.number_of_populations {
        // create a new population
        let mut agents = vec![];
        for _ in 0..adaptive_config.population_size {
            // spawn a new Agent
            let entity = commands
                .spawn(AgentBundle::default())
                // rng component forked
                .insert(source.fork_rng())
                // TODO: insert here the vector of all loaded images as a component (or 91 components, idk yet)
                .id();
            agents.push(entity);
        }
        // save them in the population
        populations.0.insert(p, agents);
    }

    next.set(AppState::InitializeNewGeneration);
}

fn intialize_genotypes_for_agents(
    mut q_agents: Query<(&mut Agent, &mut EntropyComponent<WyRand>)>,
    adaptive_config: Res<AdaptiveConfig>,
    mut next: ResMut<NextState<AppState>>,
) {
    for (mut agent, mut rng) in q_agents.iter_mut() {
        agent.genotype = Genotype::init(rng.fork_rng(), &adaptive_config)
    }
    next.set(AppState::EvaluateGenerations);
}

/// Each agent gets a list of image components (91 max) for them to manipulate so that the original images
/// dont need to be mutable
fn intialize_buffer_for_agents(
    q_agents: Query<Entity, With<Agent>>,
    q_images: Query<(&Image, &Annotation)>,
    par_commands: ParallelCommands,
) {
    q_agents.par_iter().for_each(|agent| {
        let mut buffer = vec![];
        for (image, annotation) in q_images.iter() {
            let new_buffer_image = Image {
                rgba: image.rgba.clone(),
                grey: image.grey.clone(),
                width: image.width.clone(),
                height: image.height.clone(),
                format: image.format.clone(),
                retina_positions: image.retina_positions.clone(),
                dark_pixel_positions: image.dark_pixel_positions.clone(),
            };
            let new_annotation = Annotation {
                folder: annotation.folder.clone(),
                filename: annotation.filename.clone(),
                path: annotation.path.clone(),
                source: annotation.source.clone(),
                size: annotation.size.clone(),
                segmented: annotation.segmented.clone(),
                objects: annotation.objects.clone(),
            };
            buffer.push((new_buffer_image, new_annotation));
        }
        par_commands.command_scope(|mut commands| {
            commands
                .entity(agent)
                .insert(BufferMemory { images: buffer });
        });
    });
}

/// Do all update steps for one generation
fn evaluate_agents(
    mut next: ResMut<NextState<AppState>>,
    q_images: Query<(&Image, &Annotation)>,
    adaptive_config: Res<AdaptiveConfig>,
    q_agents: Query<(&Agent, &EntropyComponent<WyRand>)>,
    parser: Res<XMLParser>,
    par_commands: ParallelCommands,
    mut generation: Res<Generation>,
) {
    q_agents.par_iter().for_each(|(agent, rng)| {
        // for every one of the 150 agents
        q_images.par_iter().for_each(|(image, annotation)| {
            par_commands.command_scope(|mut commands| {
                // for every one of the 4 images bspw.
                let mut temp_image = image.clone();
                let mut temp_agent = agent.clone();
                let new_rng = rng.clone().fork_rng();
                let fitness = temp_agent
                    .evaluate(&adaptive_config, &mut temp_image, annotation)
                    .unwrap();
                temp_agent.add_to_fitness(fitness);
                // TODO: generate a netlist for this image and this agent and save it somewhere or send an event

                // insert a stats component for another system to grab the information
                commands
                    .spawn(Stats {
                        agent: temp_agent.clone(),
                        image: temp_image.clone(),
                    })
                    .insert(new_rng);
            });
        })
    });

    // info(generation.0);
    // info(adaptive_config.max_generations);
    // if generation.0 >= adaptive_config.max_generations {
    //     info("cleanup");
    //     next.set(AppState::Cleanup);
    // }
}

fn genetic_algorithm_step(
    q_images: Query<(&Image, &Annotation)>,
    q_stats: Query<(&mut Stats, &EntropyComponent<WyRand>), Without<Source>>,
    par_commands: ParallelCommands,
    mut commands: Commands,
    adaptive_config: Res<AdaptiveConfig>,
    q_agents: Query<Entity, With<Agent>>,
    mut q_source: Query<&mut EntropyComponent<WyRand>, With<Source>>,
) {
    info(q_stats.iter().len());
    let number_of_images = q_images.iter().len();
    // collect the agents with the same image
    // let mut matched = HashMap::new();
    // q_images.iter().for_each(|(img, ann)| {
    //     let agents = q_stats
    //         .iter()
    //         .filter(|stats| &stats.image == img)
    //         .map(|stats| &stats.agent)
    //         .collect::<Vec<&Agent>>();
    //     matched.insert((img, ann), agents);
    // });

    // TODO: this below is need in the cleanup or create statistics state to create the actual image files etc.
    // // this is a mulitplier of all agents, here we have agents x images (z.b. 150 agents x 4 images = 600 agents in this loop -> 150 agents per iteration)
    // for ((image, annotation), agents) in matched.iter_mut() {
    //     // create local copies
    //     let mut local_agents = vec![];
    //     agents.iter_mut().for_each(|agent| {
    //         local_agents.push(agent.clone());
    //     });
    //     // average each agents fitness over the number of images
    //     local_agents.par_iter_mut().for_each(|agent| {
    //         agent.set_fitness(agent.fitness() / number_of_images as f32);
    //     });
    // }

    // collect all (images x agents) [4 x 150 = 600] agents and create a new population with maximum of e.g. 150 Agents
    let mut agents = vec![];
    q_stats.iter().for_each(|(stats, _)| {
        agents.push(stats.agent.clone());
    });

    // sort the population by fitness
    // agents.sort_by(|a, b| b.fitness().partial_cmp(&a.fitness()).unwrap());

    // TODO: save fitness data for plotting over generations
    // TODO: after 50 % of max generations, decrease the variance by 10 % each generation

    // select, crossover and mutate
    let new_agents = q_stats
        .iter()
        .map(|(_, rng)| {
            // select
            let mut tournament = Vec::with_capacity(adaptive_config.tournament_size);
            for _ in 0..adaptive_config.tournament_size {
                tournament.push(agents.choose(&mut rng.clone()).unwrap());
            }
            tournament.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());
            // crossover
            let mut offspring = tournament[0].crossover(rng.clone(), tournament[1]);
            // mutate
            offspring.mutate(rng.clone(), &adaptive_config);
            offspring
        })
        .collect::<Vec<Agent>>();

    // evolve the population

    let mut combined = agents
        .iter()
        .cloned()
        .chain(new_agents.iter().cloned())
        .collect::<Vec<Agent>>();
    combined.sort_by(|a, b| b.fitness().partial_cmp(&a.fitness()).unwrap());
    // combined.shuffle(rng);
    let new_population = combined
        .iter()
        .cloned()
        .take(adaptive_config.population_size)
        // // resets the fitness
        // .map(|mut a| {
        //     a.set_fitness(0.0f32);
        //     a
        // })
        .collect::<Vec<Agent>>();

    // despawn all current Agents
    q_agents.par_iter().for_each(|entity| {
        par_commands.command_scope(|mut commands| {
            commands.entity(entity).despawn_recursive();
        })
    });
    // spawn new ones from the select, crossover and mutate step
    let mut source = q_source.single_mut();
    for agent in new_population.iter() {
        commands
            .spawn(AgentBundle {
                agent: agent.clone(),
            })
            // rng component forked
            .insert(source.fork_rng());
    }
}

fn increase_generation_count(
    mut generation: ResMut<Generation>,
    adaptive_config: Res<AdaptiveConfig>,
    mut next: ResMut<NextState<AppState>>,
) {
    generation.0 += 1;
    if generation.0 >= adaptive_config.max_generations {
        next.set(AppState::Cleanup);
    }
}

fn clear_stats(
    q_stats: Query<(Entity, &mut Stats)>,
    par_commands: ParallelCommands,
    mut next: ResMut<NextState<AppState>>,
) {
    // despawn every stat entity to process the next generation
    q_stats.par_iter().for_each(|(entity, _)| {
        par_commands.command_scope(|mut commands| {
            commands.entity(entity).despawn_recursive();
        })
    });

    next.set(AppState::InitializeNewGeneration);
}

fn cleanup(mut exit: EventWriter<AppExit>) {
    // save fitness and configuration the same way as it where stuck or stale
    //     std::fs::create_dir_all(format!("iterations/{}", iteration)).unwrap();
    //     plotting::update_image(
    //         &average_fitness_data,
    //         format!("iterations/{}/fitness.png", iteration).as_str(),
    //         max_generations,
    //     );
    //     let out = serde_json::to_string_pretty(adaptive_config).unwrap();
    //     write(
    //         format!("iterations/{}/config.json", iteration).as_str(),
    //         out,
    //     )
    //     .unwrap();
    //     std::fs::OpenOptions::new()
    //         .create(true)
    //         .append(true)
    //         .open("iteration_results.txt")
    //         .unwrap()
    //         .write_fmt(format_args!(
    //             "generations survived: {} ",
    //             population.generation()
    //         ))
    //         .unwrap();

    // save netlists over time
    //     plotting::netlists_over_time(
    //         &netlists_data,
    //         format!("iterations/{}/netlist_plot.png", iteration).as_str(),
    //         population_size,
    //         max_generations,
    //     );

    // remove 'agents' directory if it exists
    //     std::fs::remove_dir_all(save_path.clone()).unwrap_or_default();

    // create files
    //     population
    //         .agents_mut()
    //         .par_iter_mut()
    //         .enumerate()
    //         .for_each(|(index, agent)| {
    //             // saves the folder name with index + fitness round2
    //             let agent_folder = format!("{}_{}", round2(agent.fitness()), index);
    //             std::fs::create_dir_all(format!("{}/{}", save_path, agent_folder)).unwrap();
    //             // save control network
    //             let folder_name = "control";
    //             std::fs::create_dir_all(format!("{}/{}/{}", save_path, agent_folder, folder_name))
    //                 .unwrap();
    //             agent
    //                 .genotype()
    //                 .control_network()
    //                 .short_term_memory()
    //                 .visualize(
    //                     format!("{}/{}/{}/memory.png", save_path, agent_folder, folder_name),
    //                     adaptive_config.number_of_network_updates,
    //                 )
    //                 .unwrap();
    //             agent
    //                 .genotype_mut()
    //                 .control_network_mut()
    //                 .to_json(format!(
    //                     "{}/{}/{}/network.json",
    //                     save_path, agent_folder, folder_name
    //                 ))
    //                 .unwrap();
    //             agent
    //                 .genotype()
    //                 .control_network()
    //                 .to_dot(format!(
    //                     "{}/{}/{}/network.dot",
    //                     save_path, agent_folder, folder_name
    //                 ))
    //                 .unwrap();

    //             // save categorize network
    //             let folder_name = "categorize";
    //             std::fs::create_dir_all(format!("{}/{}/{}", save_path, agent_folder, folder_name))
    //                 .unwrap();
    //             agent
    //                 .genotype()
    //                 .categorize_network()
    //                 .short_term_memory()
    //                 .visualize(
    //                     format!("{}/{}/{}/memory.png", save_path, agent_folder, folder_name,),
    //                     adaptive_config.number_of_network_updates,
    //                 )
    //                 .unwrap();
    //             agent
    //                 .genotype_mut()
    //                 .categorize_network_mut()
    //                 .to_json(format!(
    //                     "{}/{}/{}/network.json",
    //                     save_path, agent_folder, folder_name
    //                 ))
    //                 .unwrap();
    //             agent
    //                 .genotype()
    //                 .categorize_network()
    //                 .to_dot(format!(
    //                     "{}/{}/{}/network.dot",
    //                     save_path, agent_folder, folder_name
    //                 ))
    //                 .unwrap();
    //         });

    // for the best agent, create the images
    // population.agents().iter().take(1).for_each(|agent| {
    //         agent
    //             .statistics()
    //             .par_iter()
    //             .for_each(|(label, (image, _, netlist))| {
    //                 let folder_name = format!("best_agent_{}", round2(agent.fitness()));
    //                 std::fs::create_dir_all(format!("{}/{}/{}", save_path, folder_name, label))
    //                     .unwrap();

    //                 image
    //                     .save_with_retina(PathBuf::from(format!(
    //                         "{}/{}/{}/retina.png",
    //                         save_path, folder_name, label
    //                     )))
    //                     .unwrap();
    //                 image
    //                     .save_with_retina_upscaled(PathBuf::from(format!(
    //                         "{}/{}/{}/retina_orig.png",
    //                         save_path, folder_name, label
    //                     )))
    //                     .unwrap();

    //                 // save netlist
    //                 std::fs::write(
    //                     format!("{}/{}/{}/netlist.net", save_path, folder_name, label),
    //                     netlist.clone(),
    //                 )
    //                 .unwrap();
    //             });
    //     });
    info("DONE");
    exit.send(AppExit::Success);
}

// if we have reached the last update step, go into next state

// stop criteria met
// next.set(AppState::Cleanup);

// #[allow(dead_code)]
// fn preprocess(mut exit: EventWriter<AppExit>) {
//     // training folder
//     for entry in std::fs::read_dir("data/training").unwrap() {
//         let entry = entry.unwrap();
//         let path = entry.path();
//         XMLParser::resize_segmented_images(path).unwrap();
//     }
//     // training folder
//     for entry in std::fs::read_dir("data/testing").unwrap() {
//         let entry = entry.unwrap();
//         let path = entry.path();
//         XMLParser::resize_segmented_images(path).unwrap();
//     }
//     exit.send(AppExit::Success);
// }

// #[allow(dead_code)]
// fn test_agents(mut exit: EventWriter<AppExit>, adaptive_config: Res<AdaptiveConfig>) {
//     training::test_agents(String::from("iterations/0/agents"), adaptive_config).unwrap();
//     exit.send(AppExit::Success);
// }

// fn process_image()

#[allow(dead_code)]
fn run_one_config(mut exit: EventWriter<AppExit>, adaptive_config: Res<AdaptiveConfig>) {
    // let filepath = String::from("current_config.json");
    // let adaptive_config: AdaptiveConfig =
    //     from_str(read_to_string(filepath).unwrap().as_str()).unwrap();
    let iteration = 0;
    // training::train_agents(
    //     None,
    //     format!("iterations/{}/agents", iteration),
    //     iteration,
    //     adaptive_config,
    // )
    // .unwrap();
    exit.send(AppExit::Success);
}

// #[allow(dead_code)]
// fn test_configs(mut exit: EventWriter<AppExit>) {
//     let max_iterations = 100;
//     let mut rng = ChaCha8Rng::from_entropy();
//     let _ = std::fs::remove_dir_all("iterations");
//     let _ = std::fs::remove_file("iteration_results.txt");
//     let mut iteration = 0usize;
//     let mut adaptive_config = AdaptiveConfig::new();
//     adaptive_config.randomize(&mut rng);
//     loop {
//         training::train_agents(
//             TrainingStage::Artificial { stage: 0 },
//             None,
//             format!("iterations/{}/agents", iteration),
//             iteration,
//             &adaptive_config,
//             false,
//             true,
//         )
//         .unwrap();

//         if iteration >= max_iterations {
//             break;
//         }

//         // tweak configuration
//         adaptive_config.randomize(&mut rng);

//         iteration += 1;
//     }
//     exit.send(AppExit::Success);
// }
