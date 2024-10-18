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
use rand::seq::SliceRandom;
use rand::SeedableRng;
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
        // after this do the preprocessing of images if enabled in the config
        .add_systems(Update, (preprocess,).run_if(in_state(AppState::Preprocess)))
        // next load all images and setup the run
        .add_systems(
            Update,
            (
                load_images,
                setup_source,
                initialize_population,
                initialize_average_fitness,
            )
                .chain()
                .run_if(in_state(AppState::Setup)),
        )
        .add_systems(
            Update,
            (evaluate_agents, genetic_algorithm_step)
                .chain()
                .run_if(in_state(AppState::EvaluateGeneration)),
        )
        .add_systems(
            Update,
            (increase_generation_count, clear_stats)
                .chain()
                .run_if(in_state(AppState::PrepareNewGeneration)),
        )
        .add_systems(Update, (cleanup,).run_if(in_state(AppState::AlgorithmDone)))
        .run();
}

#[derive(Component)]
struct Source;

#[derive(States, Debug, PartialEq, Hash, Eq, Clone, Default)]
enum AppState {
    #[default]
    Preprocess,
    Setup,
    // InitializeNewGeneration,
    EvaluateGeneration,
    PrepareNewGeneration,
    AlgorithmDone,
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
            preprocess: bool::default(),
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
            agent_save_path: String::default(),
        }
    }
}

#[derive(Resource, Default)]
struct Generation(u64);

#[derive(Component, Debug, Clone)]
struct Stats {
    agent_entity: Entity,
    agent: Agent,
    image: Image,
    annotation: Annotation,
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

fn initialize_population(
    adaptive_config: Res<AdaptiveConfig>,
    mut commands: Commands,
    mut q_source: Query<&mut EntropyComponent<WyRand>, With<Source>>,
) {
    let mut source = q_source.single_mut();
    let mut agents = vec![];
    for _ in 0..adaptive_config.population_size {
        // spawn a new Agent
        let mut rng = source.fork_rng();
        let mut agent = Agent::default();
        agent.genotype = Genotype::init(rng.fork_rng(), &adaptive_config);
        let entity = commands
            .spawn(agent)
            // rng component forked
            .insert(rng)
            .id();
        agents.push(entity);
    }
}

fn initialize_average_fitness(mut commands: Commands, mut next: ResMut<NextState<AppState>>) {
    commands.spawn(AverageFitness::default());
    next.set(AppState::EvaluateGeneration);
}

// fn prepare_new_generation(mut next: ResMut<NextState<AppState>>) {
//     next.set(AppState::EvaluateGeneration);
// }

/// Do all update steps for one generation
fn evaluate_agents(
    q_images: Query<(&Image, &Annotation)>,
    adaptive_config: Res<AdaptiveConfig>,
    q_agents: Query<(Entity, &Agent, &EntropyComponent<WyRand>)>,
    par_commands: ParallelCommands,
) {
    q_agents.par_iter().for_each(|(entity, agent, rng)| {
        // for every one of the 150 agents
        // let mut statistics = HashMap::new();
        q_images.par_iter().for_each(|(image, annotation)| {
            par_commands.command_scope(|mut commands| {
                // for every one of the 4 images bspw.
                let mut temp_agent = agent.clone();
                let mut temp_image = image.clone();
                let new_rng = rng.clone().fork_rng();
                let fitness = temp_agent
                    .evaluate(&adaptive_config, &mut temp_image, annotation)
                    .unwrap();
                assert_eq!(temp_agent.fitness(), 0.);
                assert!(fitness <= 1.0);
                temp_agent.add_to_fitness(fitness);

                // TODO: generate a netlist for this image and this agent and save it somewhere or send an event

                commands
                    .spawn(Stats {
                        agent_entity: entity,
                        agent: temp_agent.clone(),
                        image: temp_image.clone(),
                        annotation: annotation.clone(),
                    })
                    .insert(new_rng);
            });
        });
    });
}

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
) {
    // collect all (images x agents) [4 x 150 = 600] agents and create a new population with maximum of e.g. 150 Agents
    let mut agents = vec![];
    q_stats.iter().for_each(|(stats, _)| {
        agents.push(stats.agent.clone());
    });

    // take the best {population_size} agents for further processing
    agents.sort_by(|a, b| b.fitness().partial_cmp(&a.fitness()).unwrap());
    agents = agents
        .iter()
        .cloned()
        .take(adaptive_config.population_size)
        .collect::<Vec<Agent>>();

    // average each agents fitness over the number of images
    // agents.par_iter_mut().for_each(|agent| {
    //     agent.set_fitness(agent.fitness() / number_of_images as f32);
    // });

    // save fitness data for plotting over generations
    let mut fitness_comp = q_average_fitness.single_mut();
    let average_fitness = agents
        .iter()
        // .inspect(|a| info(a.fitness()))
        .fold(0f32, |acc, a| acc + a.fitness())
        / agents.iter().len() as f32;
    fitness_comp.0.push(average_fitness);

    // TODO: after 50 % of max generations, decrease the variance by 10 % each generation

    // return and skip the next steps when max generations reached
    if generation.0 >= adaptive_config.max_generations {
        next.set(AppState::AlgorithmDone);
        return;
    }

    // select, crossover and mutate
    let new_agents = q_stats
        .iter()
        .map(|(_, rng)| {
            // select
            let mut tournament = Vec::with_capacity(adaptive_config.tournament_size);
            for _ in 0..adaptive_config.tournament_size {
                tournament.push(agents.choose(&mut rng.clone()).unwrap());
            }
            tournament.sort_by(|a, b| b.fitness().partial_cmp(&a.fitness()).unwrap());
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
    let new_population = combined
        .iter()
        .cloned()
        .take(adaptive_config.population_size)
        // resets the fitness
        .map(|mut a| {
            a.set_fitness(0.0f32);
            a
        })
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
        let new_agent = agent.clone();
        commands.spawn(new_agent).insert(source.fork_rng());
    }

    next.set(AppState::PrepareNewGeneration);
}

fn increase_generation_count(
    mut generation: ResMut<Generation>,
    q_images: Query<(&Image, &Annotation)>,
) {
    let image_number = q_images.iter().len();
    generation.0 += 1;
    info(format!(
        "processed generation {}, with {} images",
        generation.0, image_number
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
            commands.entity(entity).despawn_recursive();
        })
    });

    next.set(AppState::EvaluateGeneration);
}

fn cleanup(
    mut exit: EventWriter<AppExit>,
    adaptive_config: Res<AdaptiveConfig>,
    q_average_fitness: Query<&AverageFitness>,
    mut q_stats: Query<(Entity, &mut Stats), Without<Source>>,
) {
    // sort only the best 150 entities
    let mut best_agents = q_stats
        .iter_mut()
        .map(|(e, s)| (e, s.clone()))
        .take(adaptive_config.population_size)
        .collect::<Vec<(Entity, Stats)>>();

    best_agents.sort_by(|a, b| {
        b.1.agent
            .fitness()
            .partial_cmp(&a.1.agent.fitness())
            .unwrap()
    });

    // average each agents fitness over the number of images
    // let number_of_images = q_images.iter().len();
    // best_agents.par_iter_mut().for_each(|(_, stats)| {
    //     stats
    //         .agent
    //         .set_fitness(stats.agent.fitness() / number_of_images as f32);
    // });

    assert_eq!(best_agents.iter().len(), adaptive_config.population_size);

    // remove 'agents' directory if it exists
    std::fs::remove_dir_all(&adaptive_config.agent_save_path).unwrap_or_default();

    // save fitness data
    let average_fitness_data = q_average_fitness.single();
    std::fs::create_dir_all(&adaptive_config.agent_save_path).unwrap();
    plotting::plot_average_fitness(
        &average_fitness_data.0,
        format!("{}/fitness.png", adaptive_config.agent_save_path).as_str(),
        adaptive_config.max_generations,
    );

    let out = format!("{:#?}", adaptive_config);
    write(
        format!("{}/config.txt", adaptive_config.agent_save_path).as_str(),
        out,
    )
    .unwrap();

    // save netlists over time
    //     plotting::netlists_over_time(
    //         &netlists_data,
    //         format!("iterations/{}/netlist_plot.png", iteration).as_str(),
    //         population_size,
    //         max_generations,
    //     );

    info("creating files");
    // create files
    best_agents.par_iter().for_each(|(_, stats)| {
        let mut agent = stats.agent.clone();

        // saves the folder name with fitness + entity_index
        let agent_folder = format!("{}_{}", round2(agent.fitness()), stats.agent_entity.index());
        std::fs::create_dir_all(format!(
            "{}/{}",
            adaptive_config.agent_save_path, agent_folder
        ))
        .unwrap();
        // save control network
        let folder_name = "control";
        std::fs::create_dir_all(format!(
            "{}/{}/{}",
            adaptive_config.agent_save_path, agent_folder, folder_name
        ))
        .unwrap();
        agent
            .genotype()
            .control_network()
            .short_term_memory()
            .visualize(
                format!(
                    "{}/{}/{}/memory.png",
                    adaptive_config.agent_save_path, agent_folder, folder_name
                ),
                adaptive_config.number_of_network_updates,
            )
            .unwrap();
        agent
            .genotype_mut()
            .control_network_mut()
            .to_json(format!(
                "{}/{}/{}/network.json",
                adaptive_config.agent_save_path, agent_folder, folder_name
            ))
            .unwrap();
        agent
            .genotype()
            .control_network()
            .to_dot(format!(
                "{}/{}/{}/network.dot",
                adaptive_config.agent_save_path, agent_folder, folder_name
            ))
            .unwrap();

        // save categorize network
        let folder_name = "categorize";
        std::fs::create_dir_all(format!(
            "{}/{}/{}",
            adaptive_config.agent_save_path, agent_folder, folder_name
        ))
        .unwrap();
        agent
            .genotype()
            .categorize_network()
            .short_term_memory()
            .visualize(
                format!(
                    "{}/{}/{}/memory.png",
                    adaptive_config.agent_save_path, agent_folder, folder_name,
                ),
                adaptive_config.number_of_network_updates,
            )
            .unwrap();
        agent
            .genotype_mut()
            .categorize_network_mut()
            .to_json(format!(
                "{}/{}/{}/network.json",
                adaptive_config.agent_save_path, agent_folder, folder_name
            ))
            .unwrap();
        agent
            .genotype()
            .categorize_network()
            .to_dot(format!(
                "{}/{}/{}/network.dot",
                adaptive_config.agent_save_path, agent_folder, folder_name
            ))
            .unwrap();
    });

    info("creating images");
    // for the best x agents, create the images
    // TODO: enable more images creation for the best X agents via adaptive config
    let best_entity = best_agents[0].1.agent_entity;
    q_stats.par_iter().for_each(|(_, stats)| {
        if best_entity == stats.agent_entity {
            let folder_name = format!("best_agent_{}", stats.agent_entity.index());
            std::fs::create_dir_all(format!(
                "{}/{}/{}",
                adaptive_config.agent_save_path, folder_name, stats.annotation.filename
            ))
            .unwrap();

            stats
                .image
                .save_with_retina(PathBuf::from(format!(
                    "{}/{}/{}/retina.png",
                    adaptive_config.agent_save_path, folder_name, stats.annotation.filename
                )))
                .unwrap();
            stats
                .image
                .save_with_retina_upscaled(
                    PathBuf::from(format!(
                        "{}/{}/{}/retina_orig.png",
                        adaptive_config.agent_save_path, folder_name, stats.annotation.filename
                    )),
                    &adaptive_config,
                )
                .unwrap();

            // save netlist
            // std::fs::write(
            //     format!(
            //         "{}/{}/{}/netlist.net",
            //         adaptive_config.agent_save_path, folder_name, stats.annotation.filename
            //     ),
            //     netlist.clone(),
            // )
            // .unwrap();
        }
    });
    info("Finished");
    exit.send(AppExit::Success);
}
