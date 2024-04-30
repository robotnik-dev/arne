// this library is a generic implementation of a genetic algorithm
// it uses the entity component system pattern to represent the agents and to leverage 
// the parallelism for evaluating the fitness of each agent

use std::{collections::HashMap, ops::Index, time::Duration};

use bevy::{input::mouse::{MouseScrollUnit, MouseWheel}, pbr::DefaultOpaqueRendererMethod, prelude::*, transform::commands};
use bevy_prng::WyRand;
use rand_core::RngCore;
use bevy_rand::prelude::{EntropyComponent, GlobalEntropy, ForkableRng, EntropyPlugin};
use bevy_tweening::{lens::TransformPositionLens, *};

pub struct GeneticAlgorithmPlugin;

impl Plugin for GeneticAlgorithmPlugin {
    fn build(&self, app: &mut App) {

        // TODO: remove seed when testing is done
        let seed: u64 = 234;

        app.add_plugins(EntropyPlugin::<WyRand>::with_seed(seed.to_le_bytes()));
        app.add_plugins(TweeningPlugin);
        app.init_state::<GeneticAlgorithmState>();
        app.insert_resource(BackgroundColorGreyScale(255));
        app.insert_resource(Grid::default());
        // app.add_event::<PopulationInitialized>();

        app.add_systems(Startup, (
            (
                initialize_population,
                setup_scene,
            ).chain(),
        ));

        app.add_systems(Update, (
            rotate_camera,
            zoom_camera,
            fitness_evaluation
                .run_if(in_state(GeneticAlgorithmState::EvaluateFitness)),
            check_stop_criteria
                .run_if(in_state(GeneticAlgorithmState::CheckStopCriteria)),
            crossover
                .run_if(in_state(GeneticAlgorithmState::Crossover)),
            mutation
                .run_if(in_state(GeneticAlgorithmState::Mutation)),
            phenotype_mapping
                .run_if(in_state(GeneticAlgorithmState::PhenotypeMapping)),
            selection
                .run_if(in_state(GeneticAlgorithmState::Selection)),
            clean_up
                .run_if(in_state(GeneticAlgorithmState::EndAgorithm)),
            paused
                .run_if(in_state(GeneticAlgorithmState::Pause)),
            check_animate_transform_done
                .run_if(in_state(GeneticAlgorithmState::AnimateMovement)),
        ));
        app.add_systems(OnEnter(GeneticAlgorithmState::AnimateMovement), (
            animate_agent_transform,
        ));
    }
}

// UI SECTION


// GENETIC ALGORITHM SECTION

const POPULATION_SIZE: u32 = 100;
const MAXIMUM_GENERATIONS: u32 = 100;
const ANIMATION_SPEED_MULTIPLIER: f32 = 0.25;

// #[derive(Event)]
// struct PopulationInitialized;

#[derive(Resource)]
struct BackgroundColorGreyScale(u8);

/// a grid which stores the row and column positions of the agents
#[derive(Resource)]
struct Grid{
    rows: u32,
    cols: u32,
}

impl Grid {
    fn get_at(&self, index: usize) -> Vec3 {
        let row = 1 + index as u32 % self.rows;
        let col = 1 + index as u32 / self.cols;
        Vec3::new(row as f32, col as f32, 0.0)
    }
}

impl Default for Grid {
    fn default() -> Self {
        if POPULATION_SIZE % 10 != 0 {
            panic!("Population size must be a multiple of 10");
        }
        Grid {
            rows: POPULATION_SIZE / 10,
            cols: POPULATION_SIZE / 10,
        }
    }
}

#[derive(States, Default, Debug, Clone, PartialEq, Eq, Hash)]
enum GeneticAlgorithmState {
    #[default]
    Startup,
    EvaluateFitness,
    CheckStopCriteria,
    // Choose the best agents to be parents for the next generation
    AnimateMovement,
    Selection,
    Crossover,
    Mutation,
    PhenotypeMapping,
    EndAgorithm,
    // debug pause state
    Pause
}

/// The AgentBundle is a collection of components that represent an agent.
#[derive(Bundle)]
struct AgentBundle {
    // marker
    agent: Agent,
    fitness: Fitness,
    chromosome: Chromosome,
    generation: Generation,
}

#[derive(Component)]
struct Chromosome {
    genes: u8
}

impl Chromosome {
    fn new(init: u8) -> Self {
        Chromosome {
            genes: init
        }
    }
}

#[derive(Component)]
struct GridPosition(Vec3);

/// An agent is a potential solution to the problem.
#[derive(Component)]
struct Agent;

#[derive(Component)]
struct Generation(u32);

/// The fitness of an agent is a measure of how well it performs.
#[derive(Component, PartialEq, PartialOrd)]
struct Fitness(f32);

/// Marker Component for all evaluated Agents
#[derive(Component)]
struct Evaluated;

// NEURAL NETWORK SECTION

/// tangens hyperbolicus activation function maps the output to the range [-1, 1]
const TANH: ActivationFunction = ActivationFunction(|x| x.tanh());
/// ReLU activation function maps the output to the range [0, inf)
const RELU: ActivationFunction = ActivationFunction(|x| x.max(0.0));

#[derive(Bundle)]
struct NeuronBundle {
    // marker
    neuron: Neuron,
    activation_function: ActivationFunction,
    bias: Bias,
    weights: Weights,
    inputs: Inputs,
    outputs: Outputs,
    activation: Activation,
}

#[derive(Component)]
struct Neuron;

#[derive(Component)]
struct ActivationFunction(fn(f32) -> f32);

#[derive(Component)]
struct Bias(f32);

/// The weights of the input connections to the neuron.
/// The weights are used to calculate the activation of the neuron.
#[derive(Component)]
struct Weights(Vec<f32>);

/// List of entity IDs that are connected as inputs to the neuron.
#[derive(Component)]
struct Inputs(Vec<u32>);

/// List of entity IDs that are connected as outputs to the neuron.
#[derive(Component)]
struct Outputs(Vec<u32>);

/// The activation of the neuron, generated by the activation function.
#[derive(Component)]
struct Activation(f32);


fn rotate_camera(
    mut q_camera: Query<&mut Transform, With<Camera>>,
    mouse: Res<ButtonInput<MouseButton>>,
) {
    if let Ok(mut cam_transform) = q_camera.get_single_mut() {
        if mouse.pressed(MouseButton::Left) {
            cam_transform.rotate_around(Vec3::new(5.0, 5.0, 0.0), Quat::from_rotation_y(0.05))
        }
        else if mouse.pressed(MouseButton::Right) {
            cam_transform.rotate_around(Vec3::new(5.0, 5.0, 0.0), Quat::from_rotation_y(-0.05))
        }
    }
}

fn zoom_camera(
    mut q_camera: Query<&mut Projection, With<Camera>>,
    mut ev_wheel: EventReader<MouseWheel>
) {

    let Projection::Perspective(persp) = q_camera.single_mut().into_inner() else { return };
    for ev in ev_wheel.read() {
        match ev.unit {
            MouseScrollUnit::Line => {
                persp.fov -= 0.05 * ev.y;
            }
            MouseScrollUnit::Pixel => {
                persp.fov += 1.0 * ev.y;
            }
        }
    }
}

/// At the beginning, a set of solutions,
/// which is denoted as population, is initialized.
/// This runs once when the App starts.
/// When done, the next state is set to Selection.
fn initialize_population(
    mut commands: Commands,
    mut rng: ResMut<GlobalEntropy<WyRand>>,
) {
    // set an offset for x and y position so that the camera can rotate around the center
    // create the population
    for _ in 0..POPULATION_SIZE {
        let genes = rng.next_u32() as u8 % 255;
        commands.spawn((AgentBundle {
            agent: Agent,
            fitness: Fitness(0.0),
            // random between 0 and 255
            chromosome: Chromosome::new(genes),
            generation: Generation(1)
        },
        rng.fork_rng(),
        ));
    }
}

fn setup_scene(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    q_agent: Query<(Entity, &Chromosome), With<Agent>>,
    bg_color: Res<BackgroundColorGreyScale>,
    mut next_state: ResMut<NextState<GeneticAlgorithmState>>,
    mut grid: ResMut<Grid>
) {
    // setup camera
    commands.spawn(
        Camera3dBundle {
            transform: Transform::from_xyz(5.0, 5.0, 20.0).looking_at(Vec3::new(5.0, 5.0, 0.0), Vec3::Y),
            camera: Camera {
                clear_color: ClearColorConfig::Custom(Color::rgb_u8(bg_color.0, bg_color.0, bg_color.0)),
                ..default()
            },
            ..Default::default()
        }
    );

    // setup light
    commands.spawn(
        PointLightBundle {
            transform: Transform::from_xyz(4.0, 8.0, 4.0),
            ..Default::default()
        }
    );

    // attach visual to agents
    q_agent
        .iter()
        .enumerate()
        .for_each(|(idx, (entity, chromosome))| {
            // grid x position start left at 1.0
            // grid y position start top at 1.0
            let grid_x = 1.0 + (idx as f32 % grid.rows as f32);
            let grid_y = grid.cols as f32 - (idx as f32 / grid.cols as f32).floor();

            commands.entity(entity)
                .insert((
                    PbrBundle {
                        mesh: meshes.add(Cuboid::from_size(Vec3::splat(1.0))),
                        material: materials.add(Color::rgb_u8(chromosome.genes, chromosome.genes, chromosome.genes)),
                        transform: Transform::from_xyz(grid_x, grid_y, 0.0),
                        ..default()
                    },
                    GridPosition(Vec3::new(grid_x, grid_y, 0.0)),
                ));
        });
    next_state.set(GeneticAlgorithmState::EvaluateFitness);
}

/// The population is evaluated for fitness.
/// This system runs for each Agent in the population while in the selection state.
/// The fitness of each agent is evaluated. (This is a slow operation)
/// The fitness of each agent is stored in the Fitness component.
/// The next state is set to Crossover.
fn fitness_evaluation(
    mut q_agent: Query<(&mut Fitness, &Chromosome, &mut EntropyComponent<WyRand>), With<Agent>>,
    mut next_state: ResMut<NextState<GeneticAlgorithmState>>,
    bg_color: Res<BackgroundColorGreyScale>,
    // mut grid: ResMut<Grid>
) {
    // compute fitness
    q_agent
        .par_iter_mut()
        .for_each(|(mut fitness, chromosome, mut rng)| {
            // steps to evaluate fitness:
            // - train the agent with training data(operational heavy task) optional for now.
            // - use the trained agent to solve the problem
            // - evaluate the performance of the agent
            // - assign the performance as the fitness of the agent

            // The problem to solve is minimizing the error value of a function.
            // for this test its just a rgb color comparison to a background color.
            // The less the difference the better the fitness.

            let diff = chromosome.genes.abs_diff(bg_color.0);
            // normalize the difference to the range [0, 1] and assign as fitness
            fitness.0 = 255.0 - (diff as f32);
        });
    
    next_state.set(GeneticAlgorithmState::CheckStopCriteria);
}

fn check_stop_criteria(
    mut next_state: ResMut<NextState<GeneticAlgorithmState>>,
    q_generation: Query<&Generation, With<Agent>>,
) {
    // things to check:
    // - maximum number of generations
    // - reached a plateau
    // - a solution with satisfactory fitness is found
    let stop = q_generation
        .iter()
        .any(|generation|
            generation.0 >= MAXIMUM_GENERATIONS
        );
    if stop {
        next_state.set(GeneticAlgorithmState::EndAgorithm);
    } else {
        next_state.set(GeneticAlgorithmState::Pause);
    }
}

fn selection(
    mut next_state: ResMut<NextState<GeneticAlgorithmState>>,
    q_agent: Query<(Entity, &Fitness), With<Agent>>,
    mut commands: Commands,
    // mut grid: ResMut<Grid>
) {
    info!("Selection..");
    
    // sort the agents by fitness
    let mut agents = q_agent
        .iter()
        .collect::<Vec<_>>();

    agents.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    agents
        .iter()
        // just despawn the first half of the agents -> change this later?
        .take(agents.len() / 2)
        .for_each(|(entity, _)| {
            commands.entity(*entity).despawn();
        });

    next_state.set(GeneticAlgorithmState::Crossover);
}

fn crossover(
    mut next_state: ResMut<NextState<GeneticAlgorithmState>>,
    q_agent: Query<&Chromosome, With<Agent>>,
    mut commands: Commands,
) {
    info_once!("Performing crossover..");

    // right now only the best agents are 'alive'
    // so we can take a pair of agents and create a new agent from them
    // the new agent will have a mix of the genes of the parents
    // in this case its a mix of the color of the two parents
    // to mark the new agent we will attach a 'Offspring' component to it for the mutation step
    // all of the agents will increase their generation count by 1


    next_state.set(GeneticAlgorithmState::Mutation);
}

fn mutation(
    mut next_state: ResMut<NextState<GeneticAlgorithmState>>
) {
    info!("Performing mutation..");

    next_state.set(GeneticAlgorithmState::PhenotypeMapping);
}

fn phenotype_mapping(
    mut next_state: ResMut<NextState<GeneticAlgorithmState>>
) {
    info!("Performing phenotype mapping..");

    next_state.set(GeneticAlgorithmState::EvaluateFitness);
}

fn clean_up(
    time: Res<Time>,
    mut next_state: ResMut<NextState<GeneticAlgorithmState>>
) {
    info!("Collection data..");
    info!("Ending algorithm..");
    info!("Time elapsed {:?}", time.elapsed());
    // shut the programm down
    // set it to paused for now
    next_state.set(GeneticAlgorithmState::Pause);

    // std::process::exit(0);
}

fn paused(
    key: Res<ButtonInput<KeyCode>>,
    mut next_state: ResMut<NextState<GeneticAlgorithmState>>
) {
    info_once!("Paused..");
    if key.just_pressed(KeyCode::Space) {
        next_state.set(GeneticAlgorithmState::AnimateMovement);
    }
}

// run once on state enter
fn animate_agent_transform(
    grid: ResMut<Grid>,
    q_agent: Query<(Entity, &GridPosition, &Transform, &Fitness), With<Agent>>,
    mut commands: Commands
) {
    let mut agents = q_agent
        .iter()
        .map(|(entity, grid_position, _, fitness)| {
            (entity, grid_position, fitness.0)
        })
        .collect::<Vec<_>>();
    
    agents.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());

    agents
        .iter()
        .enumerate()
        .for_each(|(index,(entity, old_pos, _))| {
            let tween = Tween::new(
                EaseFunction::QuadraticInOut,
                Duration::from_secs_f32(1.0 / ANIMATION_SPEED_MULTIPLIER),
                TransformPositionLens {
                    start: old_pos.0,
                    end: grid.get_at(index),
            }).with_completed_event(1);
            commands.entity(*entity)
                .insert(
                    Animator::new(tween)
                );
        });

}

fn check_animate_transform_done(
    mut reader: EventReader<TweenCompleted>,
    mut next_state: ResMut<NextState<GeneticAlgorithmState>>
) {
    for ev in reader.read().take(1) {
        next_state.set(GeneticAlgorithmState::Selection);
    }
}
