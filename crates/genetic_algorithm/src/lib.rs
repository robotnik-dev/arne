// this library is a generic implementation of a genetic algorithm
// it uses the entity component system pattern to represent the agents and to leverage 
// the parallelism for evaluating the fitness of each agent

use std::time::Duration;

use bevy::{input::mouse::{MouseScrollUnit, MouseWheel}, pbr::DefaultOpaqueRendererMethod, prelude::*, transform::commands};
use bevy_prng::WyRand;
use rand_core::RngCore;
use bevy_rand::prelude::{EntropyComponent, GlobalEntropy, ForkableRng, EntropyPlugin};
use bevy_tweening::{lens::TransformPositionLens, *};

pub struct GeneticAlgorithmPlugin;

impl Plugin for GeneticAlgorithmPlugin {
    fn build(&self, app: &mut App) {

        // TODO: remove seed when testing is done
        let seed: u64 = 235;

        app.add_plugins(EntropyPlugin::<WyRand>::with_seed(seed.to_le_bytes()));
        app.add_plugins(TweeningPlugin);
        app.init_state::<GeneticAlgorithmState>();
        app.insert_resource(BackgroundColorGreyScale(128));
        app.insert_resource(Grid { occupied: Vec::new(), free: Vec::new()});
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
            check_animation_done
                .run_if(in_state(GeneticAlgorithmState::Animating)),
        ));
        app.add_systems(OnEnter(GeneticAlgorithmState::Animating), (
            animate_agent_transform,
        ));
    }
}

// UI SECTION


// GENETIC ALGORITHM SECTION

const POPULATION_SIZE: u32 = 100;

// #[derive(Event)]
// struct PopulationInitialized;

#[derive(Resource)]
struct BackgroundColorGreyScale(u8);

#[derive(Resource)]
struct Grid {
    occupied: Vec<(u32, u32)>,
    free: Vec<(u32, u32)>,
}

impl Grid {
    fn next_free(&mut self) -> Option<(u32, u32)> {
        // sort the free grid positions by row and col
        // the top left grid position is 1,1 and should be the first in the list
        // the bottom right grid position is 10,10 and should be the last in the list
        self.free.sort();
        self.free.first().copied()
    }
}

#[derive(States, Default, Debug, Clone, PartialEq, Eq, Hash)]
enum GeneticAlgorithmState {
    #[default]
    Startup,
    EvaluateFitness,
    CheckStopCriteria,
    // Choose the best agents to be parents for the next generation
    Selection,
    Crossover,
    Mutation,
    PhenotypeMapping,
    EndAgorithm,
    Animating,
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
struct GridPosition {
    row: u32,
    col: u32,
    x: f32,
    y: f32,
}

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
            let grid_x = 1.0 + (idx as f32 % 10.0);
            let grid_y = (idx as f32 / 10.0).floor();

            // let offset_x = -5.0;
            // let offset_y = -5.0;

            let spawn_x = grid_x;
            let spawn_y = grid_y;

            commands.entity(entity)
                .insert((
                    PbrBundle {
                        mesh: meshes.add(Cuboid::from_size(Vec3::splat(1.0))),
                        material: materials.add(Color::rgb_u8(chromosome.genes, chromosome.genes, chromosome.genes)),
                        transform: Transform::from_xyz(spawn_x, spawn_y, 0.0),
                        ..default()
                    },
                    GridPosition {
                        row: grid_x as u32,
                        col: grid_y as u32,
                        x: spawn_x,
                        y: spawn_y,
                    }
                ));
            // set all occupied grid positions
            grid.occupied.push((grid_x as u32, grid_y as u32));

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
            fitness.0 = 1.0 - (diff as f32 / 255.0);
        });

    next_state.set(GeneticAlgorithmState::CheckStopCriteria);
}

fn check_stop_criteria(
    mut next_state: ResMut<NextState<GeneticAlgorithmState>>,
    q_generation: Query<&Generation, With<Agent>>,
) {
    info!("Checking stop criteria..");

    // things to check:
    // - maximum number of generations
    // - reached a plateau
    // - a solution with satisfactory fitness is found
    // let stop = q_agent
    //     .iter()
    //     .any(|chromosome| {
    //         chromosome.genes == 255
    //     });
    let stop = q_generation
        .iter()
        .any(|generation|
            generation.0 >= 100
        );
    if stop {
        // let sum = q_agent
        //     .iter()
        //     .fold(0.0,|acc, e| {
        //         acc + e.0
        //     });
        // info!("Average fitness: {:?}", sum / POPULATION_SIZE as f32);
        next_state.set(GeneticAlgorithmState::EndAgorithm);
    } else {
        next_state.set(GeneticAlgorithmState::Selection);
    }
}

fn selection(
    mut next_state: ResMut<NextState<GeneticAlgorithmState>>,
    q_agent: Query<(Entity, &Fitness, &GridPosition), With<Agent>>,
    mut commands: Commands,
    mut grid: ResMut<Grid>
) {
    info!("Selection..");
    
    // sort the agents by fitness
    let mut agents = q_agent
        .iter()
        .collect::<Vec<_>>();

    agents.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    agents
        .iter()
        // just despawn the first half of the agents
        .take(POPULATION_SIZE as usize / 2)
        .for_each(|(entity, _, grid_position)| {
            // remove the occupied grid position and add it to free grid positions
            grid.occupied.retain(|&pos| pos != (grid_position.row, grid_position.col));
            grid.free.push((grid_position.row, grid_position.col));
            // free the entity
            commands.entity(*entity).despawn();
        });

    next_state.set(GeneticAlgorithmState::Animating);
}

fn crossover(
    mut next_state: ResMut<NextState<GeneticAlgorithmState>>,
    q_agent: Query<&Chromosome, With<Agent>>,
) {
    info_once!("Performing crossover..");
    // next_state.set(GeneticAlgorithmState::Mutation);
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
    mut key: Res<ButtonInput<KeyCode>>,
    mut next_state: ResMut<NextState<GeneticAlgorithmState>>
) {
    info_once!("Paused..");
    if key.just_pressed(KeyCode::Space) {
        info!("Resuming..");
        next_state.set(GeneticAlgorithmState::Animating);
    }

}


// run once on state enter
fn animate_agent_transform(
    mut grid: ResMut<Grid>,
    mut q_agent: Query<(Entity, &GridPosition, &Transform), With<Agent>>,
    mut commands: Commands
) {
    // the goal is to create a tuple with three elements
    // - the entity id
    // - the old transform
    // - the new transform where to move the agent
    // then create for each of these tuples a tween that moves the agent from the old to the new position

    let agents = q_agent
        .iter()
        .map(|(entity, grid_position, transform)| {
            let next_free = grid.next_free().unwrap();
            let next_x = next_free.0 as f32;
            let next_y = next_free.1 as f32;
            let next_pos = Vec3::new(next_x, next_y, 0.0);
            let old_pos = Vec3::new(grid_position.x, grid_position.y, 0.0);
            // remove old_pos from occupied and add it to free
            // remove next_pos from free and add it to occupied
            grid.occupied.retain(|&pos| pos != (grid_position.row, grid_position.col));
            grid.free.push((grid_position.row, grid_position.col));
            grid.free.retain(|&pos| pos != (next_free.0, next_free.1));
            grid.occupied.push((next_free.0, next_free.1));

            (entity, old_pos, next_pos)
        })
        .collect::<Vec<_>>();
    
    agents
        .iter()
        .for_each(|(entity, old_pos, next_pos)| {
            let tween = Tween::new(
                EaseFunction::QuadraticInOut,
                Duration::from_secs(1),
                TransformPositionLens {
                    start: *old_pos,
                    end: *next_pos
            }).with_completed_event(1);
            commands.entity(*entity)
                .insert(
                    Animator::new(tween)
                );
        });

}

fn check_animation_done(
    mut reader: EventReader<TweenCompleted>,
    mut next_state: ResMut<NextState<GeneticAlgorithmState>>
) {
    for ev in reader.read().take(1) {
        next_state.set(GeneticAlgorithmState::Crossover);
    }
}
