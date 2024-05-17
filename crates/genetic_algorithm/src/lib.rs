// // this library is a generic implementation of a genetic algorithm
// // it uses the entity component system pattern to represent the agents and to leverage 
// // the parallelism for evaluating the fitness of each agent

// use std::{marker::PhantomData, time::Duration};

// use bevy::{ecs::query::QueryData, input::mouse::{MouseScrollUnit, MouseWheel}, prelude::*, sprite::MaterialMesh2dBundle, transform::commands};
// use bevy_prng::WyRand;
// use rand_core::RngCore;
// use bevy_rand::prelude::{EntropyComponent, GlobalEntropy, ForkableRng, EntropyPlugin};
// use bevy_tweening::{lens::{TransformPositionLens, TransformScaleLens}, *};

// pub struct GeneticAlgorithmPlugin;

// impl Plugin for GeneticAlgorithmPlugin {
//     fn build(&self, app: &mut App) {

//         // TODO: remove seed when testing is done
//         let seed: u64 = 234;

//         app.add_plugins(EntropyPlugin::<WyRand>::with_seed(seed.to_le_bytes()));
//         app.add_plugins(TweeningPlugin);
//         app.init_state::<GeneticAlgorithmState>();
//         app.insert_resource(BackgroundColorGreyScale(255));
//         app.insert_resource(Generation(0));
//         app.insert_resource(Grid::default());
//         // app.add_event::<AgentOutsourced>();

//         app.add_systems(Startup, (
//             (
//                 initialize_population,
//                 setup_scene,
//             ).chain(),
//         ));

//         app.add_systems(Update, (
//             // rotate_camera,
//             // zoom_camera,
//             fitness_evaluation
//                 .run_if(in_state(GeneticAlgorithmState::EvaluateFitness)),
//             check_stop_criteria
//                 .run_if(in_state(GeneticAlgorithmState::CheckStopCriteria)),
//             crossover
//                 .run_if(in_state(GeneticAlgorithmState::Crossover)),
//             mutation
//                 .run_if(in_state(GeneticAlgorithmState::Mutation)),
//             phenotype_mapping
//                 .run_if(in_state(GeneticAlgorithmState::PhenotypeMapping)),
//             selection
//                 .run_if(in_state(GeneticAlgorithmState::Selection)),
//             clean_up
//                 .run_if(in_state(GeneticAlgorithmState::EndAgorithm)),
//             paused
//                 .run_if(in_state(GeneticAlgorithmState::Pause)),
//             check_animate_transform_done
//                 .run_if(in_state(GeneticAlgorithmState::AnimateMovement)),
//             check_animate_despawn_done
//                 .run_if(in_state(GeneticAlgorithmState::AnimateDespawn)),
//         ));
//         app.add_systems(OnEnter(GeneticAlgorithmState::AnimateMovement), (
//             animate_agent_transform,
//         ));
//     }
// }

// // UI SECTION


// // GENETIC ALGORITHM SECTION

// const POPULATION_SIZE: u32 = 100;
// const MAXIMUM_GENERATIONS: u32 = 100;
// const ANIMATION_SPEED_MULTIPLIER: f32 = 1.0;
// const DESPAWN_SPEED_MULTIPLIER: f32 = 1.0;
// const MESH2D_SCALE: f32 = 128.0;

// const RNN_SIZE: usize = 5;

// /// gets calles when the fitness of an agent is not high enough and he needs to be despawned
// // #[derive(Event)]
// // struct AgentOutsourced(Entity);

// #[derive(Resource)]
// struct BackgroundColorGreyScale(u8);

// #[derive(Resource)]
// struct Generation(u32);

// /// a grid which stores the row and column positions of the agents
// #[derive(Resource)]
// struct Grid{
//     rows: u32,
//     cols: u32,
// }

// impl Grid {
//     fn get_pos_at(&self, index: usize) -> Vec3 {
//         let row = 1 + index as u32 % self.rows;
//         let col = 1 + index as u32 / self.cols;
//         Vec3::new(row as f32, col as f32, 0.0) * MESH2D_SCALE
//     }
// }

// impl Default for Grid {
//     fn default() -> Self {
//         if POPULATION_SIZE % 10 != 0 {
//             panic!("Population size must be a multiple of 10");
//         }
//         Grid {
//             rows: POPULATION_SIZE / 10,
//             cols: POPULATION_SIZE / 10,
//         }
//     }
// }

// #[derive(States, Default, Debug, Clone, PartialEq, Eq, Hash)]
// enum GeneticAlgorithmState {
//     #[default]
//     Startup,
//     EvaluateFitness,
//     CheckStopCriteria,
//     // Choose the best agents to be parents for the next generation
//     AnimateMovement,
//     AnimateDespawn,
//     Selection,
//     Crossover,
//     Mutation,
//     PhenotypeMapping,
//     EndAgorithm,
//     // debug pause state
//     Pause
// }

// /// The AgentBundle is a collection of components that represent an agent.
// #[derive(Bundle)]
// struct AgentBundle {
//     // marker
//     agent: Agent,
//     fitness: Fitness,
//     chromosome: Rnn,
// }

// impl Default for AgentBundle {
//     fn default() -> Self {
//         Self {
//             agent: Agent,
//             fitness: Fitness(0.0),
//             chromosome: Rnn::default(),
//         }
//     }
// }

// /// Recurrent neural network. Fixed neuron count.
// #[derive(Component)]
// struct Rnn {
//     neurons: Vec<NeuronBundle>,
// }

// impl Default for Rnn {
//     fn default() -> Self {
//         Rnn {
//             neurons: vec![NeuronBundle::default(); RNN_SIZE],
//         }
//     }
// }

// // impl Rnn {
// //     fn new() -> Self {
// //         Rnn {
// //             neurons: Vec::new(),
// //         }
// //     }
// // }

// /// An agent is a potential solution to the problem.
// #[derive(Component)]
// struct Agent;

// /// The fitness of an agent is a measure of how well it performs.
// #[derive(Component, PartialEq, PartialOrd)]
// struct Fitness(f32);

// /// marker component for the offspring of two agents
// #[derive(Component)]
// struct Offspring {
//     chromosome: Rnn,
//     spawn_pos: Vec3,
// }

// // NEURAL NETWORK SECTION

// /// tangens hyperbolicus activation function maps the output to the range [-1, 1]
// const TANH: ActivationFunction = ActivationFunction(|x| x.tanh());
// /// ReLU activation function maps the output to the range [0, inf)
// const RELU: ActivationFunction = ActivationFunction(|x| x.max(0.0));

// #[derive(Bundle, Clone)]
// struct NeuronBundle {
//     neuron: Neuron,
//     activation_function: ActivationFunction,
//     bias: Bias,
//     /// The weights of the input connections to the neuron.
//     /// The weights are used to calculate the activation of the neuron.
//     weights: Weights,
//     /// List of entity IDs that are connected as inputs to the neuron.
//     inputs: Inputs,
//     /// List of entity IDs that are connected as outputs to the neuron.
//     outputs: Outputs,
//     /// The activation of the neuron, generated by the activation function.
//     activation: Activation,
// }

// #[derive(Component, Clone)]
// struct Neuron;

// #[derive(Component, Clone)]
// struct ActivationFunction(fn(f32) -> f32);

// #[derive(Component, Clone)]
// struct Bias(f32);

// #[derive(Component, Clone)]
// struct Weights(Vec<f32>);

// #[derive(Component, Clone)]
// struct Inputs(Vec<u32>);

// #[derive(Component, Clone)]
// struct Outputs(Vec<u32>);

// #[derive(Component, Clone)]
// struct Activation(f32);

// impl Default for NeuronBundle {
//     fn default() -> Self {
//         NeuronBundle {
//             neuron: Neuron,
//             activation_function: TANH,
//             bias: Bias(0.0),
//             weights: Weights(Vec::new()),
//             inputs: Inputs(Vec::new()),
//             outputs: Outputs(Vec::new()),
//             activation: Activation(0.0),
//         }
//     }
// }

// /// At the beginning, a set of solutions,
// /// which is denoted as population, is initialized.
// /// This runs once when the App starts.
// /// When done, the next state is set to Selection.
// fn initialize_population(
//     mut commands: Commands,
//     mut rng: ResMut<GlobalEntropy<WyRand>>,
//     mut generation: ResMut<Generation>,
// ) {
//     // set an offset for x and y position so that the camera can rotate around the center
//     // create the population
//     for _ in 0..POPULATION_SIZE {
//         commands.spawn((
//             AgentBundle::default(),
//             rng.fork_rng(),
//         ));
//     }
//     generation.0 += 1;
// }

// fn setup_scene(
//     mut commands: Commands,
//     mut meshes: ResMut<Assets<Mesh>>,
//     mut materials: ResMut<Assets<ColorMaterial>>,
//     q_agent: Query<(Entity, &Rnn), With<Agent>>,
//     bg_color: Res<BackgroundColorGreyScale>,
//     mut next_state: ResMut<NextState<GeneticAlgorithmState>>,
//     grid: ResMut<Grid>
// ) {

//     // setup camera
//     let mut camera_bundle = Camera2dBundle::default();
//     camera_bundle.transform = Transform::from_xyz(MESH2D_SCALE * 5., MESH2D_SCALE * 5., 0.0);
//     camera_bundle.projection.scale = 2.0;
//     camera_bundle.camera.clear_color = ClearColorConfig::Custom(Color::rgb_u8(bg_color.0, bg_color.0, bg_color.0));
//     commands.spawn(
//         camera_bundle
//     );

//     // attach visual to agents
//     q_agent
//         .iter()
//         .enumerate()
//         .for_each(|(index, (entity, rnn))| {
//             let grid_x = grid.get_pos_at(index as usize).x;
//             let grid_y = grid.get_pos_at(index as usize).y;
//             commands.entity(entity)
//                 .insert((
//                     MaterialMesh2dBundle {
//                         mesh: meshes.add(Rectangle::from_size(Vec2::splat(1.0))).into(),
//                         material: materials.add(Color::rgb_u8(1, 1, 1)),
//                         transform: Transform::from_xyz(grid_x, grid_y, 0.0).with_scale(Vec3::splat(MESH2D_SCALE)),
//                         ..default()
//                     },
//                 ));
//         });
//     next_state.set(GeneticAlgorithmState::Pause);
// }

// /// The population is evaluated for fitness.
// /// This system runs for each Agent in the population while in the selection state.
// /// The fitness of each agent is evaluated. (This is a slow operation)
// /// The fitness of each agent is stored in the Fitness component.
// /// The next state is set to Crossover.
// fn fitness_evaluation(
//     mut q_agent: Query<(&mut Fitness, &Rnn, &mut EntropyComponent<WyRand>), With<Agent>>,
//     mut next_state: ResMut<NextState<GeneticAlgorithmState>>,
//     bg_color: Res<BackgroundColorGreyScale>,
//     // mut grid: ResMut<Grid>
// ) {
//     info!("Evaluating fitness..");
//     info!("Number of agents: {}", q_agent.iter().len());
//     // compute fitness
//     q_agent
//         .par_iter_mut()
//         .for_each(|(mut fitness, rnn, mut rng)| {
//             // steps to evaluate fitness:
//             // - train the agent with training data(operational heavy task) optional for now.
//             // - use the trained agent to solve the problem
//             // - evaluate the performance of the agent
//             // - assign the performance as the fitness of the agent

//             // The problem to solve is minimizing the error value of a function.
//             // for this test its just a rgb color comparison to a background color.
//             // The less the difference the better the fitness.

//             // let diff = chromosome.genes.abs_diff(bg_color.0);
//             // normalize the difference to the range [0, 1] and assign as fitness
//             // fitness.0 = 255.0 - (diff as f32);
//             // info!("Color: {} Fitness: {}", chromosome.genes, fitness.0);
//         });
    
//     next_state.set(GeneticAlgorithmState::CheckStopCriteria);
// }

// fn check_stop_criteria(
//     mut next_state: ResMut<NextState<GeneticAlgorithmState>>,
//     q_rnn: Query<&Rnn, With<Agent>>,
//     generation: Res<Generation>,
// ) {
//     info!("Checking stop criteria..");
//     // things to check:
//     // - maximum number of generations
//     // - reached a plateau
//     // - a solution with satisfactory fitness is found
//     let stop = false; // generation.0 >= MAXIMUM_GENERATIONS || q_rnn.iter().all(|rnn| chromosome.genes == 255);

//     if stop {
//         info!("Stopping criteria met..");
//         next_state.set(GeneticAlgorithmState::EndAgorithm);
//     } else {
//         info!("Stopping criteria NOT met. Going on with algorithm..");
//         next_state.set(GeneticAlgorithmState::AnimateMovement);
//     }
// }

// fn selection(
//     mut next_state: ResMut<NextState<GeneticAlgorithmState>>,
//     q_agent: Query<(Entity, &Fitness, &Transform), With<Agent>>,
//     mut commands: Commands,
//     // mut grid: ResMut<Grid>
// ) {
//     info!("Selection..");
    
//     // sort the agents by fitness
//     let mut agents = q_agent
//         .iter()
//         .collect::<Vec<_>>();

//     agents.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

//     agents
//         .iter()
//         // just despawn the first half of the agents -> change this later?
//         .take(POPULATION_SIZE as usize / 2)
//         .for_each(|(entity, _, transform)| {
//             // attach a tween component to animate shrinking the agent
//             let tween = Tween::new(
//                 EaseFunction::QuadraticInOut,
//                 Duration::from_secs_f32(1.0 / DESPAWN_SPEED_MULTIPLIER),
//                 TransformScaleLens {
//                     start: transform.scale,
//                     end: Vec3::splat(0.0),
//             }).with_completed_event(1);
//             commands.entity(*entity)
//                 .insert(
//                     Animator::new(tween)
//                 );
//             // commands.entity(*entity).despawn();
//         });

//     next_state.set(GeneticAlgorithmState::AnimateDespawn);
// }

// fn crossover(
//     mut next_state: ResMut<NextState<GeneticAlgorithmState>>,
//     q_agent: Query<&mut Rnn, With<Agent>>,
//     mut commands: Commands,
//     grid: Res<Grid>,
//     mut rng: ResMut<GlobalEntropy<WyRand>>,
// ) {
//     info!("Performing crossover..");

//     let number_of_agents = q_agent.iter().len();
//     let mut agents = q_agent.iter().collect::<Vec<_>>();
//     let mut other_agents = q_agent.iter().collect::<Vec<_>>();
//     let n = 4;
//     let mask = (1 << n) - 1;
//     let _ = agents
//         .iter_mut()
//         .take(number_of_agents / 2)
//         .zip(other_agents.iter_mut().skip(number_of_agents / 2))
//         .enumerate()
//         .for_each(|(index, (rnn1, rnn2))| {
//             // let a_masked_1 = chromosome1.genes & mask;
//             // let b_masked_1 = chromosome2.genes & !mask;

//             // let a_masked_2 = chromosome1.genes & !mask;
//             // let b_masked_2 = chromosome2.genes & mask;

//             // let index_1 = index * 2;
//             // let index_2 = index * 2 + 1;

//             // let offspring_1 = a_masked_1 | b_masked_1;
//             // let offspring_2 = a_masked_2 | b_masked_2;

//             // let grid_pos_1 = grid.get_pos_at(index_1);
//             // let grid_pos_2 = grid.get_pos_at(index_2);

//             // commands.spawn((
//             //     AgentBundle {
//             //         agent: Agent,
//             //         fitness: Fitness(0.0),
//             //         chromosome: Chromosome::new(offspring_1),
//             //         rng: rng.fork_rng(),
//             //     },
//             //     Offspring {
//             //         chromosome: Chromosome::new(offspring_1),
//             //         spawn_pos: grid_pos_1,
//             //     },
//             // ));
//             // commands.spawn((
//             //     AgentBundle {
//             //         agent: Agent,
//             //         fitness: Fitness(0.0),
//             //         chromosome: Chromosome::new(offspring_2),
//             //         rng: rng.fork_rng(),
//             //     },
//             //     Offspring {
//             //         chromosome: Chromosome::new(offspring_2),
//             //         spawn_pos: grid_pos_2,
//             //     },
//             // ));
//         });
    
//     next_state.set(GeneticAlgorithmState::Mutation);
// }

// fn mutation(
//     mut next_state: ResMut<NextState<GeneticAlgorithmState>>,
//     mut q_offspring: Query<(&mut Offspring, &mut Rnn, &mut EntropyComponent<WyRand>), With<Agent>>,
// ) {
//     info!("Performing mutation..");

//     q_offspring
//         .iter_mut()
//         .for_each(|(mut offspring, mut rnn, mut rng)| {
//             // a 1 % chance of mutation
//             let should_mutate = rng.next_u32() % 100 == 0;
//             // if should_mutate {
//             //     let mask = 1 << (rng.next_u32() % 8);
//             //     offspring.chromosome.genes ^= mask;
//             //     chromosome.genes ^= mask;
//             // }
//         });
//     next_state.set(GeneticAlgorithmState::PhenotypeMapping);
// }

// fn phenotype_mapping(
//     mut next_state: ResMut<NextState<GeneticAlgorithmState>>,
//     mut generation: ResMut<Generation>,
//     mut commands: Commands,
//     mut q_offspring: Query<(Entity, &mut Offspring), With<Agent>>,
//     mut meshes: ResMut<Assets<Mesh>>,
//     mut materials: ResMut<Assets<ColorMaterial>>,
// ) {
//     info!("Performing phenotype mapping..");

//     q_offspring
//         .iter_mut()
//         .for_each(|(entity, offspring)| {
//             commands.entity(entity)
//                 .insert(MaterialMesh2dBundle {
//                         mesh: meshes.add(Rectangle::from_size(Vec2::splat(1.0))).into(),
//                         material: materials.add(Color::rgb_u8(1, 1, 1)),
//                         transform: Transform::from_xyz(offspring.spawn_pos.x, offspring.spawn_pos.y,0.0).with_scale(Vec3::splat(MESH2D_SCALE)),
//                         ..default()
//                 });
//             commands.entity(entity).remove::<Offspring>();
//         });

//     generation.0 += 1;
//     info!("Generation: {}", generation.0);

//     next_state.set(GeneticAlgorithmState::EvaluateFitness);
// }

// fn clean_up(
//     time: Res<Time>,
//     mut next_state: ResMut<NextState<GeneticAlgorithmState>>
// ) {
//     info!("Collection data..");
//     info!("Ending algorithm..");
//     info!("Time elapsed {:?}", time.elapsed());
//     // shut the programm down
//     // set it to paused for now
//     next_state.set(GeneticAlgorithmState::Pause);

//     // std::process::exit(0);
// }

// fn paused(
//     key: Res<ButtonInput<KeyCode>>,
//     mut next_state: ResMut<NextState<GeneticAlgorithmState>>
// ) {
//     info_once!("Paused..");
//     if key.just_pressed(KeyCode::Space) {
//         next_state.set(GeneticAlgorithmState::EvaluateFitness);
//     }
// }

// // run once on state enter
// fn animate_agent_transform(
//     grid: ResMut<Grid>,
//     q_agent: Query<(Entity, &Transform, &Fitness), With<Agent>>,
//     mut commands: Commands
// ) {
//     let mut agents = q_agent
//         .iter()
//         .map(|(entity, transform, fitness)| {
//             (entity, transform, fitness.0)
//         })
//         .collect::<Vec<_>>();
    
//     agents.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());

//     agents
//         .iter()
//         .enumerate()
//         .for_each(|(index,(entity, old_transform, _))| {
//             // info!("Index: {} Old pos: {:?}, new pos{}", index, old_pos.0, grid.get_pos_at(index));
//             let tween = Tween::new(
//                 EaseFunction::QuadraticInOut,
//                 Duration::from_secs_f32(1.0 / ANIMATION_SPEED_MULTIPLIER),
//                 TransformPositionLens {
//                     start: old_transform.translation,
//                     end: grid.get_pos_at(index),
//             }).with_completed_event(1);
//             commands.entity(*entity)
//                 .insert(
//                     Animator::new(tween)
//                 );
//         });

// }

// fn check_animate_transform_done(
//     mut reader: EventReader<TweenCompleted>,
//     mut next_state: ResMut<NextState<GeneticAlgorithmState>>,
//     q_agents: Query<(Entity, &Animator<Transform>), With<Agent>>,
//     mut commands: Commands
// ) {
//     let mut count = 0;
//     for ev in reader.read() {
//         count += 1;
//         commands.entity(ev.entity).remove::<Animator<Transform>>();
//         if q_agents.iter().len() == count {
//             next_state.set(GeneticAlgorithmState::Selection);
//         }
//     }
// }

// fn check_animate_despawn_done(
//     mut reader: EventReader<TweenCompleted>,
//     mut next_state: ResMut<NextState<GeneticAlgorithmState>>,
//     q_agents: Query<(Entity, &Animator<Transform>), With<Agent>>,
//     mut commands: Commands
// ) {
//     let mut count = 0;
//     for ev in reader.read() {
//         count += 1;
//         commands.entity(ev.entity).despawn();
//         if q_agents.iter().len() == count {
//             next_state.set(GeneticAlgorithmState::Crossover);
//         }
//     }
// }