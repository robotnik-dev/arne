// this library is a generic implementation of a genetic algorithm
// it uses the entity component system pattern to represent the agents and to leverage 
// the parallelism for evaluating the fitness of each agent

use bevy::prelude::*;

pub struct GeneticAlgorithmPlugin;

impl Plugin for GeneticAlgorithmPlugin {
    fn build(&self, app: &mut App) {
        // app.init_resource::<MyOtherResource>();
        // app.add_event::<MyEvent>();
        app.add_systems(Startup, initialize_population);
        // app.add_systems(Update, my_system);
    }
}

#[derive(Bundle)]
struct AgentBundle {
    // marker
    agent: Agent
}

#[derive(Component)]
struct Agent;

/// At the beginning, a set of solutions,
/// which is denoted as population, is initialized.
fn initialize_population() {
    todo!()
}
