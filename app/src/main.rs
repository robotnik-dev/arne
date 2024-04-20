use bevy::prelude::*;
use genetic_algorithm::GeneticAlgorithmPlugin;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(GeneticAlgorithmPlugin)
        .run();
}

// idea: using bevys states to change from initiaizing to selection to crossover to mutation
// because selection needs parralelism and should evaluate all agents at ones as fast as possible
// And use system sets to run all system at ones when in this particular state
