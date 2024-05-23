// use bevy::prelude::*;
// use genetic_algorithm::GeneticAlgorithmPlugin;

use approx::AbsDiffEq;
use plotters::prelude::*;
use rayon::prelude::*;
use rand::prelude::*;

type Error = Box<dyn std::error::Error>;
type Result = std::result::Result<(), Error>;

const POPULATION_SIZE: usize = 100;
const MAX_GENERATIONS: u32 = 1;
const NEURON_PER_RNN: usize = 5;
const NUMBER_OF_RNN_UPDATES: usize = 20;
const GREYSCALE_TO_MATCH: SimpleGrayscale = SimpleGrayscale(255);

trait Genotype {}

trait Phenotype {}

trait GenotypePhenotypeMapping<P: Phenotype> {
    fn map_to_phenotype(&self) -> P;
}

trait AgentEvaluation<T> {
    /// normalized fitness value between 0 and 1
    fn calculate_fitness(&self, data: T) -> f64;

    /// evaluate the agent with the given preferred output
    fn evaluate(&mut self, data: T, number_of_updates: usize) -> f64;
}

#[derive(Debug, Clone)]
struct SimpleGrayscale(u8);

impl Phenotype for SimpleGrayscale {}

// //TODO: Rename later
// #[derive(Debug, Clone)]
// struct RealPhenotype(f32);

// impl Phenotype for RealPhenotype {}

struct Population<G: Genotype, P: Phenotype> {
    agents: Vec<Agent<G,P>>,
    generation: u32,
}

impl std::fmt::Display for Population<RNN, SimpleGrayscale> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Generation: {}", self.generation)?;
        for agent in &self.agents {
            writeln!(f, "Agent: {}", agent.genotype)?;
            writeln!(f, "Fitness: {}", agent.fitness)?;
            writeln!(f, "Phenotype: {:?}", agent.genotype.map_to_phenotype())?;
            writeln!(f, "___")?;
        }
        Ok(())
    }
}

impl Population<RNN, SimpleGrayscale> {
    fn new(size: usize) -> Self {
        let agents = (0..size)
            .map(|_| Agent::new(NEURON_PER_RNN))
            .collect::<Vec<Agent<RNN, SimpleGrayscale>>>();
        Population {
            agents,
            generation: 0,
        }
    }
}

struct Agent<G: Genotype, P: Phenotype> {
    fitness: f64,
    genotype: G,
    phenotype: P,
}

impl Clone for Agent<RNN, SimpleGrayscale> {
    fn clone(&self) -> Self {
        Agent {
            fitness: self.fitness,
            genotype: self.genotype.clone(),
            phenotype: self.phenotype.clone(),
        }
    }
}

impl AgentEvaluation<SimpleGrayscale> for Agent<RNN, SimpleGrayscale> {
    fn calculate_fitness(&self, data: SimpleGrayscale) -> f64 {
        let correct_greyscale = data.0 as f64;
        let agent_greyscale = self.genotype.map_to_phenotype().0 as f64;
        1.0 - (correct_greyscale - agent_greyscale).abs() / 255.0
    }

    fn evaluate(&mut self, data: SimpleGrayscale, number_of_updates: usize) -> f64 {
        let mut local_fitness = 0.0;
        self.genotype.short_term_memory.clear();
        for i in 0..number_of_updates {
            self.genotype.update();
            // creating snapshot of the network at the current time step
            let snapshot = SnapShot {
                outputs: self.genotype.neurons.iter().map(|neuron| neuron.output).collect(),
                time_step: (i + 1) as u32,
            };
            self.genotype.short_term_memory.add_snapshot(snapshot);
            // CLONING here is okay because its only a u8, but for other implementations it might be a problem
            local_fitness += self.calculate_fitness(data.clone());
        }
        local_fitness / number_of_updates as f64
    }
}

// impl Clone for Agent<RNN, RealPhenotype> {
//     fn clone(&self) -> Self {
//         Agent {
//             fitness: 0.,
//             genotype: self.genotype.clone(),
//             phenotype: self.phenotype.clone(),
//         }
//     }
// }

impl Agent<RNN, SimpleGrayscale> {
    fn new(number_of_neurons: usize) -> Self {
        Agent {
            fitness: 0.0,
            genotype: RNN::new(number_of_neurons),
            phenotype: SimpleGrayscale(0),
        }
    }
}

/// A short term memory that can be used to store the state of the network
#[derive(Clone)]
struct ShortTermMemory {
    snapshots: Vec<SnapShot>,
}

impl ShortTermMemory {
    fn new() -> Self {
        ShortTermMemory {
            snapshots: vec![],
        }
    }

    fn clear(&mut self) {
        self.snapshots.clear();
    }

    fn get_neuron_count(&self) -> usize {
        self.snapshots.first().map(|snapshot| snapshot.outputs.len()).unwrap_or(0)
    }

    fn get_timesteps(&self) -> Vec<u32> {
        self.snapshots.iter().map(|snapshot| snapshot.time_step).collect()
    }

    fn add_snapshot(&mut self, snapshot: SnapShot) {
        self.snapshots.push(snapshot);
    }

    fn get_snapshot_at_timestep(&self, time_step: u32) -> Option<&SnapShot> {
        self.snapshots.iter().find(|snapshot| snapshot.time_step == time_step)
    }

    /// function that returns a list of tuples of (timestep, neuron_output) as a vector
    /// for evry snapshot that was already saved so that it can be visualized
    /// E.g. if we saved two snapshots at timestep 0 and 1, the function will return (asuuming we have 5 neurons)
    /// [
    ///     [
    ///         (1, get_snapshot_at_timestep(1).outputs[0]),
    ///         (2, get_snapshot_at_timestep(2).outputs[0])
    ///     ],
    ///     [
    ///         (1, get_snapshot_at_timestep(1).outputs[1]),
    ///         (2, get_snapshot_at_timestep(2).outputs[1])
    ///     ],
    ///    [...]
    /// ]
    fn get_visualization_data(&self) -> Option<Vec<Vec<(u32, f64)>>> {
        let mut data = vec![];
        for idx in 0..self.get_neuron_count() {
            let mut snapshot_data = vec![];
            for timestep in self.get_timesteps() {
                let snapshot = self.get_snapshot_at_timestep(timestep)?;
                snapshot_data.push((timestep, snapshot.outputs[idx]));
            }
            data.push(snapshot_data);
        }
        Some(data)
    }

    fn visualize(&self, to_filename: String) -> Result {
        let path = format!("images/{}.png", to_filename);
        let root = BitMapBackend::new(&path, (1024, 768)).into_drawing_area();
        root.fill(&WHITE)?;
        let visualization_data = self.get_visualization_data().ok_or("IndexError: cant get visualization data")?;
        
        let drawing_areas = root.split_evenly((visualization_data.len(),1));

        for i in 0..drawing_areas.len() {
            let mut chart_builder = ChartBuilder::on(&drawing_areas[i]);

            chart_builder
                // .caption(format!("N{}", i), ("sans-serif", 16).into_font())
                .margin(5)
                .set_all_label_area_size(20);
            
            let mut chart_context = chart_builder
                .build_cartesian_2d(1u32..20u32, -1.0f64..1.0f64)?;
        
            chart_context
                .configure_mesh()
                .disable_x_mesh()
                .max_light_lines(1)
                .y_labels(3)
                .x_labels(1)
                .draw()?;
    
            let color = Palette99::pick(i);
            chart_context
                .draw_series(LineSeries::new(visualization_data[i].iter().map(|(x, y)| (*x, *y)), &color))?;
        }

        root.present()?;

        Ok(())
    }
}

/// a snapshot of the network at a certain point in time.
/// This can be used to restore the network to a previous state but its mainly used
/// to store the outputs of the neurons to visulaize the network
#[derive(Clone)]
struct SnapShot {
    outputs: Vec<f64>,
    time_step: u32,
}

#[derive(Clone)]
struct RNN {
    neurons: Vec<Neuron>,
    short_term_memory: ShortTermMemory,
}

impl Genotype for RNN {}

impl std::fmt::Display for RNN {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for neuron in &self.neurons {
            write!(f, "Neuron {}:", neuron.index)?;
            write!(f, " self activation: {}", neuron.self_activation)?;
            write!(f, " bias: {}", neuron.bias)?;
            writeln!(f, " output: {}", neuron.output)?;
            writeln!(f, "Input neurons:")?;
            for (index, weight) in &neuron.input_connections {
                writeln!(f, " Neuron: {} with input weight {}", index, weight)?;
            }
            writeln!(f, "___")?;
        }
        Ok(())
    }
}

impl GenotypePhenotypeMapping<SimpleGrayscale> for RNN {
    fn map_to_phenotype(&self) -> SimpleGrayscale {
        let num_neurons = 3;
        let greyscale = self.neurons
            .iter()
            .skip(self.neurons.len() - num_neurons)
            // output of 0 should be 0 and output of 1 should be 255
            .map(|neuron| (neuron.output * 255.0) as f64)
            // take the avarage of the outcome
            .sum::<f64>() / num_neurons as f64;
        SimpleGrayscale(greyscale.round() as u8)
    }
}

impl RNN {
    fn new(neuron_count: usize) -> Self {
        let mut neurons = vec![];
        for i in 0..neuron_count {
            let neuron = Neuron::new(i);
            neurons.push(neuron);
        }

        // connect all neurons with each other
        neurons
            .iter_mut()
            .enumerate()
            .for_each(|(index, neuron)| {
                for i in 0..neuron_count {
                    if i != index {
                        neuron.input_connections.push((i, thread_rng().gen_range(-1.0..=1.0)));
                    }
                }
            });
        
        RNN {
            neurons,
            short_term_memory: ShortTermMemory::new(),
        }
    }

    fn update(&mut self) {
        // collect all outputs from neurons in a hashmap with access to the index
        let outputs = self.neurons
            .iter()
            .map(|neuron| (neuron.index, neuron.output))
            .collect::<std::collections::HashMap<usize, f64>>();

        self.neurons
            .iter_mut()
            .for_each(|neuron| {
                let mut activation = neuron.input_connections
                    .iter()
                    .map(|(index, weight)| weight * outputs[index])
                    .sum::<f64>();
                // add self activation to the activation
                activation += neuron.output * neuron.self_activation;
                // add the bias
                activation += neuron.bias;
                // apply the activation function to the neuron
                neuron.output = activation.tanh();
            })
    }
}

/// A neuron in a neural network that can have a self-connection
#[derive(Clone)]
struct Neuron {
    // a unique identifier for the neuron
    index: usize,
    /// the activation of the neuron
    // activation: f64,
    /// output of the neuron after activation_function is applied
    output: f64,
    /// the indices of the neurons that are connected to this neuron and the weight of the connection
    /// the weight of 0 means no connection
    input_connections: Vec<(usize, f64)>,
    bias: f64,
    /// to represent the memory of the neuron, we append self activation to the input vector
    /// but store it separately
    self_activation: f64,
}

impl Neuron {
    fn new(index: usize) -> Self {
        Neuron {
            index,
            output: 0.,
            input_connections: vec![],
            bias: thread_rng().gen_range(-1.0..=1.0),
            // randomize self activation between -1 and 1
            self_activation: thread_rng().gen_range(-1.0..=1.0),
        }
    }
}

fn main() -> Result {
    // intialize population
    let mut population = Population::new(POPULATION_SIZE);

    // loop until stop criterial is met
    loop {
        // increase generation counter
        population.generation += 1;

        // evaluate the fitness of each individual of the population
        population.agents
            .par_iter_mut()
            .for_each(|agent| {
                agent.fitness = agent.evaluate(GREYSCALE_TO_MATCH, NUMBER_OF_RNN_UPDATES);
            });
        
        // sort the population by fitness
        population.agents.sort_by(|a, b|b.fitness.partial_cmp(&a.fitness).unwrap());
        
        // check stop criteria
        if population.generation >= MAX_GENERATIONS || population.agents.iter().any(|agent| agent.fitness.abs_diff_eq(&1.0, 0.01) )
        {
            break;
        }
        
        // select the best 50 % individuals of the population and remove the rest
        population.agents.truncate(population.agents.len() / 2);

        // crossover the selected individuals to create new individuals to fill the population
        // TODO: for now just duplicate the population to fill the population
        population.agents.extend(population.agents.clone());
        
        // mutate the new individuals (chance of 1% to mutate at all)
    }
    
    println!("Stopped at generation {}", population.generation);
    
    // visualize the best agent
    let best_agent = population.agents.first().unwrap();
    best_agent.genotype.short_term_memory.visualize("best_agent".into())?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use rand_chacha::rand_core::CryptoRngCore;

    use super::*;

    #[test]
    fn test_map_to_phenotype_greyscale() {
        let mut rnn = RNN::new(3);
        rnn.neurons[0].output = 0.97;
        rnn.neurons[1].output = 0.88;
        rnn.neurons[2].output = 0.39;

        assert_eq!(190, rnn.map_to_phenotype().0);
    }
    #[test]
    fn test_map_to_phenotype_greyscale_1() {
        let mut rnn = RNN::new(3);
        rnn.neurons[0].output = 0.5;
        rnn.neurons[1].output = 0.5;
        rnn.neurons[2].output = 0.5;

        assert_eq!(128, rnn.map_to_phenotype().0);
    }

    #[test]
    fn test_map_to_phenotype_greyscale_2() {
        let mut rnn = RNN::new(3);
        rnn.neurons[0].output = 0.0;
        rnn.neurons[1].output = 0.0;
        rnn.neurons[2].output = 0.0;

        assert_eq!(0, rnn.map_to_phenotype().0);
    }

    #[test]
    fn test_map_to_phenotype_greyscale_3() {
        let mut rnn = RNN::new(3);
        rnn.neurons[0].output = 1.0;
        rnn.neurons[1].output = 1.0;
        rnn.neurons[2].output = 1.0;

        assert_eq!(255, rnn.map_to_phenotype().0);
    }

    #[test]
    fn test_calculate_fitness_agent() {
        use rand::prelude::*;
        use rand_chacha::ChaCha8Rng;
        use utils::round_to_decimal_places;

        let mut rng = ChaCha8Rng::seed_from_u64(2);
        let mut agent = Agent::new(3);
        agent.genotype.neurons
            .iter_mut()
            .for_each(|neuron|{
                neuron.input_connections
                .iter_mut()
                .for_each(|(_, weight)| *weight = round_to_decimal_places(rng.gen_range(-1.0..=1.0), 2));
                neuron.self_activation = round_to_decimal_places(rng.gen_range(-1.0..=1.0), 2);
                neuron.bias = 1.;
            });
        agent.genotype.update();
        agent.genotype.update();
        println!("{}", agent.genotype);

        let correct_greyscale = SimpleGrayscale(127);
        let fitness = agent.calculate_fitness(correct_greyscale);

        assert_eq!(round_to_decimal_places(fitness, 2), 0.75);
    }

    #[test]
    fn test_update_rnn_two_neurons() {
        use rand::prelude::*;
        use rand_chacha::ChaCha8Rng;
        use utils::round_to_decimal_places;

        let mut rng = ChaCha8Rng::seed_from_u64(2);
        let mut rnn = RNN::new(2);

        // randomize the weights and self activations with a custom seed and set the bias to 1.0
        rnn.neurons
            .iter_mut()
            .for_each(|neuron|{
                neuron.input_connections
                .iter_mut()
                .for_each(|(_, weight)| *weight = round_to_decimal_places(rng.gen_range(-1.0..=1.0), 2));
                neuron.self_activation = round_to_decimal_places(rng.gen_range(-1.0..=1.0), 2);
                neuron.bias = 1.;
            });

        // first iteration
        rnn.update();

        assert_eq!(round_to_decimal_places(rnn.neurons[0].output, 2), 0.76);
        assert_eq!(round_to_decimal_places(rnn.neurons[1].output, 2), 0.76);
        
        // second iteration
        rnn.update();

        assert_eq!(round_to_decimal_places(rnn.neurons[0].output, 2), 0.93);
        assert_eq!(round_to_decimal_places(rnn.neurons[1].output, 2), 0.96);
    }

    #[test]
    fn test_update_rnn_three_neurons() {
        use rand::prelude::*;
        use rand_chacha::ChaCha8Rng;
        use utils::round_to_decimal_places;

        let mut rng = ChaCha8Rng::seed_from_u64(2);
        let mut rnn = RNN::new(3);

        // randomize the weights and self activations with a custom seed and set the bias to 1.0
        rnn.neurons
            .iter_mut()
            .for_each(|neuron|{
                neuron.input_connections
                .iter_mut()
                .for_each(|(_, weight)| *weight = round_to_decimal_places(rng.gen_range(-1.0..=1.0), 2));
                neuron.self_activation = round_to_decimal_places(rng.gen_range(-1.0..=1.0), 2);
                neuron.bias = 1.;
            });
            
        // first iteration
        rnn.update();

        assert_eq!(round_to_decimal_places(rnn.neurons[0].output, 2), 0.76);
        assert_eq!(round_to_decimal_places(rnn.neurons[1].output, 2), 0.76);
        assert_eq!(round_to_decimal_places(rnn.neurons[2].output, 2), 0.76);
        
        // second iteration
        rnn.update();

        assert_eq!(round_to_decimal_places(rnn.neurons[0].output, 2), 0.97);
        assert_eq!(round_to_decimal_places(rnn.neurons[1].output, 2), 0.88);
        assert_eq!(round_to_decimal_places(rnn.neurons[2].output, 2), 0.39);
    }

    #[test]
    fn test_evaluate_agent() {
        use rand::prelude::*;
        use rand_chacha::ChaCha8Rng;
        use utils::round_to_decimal_places;

        let mut rng = ChaCha8Rng::seed_from_u64(2);
        let mut agent = Agent::new(3);
        let mut agent2 = Agent::new(3);

        // randomize the weights and self activations with a custom seed and set the bias to 1.0
        agent.genotype.neurons
            .iter_mut()
            .for_each(|neuron|{
                neuron.input_connections
                .iter_mut()
                .for_each(|(_, weight)| *weight = round_to_decimal_places(rng.gen_range(-1.0..=1.0), 2));
                neuron.self_activation = round_to_decimal_places(rng.gen_range(-1.0..=1.0), 2);
                neuron.bias = -0.6;
            });
        agent2.genotype.neurons
            .iter_mut()
            .for_each(|neuron|{
                neuron.input_connections
                .iter_mut()
                .for_each(|(_, weight)| *weight = round_to_decimal_places(rng.gen_range(-1.0..=1.0), 2));
                neuron.self_activation = round_to_decimal_places(rng.gen_range(-1.0..=1.0), 2);
                neuron.bias = 1.;
            });
        
        let fitness = agent.evaluate(SimpleGrayscale(127), 2);
        let fitness2 = agent2.evaluate(SimpleGrayscale(127), 2);
        

    }

    #[test]
    fn test_create_snapshots() {
        // create a new rnn with 3 neurons
        // update the rnn 5 times
        // after each update create a snapshot
        // check if the snapshots are correct
        
        let mut rnn = RNN::new(3);
        rnn.neurons[0].output = 0.97;
        rnn.neurons[1].output = 0.88;
        rnn.neurons[2].output = 0.39;
        
        for i in 0..1 {
            let snapshot = SnapShot {
                outputs: rnn.neurons.iter().map(|neuron| neuron.output).collect(),
                time_step: i,
            };
            rnn.short_term_memory.add_snapshot(snapshot);
            let saved_snapshot = rnn.short_term_memory.get_snapshot_at_timestep(i).unwrap();
            assert_eq!(saved_snapshot.outputs[0], 0.97);
            assert_eq!(saved_snapshot.outputs[1], 0.88);
            assert_eq!(saved_snapshot.outputs[2], 0.39);
        }
    }
    
    #[test]
    fn test_visualize() {
        // create a new rnn with 3 neurons
        // update the rnn 5 times
        // after each update create a snapshot
        // visualize the rnn
        let mut rnn = RNN::new(3);

        for i in 1..=5 {
            rnn.update();
            let snapshot = SnapShot {
                outputs: rnn.neurons.iter().map(|neuron| neuron.output).collect(),
                time_step: i,
            };
            rnn.short_term_memory.add_snapshot(snapshot);
        }
        rnn.short_term_memory.visualize("test_output".into()).unwrap();

    }

}