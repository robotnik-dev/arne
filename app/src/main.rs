use std::collections::HashMap;
use approx::AbsDiffEq;
use plotters::prelude::*;
use rayon::prelude::*;
use rand::prelude::*;
use rand_distr::{Distribution, Normal};
use petgraph::{dot::Dot, Graph};
use rand_chacha::ChaCha8Rng;
use lazy_static::lazy_static;


type Error = Box<dyn std::error::Error>;
type Result = std::result::Result<(), Error>;

const POPULATION_SIZE: usize = 100;
const MAX_GENERATIONS: u32 = 1;
const NEURONS_PER_RNN: usize = 5;
const NUMBER_OF_RNN_UPDATES: usize = 20;
const GREYSCALE_TO_MATCH: SimpleGrayscale = SimpleGrayscale(255);

lazy_static! {
    static ref MUTATION_PROBABILITIES: HashMap<String, f32> = {
        let mut m = HashMap::new();
        // global mutation rate which can be changed later
        m.insert("global_variance".to_string(), 0.2);
        // setting all incoming weights, self activation and bias to 0
        m.insert("delete_neuron".to_string(), 0.1);
        // setting all incoming weights to 0
        m.insert("delete_weights".to_string(), 0.1);
        // setting bias to 0
        m.insert("delete_bias".to_string(), 0.1);
        // setting self activation to 0
        m.insert("delete_self_activation".to_string(), 0.1);
        // randomize the weights self activation and bias
        m.insert("mutate_neuron".to_string(), 0.2);
        // randomize all incoming weights
        m.insert("mutate_weights".to_string(), 0.2);
        // randomize the bias
        m.insert("mutate_bias".to_string(), 0.1);
        // randomize the self activation
        m.insert("mutate_self_activation".to_string(), 0.1);
        m
    };
}

// Stuff to change and experiment with:
// - crossover method is uniform, try other methods
// - selection method is roulette wheel, try other methods
// - mutation chances
// - number of neurons in the RNN
// - Population size

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

#[derive(Debug, Clone)]
struct Position {
    x: f32,
    y: f32,
}

/// This phenotype/solution to the problem is a line follower.
/// If the agent/retina(TBD) can stay in each iteration step on the line(the center pixel of the image)
/// the higher the fitness value will be.
#[derive(Debug, Clone)]
struct FollowLine {
    // store only the difference between the current position and the position
    // of the last iteration
    delta_position: Position,
}

impl Phenotype for FollowLine {}

struct Population {
    agents: Vec<Agent>,
    generation: u32,
}

impl Population {
    fn new(rng: &mut dyn RngCore, size: usize, neurons_per_rnn: usize) -> Self {
        let agents = (0..size)
            .map(|_| Agent::new(rng, neurons_per_rnn))
            .collect();
        Population {
            agents,
            generation: 0,
        }
    }

    fn evolve(&mut self, new_agents: Vec<Agent>) {
        self.agents = new_agents;
        self.generation += 1;
    }

    /// roulette wheel selection
    fn select_weighted(&self, rng: &mut dyn RngCore) -> &Agent {
        self.agents.choose_weighted(rng, |agent| agent.fitness.max(0.000001)).unwrap()
    }

    /// tournament selection
    fn select_tournament(&self, rng: &mut dyn RngCore) -> &Agent {
        todo!()
    }
}

struct Agent {
    fitness: f64,
    genotype: Rnn,
}

impl Clone for Agent {
    fn clone(&self) -> Self {
        Agent {
            fitness: self.fitness,
            genotype: self.genotype.clone(),
        }
    }
}

impl AgentEvaluation<SimpleGrayscale> for Agent {
    fn calculate_fitness(&self, data: SimpleGrayscale) -> f64 {
        let correct_greyscale = data.0 as f64;
        let phenotype = self.genotype.map_to_phenotype().0 as f64;
        1.0 - (correct_greyscale - phenotype).abs() / 255.0
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

impl Agent {
    fn new(rng: &mut dyn RngCore, number_of_neurons: usize) -> Self {
        Agent {
            fitness: 0.0,
            genotype: Rnn::new(rng, number_of_neurons),
        }
    }

    pub fn crossover(&self, rng: &mut dyn RngCore, with: &Agent) -> Agent {
        let offspring = self.genotype.crossover_uniform(rng, &with.genotype);
        Agent::from(offspring)
    }

    pub fn mutate(&mut self, rng: &mut dyn RngCore) {
        self.genotype.mutate(rng);
    }
}

impl From<Rnn> for Agent {
    fn from(rnn: Rnn) -> Self {
        Agent {
            fitness: 0.0,
            genotype: rnn,
        }
    }
}

/// A short term memory that can be used to store the state of the network
#[derive(Clone, Debug)]
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

impl PartialEq for ShortTermMemory {
    fn eq(&self, other: &Self) -> bool {
        self.snapshots
            .iter()
            .zip(other.snapshots.iter())
            .all(|(a, b)| a == b)
    }
}

/// a snapshot of the network at a certain point in time.
/// This can be used to restore the network to a previous state but its mainly used
/// to store the outputs of the neurons to visulaize the network
#[derive(Clone, Debug)]
struct SnapShot {
    outputs: Vec<f64>,
    time_step: u32,
}

impl PartialEq for SnapShot {
    fn eq(&self, other: &Self) -> bool {
        self.outputs
            .iter()
            .zip(other.outputs.iter())
            .all(|(a, b)| a.abs_diff_eq(b, 0.01))
    }
}

#[derive(Clone, Debug)]
struct Rnn {
    neurons: Vec<Neuron>,
    short_term_memory: ShortTermMemory,
}

impl PartialEq for Rnn {
    fn eq(&self, other: &Self) -> bool {
        self.neurons
            .iter()
            .zip(other.neurons.iter())
            .all(|(a, b)| a == b)
    }
}

impl GenotypePhenotypeMapping<SimpleGrayscale> for Rnn {
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

impl From<Graph<(usize,f64),f64>> for Rnn {
    fn from(graph: Graph<(usize,f64),f64>) -> Self {
        let mut neurons = vec![];
        for node in graph.node_indices() {
            let mut neuron = Neuron {
                index: graph.node_weight(node).unwrap().0,
                output: 0.0,
                input_connections: vec![],
                bias: graph.node_weight(node).unwrap().1,
                self_activation: *graph.edge_weight(graph.find_edge(node, node).expect("msg")).unwrap(),
            };
            for neighbor in graph.neighbors(node) {
                // skip self connection
                if neighbor == node {
                    continue;
                }
                neuron.input_connections.push((graph.node_weight(neighbor).unwrap().0, *graph.edge_weight(graph.find_edge(neighbor, node).expect("msg")).unwrap()));
            }
            neurons.push(neuron);
        }
        Rnn {
            neurons,
            short_term_memory: ShortTermMemory::new(),
        }
    }
}

impl Rnn {
    fn new(rng: &mut dyn RngCore, neuron_count: usize) -> Self {
        let mut neurons = vec![];
        for i in 0..neuron_count {
            let neuron = Neuron::new(rng, i, neuron_count);
            neurons.push(neuron);
        }

        // connect all neurons with each other
        // with the Xavier initialization
        let (lower, upper) = (-1.0 / (neuron_count as f64).sqrt(), 1.0 / (neuron_count as f64).sqrt());
        neurons
            .iter_mut()
            .enumerate()
            .for_each(|(index, neuron)| {
                for i in 0..neuron_count {
                    if i != index {
                        neuron.input_connections.push((i, rng.gen_range(lower..=upper)));
                    }
                }
            });
        
        Rnn {
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
            });
    }

    /// generates a new RNN by performing a uniform crossover operation with another RNN, returning new genotype
    fn crossover_uniform(&self, rng: &mut dyn RngCore, with: &Rnn) -> Rnn {
        let mut new_rnn = self.clone();
        for (neuron, other_neuron) in new_rnn.neurons.iter_mut().zip(with.neurons.iter()) {
            if rng.gen_bool(0.5) {
                // crossover the self activation
                neuron.self_activation = other_neuron.self_activation;
            }
            if rng.gen_bool(0.5) {
                // crossover the bias
                neuron.bias = other_neuron.bias;
            }
            // crossover the weights
            for (i, weight) in neuron.input_connections.iter_mut().enumerate() {
                if rng.gen_bool(0.5) {
                    weight.1 = other_neuron.input_connections[i].1;
                }
            }
        }
        new_rnn
    }

    fn mutate(&mut self, rng: &mut dyn RngCore) -> Self {
        // check for each entry in the global MUTATION_PROBABILITIES if the mutation should be applied
        // then call the corresponding function
        for (mutation, probability) in MUTATION_PROBABILITIES.iter() {
            if rng.gen_bool(*probability as f64) {
                match mutation.as_str() {
                    "delete_neuron" => self.delete_neuron(rng),
                    "delete_weights" => self.delete_weights(rng),
                    "delete_bias" => self.delete_bias(rng),
                    "delete_self_activation" => self.delete_self_activation(rng),
                    "mutate_neuron" => self.mutate_neuron(rng),
                    "mutate_weights" => self.mutate_weights(rng),
                    "mutate_bias" => self.mutate_bias(rng),
                    "mutate_self_activation" => self.mutate_self_activation(rng),
                    _ => (),
                }
            }
        }
        self.clone()
    }
    
    /// setting all incoming weights, self activation and bias to 0 from a random neuron
    fn delete_neuron(&mut self, rng: &mut dyn RngCore) {
        self.neurons
            .iter_mut()
            .choose(rng)
            .map(|neuron| {
                neuron.input_connections
                    .iter_mut()
                    .for_each(|(_, weight)| *weight = 0.0);
                neuron.self_activation = 0.0;
                neuron.bias = 0.0;
            });
    }
    
    /// setting all incoming weights to 0 from a random neuron
    fn delete_weights(&mut self, rng: &mut dyn RngCore) {
        self.neurons
            .iter_mut()
            .choose(rng)
            .map(|neuron| {
                neuron.input_connections
                    .iter_mut()
                    .for_each(|(_, weight)| *weight = 0.0);
            });
    }
    
    /// setting bias to 0 from a random neuron
    fn delete_bias(&mut self, rng: &mut dyn RngCore) {
        self.neurons
            .iter_mut()
            .choose(rng)
            .map(|neuron| neuron.bias = 0.0);
    }

    /// setting self activation to 0 from a random neuron
    fn delete_self_activation(&mut self, rng: &mut dyn RngCore) {
        self.neurons
            .iter_mut()
            .choose(rng)
            .map(|neuron| neuron.self_activation = 0.0);
    }

    /// randomize the weights self activation and bias from a random neuron
    /// randomize with a normal distribution with mean 0 and variance 0.2
    fn mutate_neuron(&mut self, rng: &mut dyn RngCore) {
        let variance = *MUTATION_PROBABILITIES.get("global_variance").unwrap() as f64;
        self.neurons
            .iter_mut()
            .choose(rng)
            .map(|neuron| {
                neuron.input_connections
                    .iter_mut()
                    .for_each(|(_, weight)| *weight = Normal::new(0.0, variance).unwrap().sample(rng));
                neuron.self_activation = Normal::new(0.0, variance).unwrap().sample(rng);
                neuron.bias = Normal::new(0.0, variance).unwrap().sample(rng);
            });
    }

    /// randomize all incoming weights from a random neuron
    /// randomize with a normal distribution with mean 0 and variance 0.2
    fn mutate_weights(&mut self, rng: &mut dyn RngCore) {
        let variance = *MUTATION_PROBABILITIES.get("global_variance").unwrap() as f64;
        self.neurons
            .iter_mut()
            .choose(rng)
            .map(|neuron| {
                neuron.input_connections
                    .iter_mut()
                    .for_each(|(_, weight)| *weight = Normal::new(0.0, variance).unwrap().sample(rng));
            });
    }

    /// randomize the bias from a random neuron
    /// randomize with a normal distribution with mean 0 and variance 0.2
    fn mutate_bias(&mut self, rng: &mut dyn RngCore) {
        let variance = *MUTATION_PROBABILITIES.get("global_variance").unwrap() as f64;
        self.neurons
            .iter_mut()
            .choose(rng)
            .map(|neuron| neuron.bias = Normal::new(0.0, variance).unwrap().sample(rng));
    }

    /// randomize the self activation from a random neuron
    /// randomize with a normal distribution with mean 0 and variance 0.2
    fn mutate_self_activation(&mut self, rng: &mut dyn RngCore) {
        let variance = *MUTATION_PROBABILITIES.get("global_variance").unwrap() as f64;
        self.neurons
            .iter_mut()
            .choose(rng)
            .map(|neuron| neuron.self_activation = Normal::new(0.0, variance).unwrap().sample(rng));
    }

}

impl From<Rnn> for Graph<(usize,f64),f64> {
    fn from(rnn: Rnn) -> Self {
        // converting the RNN to a graph
        let mut graph = Graph::<(usize,f64), f64>::new();
        let mut nodes = vec![];

        // creating nodes and adding self activation
        for neuron in &rnn.neurons {
            let index = graph.add_node((neuron.index, neuron.bias));
            graph.add_edge(index, index, neuron.self_activation);
            nodes.push(index);
        }
        // adding edges
        for index in 0..nodes.len() {
            for (connected_index, weight) in &rnn.neurons[index].input_connections {
                graph.add_edge(nodes[*connected_index], nodes[index], *weight);
            }
        }
        graph
    }
}

/// A neuron in a neural network that can have a self-connection
#[derive(Clone, Debug)]
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

impl PartialEq for Neuron {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index &&
        self.output.abs_diff_eq(&other.output, 0.01) &&
        self.input_connections.iter().all(|con | other.input_connections.iter().find(|other_con| con == *other_con ).is_some()) &&
        self.bias.abs_diff_eq(&other.bias, 0.01) &&
        self.self_activation.abs_diff_eq(&other.self_activation, 0.01)
    }
}

impl Neuron {
    fn new(rng: &mut dyn RngCore, index: usize, neuron_count: usize) -> Self {
        let (lower, upper) = (-1.0 / (neuron_count as f64).sqrt(), 1.0 / (neuron_count as f64).sqrt());
        Neuron {
            index,
            output: 0.,
            input_connections: vec![],
            bias: rng.gen_range(-1.0..=1.0),
            // randomize self activation with the Xavier initialization
            self_activation: rng.gen_range(lower..=upper),
        }
    }
}

fn main() -> Result {
    let mut rng = ChaCha8Rng::seed_from_u64(2);

    // intialize population
    let mut population = Population::new(&mut rng, POPULATION_SIZE, NEURONS_PER_RNN);

    // loop until stop criterial is met
    loop {
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

        let new_agents = (0..population.agents.len())
            .map(|_| {
                let parent1 = population.select_weighted(&mut rng);
                let parent2 = population.select_weighted(&mut rng);
                let mut offspring = parent1.crossover(&mut rng, parent2);
                offspring.mutate(&mut rng);
                offspring
            })
            .collect::<Vec<Agent>>();
            
        population.evolve(new_agents);
    }
    
    println!("Stopped at generation {}", population.generation);
    
    // visualize the best agent
    let best_agent = population.agents.first().unwrap();
    best_agent.genotype.short_term_memory.visualize("best_agent".into())?;
    let graph = Graph::from(best_agent.genotype.clone());
    let dot = Dot::new(&graph);
    println!("{:?}", dot);

    Ok(())
}

#[cfg(test)]
mod tests {
    use petgraph::{dot::Dot, visit::NodeRef};
    use rand_chacha::ChaCha8Rng;

    use super::*;

    #[test]
    fn test_map_to_phenotype_greyscale() {
        let mut rng = ChaCha8Rng::seed_from_u64(2);
        let mut rnn = Rnn::new(&mut rng, 3);
        rnn.neurons[0].output = 0.97;
        rnn.neurons[1].output = 0.88;
        rnn.neurons[2].output = 0.39;

        assert_eq!(190, rnn.map_to_phenotype().0);
    }
    #[test]
    fn test_map_to_phenotype_greyscale_1() {
        let mut rng = ChaCha8Rng::seed_from_u64(2);
        let mut rnn = Rnn::new(&mut rng, 3);
        rnn.neurons[0].output = 0.5;
        rnn.neurons[1].output = 0.5;
        rnn.neurons[2].output = 0.5;

        assert_eq!(128, rnn.map_to_phenotype().0);
    }

    #[test]
    fn test_map_to_phenotype_greyscale_2() {
        let mut rng = ChaCha8Rng::seed_from_u64(2);
        let mut rnn = Rnn::new(&mut rng, 3);
        rnn.neurons[0].output = 0.0;
        rnn.neurons[1].output = 0.0;
        rnn.neurons[2].output = 0.0;

        assert_eq!(0, rnn.map_to_phenotype().0);
    }

    #[test]
    fn test_map_to_phenotype_greyscale_3() {
        let mut rng = ChaCha8Rng::seed_from_u64(2);
        let mut rnn = Rnn::new(&mut rng, 3);
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
        let mut agent = Agent::new(&mut rng, 3);
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

        let correct_greyscale = SimpleGrayscale(127);
        let fitness = agent.calculate_fitness(correct_greyscale);

        assert_eq!(round_to_decimal_places(fitness, 2), 0.81);
    }

    #[test]
    fn test_update_rnn_two_neurons() {
        use rand::prelude::*;
        use rand_chacha::ChaCha8Rng;
        use utils::round_to_decimal_places;

        let mut rng = ChaCha8Rng::seed_from_u64(2);
        let mut rnn = Rnn::new(&mut rng, 3);

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

        assert_eq!(round_to_decimal_places(rnn.neurons[0].output, 2), 0.4);
        assert_eq!(round_to_decimal_places(rnn.neurons[1].output, 2), 0.86);
    }

    #[test]
    fn test_update_rnn_three_neurons() {
        use rand::prelude::*;
        use rand_chacha::ChaCha8Rng;
        use utils::round_to_decimal_places;

        let mut rng = ChaCha8Rng::seed_from_u64(2);
        let mut rnn = Rnn::new(&mut rng, 3);

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

        assert_eq!(round_to_decimal_places(rnn.neurons[0].output, 2), 0.4);
        assert_eq!(round_to_decimal_places(rnn.neurons[1].output, 2), 0.86);
        assert_eq!(round_to_decimal_places(rnn.neurons[2].output, 2), 0.79);
    }

    #[test]
    fn test_evaluate_agent() {
        use rand::prelude::*;
        use rand_chacha::ChaCha8Rng;
        use utils::round_to_decimal_places;

        let mut rng = ChaCha8Rng::seed_from_u64(2);

        let mut agent = Agent::new(&mut rng, 3);
        let mut agent2 = Agent::new(&mut rng, 3);

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
        
        let mut rng = ChaCha8Rng::seed_from_u64(2);
        let mut rnn = Rnn::new(&mut rng, 3);
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
    fn test_snapshot_eq() {
        let snapshot = SnapShot {
            outputs: vec![0.97, 0.88, 0.39],
            time_step: 1,
        };
        let snapshot2 = SnapShot {
            outputs: vec![0.97, 0.88, 0.39],
            time_step: 1,
        };
        assert_eq!(snapshot, snapshot2);
    }

    #[test]
    fn test_snapshot_not_eq() {
        let snapshot = SnapShot {
            outputs: vec![1.97, 0.88, 0.39],
            time_step: 1,
        };
        let snapshot2 = SnapShot {
            outputs: vec![0.97, 0.88, 0.39],
            time_step: 1,
        };
        assert_ne!(snapshot, snapshot2);
    }

    #[test]
    fn test_short_term_memory_eq() {
        let stm = ShortTermMemory {
            snapshots: vec![
                SnapShot {
                    outputs: vec![0.97, 0.88222222, 0.39],
                    time_step: 1,
                },
                SnapShot {
                    outputs: vec![1.97, -0.88, 0.0],
                    time_step: 2,
                },
                SnapShot {
                    outputs: vec![2.955555557, 0.88, -0.39],
                    time_step: 3,
                },
                SnapShot {
                    outputs: vec![0.922227, 0.0, 0.39],
                    time_step: 4,
                },
            ],
        };
        let stm2 = ShortTermMemory {
            snapshots: vec![
                SnapShot {
                    outputs: vec![0.97, 0.88222222, 0.39],
                    time_step: 1,
                },
                SnapShot {
                    outputs: vec![1.97, -0.88, 0.0],
                    time_step: 2,
                },
                SnapShot {
                    outputs: vec![2.955555557, 0.88, -0.39],
                    time_step: 3,
                },
                SnapShot {
                    outputs: vec![0.922227, 0.0, 0.39],
                    time_step: 4,
                },
            ],
        };

        assert_eq!(stm, stm2);
    }

    #[test]
    fn test_from_rnn_to_graph() {
        use rand::prelude::*;
        use rand_chacha::ChaCha8Rng;
        use utils::round_to_decimal_places;

        let mut rng = ChaCha8Rng::seed_from_u64(2);

        let mut agent = Agent::new(&mut rng, 3);
        agent.genotype.neurons
            .iter_mut()
            .for_each(|neuron|{
                neuron.input_connections
                .iter_mut()
                .for_each(|(_, weight)| *weight = round_to_decimal_places(rng.gen_range(-1.0..=1.0), 2));
                neuron.self_activation = round_to_decimal_places(rng.gen_range(-1.0..=1.0), 2);
                neuron.bias = -0.6;
            });

        let graph = Graph::<(usize, f64), f64>::from(agent.genotype.clone());

        graph
            .node_indices()
            .for_each(|node| {
                graph
                    .neighbors(node)
                    .for_each(|neighbor| {
                        // get weight from neuron at index "node" from the agent and the neuron at index "neighbor"
                        if let Some(correct_weight) = agent.genotype.neurons[node.index()].input_connections
                            .iter()
                            .find(|(index, _)| *index == neighbor.index()) {
                                assert_eq!(correct_weight.1, *graph.edge_weight(graph.find_edge(neighbor, node).expect("msg")).unwrap());
                            }
                    });
            });
    }

    #[test]
    fn test_from_graph_to_rnn() {
        let mut graph = Graph::<(usize, f64), f64>::new();
        let node1 = graph.add_node((0, 1.0));
        let node2 = graph.add_node((1, -0.5));
        let node3 = graph.add_node((2, 0.0));

        // self connections
        graph.add_edge(node1, node1, 0.2);
        graph.add_edge(node2, node2, -0.2);
        graph.add_edge(node3, node3, 0.0);

        // connections between neurons
        graph.add_edge(node1, node2, 0.5);
        graph.add_edge(node1, node3, 0.3);
        graph.add_edge(node2, node1, 0.1);
        graph.add_edge(node2, node3, 0.55);
        graph.add_edge(node3, node1, 0.98);
        graph.add_edge(node3, node2, 0.11);

        let rnn = Rnn::from(graph.clone());

        graph
            .node_indices()
            .for_each(|node| {
                graph
                    .neighbors(node)
                    .for_each(|neighbor| {
                        // get weight from neuron at index "node" from the agent and the neuron at index "neighbor"
                        if let Some(correct_weight) = rnn.neurons[node.index()].input_connections
                            .iter()
                            .find(|(index, _)| *index == neighbor.index()) {
                                assert_eq!(correct_weight.1, *graph.edge_weight(graph.find_edge(neighbor, node).expect("msg")).unwrap());
                            }
                    });
            });

    }

    #[test]
    fn test_rnn_to_graph_conversion_and_back() {
        let mut rng = ChaCha8Rng::seed_from_u64(2);
        let mut rnn = Rnn::new(&mut rng, 3);

        let graph = Graph::<(usize, f64), f64>::from(rnn.clone());
        let rnn2 = Rnn::from(graph.clone());

        assert_eq!(rnn, rnn2);
    }

    #[test]
    fn test_crossover_uniform() {
        let mut rng = ChaCha8Rng::seed_from_u64(2);
        let mut agent = Agent::new(&mut rng, 10);
        let mut agent2 = Agent::new(&mut rng, 10);

        agent.genotype.neurons
            .iter_mut()
            .for_each(|neuron|{
                neuron.input_connections
                    .iter_mut()
                    .for_each(|(_, weight)| *weight = 0.5);
                neuron.self_activation = 0.1;
                neuron.bias = 1.;
            });
        
        agent2.genotype.neurons
            .iter_mut()
            .for_each(|neuron|{
                neuron.input_connections
                    .iter_mut()
                    .for_each(|(_, weight)| *weight = -0.5);
                neuron.self_activation = -0.1;
                neuron.bias = -1.;
            });

        let offspring = agent.crossover(&mut rng, &agent2);
        
        // check if the offspring is different from the parents
        assert_ne!(agent.genotype, offspring.genotype);
        assert_ne!(agent2.genotype, offspring.genotype);

        // print the parents and then the offspring as graph
        let parent1_graph = Graph::from(agent.genotype.clone());
        let dot1 = Dot::new(&parent1_graph);
        let parent2_graph = Graph::from(agent2.genotype.clone());
        let dot2 = Dot::new(&parent2_graph);
        let offspring_graph = Graph::from(offspring.genotype.clone());
        let dot3 = Dot::new(&offspring_graph);
        println!("Parent1 \n {:?}", dot1);
        println!("Parent2 \n {:?}", dot2);
        println!("Offspring \n {:?}", dot3);

        // check if the number count of all negative numbers in the offsrping are approximately the saame as the psotive numbers
        let negative_count = offspring.genotype.neurons
            .iter()
            .map(|neuron| neuron.input_connections.iter().filter(|(_, weight)| *weight < 0.0).count())
            .sum::<usize>();
        let positive_count = offspring.genotype.neurons
            .iter()
            .map(|neuron| neuron.input_connections.iter().filter(|(_, weight)| *weight > 0.0).count())
            .sum::<usize>();
        
        assert_eq!(positive_count, 43);
        assert_eq!(negative_count, 47);
    }


    #[test]
    fn test_delete_neuron() {
        let mut rng = ChaCha8Rng::seed_from_u64(2);
        let mut rnn = Rnn::new(&mut rng, 3);

        rnn.delete_neuron(&mut rng);

        assert_eq!(rnn.neurons[0].input_connections.iter().map(|(_, weight)| *weight).sum::<f64>(), 0.0);
        assert_eq!(rnn.neurons[0].self_activation, 0.0);
        assert_eq!(rnn.neurons[0].bias, 0.0);
    }

    #[test]
    fn test_delete_weights() {
        let mut rng = ChaCha8Rng::seed_from_u64(2);
        let mut rnn = Rnn::new(&mut rng, 3);

        rnn.delete_weights(&mut rng);

        assert_eq!(rnn.neurons[0].input_connections.iter().map(|(_, weight)| *weight).sum::<f64>(), 0.0);
    }

    #[test]
    fn test_delete_bias() {
        let mut rng = ChaCha8Rng::seed_from_u64(2);
        let mut rnn = Rnn::new(&mut rng, 3);

        rnn.delete_bias(&mut rng);

        assert_eq!(rnn.neurons[0].bias, 0.0);
    }

    #[test]
    fn test_delete_self_activation() {
        let mut rng = ChaCha8Rng::seed_from_u64(2);
        let mut rnn = Rnn::new(&mut rng, 3);

        rnn.delete_self_activation(&mut rng);

        assert_eq!(rnn.neurons[0].self_activation, 0.0);
    }

    #[test]
    fn test_mutate_neuron() {
        let mut rng = ChaCha8Rng::seed_from_u64(2);
        let mut rnn = Rnn::new(&mut rng, 3);
        let bias = rnn.neurons[0].bias;
        let self_activation = rnn.neurons[0].self_activation;
        let weights = rnn.neurons[0].input_connections.iter().map(|(_, weight)| *weight).collect::<Vec<f64>>();

        rnn.mutate_neuron(&mut rng);

        // Check that the properties of the neuron have been changed.
        assert_ne!(bias, rnn.neurons[0].bias);
        assert_ne!(self_activation, rnn.neurons[0].self_activation);
        assert_ne!(weights, rnn.neurons[0].input_connections.iter().map(|(_, weight)| *weight).collect::<Vec<f64>>());
    }

    #[test]
    fn test_mutate_weights() {
        let mut rng = ChaCha8Rng::seed_from_u64(2);
        let mut rnn = Rnn::new(&mut rng, 3);
        let weights = rnn.neurons[0].input_connections.iter().map(|(_, weight)| *weight).collect::<Vec<f64>>();

        rnn.mutate_weights(&mut rng);

        // Check that the properties of the neuron have been changed.
        assert_ne!(weights, rnn.neurons[0].input_connections.iter().map(|(_, weight)| *weight).collect::<Vec<f64>>());
    }

    #[test]
    fn test_mutate_bias() {
        let mut rng = ChaCha8Rng::seed_from_u64(2);
        let mut rnn = Rnn::new(&mut rng, 3);
        let bias = rnn.neurons[0].bias;

        rnn.mutate_bias(&mut rng);

        // Check that the properties of the neuron have been changed.
        assert_ne!(bias, rnn.neurons[0].bias);
    }

    #[test]
    fn test_mutate_self_activation() {
        let mut rng = ChaCha8Rng::seed_from_u64(2);
        let mut rnn = Rnn::new(&mut rng, 3);
        let self_activation = rnn.neurons[0].self_activation;

        rnn.mutate_self_activation(&mut rng);

        // Check that the properties of the neuron have been changed.
        assert_ne!(self_activation, rnn.neurons[0].self_activation);
    }

}