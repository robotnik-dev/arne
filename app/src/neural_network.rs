use std::{fmt::Debug, fs::OpenOptions, io::{self, Read, Write}};
use approx::AbsDiffEq;
use plotters::prelude::*;
use rand::prelude::*;
use rand_distr::{Distribution, Normal};
use petgraph::Graph;
use serde::{Serialize, Deserialize};

use crate::Result;
use crate::Error;
use crate::utils::round2;
use crate::image_processing::Retina;
use crate::genetic_algorithm::MUTATION_PROBABILITIES;

pub const NEURONS_PER_RNN: usize = 5;
pub const NUMBER_OF_RNN_UPDATES: usize = 40;


/// A short term memory that can be used to store the state of the network
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ShortTermMemory {
    snapshots: Vec<SnapShot>,
}

impl ShortTermMemory {
    pub fn new() -> Self {
        ShortTermMemory {
            snapshots: vec![],
        }
    }

    pub fn clear(&mut self) {
        self.snapshots.clear();
    }

    pub fn get_neuron_count(&self) -> usize {
        self.snapshots.first().map(|snapshot| snapshot.outputs.len()).unwrap_or(0)
    }

    pub fn get_timesteps(&self) -> Vec<u32> {
        self.snapshots.iter().map(|snapshot| snapshot.time_step).collect()
    }

    pub fn add_snapshot(&mut self, outputs: Vec<f64>, time_step: u32) {
        let snapshot = SnapShot::new(outputs, time_step);
        self.snapshots.push(snapshot);
    }

    pub fn get_snapshot_at_timestep(&self, time_step: u32) -> Option<&SnapShot> {
        self.snapshots.iter().find(|snapshot| snapshot.time_step == time_step)
    }

    /// function that returns a list of tuples of (timestep, neuron_output) as a vector
    /// for evry snapshot that was already saved so that it can be visualized
    pub fn get_visualization_data(&self) -> Option<Vec<Vec<(u32, f64)>>> {
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

    pub fn visualize(&self, path: String) -> Result {
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
                .build_cartesian_2d(1u32..NUMBER_OF_RNN_UPDATES as u32, -1.0f64..1.0f64)?;
        
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
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SnapShot {
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

impl SnapShot {
    pub fn new(outputs: Vec<f64>, time_step: u32) -> Self {
        SnapShot {
            outputs,
            time_step,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Rnn {
    neurons: Vec<Neuron>,
    short_term_memory: ShortTermMemory,
    /// visual representation of the network
    graph: Graph<(usize, f64), f64>,
}

impl PartialEq for Rnn {
    fn eq(&self, other: &Self) -> bool {
        self.neurons
            .iter()
            .zip(other.neurons.iter())
            .all(|(a, b)| a == b)
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
                // if cant find self activation edge, set it to 0
                self_activation: graph.find_edge(node, node).map(|edge| *graph.edge_weight(edge).unwrap()).unwrap_or(0.0),
                retina_inputs: vec![],
            };
            for neighbor in graph.neighbors(node) {
                // skip self connection
                if neighbor == node {
                    continue;
                }
                // if cant find edge, set weight to 0
                let weight = graph.find_edge(neighbor, node).map(|edge| *graph.edge_weight(edge).unwrap()).unwrap_or(0.0);
                neuron.input_connections.push((graph.node_weight(neighbor).unwrap().0, weight));
            }
            neurons.push(neuron);
        }
        Rnn {
            neurons,
            short_term_memory: ShortTermMemory::new(),
            graph,
        }
    }
}

impl Rnn {
    pub fn new(rng: &mut dyn RngCore, neuron_count: usize) -> Self {
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
            graph: Graph::default(),
        }
    }

    /// builds a new RNN from a json file located at 'path'
    pub fn from_json(path: String) -> std::result::Result<Self, Error> {
        let mut file = OpenOptions::new()
            .read(true)
            .open(path)?;
        let mut buffer = String::new();
        file.read_to_string(&mut buffer)?;
        let rnn: Rnn = serde_json::from_str(buffer.trim())?;

        Ok(rnn)
    }

    pub fn short_term_memory(&self) -> &ShortTermMemory {
        &self.short_term_memory
    }

    pub fn short_term_memory_mut(&mut self) -> &mut ShortTermMemory {
        &mut self.short_term_memory
    }

    pub fn add_snapshot(&mut self, outputs: Vec<f64>, time_step: u32) {
        self.short_term_memory.add_snapshot(outputs, time_step);
    }

    pub fn neurons(&self) -> &Vec<Neuron> {
        &self.neurons
    }

    /// each neruon has a connection to all of the 25 retina pixels and these need to be updated each rnn update step
    pub fn update_inputs_from_retina(&mut self, retina: &Retina) {
        self.neurons
            .iter_mut()
            .for_each(|neuron| {
                neuron.retina_inputs.clear();
                for i in 0..5 {
                    for j in 0..5 {
                        neuron.retina_inputs.push(retina.get_value(i, j));
                    }
                }
            });
    }

    pub fn update(&mut self) {
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
    pub fn crossover_uniform(&self, rng: &mut dyn RngCore, with: &Rnn) -> Rnn {
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

    pub fn mutate(&mut self, rng: &mut dyn RngCore) -> Rnn {
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
    pub fn delete_neuron(&mut self, rng: &mut dyn RngCore) {
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
    pub fn delete_weights(&mut self, rng: &mut dyn RngCore) {
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
    pub fn delete_bias(&mut self, rng: &mut dyn RngCore) {
        self.neurons
            .iter_mut()
            .choose(rng)
            .map(|neuron| neuron.bias = 0.0);
    }

    /// setting self activation to 0 from a random neuron
    pub fn delete_self_activation(&mut self, rng: &mut dyn RngCore) {
        self.neurons
            .iter_mut()
            .choose(rng)
            .map(|neuron| neuron.self_activation = 0.0);
    }

    /// randomize the weights self activation and bias from a random neuron
    /// randomize with a normal distribution with mean 0 and variance 0.2
    pub fn mutate_neuron(&mut self, rng: &mut dyn RngCore) {
        let variance = *MUTATION_PROBABILITIES.get("global_variance").ok_or("Cant read 'global_variance'").unwrap() as f64;
        self.neurons
            .iter_mut()
            .choose(rng)
            .map(|neuron| {
                neuron.input_connections
                    .iter_mut()
                    .for_each(|(_, weight)| {
                        *weight = Normal::new(0.0, variance).unwrap().sample(rng);
                    });
                neuron.self_activation = Normal::new(0.0, variance).unwrap().sample(rng);
                neuron.bias = Normal::new(0.0, variance).unwrap().sample(rng);
            });
    }

    /// randomize all incoming weights from a random neuron
    /// randomize with a normal distribution with mean 0 and variance 0.2
    pub fn mutate_weights(&mut self, rng: &mut dyn RngCore) {
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
    pub fn mutate_bias(&mut self, rng: &mut dyn RngCore) {
        let variance = *MUTATION_PROBABILITIES.get("global_variance").unwrap() as f64;
        self.neurons
            .iter_mut()
            .choose(rng)
            .map(|neuron| neuron.bias = Normal::new(0.0, variance).unwrap().sample(rng));
    }

    /// randomize the self activation from a random neuron
    /// randomize with a normal distribution with mean 0 and variance 0.2
    pub fn mutate_self_activation(&mut self, rng: &mut dyn RngCore) {
        let variance = *MUTATION_PROBABILITIES.get("global_variance").unwrap() as f64;
        self.neurons
            .iter_mut()
            .choose(rng)
            .map(|neuron| neuron.self_activation = Normal::new(0.0, variance).unwrap().sample(rng));
    }

    /// saves the RNN to a json file at the saves/rnn folder or if a path is provided at the given path
    pub fn to_json(&mut self, path: Option<&String>) -> Result {
        // save Dot in Rnn
        self.graph = Graph::from(self.clone());
   
        let json = serde_json::to_string_pretty(self)?;
        let file_path: String;

        if let Some(path) = path {
            file_path = path.clone();
        } else {
            let mut entries = std::fs::read_dir("saves/rnn")?
                .map(|res| res.map(|e| e.path()))
                .collect::<std::result::Result<Vec<_>, io::Error>>()?;
            entries.sort();
            let new_file_name = entries
                .iter()
                .last()
                .map(|path| {
                    let last_file_index = path
                        .to_str()
                        .unwrap()
                        .replace("saves/rnn", "")
                        .replace(".json", "")
                        .replace("\\", "")
                        .parse::<usize>()
                        .unwrap();
                    format!("{}.json", last_file_index + 1)
                })
                .unwrap_or("0.json".to_string());
            file_path = format!("saves/rnn/{}", new_file_name);
        }
        
        OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(file_path)?
            .write_fmt(format_args!("{}\n", json))?;

        Ok(())
    }

}

impl From<Rnn> for Graph<(usize,f64),f64> {
    fn from(rnn: Rnn) -> Self {
        // converting the RNN to a graph
        let mut graph = Graph::<(usize,f64), f64>::new();
        let mut nodes = vec![];

        // creating nodes and adding self activation
        for neuron in &rnn.neurons {
            let index = graph.add_node((neuron.index, round2(neuron.bias)));
            // only add self activation if it is not 0
            if round2(neuron.self_activation) != 0.0 {
                graph.add_edge(index, index, round2(neuron.self_activation));
            }
            nodes.push(index);
        }
        // adding edges
        for index in 0..nodes.len() {
            for (connected_index, weight) in &rnn.neurons[index].input_connections {
                // only add the edge if the weight is not 0
                if round2(*weight) != 0.0 {
                    graph.add_edge(nodes[*connected_index], nodes[index], round2(*weight));
                }
            }
        }
        graph
    }
}

/// A neuron in a neural network that can have a self-connection
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Neuron {
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
    retina_inputs: Vec<f32>
}

impl PartialEq for Neuron {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index &&
        round2(self.output) == round2(other.output) &&
        self.input_connections
            .iter()
            .all(|con |
                other.input_connections
                    .iter()
                    .find(|other_con|
                        (con.0, round2(con.1)) == (other_con.0, round2(other_con.1)))
                    .is_some()
            ) &&
        round2(self.bias) == round2(other.bias) &&
        round2(self.self_activation) == round2(other.self_activation)
    }
}

impl Neuron {
    pub fn new(rng: &mut dyn RngCore, index: usize, neuron_count: usize) -> Self {
        let (lower, upper) = (-1.0 / (neuron_count as f64).sqrt(), 1.0 / (neuron_count as f64).sqrt());
        Neuron {
            index,
            output: 0.,
            input_connections: vec![],
            bias: rng.gen_range(lower..=upper),
            // randomize self activation with the Xavier initialization
            self_activation: rng.gen_range(lower..=upper),
            retina_inputs: vec![],
        }
    }

    pub fn output(&self) -> f64 {
        self.output
    }
}
