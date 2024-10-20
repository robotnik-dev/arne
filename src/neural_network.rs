use crate::image::Retina;
use crate::utils::round2;
use crate::{genetic_algorithm::Statistics, image::Position, Result};
use crate::{AdaptiveConfig, Error};
use approx::AbsDiffEq;
use bevy::prelude::*;
use bevy_prng::WyRand;
use bevy_rand::prelude::EntropyComponent;
use bevy_rand::traits::{ForkableInnerRng, ForkableRng};
use petgraph::{dot::Dot, Graph};
use plotters::prelude::*;
use rand::seq::IteratorRandom;
use rand::Rng;
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};
use std::{
    fmt::Debug,
    fs::OpenOptions,
    io::{Read, Write},
};

/// A short term memory that can be used to store the state of the network
#[derive(Clone, Debug, Serialize, Deserialize, Component)]
pub struct ShortTermMemory {
    snapshots: Vec<SnapShot>,
}

impl Default for ShortTermMemory {
    fn default() -> Self {
        Self::new()
    }
}

impl ShortTermMemory {
    pub fn new() -> Self {
        ShortTermMemory { snapshots: vec![] }
    }

    pub fn clear(&mut self) {
        self.snapshots.clear();
    }

    pub fn get_neuron_count(&self) -> usize {
        self.snapshots
            .first()
            .map(|snapshot| snapshot.outputs.len())
            .unwrap_or(0)
    }

    pub fn get_timesteps(&self) -> Vec<u32> {
        self.snapshots
            .iter()
            .map(|snapshot| snapshot.time_step)
            .collect()
    }

    pub fn add_snapshot(&mut self, outputs: Vec<f32>, time_step: u32) {
        let snapshot = SnapShot::new(outputs, time_step);
        self.snapshots.push(snapshot);
    }

    pub fn get_snapshot_at_timestep(&self, time_step: u32) -> Option<&SnapShot> {
        self.snapshots
            .iter()
            .find(|snapshot| snapshot.time_step == time_step)
    }

    /// function that returns a list of tuples of (timestep, neuron_output) as a vector
    /// for evry snapshot that was already saved so that it can be visualized
    pub fn get_visualization_data(&self) -> Option<Vec<Vec<(u32, f32)>>> {
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

    pub fn visualize(&self, path: String, nr_of_network_updates: usize) -> Result {
        let root = BitMapBackend::new(&path, (1024, 768)).into_drawing_area();
        root.fill(&WHITE)?;
        let visualization_data = self
            .get_visualization_data()
            .ok_or("IndexError: cant get visualization data")?;

        let drawing_areas = root.split_evenly((visualization_data.len(), 1));

        for i in 0..drawing_areas.len() {
            let mut chart_builder = ChartBuilder::on(&drawing_areas[i]);

            chart_builder
                // .caption(format!("N{}", i), ("sans-serif", 16).into_font())
                .margin(5)
                .set_all_label_area_size(20);

            let mut chart_context = chart_builder
                .build_cartesian_2d(0u32..nr_of_network_updates as u32, -1.0f32..1.0f32)?;

            chart_context
                .configure_mesh()
                .disable_x_mesh()
                .max_light_lines(1)
                .y_labels(3)
                .x_labels(1)
                .draw()?;

            let color = Palette99::pick(i);
            chart_context.draw_series(LineSeries::new(
                visualization_data[i].iter().map(|(x, y)| (*x, *y)),
                &color,
            ))?;
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
    outputs: Vec<f32>,
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
    pub fn new(outputs: Vec<f32>, time_step: u32) -> Self {
        SnapShot { outputs, time_step }
    }

    pub fn outputs(&self) -> &Vec<f32> {
        &self.outputs
    }

    pub fn outputs_mut(&mut self) -> &mut Vec<f32> {
        &mut self.outputs
    }

    pub fn time_step(&self) -> u32 {
        self.time_step
    }

    pub fn time_step_mut(&mut self) -> &mut u32 {
        &mut self.time_step
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Rnn {
    neurons: Vec<Neuron>,
    // 1. weights from input to hidden state (retina weights + neuron to neruon weights)
    // 2. weights from previous hidden to current hidden state
    // 3. weights from hidden to output
    // weights: (Vec<f32>, Vec<f32>, Vec<f32>),
    short_term_memory: ShortTermMemory,
    /// visual representation of the network
    graph: Graph<(usize, f32), f32>,
    statistics: Statistics,
    /// lower variance when later in the genetic algorithm
    mutation_variance: f32,
    mean: f32,
}

impl PartialEq for Rnn {
    fn eq(&self, other: &Self) -> bool {
        self.neurons
            .iter()
            .zip(other.neurons.iter())
            .all(|(a, b)| a == b)
    }
}

impl From<Graph<(usize, f32), f32>> for Rnn {
    fn from(graph: Graph<(usize, f32), f32>) -> Self {
        let mut neurons = vec![];
        for node in graph.node_indices() {
            let mut neuron = Neuron {
                index: graph.node_weight(node).unwrap().0,
                output: 0.0,
                input_connections: vec![],
                bias: graph.node_weight(node).unwrap().1,
                // if cant find self activation edge, set it to 0
                self_activation: graph
                    .find_edge(node, node)
                    .map(|edge| *graph.edge_weight(edge).unwrap())
                    .unwrap_or(0.0),
                retina_inputs: vec![],
                // TODO: maybe want to display in a graph??
                retina_weights: vec![],
            };
            for neighbor in graph.neighbors(node) {
                // skip self connection
                if neighbor == node {
                    continue;
                }
                // if cant find edge, set weight to 0
                let weight = graph
                    .find_edge(neighbor, node)
                    .map(|edge| *graph.edge_weight(edge).unwrap())
                    .unwrap_or(0.0);
                neuron
                    .input_connections_mut()
                    .push((graph.node_weight(neighbor).unwrap().0, weight));
            }
            neurons.push(neuron);
        }
        Rnn {
            neurons,
            short_term_memory: ShortTermMemory::new(),
            graph,
            statistics: Statistics::new(),
            mutation_variance: 0.2,
            mean: 0.0,
        }
    }
}

impl Rnn {
    pub fn new(
        mut rng: EntropyComponent<WyRand>,
        neuron_count: usize,
        adaptive_config: &Res<AdaptiveConfig>,
    ) -> Self {
        let mut neurons = vec![];

        for i in 0..neuron_count {
            let neuron = Neuron::new(rng.fork_rng(), i, neuron_count, adaptive_config);
            neurons.push(neuron);
        }

        neurons.iter_mut().enumerate().for_each(|(index, neuron)| {
            for i in 0..neuron_count {
                if i != index {
                    neuron.input_connections_mut().push((
                        i,
                        rng.gen_range(adaptive_config.neuron_lower..=adaptive_config.neuron_upper),
                    ));
                }
            }
        });

        Rnn {
            neurons,
            short_term_memory: ShortTermMemory::new(),
            graph: Graph::default(),
            statistics: Statistics::new(),
            mutation_variance: adaptive_config.variance,
            mean: adaptive_config.mean,
        }
    }

    /// builds a new RNN from a json file located at 'path'
    pub fn from_json(path: String) -> std::result::Result<Self, Error> {
        let mut file = OpenOptions::new().read(true).open(path)?;
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

    pub fn add_snapshot(&mut self, outputs: Vec<f32>, time_step: u32) {
        self.short_term_memory.add_snapshot(outputs, time_step);
    }

    pub fn neurons(&self) -> &Vec<Neuron> {
        &self.neurons
    }

    pub fn neurons_mut(&mut self) -> &mut Vec<Neuron> {
        &mut self.neurons
    }

    /// the scaled output of neuron 0 and neuron 1 are the next x and y position of the agent.
    /// scaling factor can be set in the config file
    pub fn next_delta_position(&self, adaptive_config: &Res<AdaptiveConfig>) -> Position {
        Position::new(
            (self.neurons()[0].output() * adaptive_config.retina_movement_speed as f32) as i32,
            (self.neurons()[1].output() * adaptive_config.retina_movement_speed as f32) as i32,
        )
    }

    /// each neuron has a connection to all of the retina SUPERpixels and these need to be updated each rnn update step
    pub fn update_inputs_from_retina(&mut self, retina: &Retina) {
        self.neurons_mut().iter_mut().for_each(|neuron| {
            neuron.retina_inputs_mut().clear();
            for i in 0..retina.superpixel_rows_or_col() {
                for j in 0..retina.superpixel_rows_or_col() {
                    neuron.add_retina_input(retina.get_superpixel_value(j, i));
                }
            }
        });
    }

    pub fn update(&mut self) {
        // collect hidden state (all outputs of the RNN at this time step)
        let h_t = self
            .neurons()
            .iter()
            .map(|neuron| (neuron.index, neuron.output()))
            .collect::<std::collections::HashMap<usize, f32>>();

        // calculate new hidden state.
        // w_xh: weights from input to hidden state -> retina_weights
        // w_hh: weights from hidden to hidden
        // w_ho: weights from hidden to output
        self.neurons_mut().iter_mut().for_each(|neuron| {
            let mut activation = neuron
                // inputs TO this neuron
                .input_connections()
                .iter()
                .map(|(index, weight)| weight * h_t[index])
                .sum::<f32>();
            // });

            // add sum of retina inputs times each retina weight to the activation
            let retina_sum = neuron
                .retina_inputs()
                .iter()
                .zip(neuron.retina_weights().iter())
                .map(|(input, weight)| *input * weight)
                .sum::<f32>();
            activation += retina_sum;

            // add self activation to the activation
            activation += neuron.self_activation() * neuron.output();
            // add the bias
            activation += neuron.bias();
            // apply the activation function to the neuron
            neuron.output = activation.tanh();
        });
    }

    /// generates a new RNN by performing a uniform crossover operation with another RNN, returning new genotype
    pub fn crossover_uniform(&self, mut rng: EntropyComponent<WyRand>, with: &Rnn) -> Rnn {
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
            for (i, weight) in neuron.input_connections_mut().iter_mut().enumerate() {
                if rng.gen_bool(0.5) {
                    weight.1 = other_neuron.input_connections()[i].1;
                }
            }
        }
        new_rnn
    }

    pub fn mutate(
        &mut self,
        mut rng: EntropyComponent<WyRand>,
        adaptive_config: &AdaptiveConfig,
    ) -> Rnn {
        if rng.gen_bool(adaptive_config.delete_neuron as f64) {
            self.delete_neuron(rng.fork_rng());
            self.statistics.deleted_neurons += 1;
        }
        if rng.gen_bool(adaptive_config.delete_weights as f64) {
            self.delete_weights(rng.fork_rng());
            self.statistics.deleted_weights += 1;
        }
        if rng.gen_bool(adaptive_config.delete_bias as f64) {
            self.delete_bias(rng.fork_rng());
            self.statistics.deleted_biases += 1;
        }
        if rng.gen_bool(adaptive_config.delete_self_activation as f64) {
            self.delete_self_activation(rng.fork_rng());
            self.statistics.deleted_self_activations += 1;
        }
        if rng.gen_bool(adaptive_config.mutate_neuron as f64) {
            self.mutate_neuron(rng.fork_rng());
            self.statistics.mutated_neurons += 1;
        }
        if rng.gen_bool(adaptive_config.mutate_weights as f64) {
            self.mutate_weights(rng.fork_rng());
            self.statistics.mutated_weights += 1;
        }
        if rng.gen_bool(adaptive_config.mutate_bias as f64) {
            self.mutate_bias(rng.fork_rng());
            self.statistics.mutated_biases += 1;
        }
        if rng.gen_bool(adaptive_config.mutate_self_activation as f64) {
            self.mutate_self_activation(rng.fork_rng());
            self.statistics.mutated_self_activations += 1;
        }
        self.clone()
    }

    /// setting all incoming weights, self activation and bias to 0 from a random neuron
    pub fn delete_neuron(&mut self, mut rng: EntropyComponent<WyRand>) {
        if let Some(neuron) = self
            .neurons_mut()
            .iter_mut()
            // .filter(|n| !n.is_deleted())
            .choose(&mut rng.fork_inner())
        {
            neuron
                .input_connections_mut()
                .iter_mut()
                .for_each(|(_, weight)| *weight = 0.0);
            neuron.self_activation = 0.0;
            neuron.bias = 0.0;
            neuron
                .retina_weights_mut()
                .iter_mut()
                .for_each(|weight| *weight = 0.0);
        };
    }

    /// setting one weight to 0 from a random neuron
    pub fn delete_weights(&mut self, mut rng: EntropyComponent<WyRand>) {
        if let Some(neuron) = self.neurons.iter_mut().choose(&mut rng.fork_inner()) {
            if rng.gen_bool(0.5) {
                if let Some((_, weight)) = neuron
                    .input_connections_mut()
                    .iter_mut()
                    // .filter(|(_, w)| *w != 0.0)
                    .choose(&mut rng.fork_inner())
                {
                    *weight = 0.0;
                }
            } else if let Some(weight) = neuron
                .retina_weights_mut()
                .iter_mut()
                // .filter(|w| *w != &0.0)
                .choose(&mut rng.fork_inner())
            {
                *weight = 0.0;
            }
        };
    }

    /// setting bias to 0 from a random neuron
    pub fn delete_bias(&mut self, mut rng: EntropyComponent<WyRand>) {
        if let Some(neuron) = self
            .neurons_mut()
            .iter_mut()
            // .filter(|n: &&mut Neuron| n.bias() != 0.0)
            .choose(&mut rng.fork_inner())
        {
            neuron.bias = 0.0
        };
    }

    /// setting self activation to 0 from a random neuron
    pub fn delete_self_activation(&mut self, mut rng: EntropyComponent<WyRand>) {
        if let Some(neuron) = self
            .neurons_mut()
            .iter_mut()
            // .filter(|n| n.self_activation() != 0.0)
            .choose(&mut rng.fork_inner())
        {
            neuron.self_activation = 0.0
        };
    }

    /// randomize one weight, self activation and bias from a random neuron
    pub fn mutate_neuron(&mut self, mut rng: EntropyComponent<WyRand>) {
        let std_dev = self.mutation_variance;
        let mean = self.mean;
        if let Some(neuron) = self.neurons_mut().iter_mut().choose(&mut rng.fork_inner()) {
            neuron.self_activation = Normal::new(mean, std_dev)
                .unwrap()
                .sample(&mut rng.fork_inner());
            neuron.bias = Normal::new(mean, std_dev)
                .unwrap()
                .sample(&mut rng.fork_inner());
            if let Some((_, weight)) = neuron
                .input_connections_mut()
                .iter_mut()
                .choose(&mut rng.fork_inner())
            {
                *weight = Normal::new(mean, std_dev)
                    .unwrap()
                    .sample(&mut rng.fork_inner());
            }
            if let Some(weight) = neuron
                .retina_weights_mut()
                .iter_mut()
                .choose(&mut rng.fork_inner())
            {
                *weight = Normal::new(mean, std_dev)
                    .unwrap()
                    .sample(&mut rng.fork_inner());
            }

            // if rng.gen_bool(0.5) {
            //     if let Some((_, weight)) = neuron.input_connections_mut().iter_mut().choose(rng) {
            //         *weight = Normal::new(mean, std_dev).unwrap().sample(rng);
            //     }
            // } else if let Some(weight) = neuron.retina_weights_mut().iter_mut().choose(rng) {
            //     *weight = Normal::new(mean, std_dev).unwrap().sample(rng);
            // }
        };
    }

    /// randomize one incoming weight from a random neuron
    pub fn mutate_weights(&mut self, mut rng: EntropyComponent<WyRand>) {
        let std_dev = self.mutation_variance;
        let mean = self.mean;
        if let Some(neuron) = self.neurons_mut().iter_mut().choose(&mut rng.fork_inner()) {
            if let Some((_, weight)) = neuron
                .input_connections_mut()
                .iter_mut()
                .choose(&mut rng.fork_inner())
            {
                *weight = Normal::new(mean, std_dev)
                    .unwrap()
                    .sample(&mut rng.fork_inner());
            }
            if let Some(weight) = neuron
                .retina_weights_mut()
                .iter_mut()
                .choose(&mut rng.fork_inner())
            {
                *weight = Normal::new(mean, std_dev)
                    .unwrap()
                    .sample(&mut rng.fork_inner());
            }
        };
    }

    /// randomize the bias from a random neuron
    /// randomize with a normal distribution with mean 0 and variance 0.2
    pub fn mutate_bias(&mut self, mut rng: EntropyComponent<WyRand>) {
        let std_dev = self.mutation_variance;
        let mean = self.mean;
        if let Some(neuron) = self.neurons_mut().iter_mut().choose(&mut rng.fork_inner()) {
            neuron.bias = Normal::new(mean, std_dev)
                .unwrap()
                .sample(&mut rng.fork_inner())
        };
    }

    /// randomize the self activation from a random neuron
    /// randomize with a normal distribution with mean 0 and variance 0.2
    pub fn mutate_self_activation(&mut self, mut rng: EntropyComponent<WyRand>) {
        let std_dev = self.mutation_variance;
        let mean = self.mean;
        if let Some(neuron) = self.neurons_mut().iter_mut().choose(&mut rng.fork_inner()) {
            neuron.self_activation = Normal::new(mean, std_dev)
                .unwrap()
                .sample(&mut rng.fork_inner())
        };
    }

    pub fn update_variance(&mut self, variance: f32) {
        self.mutation_variance = variance;
        self.statistics.variance = variance;
    }

    pub fn variance(&self) -> f32 {
        self.mutation_variance
    }

    pub fn update_fitness(&mut self, fitness: f32) {
        self.statistics.fitness = fitness;
    }

    /// saves the RNN to a json file at the saves/rnn folder or if a path is provided at the given path
    pub fn to_json(&mut self, path: String) -> Result {
        self.graph = Graph::from(self.clone());

        let json = serde_json::to_string_pretty(self)?;

        OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(path)?
            .write_fmt(format_args!("{}\n", json))?;

        Ok(())
    }

    pub fn to_dot(&self, path: String) -> Result {
        let graph = Graph::from(self.clone());
        let dot = Dot::new(&graph);
        OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(path)
            .unwrap()
            .write_fmt(format_args!("{:?}\n", dot))
            .unwrap();
        Ok(())
    }
}

impl From<Rnn> for Graph<(usize, f32), f32> {
    fn from(rnn: Rnn) -> Self {
        // converting the RNN to a graph
        let mut graph = Graph::<(usize, f32), f32>::new();
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
    /// output of the neuron after activation_function is applied
    output: f32,
    /// the indices of the neurons that are connected to this neuron and the weight of the connection
    /// the weight of 0 means no connection
    input_connections: Vec<(usize, f32)>,
    bias: f32,
    /// to represent the memory of the neuron, we append self activation to the input vector
    /// but store it separately
    self_activation: f32,
    /// the superpixel values of the retina
    #[serde(skip)]
    retina_inputs: Vec<f32>,
    /// the weights of the retina inputs
    retina_weights: Vec<f32>,
}

impl PartialEq for Neuron {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index
            && round2(self.output) == round2(other.output)
            && self.input_connections().iter().all(|con| {
                other
                    .input_connections()
                    .iter()
                    .any(|other_con| (con.0, round2(con.1)) == (other_con.0, round2(other_con.1)))
            })
            && self
                .retina_weights
                .iter()
                .zip(other.retina_weights.iter())
                .all(|(a, b)| round2(*a) == round2(*b))
            && round2(self.bias) == round2(other.bias)
            && round2(self.self_activation) == round2(other.self_activation)
    }
}

impl Neuron {
    pub fn new(
        mut rng: EntropyComponent<WyRand>,
        index: usize,
        _neuron_count: usize,
        adaptive_config: &AdaptiveConfig,
    ) -> Self {
        let superpixel_size = adaptive_config.superpixel_size as usize;
        let retina_size = adaptive_config.retina_size as usize;
        let retina_weights = (0..(retina_size / superpixel_size).pow(2))
            // set 90 % of retina weights to 0
            .map(|_| {
                if rng.gen_bool(adaptive_config.init_non_zero_retina_weights as f64) {
                    rng.gen_range(adaptive_config.retina_lower..=adaptive_config.retina_upper)
                } else {
                    0.0
                }
            })
            .collect::<Vec<f32>>();

        Neuron {
            index,
            output: 0.,
            input_connections: vec![],
            bias: rng.gen_range(adaptive_config.neuron_lower..=adaptive_config.neuron_upper),
            // randomize self activation with the Xavier initialization
            self_activation: rng
                .gen_range(adaptive_config.neuron_lower..=adaptive_config.neuron_upper),
            retina_inputs: vec![],
            retina_weights,
        }
    }

    pub fn output(&self) -> f32 {
        self.output
    }

    pub fn set_output(&mut self, output: f32) {
        self.output = output;
    }

    pub fn input_connections(&self) -> &Vec<(usize, f32)> {
        &self.input_connections
    }

    pub fn input_connections_mut(&mut self) -> &mut Vec<(usize, f32)> {
        &mut self.input_connections
    }

    pub fn self_activation(&self) -> f32 {
        self.self_activation
    }

    pub fn set_self_activation(&mut self, self_activation: f32) {
        self.self_activation = self_activation;
    }

    pub fn bias(&self) -> f32 {
        self.bias
    }

    pub fn set_bias(&mut self, bias: f32) {
        self.bias = bias;
    }

    pub fn retina_inputs(&self) -> &Vec<f32> {
        &self.retina_inputs
    }

    pub fn retina_inputs_mut(&mut self) -> &mut Vec<f32> {
        &mut self.retina_inputs
    }

    pub fn retina_weights(&self) -> &Vec<f32> {
        &self.retina_weights
    }

    pub fn retina_weights_mut(&mut self) -> &mut Vec<f32> {
        &mut self.retina_weights
    }

    pub fn add_retina_input(&mut self, input: f32) {
        self.retina_inputs.push(input);
    }

    /// after the retian size changes, we need to update the retina weights with the new size
    /// for the we need to take the first n weights from the retina_weights_buffer and overwrite the retina_weights
    pub fn update_retina_weights(&mut self, new_weights: Vec<f32>) {
        self.retina_weights = new_weights;
    }

    /// checks if every weights connections in or outgoing from this neuron is 0
    pub fn is_deleted(&self) -> bool {
        self.input_connections()
            .iter()
            .all(|(_, weight)| *weight == 0.0)
            && self.self_activation == 0.0
            && self.bias == 0.0
            && self.retina_weights().iter().all(|weight| *weight == 0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn test_from_graph_to_rnn() {
        let mut graph = Graph::<(usize, f32), f32>::new();
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

        graph.node_indices().for_each(|node| {
            graph.neighbors(node).for_each(|neighbor| {
                // get weight from neuron at index "node" from the agent and the neuron at index "neighbor"
                if let Some(correct_weight) = rnn.neurons[node.index()]
                    .input_connections()
                    .iter()
                    .find(|(index, _)| *index == neighbor.index())
                {
                    assert_eq!(
                        round2(correct_weight.1),
                        round2(
                            *graph
                                .edge_weight(graph.find_edge(neighbor, node).expect("msg"))
                                .unwrap()
                        )
                    );
                }
            });
        });
    }

    // #[test]
    // fn update() {
    //     let image = Image::from_vec(vec![0.; 1000 * 1000]).unwrap();
    //     let retina = image
    //         .create_retina_at(Position::new(100, 100), 45, 5, String::from(""))
    //         .unwrap();
    //     let filepath = String::from("current_config.json");
    //     let adaptive_config: AdaptiveConfig =
    //         from_str(read_to_string(filepath).unwrap().as_str()).unwrap();
    //     let mut agent = Agent::new(&adaptive_config);
    //     let save_path = String::from("tests/rnn");
    //     let time_steps = 10;

    //     for t in 0..time_steps {
    //         agent
    //             .genotype_mut()
    //             .networks_mut()
    //             .iter_mut()
    //             .enumerate()
    //             .for_each(|(idx, network)| {
    //                 let _ = create_dir_all(format!("{}/{}", save_path, idx));

    //                 // saving before update step
    //                 network
    //                     .clone()
    //                     .to_json(format!("{}/{}/t_{}.json", save_path, idx, t))
    //                     .unwrap();

    //                 // doing the update
    //                 network.update_inputs_from_retina(&retina);
    //                 network.update();

    //                 // saving after update
    //                 // network
    //                 //     .clone()
    //                 //     .to_json(format!("{}/{}/t_{}_after.json", save_path, idx, t))
    //                 //     .unwrap();
    //             });
    //     }
    // }
}
