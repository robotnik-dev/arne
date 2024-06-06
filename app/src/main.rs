use std::{fs::OpenOptions, io::Write};
use approx::AbsDiffEq;
use rayon::prelude::*;
use rand::prelude::*;
use petgraph::{dot::Dot, Graph};
use rand_chacha::ChaCha8Rng;

mod utils;
pub use utils::round2;

mod image_processing;
pub use image_processing::Retina;

mod neural_network;
pub use neural_network::{Rnn, SnapShot, ShortTermMemory, NEURONS_PER_RNN, NUMBER_OF_RNN_UPDATES};

mod genetic_algorithm;
pub use genetic_algorithm::{Agent, Population, POPULATION_SIZE, MAX_GENERATIONS, AgentEvaluation, SimpleGrayscale, GREYSCALE_TO_MATCH};

type Error = Box<dyn std::error::Error>;
type Result = std::result::Result<(), Error>;


// Stuff to change and experiment with:
// - crossover method is uniform, try other methods
// - selection method is roulette wheel, try other methods
// - mutation chances
// - number of neurons in the RNN
// - Population size

fn main() -> Result {
    let mut rng = ChaCha8Rng::seed_from_u64(2);

    // intialize population
    let mut population = Population::new(&mut rng, POPULATION_SIZE, NEURONS_PER_RNN);

    // loop until stop criterial is met
    loop {
        // evaluate the fitness of each individual of the population
        population
            .agents_mut()
            .par_iter_mut()
            .for_each(|agent| {
                let fitness = agent.evaluate(GREYSCALE_TO_MATCH, NUMBER_OF_RNN_UPDATES);
                agent.set_fitness(fitness);
            });
        
        // sort the population by fitness
        population.agents_mut().sort_by(|a, b|b.fitness().partial_cmp(&a.fitness()).unwrap());
        
        // check stop criteria
        if population.generation() >= MAX_GENERATIONS || population.agents().iter().any(|agent| agent.fitness().abs_diff_eq(&1.0, 0.01) )
        {
            break;
        }

        let new_agents = (0..population.agents().len())
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
    
    println!("Stopped at generation {}", population.generation());
    
    // visualize the best agent as png image
    let best_agent = population.agents_mut().first_mut().unwrap();
    best_agent.genotype().short_term_memory().visualize("test/images/agents/best_agent.png".into())?;

    // save visualization as .dot file
    let graph = Graph::from(best_agent.genotype().clone());
    let dot = Dot::new(&graph);
    OpenOptions::new()
        .create(true)
        .write(true)
        .open("test/images/agents/best_agent.dot")?
        .write_fmt(format_args!("{:?}\n", dot))?;

    // save as json file in "saves/rnn/"
    best_agent.genotype_mut().to_json(None)?;

    Ok(())
}

// #[cfg(test)]
// mod tests {
//     use graph::NodeIndex;
//     use petgraph::{dot::Dot, visit::{IntoEdges, NodeRef}};
//     use plotters::style::text_anchor::Pos;
//     use rand_chacha::ChaCha8Rng;

//     use super::*;

//     #[test]
//     fn test_map_to_phenotype_greyscale() {
//         let mut rng = ChaCha8Rng::seed_from_u64(2);
//         let mut rnn = Rnn::new(&mut rng, 3);
//         rnn.neurons[0].output = 0.97;
//         rnn.neurons[1].output = 0.88;
//         rnn.neurons[2].output = 0.39;

//         assert_eq!(190, rnn.map_to_phenotype().0);
//     }
//     #[test]
//     fn test_map_to_phenotype_greyscale_1() {
//         let mut rng = ChaCha8Rng::seed_from_u64(2);
//         let mut rnn = Rnn::new(&mut rng, 3);
//         rnn.neurons[0].output = 0.5;
//         rnn.neurons[1].output = 0.5;
//         rnn.neurons[2].output = 0.5;

//         assert_eq!(128, rnn.map_to_phenotype().0);
//     }

//     #[test]
//     fn test_map_to_phenotype_greyscale_2() {
//         let mut rng = ChaCha8Rng::seed_from_u64(2);
//         let mut rnn = Rnn::new(&mut rng, 3);
//         rnn.neurons[0].output = 0.0;
//         rnn.neurons[1].output = 0.0;
//         rnn.neurons[2].output = 0.0;

//         assert_eq!(0, rnn.map_to_phenotype().0);
//     }

//     #[test]
//     fn test_map_to_phenotype_greyscale_3() {
//         let mut rng = ChaCha8Rng::seed_from_u64(2);
//         let mut rnn = Rnn::new(&mut rng, 3);
//         rnn.neurons[0].output = 1.0;
//         rnn.neurons[1].output = 1.0;
//         rnn.neurons[2].output = 1.0;

//         assert_eq!(255, rnn.map_to_phenotype().0);
//     }

//     #[test]
//     fn test_calculate_fitness_agent() {
//         use rand::prelude::*;
//         use rand_chacha::ChaCha8Rng;

//         let mut rng = ChaCha8Rng::seed_from_u64(2);
//         let mut agent = Agent::new(&mut rng, 3);
//         agent.genotype.neurons
//             .iter_mut()
//             .for_each(|neuron|{
//                 neuron.input_connections
//                 .iter_mut()
//                 .for_each(|(_, weight)| *weight = rng.gen_range(-1.0..=1.0));
//                 neuron.self_activation = rng.gen_range(-1.0..=1.0);
//                 neuron.bias = 1.;
//             });
//         agent.genotype.update();
//         agent.genotype.update();

//         let correct_greyscale = SimpleGrayscale(127);
//         let fitness = agent.calculate_fitness(correct_greyscale);

//         assert_eq!(round2(fitness), 0.82);
//     }

//     #[test]
//     fn test_update_rnn_two_neurons() {
//         use rand::prelude::*;
//         use rand_chacha::ChaCha8Rng;

//         let mut rng = ChaCha8Rng::seed_from_u64(2);
//         let mut rnn = Rnn::new(&mut rng, 3);

//         // randomize the weights and self activations with a custom seed and set the bias to 1.0
//         rnn.neurons
//             .iter_mut()
//             .for_each(|neuron|{
//                 neuron.input_connections
//                 .iter_mut()
//                 .for_each(|(_, weight)| *weight = rng.gen_range(-1.0..=1.0));
//                 neuron.self_activation = rng.gen_range(-1.0..=1.0);
//                 neuron.bias = 1.;
//             });

//         // first iteration
//         rnn.update();

//         assert_eq!(round2(rnn.neurons[0].output), 0.76);
//         assert_eq!(round2(rnn.neurons[1].output), 0.76);
        
//         // second iteration
//         rnn.update();

//         assert_eq!(round2(rnn.neurons[0].output), 0.4);
//         assert_eq!(round2(rnn.neurons[1].output), 0.86);
//     }

//     #[test]
//     fn test_update_rnn_three_neurons() {
//         use rand::prelude::*;
//         use rand_chacha::ChaCha8Rng;

//         let mut rng = ChaCha8Rng::seed_from_u64(2);
//         let mut rnn = Rnn::new(&mut rng, 3);

//         // randomize the weights and self activations with a custom seed and set the bias to 1.0
//         rnn.neurons
//             .iter_mut()
//             .for_each(|neuron|{
//                 neuron.input_connections
//                 .iter_mut()
//                 .for_each(|(_, weight)| *weight = rng.gen_range(-1.0..=1.0));
//                 neuron.self_activation = rng.gen_range(-1.0..=1.0);
//                 neuron.bias = 1.;
//             });
            
//         // first iteration
//         rnn.update();

//         assert_eq!(round2(rnn.neurons[0].output), 0.76);
//         assert_eq!(round2(rnn.neurons[1].output), 0.76);
//         assert_eq!(round2(rnn.neurons[2].output), 0.76);
        
//         // second iteration
//         rnn.update();

//         assert_eq!(round2(rnn.neurons[0].output), 0.4);
//         assert_eq!(round2(rnn.neurons[1].output), 0.86);
//         assert_eq!(round2(rnn.neurons[2].output), 0.79);
//     }

//     #[test]
//     fn test_evaluate_agent() {
//         use rand::prelude::*;
//         use rand_chacha::ChaCha8Rng;

//         let mut rng = ChaCha8Rng::seed_from_u64(2);

//         let mut agent = Agent::new(&mut rng, 3);
//         let mut agent2 = Agent::new(&mut rng, 3);

//         // randomize the weights and self activations with a custom seed and set the bias to 1.0
//         agent.genotype.neurons
//             .iter_mut()
//             .for_each(|neuron|{
//                 neuron.input_connections
//                 .iter_mut()
//                 .for_each(|(_, weight)| *weight = rng.gen_range(-1.0..=1.0));
//                 neuron.self_activation = rng.gen_range(-1.0..=1.0);
//                 neuron.bias = -0.6;
//             });
//         agent2.genotype.neurons
//             .iter_mut()
//             .for_each(|neuron|{
//                 neuron.input_connections
//                 .iter_mut()
//                 .for_each(|(_, weight)| *weight = rng.gen_range(-1.0..=1.0));
//                 neuron.self_activation = rng.gen_range(-1.0..=1.0);
//                 neuron.bias = 1.;
//             });
        
//         let fitness = agent.evaluate(SimpleGrayscale(127), 2);
//         let fitness2 = agent2.evaluate(SimpleGrayscale(127), 2);
        

//     }

//     #[test]
//     fn test_create_snapshots() {
//         // create a new rnn with 3 neurons
//         // update the rnn 5 times
//         // after each update create a snapshot
//         // check if the snapshots are correct
        
//         let mut rng = ChaCha8Rng::seed_from_u64(2);
//         let mut rnn = Rnn::new(&mut rng, 3);
//         rnn.neurons[0].output = 0.97;
//         rnn.neurons[1].output = 0.88;
//         rnn.neurons[2].output = 0.39;
        
//         for i in 0..1 {
//             let snapshot = SnapShot {
//                 outputs: rnn.neurons.iter().map(|neuron| neuron.output).collect(),
//                 time_step: i,
//             };
//             rnn.short_term_memory.add_snapshot(snapshot);
//             let saved_snapshot = rnn.short_term_memory.get_snapshot_at_timestep(i).unwrap();
//             assert_eq!(saved_snapshot.outputs[0], 0.97);
//             assert_eq!(saved_snapshot.outputs[1], 0.88);
//             assert_eq!(saved_snapshot.outputs[2], 0.39);
//         }
//     }

//     #[test]
//     fn test_snapshot_eq() {
//         let snapshot = SnapShot {
//             outputs: vec![0.97, 0.88, 0.39],
//             time_step: 1,
//         };
//         let snapshot2 = SnapShot {
//             outputs: vec![0.97, 0.88, 0.39],
//             time_step: 1,
//         };
//         assert_eq!(snapshot, snapshot2);
//     }

//     #[test]
//     fn test_snapshot_not_eq() {
//         let snapshot = SnapShot {
//             outputs: vec![1.97, 0.88, 0.39],
//             time_step: 1,
//         };
//         let snapshot2 = SnapShot {
//             outputs: vec![0.97, 0.88, 0.39],
//             time_step: 1,
//         };
//         assert_ne!(snapshot, snapshot2);
//     }

//     #[test]
//     fn test_short_term_memory_eq() {
//         let stm = ShortTermMemory {
//             snapshots: vec![
//                 SnapShot {
//                     outputs: vec![0.97, 0.88222222, 0.39],
//                     time_step: 1,
//                 },
//                 SnapShot {
//                     outputs: vec![1.97, -0.88, 0.0],
//                     time_step: 2,
//                 },
//                 SnapShot {
//                     outputs: vec![2.955555557, 0.88, -0.39],
//                     time_step: 3,
//                 },
//                 SnapShot {
//                     outputs: vec![0.922227, 0.0, 0.39],
//                     time_step: 4,
//                 },
//             ],
//         };
//         let stm2 = ShortTermMemory {
//             snapshots: vec![
//                 SnapShot {
//                     outputs: vec![0.97, 0.88222222, 0.39],
//                     time_step: 1,
//                 },
//                 SnapShot {
//                     outputs: vec![1.97, -0.88, 0.0],
//                     time_step: 2,
//                 },
//                 SnapShot {
//                     outputs: vec![2.955555557, 0.88, -0.39],
//                     time_step: 3,
//                 },
//                 SnapShot {
//                     outputs: vec![0.922227, 0.0, 0.39],
//                     time_step: 4,
//                 },
//             ],
//         };

//         assert_eq!(stm, stm2);
//     }

//     #[test]
//     fn test_from_rnn_to_graph() {
//         use rand::prelude::*;
//         use rand_chacha::ChaCha8Rng;

//         let mut rng = ChaCha8Rng::seed_from_u64(2);

//         let mut agent = Agent::new(&mut rng, 3);
//         agent.genotype.neurons
//             .iter_mut()
//             .for_each(|neuron|{
//                 neuron.input_connections
//                 .iter_mut()
//                 .for_each(|(_, weight)| *weight = rng.gen_range(-1.0..=1.0));
//                 neuron.self_activation = rng.gen_range(-1.0..=1.0);
//                 neuron.bias = -0.6;
//             });

//         let graph = Graph::<(usize, f64), f64>::from(agent.genotype.clone());

//         graph
//             .node_indices()
//             .for_each(|node| {
//                 graph
//                     .neighbors(node)
//                     .for_each(|neighbor| {
//                         // get weight from neuron at index "node" from the agent and the neuron at index "neighbor"
//                         if let Some(correct_weight) = agent.genotype.neurons[node.index()].input_connections
//                             .iter()
//                             .find(|(index, _)| *index == neighbor.index()) {
//                                 assert_eq!(round2(correct_weight.1), round2(*graph.edge_weight(graph.find_edge(neighbor, node).expect("msg")).unwrap()));
//                             }
//                     });
//             });
//     }

//     #[test]
//     fn test_from_graph_to_rnn() {
//         let mut graph = Graph::<(usize, f64), f64>::new();
//         let node1 = graph.add_node((0, 1.0));
//         let node2 = graph.add_node((1, -0.5));
//         let node3 = graph.add_node((2, 0.0));

//         // self connections
//         graph.add_edge(node1, node1, 0.2);
//         graph.add_edge(node2, node2, -0.2);
//         graph.add_edge(node3, node3, 0.0);

//         // connections between neurons
//         graph.add_edge(node1, node2, 0.5);
//         graph.add_edge(node1, node3, 0.3);
//         graph.add_edge(node2, node1, 0.1);
//         graph.add_edge(node2, node3, 0.55);
//         graph.add_edge(node3, node1, 0.98);
//         graph.add_edge(node3, node2, 0.11);

//         let rnn = Rnn::from(graph.clone());

//         graph
//             .node_indices()
//             .for_each(|node| {
//                 graph
//                     .neighbors(node)
//                     .for_each(|neighbor| {
//                         // get weight from neuron at index "node" from the agent and the neuron at index "neighbor"
//                         if let Some(correct_weight) = rnn.neurons[node.index()].input_connections
//                             .iter()
//                             .find(|(index, _)| *index == neighbor.index()) {
//                                 assert_eq!(round2(correct_weight.1), round2(*graph.edge_weight(graph.find_edge(neighbor, node).expect("msg")).unwrap()));
//                             }
//                     });
//             });

//     }

//     #[test]
//     fn test_rnn_to_graph_conversion_and_back() {
//         let mut rng = ChaCha8Rng::seed_from_u64(2);
//         let mut rnn = Rnn::new(&mut rng, 3);

//         let graph = Graph::<(usize, f64), f64>::from(rnn.clone());
//         let rnn2 = Rnn::from(graph.clone());

//         assert_eq!(rnn, rnn2);
//     }

//     #[test]
//     fn test_crossover_uniform() {
//         let mut rng = ChaCha8Rng::seed_from_u64(2);
//         let mut agent = Agent::new(&mut rng, 10);
//         let mut agent2 = Agent::new(&mut rng, 10);

//         agent.genotype.neurons
//             .iter_mut()
//             .for_each(|neuron|{
//                 neuron.input_connections
//                     .iter_mut()
//                     .for_each(|(_, weight)| *weight = 0.5);
//                 neuron.self_activation = 0.1;
//                 neuron.bias = 1.;
//             });
        
//         agent2.genotype.neurons
//             .iter_mut()
//             .for_each(|neuron|{
//                 neuron.input_connections
//                     .iter_mut()
//                     .for_each(|(_, weight)| *weight = -0.5);
//                 neuron.self_activation = -0.1;
//                 neuron.bias = -1.;
//             });

//         let offspring = agent.crossover(&mut rng, &agent2);
        
//         // check if the offspring is different from the parents
//         assert_ne!(agent.genotype, offspring.genotype);
//         assert_ne!(agent2.genotype, offspring.genotype);

//         // print the parents and then the offspring as graph
//         let parent1_graph = Graph::from(agent.genotype.clone());
//         let dot1 = Dot::new(&parent1_graph);
//         let parent2_graph = Graph::from(agent2.genotype.clone());
//         let dot2 = Dot::new(&parent2_graph);
//         let offspring_graph = Graph::from(offspring.genotype.clone());
//         let dot3 = Dot::new(&offspring_graph);
//         println!("Parent1 \n {:?}", dot1);
//         println!("Parent2 \n {:?}", dot2);
//         println!("Offspring \n {:?}", dot3);

//         // check if the number count of all negative numbers in the offsrping are approximately the saame as the psotive numbers
//         let negative_count = offspring.genotype.neurons
//             .iter()
//             .map(|neuron| neuron.input_connections.iter().filter(|(_, weight)| *weight < 0.0).count())
//             .sum::<usize>();
//         let positive_count = offspring.genotype.neurons
//             .iter()
//             .map(|neuron| neuron.input_connections.iter().filter(|(_, weight)| *weight > 0.0).count())
//             .sum::<usize>();
        
//         assert_eq!(positive_count, 43);
//         assert_eq!(negative_count, 47);
//     }

//     #[test]
//     fn test_delete_neuron() {
//         let mut rng = ChaCha8Rng::seed_from_u64(2);
//         let mut rnn = Rnn::new(&mut rng, 3);

//         rnn.delete_neuron(&mut rng);

//         assert_eq!(rnn.neurons[0].input_connections.iter().map(|(_, weight)| *weight).sum::<f64>(), 0.0);
//         assert_eq!(rnn.neurons[0].self_activation, 0.0);
//         assert_eq!(rnn.neurons[0].bias, 0.0);
//     }

//     #[test]
//     fn test_delete_weights() {
//         let mut rng = ChaCha8Rng::seed_from_u64(2);
//         let mut rnn = Rnn::new(&mut rng, 3);

//         rnn.delete_weights(&mut rng);

//         assert_eq!(rnn.neurons[0].input_connections.iter().map(|(_, weight)| *weight).sum::<f64>(), 0.0);
//     }

//     #[test]
//     fn test_delete_bias() {
//         let mut rng = ChaCha8Rng::seed_from_u64(2);
//         let mut rnn = Rnn::new(&mut rng, 3);

//         rnn.delete_bias(&mut rng);

//         assert_eq!(rnn.neurons[0].bias, 0.0);
//     }

//     #[test]
//     fn test_delete_self_activation() {
//         let mut rng = ChaCha8Rng::seed_from_u64(2);
//         let mut rnn = Rnn::new(&mut rng, 3);

//         rnn.delete_self_activation(&mut rng);

//         assert_eq!(rnn.neurons[0].self_activation, 0.0);
//     }

//     #[test]
//     fn test_mutate_neuron() {
//         let mut rng = ChaCha8Rng::seed_from_u64(2);
//         let mut rnn = Rnn::new(&mut rng, 3);
//         let bias = rnn.neurons[0].bias;
//         let self_activation = rnn.neurons[0].self_activation;
//         let weights = rnn.neurons[0].input_connections.iter().map(|(_, weight)| *weight).collect::<Vec<f64>>();

//         rnn.mutate_neuron(&mut rng);

//         // Check that the properties of the neuron have been changed.
//         assert_ne!(bias, rnn.neurons[0].bias);
//         assert_ne!(self_activation, rnn.neurons[0].self_activation);
//         assert_ne!(weights, rnn.neurons[0].input_connections.iter().map(|(_, weight)| *weight).collect::<Vec<f64>>());
//     }

//     #[test]
//     fn test_mutate_weights() {
//         let mut rng = ChaCha8Rng::seed_from_u64(2);
//         let mut rnn = Rnn::new(&mut rng, 3);
//         let weights = rnn.neurons[0].input_connections.iter().map(|(_, weight)| *weight).collect::<Vec<f64>>();

//         rnn.mutate_weights(&mut rng);

//         // Check that the properties of the neuron have been changed.
//         assert_ne!(weights, rnn.neurons[0].input_connections.iter().map(|(_, weight)| *weight).collect::<Vec<f64>>());
//     }

//     #[test]
//     fn test_mutate_bias() {
//         let mut rng = ChaCha8Rng::seed_from_u64(2);
//         let mut rnn = Rnn::new(&mut rng, 3);
//         let bias = rnn.neurons[0].bias;

//         rnn.mutate_bias(&mut rng);

//         // Check that the properties of the neuron have been changed.
//         assert_ne!(bias, rnn.neurons[0].bias);
//     }

//     #[test]
//     fn test_mutate_self_activation() {
//         let mut rng = ChaCha8Rng::seed_from_u64(2);
//         let mut rnn = Rnn::new(&mut rng, 3);
//         let self_activation = rnn.neurons[0].self_activation;

//         rnn.mutate_self_activation(&mut rng);

//         // Check that the properties of the neuron have been changed.
//         assert_ne!(self_activation, rnn.neurons[0].self_activation);
//     }

//     #[test]
//     fn test_build_from_json() {
//         let mut rng = ChaCha8Rng::seed_from_u64(2);
//         let mut rnn = Rnn::new(&mut rng, 3);
//         rnn.neurons[0].output = -0.44;
//         rnn.neurons[1].bias = -0.22;
//         let file_path = "../test/saves/rnn/test_rnn.json".to_string();

//         // save to disk
//         rnn.to_json(Some(&file_path)).unwrap();

//         // load from disk
//         let new_rnn = Rnn::from_json(file_path).unwrap();

//         assert_eq!(round2(new_rnn.neurons[0].output), -0.44);
//         assert_eq!(round2(new_rnn.neurons[1].bias), -0.22);
//         assert_eq!(new_rnn.graph.node_count(), 3);
//     }

//     #[test]
//     fn test_load_image() {
//         // using a image size of 33x25 px (nearly 4:3)
//         let image: ImageBuffer<LumaA<u8>, Vec<u8>> = image_processing::io::Reader::open("../images/artificial/checkboard.png").unwrap().decode().unwrap().into_luma_alpha8();
//         // white
//         assert_eq!(*image.get_pixel(0, 0), LumaA([255, 255]));
//         // black
//         assert_eq!(*image.get_pixel(3, 0), LumaA([0, 255]));
//         // black
//         assert_eq!(*image.get_pixel(32, 24), LumaA([0, 255]));
//     }

//     #[test]
//     #[should_panic]
//     fn test_invalid_load_image() {
//         let image: ImageBuffer<LumaA<u8>, Vec<u8>> = image_processing::io::Reader::open("../images/artificial/checkboard.png").unwrap().decode().unwrap().into_luma_alpha8();
//         // using a image size of 33x25 px (nearly 4:3) so this should panic
//         let _ = *image.get_pixel(33, 25);
//     }

//     #[test]
//     fn test_get_retina() {
//         let mut image = Image::from_path("../images/artificial/checkboard.png".to_string()).unwrap();

//         // using a image size of 33x25 px that the center pixel is at position 16, 12 (countning from 1 not 0)
//         let retina = image.create_retina_at(Position::new(5, 5), RETINA_SIZE).unwrap();

//         retina.create_png_at("../test/images/only_retina.png".to_string()).unwrap();
//         image.show_with_retina_movement_mut(&retina, "../test/images/with_retina_movement.png".to_string()).unwrap();

//         assert_eq!(retina.data[12], retina.get_value(2, 2));
//         assert_eq!(retina.data[0], retina.get_value(0, 0));
//         assert_eq!(retina.data[4], retina.get_value(4, 0));

//         // all corners are white
//         assert_eq!(retina.get_value(0, 0), 1.);
//         assert_eq!(retina.get_value(0, 4), 1.);
//         assert_eq!(retina.get_value(4, 0), 1.);
//         assert_eq!(retina.get_value(4, 4), 1.);
//     }

//     #[test]
//     fn test_get_retina_out_of_bounds() {
//         let image = Image::from_path("../images/artificial/checkboard.png".to_string()).unwrap();
//         // getting the first pixel in the top left corner should give an error
//         let retina = image.create_retina_at(Position::new(1, 1), RETINA_SIZE);
//         assert!(retina.is_err());
//     }

//     #[test]
//     fn test_retina_movement() {
//         let mut image = Image::from_path("../images/artificial/checkboard.png".to_string()).unwrap();
//         let mut retina = image.create_retina_at(Position::new(10, 10), RETINA_SIZE).unwrap();

//         image.show_with_retina_movement_mut(&retina, "../test/images/with_retina_movement.png".to_string()).unwrap();

//         retina.move_retina_mut(15, 0).unwrap();

//         image.show_with_retina_movement_mut(&retina, "../test/images/with_retina_movement.png".to_string()).unwrap();
//     }

//     #[test]
//     #[should_panic]
//     fn test_invalid_retina_movement_to_the_right() {
//         let image = Image::from_path("../images/artificial/checkboard.png".to_string()).unwrap();
//         let mut retina = image.create_retina_at(Position::new(17, 13), RETINA_SIZE).unwrap();

//         retina.move_retina_mut(20, 0).unwrap();
//     }

//     #[test]
//     #[should_panic]
//     fn test_invalid_retina_movement_to_the_left() {
//         let image = Image::from_path("../images/artificial/checkboard.png".to_string()).unwrap();
//         let mut retina = image.create_retina_at(Position::new(5, 13), RETINA_SIZE).unwrap();

//         retina.move_retina_mut(-5, 0).unwrap();
//     }

//     #[test]
//     #[should_panic]
//     fn test_invalid_retina_movement_to_the_top() {
//         let image = Image::from_path("../images/artificial/checkboard.png".to_string()).unwrap();
//         let mut retina = image.create_retina_at(Position::new(5, 13), RETINA_SIZE).unwrap();

//         retina.move_retina_mut(0, -20).unwrap();
//     }

//     #[test]
//     #[should_panic]
//     fn test_invalid_retina_movement_to_the_bottom() {
//         let image = Image::from_path("../images/artificial/checkboard.png".to_string()).unwrap();
//         let mut retina = image.create_retina_at(Position::new(5, 13), RETINA_SIZE).unwrap();

//         retina.move_retina_mut(0, 20).unwrap();
//     }

//     #[test]
//     fn test_update_from_retina_inputs() {
//         let mut rng = ChaCha8Rng::seed_from_u64(2);
//         let mut rnn = Rnn::new(&mut rng, 3);
//         let image = Image::from_path("../images/artificial/checkboard.png".to_string()).unwrap();
//         let retina = image.create_retina_at(Position::new(10, 10), RETINA_SIZE).unwrap();

//         rnn.update_inputs_from_retina(&retina);

//         assert_eq!(round2(rnn.neurons[0].retina_inputs[0] as f64), 1.0);
//         assert_eq!(round2(rnn.neurons[0].retina_inputs[4] as f64), 0.0);
//         assert_eq!(round2(rnn.neurons[0].retina_inputs[24] as f64), 1.0);
//     }

//     #[test]
//     fn test_binarize_image() {
//         let mut image = Image::from_path("../images/artificial/gradient.png".to_string()).unwrap();
//         image.save_binarized("../test/images/binarized.png".to_string()).unwrap();
//     }

// }