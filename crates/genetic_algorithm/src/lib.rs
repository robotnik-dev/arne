
pub struct GeneticAlgorithm;

impl GeneticAlgorithm {
    pub fn evolve<T>(&self, population: &Vec<T>) -> Vec<T> {
        population
            .iter()
            .map(|_| {
                // selection
                // crossover
                // mutation
                todo!()
            })
            .collect()
    }
}
