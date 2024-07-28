# Ergebnisse der Experimente

initial_retina_size = 33
min_retina_size = 5
max_retina_size = 239
retina_circle_radius = 1.0
sobel_threshold = 0.3
erode_pixels = 1

[neural_network]
neurons_per_network = 4
number_of_network_updates = 40
retina_movement_speed = 10.0
retina_resize_speed = 10.0

[neural_network.weight_bounds]
neuron_lower = -1.0
neuron_upper = 1.0
retina_lower = -1.0
retina_upper = 1.0

[genetic_algorithm]
population_size = 100
max_generations = 100

