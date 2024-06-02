import numpy as np
import networkx as nx
import math
from matplotlib import pyplot as plt
from tqdm import tqdm

# Read data from file
data = []
with open('ProgramV2\c101.txt') as datafile:
    for line in datafile:
        line = line.strip().split('\n')
        data.append(line)

# Convert data to numpy array
numpy_data = np.array(data)

# Remove the first line (header)
numpy_data = numpy_data[1:]

# Split each line into individual elements
split_data = [line[0].split() for line in numpy_data]

# Number of cities
nb_villes = len(split_data)

# Number of ants
nb_fourmis = 10

# Number of iterations
ITERATIONS = 1000

# Parameters for ant colony optimization algorithm
alpha = 1
beta = 4
rho = 0.64
Q = 4

# Maximum capacity for the truck
capacity = 200

# Generate distance matrix
distance_matrix = np.zeros((nb_villes, nb_villes))
for i in range(nb_villes):
    for j in range(nb_villes):
        if i != j:
            distance_matrix[i, j] = math.sqrt((float(split_data[i][1]) - float(split_data[j][1])) ** 2 +
                                              (float(split_data[i][2]) - float(split_data[j][2])) ** 2)

# Extract demand list from split_data and identify the depot
demand_list = []
depot_index = -1
for i, info in enumerate(split_data):
    demand = int(float(info[3]))  # Convert demand to integer
    demand_list.append(demand)
    if demand == 0:
        depot_index = i

if depot_index == -1:
    raise ValueError("No depot found with demand of 0")

# Define a list of colors for the segments
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple', 'pink']


# Ant Colony Optimization Algorithm
def ant_colony_optimization(distance_matrix, demand_list, capacity, nb_fourmis, iterations, alpha, beta, rho, Q, depot_index):
    # Initialize pheromone matrix
    pheromone_matrix = np.ones(distance_matrix.shape) / \
        (distance_matrix.shape[0] * 0.1)

    shortest_path_length = float('inf')
    shortest_path = []
    shortest_path_segments = []

    nb_villes = len(distance_matrix)
    colors = ['red', 'blue', 'green', 'yellow', 'orange',
              'purple']  # Define colors for visualization

    for iteration in tqdm(range(iterations)):
        paths = []
        path_lengths = []
        path_segments = []

        for ant in range(nb_fourmis):
            path = []
            visited = [False] * nb_villes
            unvisited_cities = set(range(nb_villes))  # Track unvisited cities
            current_capacity = 0
            current_city = depot_index  # Start at the depot
            path.append(current_city)
            visited[current_city] = True
            unvisited_cities.remove(current_city)
            segment = [current_city]  # Track the current segment
            segment_color = 0  # Start with the first color

            while unvisited_cities:
                probabilities = np.zeros(nb_villes)
                for city in unvisited_cities:
                    if current_capacity + demand_list[city] <= capacity:
                        probabilities[city] = (pheromone_matrix[current_city, city] ** alpha) * \
                                              ((1 /
                                               distance_matrix[current_city, city]) ** beta)

                if np.sum(probabilities) == 0:
                    break  # All reachable cities have been visited or cannot be visited due to capacity

                probabilities /= np.sum(probabilities)

                next_city = np.random.choice(
                    np.arange(nb_villes), p=probabilities)

                path.append(next_city)
                segment.append(next_city)
                visited[next_city] = True
                unvisited_cities.remove(next_city)
                current_capacity += demand_list[next_city]
                current_city = next_city

                if current_capacity >= capacity:
                    # Return to the depot due to capacity constraints
                    path.append(depot_index)
                    segment.append(depot_index)
                    path_segments.append(
                        (segment, colors[segment_color % len(colors)]))
                    segment = [depot_index]
                    current_capacity = 0
                    current_city = depot_index
                    segment_color += 1

            if current_city != depot_index:
                path.append(depot_index)  # Return to the depot at the end
                segment.append(depot_index)
                path_segments.append(
                    (segment, colors[segment_color % len(colors)]))

            paths.append(path)
            path_length = sum(
                distance_matrix[path[i], path[i + 1]] for i in range(len(path) - 1))
            path_lengths.append(path_length)

            if path_length < shortest_path_length:
                shortest_path_length = path_length
                shortest_path = path
                shortest_path_segments = path_segments

            # Update pheromone only on visited edges (local pheromone update)
            for i in range(len(path) - 1):
                pheromone_matrix[path[i], path[i + 1]] += Q / path_length

        # Global pheromone update (pheromone evaporation)
        pheromone_matrix *= (1 - rho)

        if iteration % 10 == 0:
            print(f"Iteration {iteration}: Shortest Path Length = {
                  shortest_path_length}")

    return shortest_path, shortest_path_length, shortest_path_segments


# Run Ant Colony Optimization Algorithm
shortest_path, shortest_path_length, shortest_path_segments = ant_colony_optimization(
    distance_matrix, demand_list, capacity, nb_fourmis, ITERATIONS, alpha, beta, rho, Q, depot_index)

print("Shortest Path:", shortest_path)
print("Shortest Path Length:", shortest_path_length)

# Create a graph
G = nx.Graph()

# Add nodes for each city
for i in range(nb_villes):
    G.add_node(i, pos=(float(split_data[i][1]), float(split_data[i][2])))

# Draw the graph
pos = nx.get_node_attributes(G, 'pos')

# Add edges for the shortest path with colors
edge_color_map = []
edge_width_map = []

for segment, color in shortest_path_segments:
    for i in range(len(segment) - 1):
        G.add_edge(segment[i], segment[i + 1], color=color, weight=2.0)
        edge_color_map.append(color)
        edge_width_map.append(2.0)

# Draw nodes and edges
edges = G.edges(data=True)
edge_colors = [e[2]['color'] for e in edges]
edge_weights = [e[2]['weight'] for e in edges]

nx.draw(G, pos, with_labels=True, node_size=300, node_color='skyblue',
        font_size=8, font_color='black', font_weight='bold')
nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=edge_colors,
                       width=edge_weights, alpha=0.7, connectionstyle='arc3,rad=0.2')

# Show the plot
plt.title('Shortest Path Found by Ant Colony Optimization')
plt.show()
