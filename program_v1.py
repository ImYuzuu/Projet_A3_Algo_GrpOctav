import numpy as np
import random
import networkx as nx
import math
from matplotlib import pyplot as plt
from tqdm import tqdm
import time



# définition des constantes

# nombre de villes
nb_villes = 6

# nombre de fourmis
nb_fourmis = 4

# nombre d'itérations
ITERATIONS = 250

# paramètre algorithme des fourmis
alpha = 1
beta = 4
rho = 0.64
Q = 4


# génération des villes
def distance(i, j):
    return math.sqrt((i[0] - j[0]) ** 2 + (i[1] - j[1]) ** 2)


def generate_villes(nb_villes):
    return [(random.random(), random.random()) for _ in range(nb_villes)]


villes = generate_villes(nb_villes)
distance_matrix = np.zeros((nb_villes, nb_villes))
for i in range(nb_villes):
    degre = random.randint(1, nb_villes if nb_villes > 10 else nb_villes - 1)
    voisins = random.sample(range(nb_villes), degre)
    for voisin in voisins:
        distance_matrix[i, voisin] = distance(villes[i], villes[voisin])
        np.sqrt((villes[i][0] - villes[voisin][0]) ** 2 + (villes[i][1] - villes[voisin][1]) ** 2)

    # The distance from a city to non neighbors is infinite
    for j in range(nb_villes):
        distance_matrix[i, j] = np.inf if (j not in voisins) or (i == j) else distance_matrix[i, j]
        distance_matrix[j, i] = distance_matrix[i, j]

# If a city has only one or no neighbors, we add a random neighbor
for i in range(nb_villes):
    nb_neighbors = np.sum(distance_matrix[i] != np.inf)
    if nb_neighbors < 2:

        for _ in range(2 - nb_neighbors):
            j = random.randint(0, nb_villes - 1)
            while j == i or j not in np.where(distance_matrix[i] == np.inf)[
                0]:  # we don't want to add a neighbor that already exists
                j = random.randint(0, nb_villes - 1)
            # if nb_neighbors == 0:
            # print(f"city {i} had 0 neighbors, adding a random one, ({j})")
            # else:
            # print(f"city {i} had {nb_neighbors} neighbors ({np.where(distance_matrix[i] != np.inf)[0][0]}), adding a random one, ({j})")
            distance_matrix[i, j] = distance(villes[i], villes[j])
            distance_matrix[j, i] = distance_matrix[i, j]

# print(f"new degrees: {[np.sum(distance_matrix[i] != np.inf) for i in range(nb_villes)]}")

# show the distance matrix
# display(
#     HTML(
#         f"""<table><tr>{'</tr><tr>'.join(f"<td>{'</td><td>'.join(str(_) for _ in row)}</td>" for row in distance_matrix)}</tr></table>"""
#     )
# )
# print(distance_matrix.tolist())

# fonction de calcul de la longueur d'un chemin
def longueur_chemin(chemin):
    return sum(distance_matrix[chemin[i]][chemin[i + 1]] for i in range(len(chemin) - 1))


def getBestPath(chemins):
    best_distance = min(longueur_chemin(tuple(chemin) if len(chemin) >= nb_villes else tuple(chemin) + (0,)) for chemin in chemins)
    best_path = [chemin for chemin in chemins if longueur_chemin(tuple(chemin)) == best_distance][0]
    return best_path, best_distance


# fonction de calcul des probabilités de transition
def proba_transition(i, j, pheromones):
    if i == j or distance_matrix[i][j] == np.inf:
        return 0
    return (distance_matrix[i][j] ** (-beta)) * (pheromones[i][j] ** alpha) / sum(
        (distance_matrix[i][k] ** (-beta)) * (pheromones[i][k] ** alpha) for k in range(nb_villes) if k != i)


# fonctions de mise à jour de la matrice de phéromones
def calcul_pheromones(actual_pheromones, path_length):
    return (1 - rho) * actual_pheromones + rho * Q / path_length


def maj_pheromones(pheromones: list[list[int]], chemins: list[list[int]]):
    tuple_pheromones = tuple(tuple(pheromones[__][_] for _ in range(nb_villes)) for __ in range(nb_villes))
    for chemin in chemins:
        for index in range(len(chemins) - 1):
            pheromones[chemin[index], chemin[index + 1]] = calcul_pheromones(
                tuple_pheromones[chemin[index]][chemin[index + 1]], longueur_chemin(tuple(chemin)))
    return pheromones


# fonction de mise à jour de la position des fourmis
def maj_fourmis(start_node, pheromones, start_time, time_limit):
    chemins = [[] for _ in range(nb_fourmis)]
    for fourmi in range(nb_fourmis):
        if time_limit is not None and time.time() - start_time > time_limit:
            print("Time limit reached")
            return chemins, pheromones, True
        # print(f"fourmi: {fourmi}")
        # print(f"chemins: {chemins}")
        # print(f"chemins[fourmi]: {chemins[fourmi]}")
        chemins[fourmi].append(start_node)
        not_visited = list(range(nb_villes))
        visited_cities = set()
        not_visited.remove(start_node)
        # print(f"chemins[fourmi]: {chemins[fourmi]}")
        while not_visited or chemins[fourmi][-1] != start_node:
            city = chemins[fourmi][-1]
            voisins_city = np.where(distance_matrix[city] != np.inf)[0]
            probas = [proba_transition(chemins[fourmi][-1], voisins_city[ville], pheromones) for ville in
                      range(len(voisins_city))]
            # print("-----")
            # print(f"probas: {probas}")
            # print(f"fourmi: {fourmi}")
            # print(f"not_visited: {not_visited}")
            # print(f"voisins_city: {voisins_city}")
            # print(f"city: {city}")
            # print(f"voisins: {voisins_city}")
            # print(f"pheromones[city]: {pheromones[city]}")
            # print("-----")
            if sum(probas) == 0:
                next_node = np.random.choice(voisins_city)
            else:
                probas /= sum(probas)
                # print(f"probas: {probas}")
                # print(f"voisins: {voisins_city}")
                next_node = np.random.choice(voisins_city, p=probas)
            chemins[fourmi].append(next_node)
            if next_node in visited_cities:  # decrease pheromone level if next city has already been visited
                pheromones[city][next_node] *= 0.5
                pheromones[next_node][city] *= 0.5
            else:
                visited_cities.add(next_node)
            if next_node in not_visited:
                not_visited.remove(next_node)
            # print(f"next_node: {next_node}")
    return chemins, pheromones, False


def fourmis(time_limit, start_node=0):
    final_path = []
    final_distance = 0
    pheromones = np.ones((nb_villes, nb_villes))
    print(f"Time limit: {time_limit} seconds") if time_limit else print("No time limit")
    start_time = time.time() if time_limit else None
    for i in tqdm(range(ITERATIONS), desc="iterations"):  # on fait ITERATIONS itérations de l'algorithme des fourmis
        # Initialize pheromones to 1 for all neighbors of all cities, 0 for the others
        if i != 0:
            for j in range(len(final_path) - 1):
                pheromones[final_path[j]][final_path[j + 1]] += Q ** 2 / final_distance
                pheromones[final_path[j + 1]][final_path[j]] += Q ** 2 / final_distance

        tuple_pheromones = tuple(tuple(pheromones[i][j] for j in range(nb_villes)) for i in range(nb_villes))
        chemins, pheromones, timeout = maj_fourmis(start_node, pheromones, start_time, time_limit)
        pheromones = pheromones if timeout else maj_pheromones(pheromones, chemins)
        best_path, best_distance = getBestPath(chemins)
        if best_distance < final_distance or final_distance == 0:
            final_distance = best_distance
            final_path = best_path
        if timeout:
            break
    return final_path, final_distance


# final_path, final_distance = fourmis(3600)
final_path, final_distance = fourmis(None)
print(f"Le meilleur chemin trouvé par les fourmis est : {final_path} il passe par {len(final_path)} villes")
print(f"La distance parcourue par les fourmis est : {final_distance}")

# tracer le graphe
G = nx.from_numpy_array(distance_matrix)
# only draw the edges that exist (no infinite edges)
edges = [(i, j) for i, j in G.edges() if G[i][j]["weight"] != np.inf]
final_edges = [(final_path[i], final_path[i + 1]) for i in range(len(final_path) - 1)]
pos = nx.circular_layout(G)


nx.draw_networkx(
    G,
    pos,
    # with_labels=True,
    edgelist=edges,
    edge_color="darkorange",
    width=5,
    alpha=0.5,
)
nx.draw_networkx(
    G,
    pos,
    # with_labels=True,
    edgelist=final_edges,
    edge_color="black",
    width=1,
)
# draw the starting node in red
nx.draw_networkx_nodes(
    G,
    pos,
    nodelist=[final_path[0]],
    node_color="red",
    node_size=500,
)
# draw weights as float number (2 decimals) on the edges
nx.draw_networkx_edge_labels(
    G,
    pos,
    edge_labels={(i, j): f"{G[i][j]['weight']:.2f}" for i, j in G.edges() if G[i][j]["weight"] != np.inf},
    font_color="black",
)

plt.show()