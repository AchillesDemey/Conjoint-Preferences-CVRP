import copy
import math
import random
from src.TwoStageCVRP.PARAMETERS import CVRP_INSTANCE

import numpy as np


class ClusteringSolver:

    def __init__(self, problem, model):
        # Algorithm parameters
        self.nb_chromosomes = 10
        self.nb_offspring = 10
        self.model = model
        # Problem specific variables
        self.stops = problem.stops[1:]
        self.demands = problem.demands
        self.vehicles = problem.vehicles
        self.capacity = problem.capacity
        if sum(self.demands) > (self.vehicles * self.capacity):
            self.capacity = math.ceil(sum(self.demands) / self.vehicles) + max(self.demands)
            print("Capacity increased from {} to {}".format(problem.capacity, self.capacity))
        else:
            self.capacity = problem.capacity
        # Algorithm variables
        self.parents = self.init_population()
        self.offspring = None
        self.scores = np.asarray([self.fitness(chromosome) for chromosome in self.parents])
        self.average = np.average(self.scores)
        self.best = np.min(self.scores)
        print(self.average, self.best, self.parents[0])

    def solve(self):
        count = 0
        while count < 5:
            # Create the next generation
            self.next_generation()
            # Update fitness values
            self.scores = np.asarray([self.fitness(chromosome) for chromosome in self.parents])
            if np.average(self.scores) < self.average:
                count = 0
            else:
                count += 1
            self.average = np.average(self.scores)
            self.best = self.scores[0]
            print(self.average, self.best, self.parents[0])
        return self.parents[0]

    def next_generation(self):
        self.offspring = []
        for offspring_nb in range(self.nb_offspring):
            p1, p2 = self.select()
            offspring = self.crossover2(p1, p2)
            offspring = self.mutate(offspring)
            #offspring = self.local_search2(offspring)
            self.offspring.append(offspring)
        self.eliminate()

    # Random initialization
    def init_population(self):
        population = []
        for _ in range(self.nb_chromosomes):
            division = [i * math.floor(len(self.stops) / self.vehicles) for i in range(0, self.vehicles)] + [
                len(self.stops)]
            permutation = np.random.permutation(self.stops)
            individual = [sorted(list(permutation[division[i]:division[i + 1]])) for i in range(len(division) - 1)]
            individual = sorted(individual, key=lambda x: x[0])
            population.append(individual)
        return population

    # Fitness based on model
    def fitness(self, chromosome):
        score = 0
        for cluster in chromosome:
            score += self.model.predict(cluster)
        return score

    # K-Tournament Selection
    def select(self, tournament=4):
        tournament = min(len(self.parents), tournament)
        first_score = math.inf
        second_score = math.inf
        first_parent = None
        second_parent = None
        chosen = np.random.choice(list(range(len(self.parents))), tournament, replace=False)
        for index in chosen:
            score = self.fitness(self.parents[index])
            if score < first_score:
                second_parent = first_parent
                second_score = first_score
                first_parent = self.parents[index]
                first_score = score
            elif score < second_score:
                second_parent = self.parents[index]
                second_score = score
        return first_parent, second_parent

    # K-Tournament Selection
    def select2(self):
        i1, i2 = np.random.choice(list(range(len(self.parents))), 2, replace=False)
        return self.parents[i1], self.parents[i2]

    # Crossover
    def crossover1(self, p1, p2):
        clusters = []
        non_added_nodes = set()
        added_nodes = set()
        for cluster_index in range(len(p1)):
            cluster_1 = set(p1[cluster_index])
            cluster_2 = set(p2[cluster_index])
            # Determine the nodes that must be in the child cluster
            # 1) The intersecting cluster of both parents
            cluster_intersection = cluster_1.intersection(cluster_2)
            added_nodes = added_nodes.union(cluster_intersection)
            # 2) The remaining nodes that are in either one of the parents and not added yet
            cluster_union = cluster_1.union(cluster_2).difference(added_nodes)
            cluster_must_add = non_added_nodes.intersection(cluster_union)
            added_nodes = added_nodes.union(cluster_must_add)
            # Combine (1) and (2)
            deterministic_nodes = cluster_intersection.union(cluster_must_add)
            # Add the remaining nodes randomly
            random_nodes = set()
            for node in cluster_union.difference(deterministic_nodes):
                if random.random() < 0.5:
                    random_nodes.add(node)
                    added_nodes.add(node)
                else:
                    non_added_nodes.add(node)
            cluster_child = deterministic_nodes.union(random_nodes)
            clusters.append(cluster_child)
        return [list(cluster) for cluster in clusters]

    def crossover2(self, p1, p2):
        # 1) Get mapping of the closest cluster centers
        centers_1 = [np.average([CVRP_INSTANCE.customer_positions[node] for node in cluster], axis=0) for cluster in p1]
        centers_2 = [np.average([CVRP_INSTANCE.customer_positions[node] for node in cluster], axis=0) for cluster in p2]
        dists = np.zeros((len(centers_1), len(centers_2)))
        m1 = []
        m2 = []
        for i1, c_1 in enumerate(centers_1):
            for i2, c_2 in enumerate(centers_2):
                dists[i1][i2] = np.linalg.norm(c_1 - c_2)
        while len(m1) < len(centers_1):
            min_1 = None
            min_2 = None
            min_v = math.inf
            for i1 in range(len(dists)):
                if i1 not in m1:
                    for i2 in range(len(dists[i1])):
                        if i2 not in m2:
                            if dists[i1][i2] < min_v:
                                min_v = dists[i1][i2]
                                min_1 = i1
                                min_2 = i2
            m1.append(min_1)
            m2.append(min_2)
        # Recombine the closest clusters
        result = []
        skipped = set()
        for c_1, c_2 in zip(m1, m2):
            intersection = set(p1[c_1]).intersection(set(p2[c_2]))
            chosen = set()
            for element in intersection:
                if random.random() < 1:
                    chosen.add(element)
                else:
                    skipped.add(element)
            skipped = skipped.union(set(p1[c_1]).symmetric_difference(set(p2[c_2])))
            result.append(list(chosen))
        # Divide remaining nodes
        for node in skipped:
            for index in np.random.permutation(range(len(result))):
                if np.sum(self.demands[result[index]]) + self.demands[node] <= self.capacity:
                    result[index].append(node)
                    break
        return result

    def mutate(self, offspring, p=0.5):
        for _ in range(1):
            if random.random() < p:
                c_1, c_2 = np.random.choice(list(range(len(offspring))), 2, replace=False)
                n_1, n_2 = random.randint(0, len(offspring[c_1]) - 1), random.randint(0, len(offspring[c_2]) - 1)
                offspring[c_1][n_1], offspring[c_2][n_2] = offspring[c_2][n_2], offspring[c_1][n_1]
            return offspring

    def local_search(self, chromosome):
        improved = True
        while improved:
            improved = False
            # print("Before LS:", chromosome, np.sum([self.model.predict(cluster) for cluster in chromosome]))
            removal_difference = math.inf
            removal_index, node_index = None, None
            # Find best element to remove
            for c_i, cluster in enumerate(chromosome):
                original_score = self.model.predict(cluster)
                for n_i, node in enumerate(cluster):
                    perturbed_cluster = cluster[:n_i] + cluster[n_i + 1:]
                    perturbed_score = self.model.predict(perturbed_cluster)
                    if perturbed_score - original_score < removal_difference and random.random() < 0.5:
                        removal_difference = perturbed_score - original_score
                        removal_index, node_index = c_i, n_i

            # Find best cluster to insert
            insertion_difference = math.inf
            insertion_index = None
            for c_i, cluster in enumerate(chromosome):
                original_score = self.model.predict(cluster)
                if c_i != removal_index:
                    perturbed_cluster = cluster + [chromosome[removal_index][node_index]]
                    perturbed_score = self.model.predict(perturbed_cluster)
                    if perturbed_score - original_score < insertion_difference:
                        insertion_difference = perturbed_score - original_score
                        insertion_index = c_i
            if removal_difference + insertion_difference < 0:
                chromosome[insertion_index].append(chromosome[removal_index][node_index])
                chromosome[removal_index] = chromosome[removal_index][:node_index] + chromosome[removal_index][
                                                                                     node_index + 1:]
                improved = True
            # print("After LS: ", chromosome, np.sum([self.model.predict(cluster) for cluster in chromosome]))
        return chromosome

    def local_search2(self, chromosome):
        improved = True
        best = chromosome
        best_score = self.fitness(chromosome)
        while improved:
            improved = False
            for c_i, cluster_i in enumerate(chromosome):
                for n_i, node in enumerate(cluster_i):
                    for c_j, cluster_j in enumerate(chromosome):
                        if c_i != c_j:
                            new_chromosome = copy.copy(chromosome)
                            new_chromosome[c_j] = cluster_j + [node]
                            new_chromosome[c_i] = cluster_i[:n_i] + cluster_i[n_i+1:]
                            score = self.fitness(new_chromosome)
                            if score < best_score:
                                improved = True
                                best = new_chromosome
                                best_score = score
        return best

    def local_search_3(self, chromosome):
        improved = True
        best = chromosome
        best_score = self.fitness(chromosome)
        while improved:
            improved = False
            for c_i, cluster_i in enumerate(chromosome):
                for c_j, cluster_j in enumerate(chromosome):
                    if c_i != c_j:
                        for n_i, node_i in enumerate(cluster_i):
                            for n_j, node_j in enumerate(cluster_j):
                                new_chromosome = copy.copy(chromosome)
                                new_chromosome[c_i][n_i], new_chromosome[c_j][n_j] = new_chromosome[c_j][n_j], new_chromosome[c_i][n_i]
                                score = self.fitness(new_chromosome)
                                if score < best_score:
                                    improved = True
                                    best = new_chromosome
                                    best_score = score
        return best

    # Elimination
    def eliminate(self):
        # Merge offpring and parents
        self.parents = sorted(self.parents + self.offspring, key=lambda x: self.fitness(x))
        # Get the population
        self.parents = self.parents[:self.nb_chromosomes]
