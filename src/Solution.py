import copy
import itertools
import math
import random
import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from PARAMETERS import CVRP_INSTANCE
from DriverTour import DriverTour


class Solution:

    # PROBLEM INSTANCE INITIALIZATION
    def __init__(self, problem_id):
        # Instance
        self.problem_id = problem_id
        self.stops = CVRP_INSTANCE.stops[problem_id]
        self.demands = CVRP_INSTANCE.demands[problem_id]
        self.nb_vehicles = CVRP_INSTANCE.n_vehicles[problem_id]
        self.weekday = CVRP_INSTANCE.weekday[problem_id]
        self.capacity = CVRP_INSTANCE.capacities[problem_id]
        # Solution
        self.incidence_matrix = CVRP_INSTANCE.incidence_matrices[problem_id]
        self.driver_tours = self.recreateDriverToursEdge()
        self.paths = self.get_paths()
        self.clusters = self.get_clusters()
        self.violates_capacity = sum(self.demands) > self.capacity * self.nb_vehicles
        self.violates_cvrp = any([sum(row) > 1 for row in self.incidence_matrix[1:, :]])

    def get_node_information(self, node_id, driver_tours):
        times_visited = 0
        data_records = []
        node_meta_information = {'problem_id': self.problem_id,
                                 'weekday': self.weekday,
                                 'node_id': node_id}
        for driver_tour in driver_tours:
            if driver_tour.has_node(node_id):
                times_visited += 1
                node_information = driver_tour.get_node_information(node_id)
                data_records.append(node_meta_information | node_information)
        return [data_record | {'times_visited': times_visited} for data_record in data_records]

    # HELPER FUNCTIONS
    def plot(self):
        fig, ax = plt.subplots()
        colors = [tuple(matplotlib.colors.hsv_to_rgb((i / len(self.driver_tours), 1, 0.75))) for i in
                  list(range(0, len(self.driver_tours)))]
        for i, driverTour in enumerate(self.driver_tours):
            driverTour.plotDriverRouteOnFigure(CVRP_INSTANCE.customer_positions, ax, color=colors[i])

        ax.scatter([0], [0], color='black', marker='s')
        x = [CVRP_INSTANCE.customer_positions[i][0] for i in self.stops]
        y = [CVRP_INSTANCE.customer_positions[j][1] for j in self.stops]
        plt.xlim([min(x) - 2, max(x) + 3])
        plt.ylim([min(y) - 2, max(y) + 3])
        plt.xlabel("x")
        plt.ylabel("y")
        plt.savefig("./figures/solution.pdf")
        #plt.show()

    def plot_tours_with_node(self, node_id):
        fig, ax = plt.subplots()
        colors = [tuple(matplotlib.colors.hsv_to_rgb((i / len(self.driver_tours), 1, 0.75))) for i in
                  list(range(0, len(self.driver_tours)))]
        for i, driverTour in enumerate(self.driver_tours):
            if node_id in driverTour.path:
                print(self.problem_id, driverTour.path)
                driverTour.plotDriverRouteOnFigure(CVRP_INSTANCE.customer_positions, ax, color=colors[i])
        plt.xlim([-100, 100])
        plt.ylim([-100, 100])
        plt.show()

    def recreateDriverToursNode(self, executeRemoval=True):
        driverTours = []
        partialTours = [[0]]
        while len(partialTours) > 0:
            nextPartialTours = []
            for partialTour in partialTours:
                nextNodes = np.where(self.incidence_matrix[partialTour[-1]] == 1)[0]
                for nextNode in nextNodes:
                    # Case 1: Complete a tour
                    if nextNode == 0:
                        driverTours.append(partialTour + [0])
                    # Case 2: Loop is being formed -> Reject
                    elif nextNode in partialTour:
                        continue
                    # Case 3: Normal case
                    else:
                        nextPartialTours.append(partialTour + [nextNode])
            partialTours = nextPartialTours
        # If the number of tours created exceeds the number of vehicles,
        # remove the longest tours of which all edges are visited by shorter tours.
        if executeRemoval:
            driverTours = sorted(driverTours, key=len, reverse=True)
            nbOfToursToRemove = len(driverTours) - self.nb_vehicles
            for i in range(nbOfToursToRemove):
                # Choose the first tour that has the potential to be removed sorted by length
                for indexTour1 in range(len(driverTours)):
                    stopsTour1 = driverTours[indexTour1][1:-1]
                    # An array of the length of the first tour. Each value indicates if the stop is visited in another tour
                    nodesInOtherTour = [False for t in range(len(driverTours[indexTour1][1:-1]))]
                    # Choose the second tour
                    for indexTour2 in range(len(driverTours)):
                        # Second must be different from first
                        if indexTour1 != indexTour2:
                            stopsTour2 = driverTours[indexTour2][1:-1]
                            for indexStop1, stop1 in enumerate(stopsTour1):
                                # Check if the stop occurs in another tour
                                if stop1 in stopsTour2:
                                    nodesInOtherTour[indexStop1] = True
                    if all(nodesInOtherTour):
                        del driverTours[indexTour1]
                        break
        return [DriverTour(self.problem_id, driverTour) for driverTour in driverTours]

    def recreateDriverToursEdge(self):
        paths = []
        partial_paths = [[0]]
        while len(partial_paths) > 0:
            next_partial_paths = []
            for partial_path in partial_paths:
                next_nodes = np.where(self.incidence_matrix[partial_path[-1]] == 1)[0]
                for next_node in next_nodes:
                    # Case 1: Complete a tour
                    if next_node == 0:
                        paths.append(partial_path + [0])
                    # Case 2: Loop is being formed -> Reject
                    elif next_node in partial_path:
                        continue
                    # Case 3: Normal case
                    else:
                        next_partial_paths.append(partial_path + [next_node])
            partial_paths = next_partial_paths
        # If the number of tours created exceeds the number of vehicles,
        # remove the longest tours of which all edges are visited by shorter tours.
        paths = sorted(paths, key=len, reverse=True)
        nb_tours_to_remove = len(paths) - self.nb_vehicles
        for t_nb in range(nb_tours_to_remove):
            for path1 in paths:
                edges_1 = set([(path1[i], path1[i+1]) for i in range(len(path1) - 1)])
                for path2 in paths:
                    if path1 != path2:
                        edges_2 = set([(path2[i], path2[i+1]) for i in range(len(path2) - 1)])
                        # Remove common edges
                        edges_1 = edges_1.difference(edges_2)
                if len(edges_1) == 0:
                    paths.remove(path1)
                    break
        return [DriverTour(self.problem_id, driverTour) for driverTour in paths]

    def plotComparison(self, other_solutions):
        fig, axs = plt.subplots(nrows=len(other_solutions) + 1)
        colors = [tuple(matplotlib.colors.hsv_to_rgb((i / len(self.driver_tours), 1, 0.75))) for i in
                  list(range(0, len(self.driver_tours)))]
        for i, driverTour in enumerate(DriverTour(self.problem_id, dt.path) for dt in self.driver_tours):
            driverTour.plotDriverRouteOnFigure(CVRP_INSTANCE.customer_positions, axs[0], color=colors[i])
        for sol_i, y_pred in enumerate(other_solutions):
            for i, driverTour in enumerate([DriverTour(self.problem_id, path) for path in y_pred]):
                driverTour.plotDriverRouteOnFigure(CVRP_INSTANCE.customer_positions, axs[sol_i + 1], color=colors[i])
        fig.tight_layout()
        plt.show()

    def get_arc_difference(self, path_pred):
        P_set = set()
        for sublist in path_pred:
            for i in range(len(sublist) - 1):
                P_set.add((sublist[i], sublist[i + 1]))

        A_set = set()
        for sublist in [driver_tour.path for driver_tour in self.driver_tours]:
            for i in range(len(sublist) - 1):
                A_set.add((sublist[i], sublist[i + 1]))

        result = set(A_set).difference((set(P_set)))
        return len(result), len(result) / len(A_set)

    def get_route_difference(self, route_pred):
        P_set = []
        for route in route_pred:
            P_set.append(set(route).difference({0}))

        A_set = []
        for route in self.paths:
            A_set.append(set(route).difference({0}))
        min_route_diff = math.inf

        for P_set_perm in list(itertools.permutations(P_set)):
            route_diff = 0
            for i in range(len(P_set_perm)):
                diff = A_set[i].difference(P_set_perm[i])
                route_diff += len(diff)
            if route_diff < min_route_diff:
                min_route_diff = route_diff
        return min_route_diff

    def get_paths(self):
        paths = []
        for driver_tour in self.driver_tours:
            paths.append(driver_tour.path)
        return paths

    def get_best_route_mapping(self, A, P):
        # create initial matrix of symmetric differences
        np_matrix = np.zeros((len(P), len(A)))
        for x in range(len(P)):
            for y in range(len(A)):
                np_matrix[x][y] = len(set(P[x]).symmetric_difference(set(A[y])))

        idx_list = []
        while len(idx_list) < len(A):
            # find smallest (x,y) in matrix
            (idx_r, idx_c) = np.where(np_matrix == np.nanmin(np_matrix))
            (r, c) = (idx_r[0], idx_c[0])  # avoid duplicates
            idx_list.append((r, c))

            # blank out row/column selected
            np_matrix[r, :] = np.NaN
            np_matrix[:, c] = np.NaN
            # print(np_matrix)
            # print(len(idx_list), len(Act))

        return idx_list

    def allstops(self, R):
        result = set()
        for route in R:
            result.update(route)
        return result

    def eval_sd(self, P):
        # get paired mapping
        A = self.clusters
        print(P)
        print(A)
        idx_list = self.get_best_route_mapping(A, P)

        diff = set()
        for (idx_P, idx_A) in idx_list:
            diff.update(set(P[idx_P]).symmetric_difference(set(A[idx_A])))

        nr_stops = len(self.allstops(A))
        # diffset, diffcount, diffrelative
        return len(diff), len(diff) / nr_stops

    # CONJOINT MODEL: Obtain attributes
    """
    Get the attributes of the given node:
    - Tour length
    - Stop number in tour
    - Angle to next stop
    - Distance to next stop
    - Intersects to next node with own tour
    """
    def get_attributes_of_node(self, node_id):
        records = []
        for driver_tour in self.driver_tours:
            if driver_tour.has_node(node_id):
                records.append(driver_tour.get_attributes(node_id))
        return records

    def get_conjoint_data(self, node_id):
        records = []
        for driver_tour in self.driver_tours:
            if driver_tour.has_node(node_id):
                driver_tour.get_conjoint_data(node_id, self.stops)
        return records

    def get_problem_data(self):
        record = {
            'problem_id': self.problem_id,
            'weekday': [1 if day == self.weekday else 0 for day in range(7)],
            'nb_vehicles': self.nb_vehicles,
            'capacity': self.capacity,
            'demands': self.demands,
            'stops': [1 if stop in self.stops else 0 for stop in range(CVRP_INSTANCE.nb_nodes)],
        }
        return record

    def get_clusters(self):
        result = []
        for path in self.paths:
            path = sorted(filter(lambda x: x != 0, path))
            result.append(path)
        return result

