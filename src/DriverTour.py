import math
import numpy as np
from matplotlib import patches
from PARAMETERS import CVRP_INSTANCE

"""
This class represents the tour of one driver in a solution
"""
class DriverTour:

    def __init__(self, problem_id, path):
        self.path = path
        self.demands = [CVRP_INSTANCE.demands[problem_id][stop] for stop in path]
        self.distances = [CVRP_INSTANCE.distance_matrix[path[i]][path[i + 1]] for i in range(len(path) - 1)]
        self.positions = [CVRP_INSTANCE.customer_positions[i] for i in path]

    def has_node(self, node_id):
        return node_id in self.path

    def get_attributes(self, node_id):
        stop_nb = self.path.index(node_id)
        next_node_id = self.path[stop_nb + 1]
        next_distance = self.distances[stop_nb]
        next_angle = math.degrees(math.atan2(self.positions[stop_nb + 1][1] - self.positions[stop_nb][1],
                                             self.positions[stop_nb + 1][0] - self.positions[stop_nb][0]))
        tour_distance = sum(self.distances)
        intersects = self.intersects_to_next_node(node_id)
        return {'stop_nb': stop_nb,
                'next_node_id': next_node_id,
                'next_distance': next_distance,
                'next_angle': next_angle,
                'tour_distance': tour_distance,
                'intersects': intersects,
                'tour_stops': len(self.path) - 2}

    def get_conjoint_data(self, node_id, stops):
        print(node_id, '-------------')
        print(self.path)
        print(stops)
        records = []
        node_index = self.path.index(node_id)
        next_node_id = self.path[node_index + 1]
        for stop in stops:
            if stop == node_id:
                continue
            else:
                if stop == 0:
                    # End in depot immediately
                    perturbed_path = self.path[:node_index + 1] + [0]
                elif stop in self.path:
                    # Remove the stop from the path
                    stop_index = self.path.index(stop)
                    perturbed_path = self.path[:stop_index] + self.path[stop_index + 1:]
                    # Replace the next node by the stop
                    perturbed_path = perturbed_path[:perturbed_path.index(node_id) + 1] + [stop] + perturbed_path[perturbed_path.index(node_id) + 1:]
                    # End in the depot if the last node is not the depot
                    if perturbed_path[-1] != 0:
                        perturbed_path += [0]
                else:
                    # Replace the next node by the stop
                    perturbed_path = self.path[:node_index + 1] + [stop] + self.path[node_index + 1:]
                    # End in the depot if the last node is not the depot
                    if perturbed_path[-1] != 0:
                        perturbed_path += [0]
                print(stop, perturbed_path)


    # FUNCTIONS FOR ATTRIBUTES
    def getTotalCapacity(self):
        return sum(self.demands)

    def getTotalDistance(self):
        total = np.sum(self.distances)
        """
        avg = np.average(self.distances)
        stDev = np.std(self.distances)
        min = np.min(self.distances)
        max = np.max(self.distances)
        """
        return total

    def intersects_to_next_node(self, node_id):
        node_index = self.path.index(node_id)
        x1, y1 = CVRP_INSTANCE.customer_positions[self.path[node_index]]
        x2, y2 = CVRP_INSTANCE.customer_positions[self.path[node_index + 1]]
        for from_index in range(len(self.path) - 1):
            if from_index != node_index:
                x3, y3 = CVRP_INSTANCE.customer_positions[self.path[from_index]]
                x4, y4 = CVRP_INSTANCE.customer_positions[self.path[from_index + 1]]
                denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
                if denom == 0:  # parallel
                    continue
                ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom
                if ua <= 0 or ua >= 1:  # out of range
                    continue
                ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denom
                if ub <= 0 or ub >= 1:  # out of range
                    continue
                return True
        return False

    # VISUALIZATION FUNCTIONS
    def plotDriverRouteOnFigure(self, coordinates, ax, color=(0, 0, 0)):
        # PLOT VARS
        # np.random.choice(range(255), size=3) / 255
        randomColor = color
        style = 'Simple, tail_width=0.5, head_width=5, head_length=6'
        x, y = zip(*[coordinates[stop] for stop in self.path])
        ax.scatter(x, y, color='black', s=0.3)
        for stop in self.path:
            ax.annotate(stop, coordinates[stop], fontsize=6.5)
        for i in range(len(x) - 1):
            xi1 = x[i]
            yi1 = y[i]
            xi2 = x[i + 1]
            yi2 = y[i + 1]
            ax.add_patch(patches.FancyArrowPatch((xi1, yi1),
                                                 (xi2, yi2),
                                                 connectionstyle="arc3,rad=0.0",
                                                 arrowstyle=style,
                                                 color=randomColor,
                                                 alpha=1))
