import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import MDS
from sklearn.cluster import KMeans

"""
This helper class extracts all information from the datasets
- Historical problems
- Historical solutions
- Distance matrix
- Coordinate estimation
"""
class CVRPInstance:

    def __init__(self):
        self.distance_matrix = np.load("data/Distancematrix.npy")  # 73x73 matrix
        self.customer_positions = self.getCoordinatesFromDistanceMatrix()

        # All problem instances
        daily_stops_npzfile = np.load("data/daily_stops.npz", allow_pickle=True)
        self.stops = daily_stops_npzfile['stops_list']  # 201 length list indicating which stops are active for each day
        self.n_vehicles = daily_stops_npzfile['nr_vehicles']  # n_vehicles for each day
        self.weekday = daily_stops_npzfile['weekday']  # categorical input
        self.capacities = daily_stops_npzfile['capacities_list']  # vehicle capacity
        self.demands = daily_stops_npzfile['demands_list']  # demands of each active stops
        # All solutions
        daily_route_npzfile = np.load("data/daily_routematrix.npz", allow_pickle=True)
        self.incidence_matrices = daily_route_npzfile['incidence_matrices']
        self.nextStops = [[y for y in x] for x in daily_route_npzfile['next_stops']]

        # Put all the stops in clusters
        self.stop_clusters = self.cluster(plot=False)
        self.nb_nodes = self.getNumberOfNodes()

    def get_arc(self, problem_id, node_from, node_to):
        return self.incidence_matrices[problem_id, node_from, node_to]

    def getCoordinatesFromDistanceMatrix(self):
        """
        Using multidimensional scaling
        :return:
        """
        seed = np.random.RandomState(seed=3)
        mds = MDS(
            n_components=2,
            max_iter=3000,
            eps=1e-9,
            random_state=seed,
            dissimilarity="precomputed",
            n_jobs=1,
        )
        pos = mds.fit(self.distance_matrix).embedding_
        # Set the depot to coordinate (0,0)
        x0, y0 = pos[0]
        for i, (x1, y1) in enumerate(pos):
            pos[i] = [x1 - x0, y1 - y0]

        # OPTIONAL: Retrieve the error between the real distances and the predicted positions
        """        
        error = []
        for i in range(self.distanceMatrix.shape[0]):
            for j in range(self.distanceMatrix.shape[1]):
                mds_dist = math.sqrt((pos[i][0]-pos[j][0])**2+(pos[i][1]-pos[j][1])**2)
                real_dist = self.distanceMatrix[i, j]
                err = mds_dist-real_dist
                print("err", err)
                error.append(err)
        print("avg err:", np.average(err))
        """

        return pos

    def getProblem(self, problem_id):
        return [self.stops[problem_id],
                self.n_vehicles[problem_id],
                self.weekday[problem_id],
                self.capacities[problem_id],
                self.demands[problem_id]]


    def getNumberOfNodes(self):
        return self.distance_matrix.shape[0]

    def cluster(self, plot=False):
        kmeans = KMeans(n_clusters=10)
        # predict the labels of clusters.
        label = kmeans.fit_predict(self.customer_positions)
        if plot:
            u_labels = np.unique(label)
            for i in u_labels:
                pos = self.customer_positions[np.where(label == i)]
                plt.scatter(pos[:, 0], pos[:, 1], label=i)
            plt.legend()
            plt.show()
        return label

    def weekday_plot(self):
        fig = plt.figure(figsize=(8, 5.5))
        x = [[] for _ in range(7)]
        y = [[] for _ in range(7)]
        for problem_id in range(len(self.stops)):
            stops = len(self.stops[problem_id])
            day = self.weekday[problem_id]
            x[day].append(problem_id)
            y[day].append(stops)
        for i in range(7):
            plt.plot(x[i], y[i], marker='x', label='WD '+str(i))
        plt.xlabel('Probleemnummer')
        plt.ylabel('Aantal stops')
        plt.legend()
        plt.savefig('./figures/stops_weekday.pdf')

    def histogram_plot(self):
        fig = plt.figure(figsize=(8, 5.5))
        counts = [0 for _ in range(len(self.distance_matrix))]

        for stops in self.stops:
            for stop in stops:
                counts[stop] += 1

        counts = sorted(counts, reverse=True)
        plt.bar([i for i in range(len(counts))], counts)
        plt.xticks([])
        plt.xlabel('Stops')
        plt.ylabel('Frequentie')
        plt.legend()
        #plt.show()
        plt.savefig('./figures/stops_histogram.pdf')

"""
cvrp = CVRPInstance()
cvrp.weekday_plot()
"""
