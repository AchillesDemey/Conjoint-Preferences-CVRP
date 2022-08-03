import pickle
import time
from ClusteringSolverILS import solve_cluster_ils
from ClusteringModel import ClusteringModel

from Solution import Solution
from RoutingModel import RoutingModel
from RoutingSolverGenetic import RoutingSolver
from RoutingSolverExhaustive import solve_exhaustive


def read_cluster_test_data():
    with open('./data/cluster_eval_after.data', 'rb') as input_file:
        ltdp = pickle.load(input_file)
    print(ltdp)

def write_cluster_test_data(data):
    with open('./data/cluster_eval_after_6.data', 'wb') as output_file:
        pickle.dump(data, output_file)


m = ClusteringModel()
data = []
for problem_id in range(0, 201):
    if problem_id > 160:
        record = solve_cluster_ils(0, problem_id, m)
        data.append(record)
        write_cluster_test_data(data)
    m.train(Solution(problem_id))

read_cluster_test_data()

