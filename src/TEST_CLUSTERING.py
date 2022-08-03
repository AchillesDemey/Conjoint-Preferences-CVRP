import pickle
from solvers.ClusteringSolverILS import solve_cluster_ils
from models.ClusteringModel import ClusteringModel
from Solution import Solution
from PARAMETERS import TAU, RHO, OMEGA

def read_cluster_test_data():
    with open('./data/cluster_eval_after.data', 'rb') as input_file:
        ltdp = pickle.load(input_file)
    print(ltdp)

def write_cluster_test_data(data):
    with open('./data/cluster_eval_after_6.data', 'wb') as output_file:
        pickle.dump(data, output_file)


# Incremental learning testsc
preference_model = ClusteringModel()
data = []
for problem_id in range(TAU, OMEGA):
    if problem_id > RHO:
        record = solve_cluster_ils(0, problem_id, preference_model)
        data.append(record)
        write_cluster_test_data(data)
    preference_model.train(Solution(problem_id))


