import numpy as np
from gurobipy import Model, quicksum, GRB
from src.TwoStageCVRP.PARAMETERS import LAMBDA, WEIGHING_SCHEME, ALPHA, CVRP_INSTANCE
from src.TwoStageCVRP.Problem import Problem


def get_clusters(problem_id_from, problem_id_to):
    problem = Problem(problem_id_to)
    stops = problem.stops
    vehicle_capacity = problem.capacity
    num_of_routes = problem.vehicles
    cost_matrix = get_transition_probability_matrix(problem_id_from, problem_id_to)

    n = len(stops) - 1  # number of clients
    Q = vehicle_capacity
    N = [stops[i] for i in range(1, n + 1)]
    V = [0] + N
    q = {i: 1 for i in N}  # dictionary of demands
    rt_count = num_of_routes

    # create set of arcs
    A = [(i, j) for i in V for j in V if i != j]

    # solve using GUROBI
    mdl = Model('CVRP')
    mdl.setParam('OutputFlag', 0)
    mdl.setParam('MIPGap', 1e-2)
    x = mdl.addVars(A, vtype=GRB.BINARY)
    u = mdl.addVars(N, vtype=GRB.CONTINUOUS)

    # objective function
    mdl.setObjective(quicksum(cost_matrix[i][j] * x[i, j] for i, j in A), GRB.MINIMIZE)

    # constraints
    mdl.addConstrs(sum(x[i, j] for j in V if j != i) == 1 for i in N)
    mdl.addConstrs(sum(x[i, j] for i in V if i != j) == 1 for j in N)
    mdl.addConstrs((x[i, j] == True) >> (u[i] + q[j] == u[j]) for i, j in A if i != 0 and j != 0)
    mdl.addConstrs(u[i] >= q[i] for i in N)
    mdl.addConstrs(u[i] <= Q for i in N)

    # fix number of routes
    mdl.addConstr(quicksum(x[0, j] for j in N) == rt_count)
    mdl.addConstr(quicksum(x[0, j] for j in N) == rt_count)

    # to show computation: log_output=True
    mdl.optimize()

    active_arcs = [a for a in A if x[a].x > 0.99]

    beginning_paths = active_arcs[:num_of_routes]
    completed_tours = []
    for path in beginning_paths:
        curr_path = list(path)
        while curr_path[-1] != 0:
            for arc in active_arcs:
                if arc[0] == curr_path[-1]:
                    curr_path.append(arc[1])
                    break
        completed_tours.append(curr_path)

    clusters = [sorted(filter(lambda node: node != 0, path)) for path in completed_tours]
    return clusters

def get_transition_probability_matrix(problem_id_from, problem_id_to, logarithm=True):
    matrix = np.zeros(CVRP_INSTANCE.distance_matrix.shape)
    # Build arc transition frequency matrix with weights
    for i, problem_id in enumerate(list(range(problem_id_from, problem_id_to))):
        adj_matrix = CVRP_INSTANCE.incidence_matrices[problem_id]
        weight = get_weight(i + 1, problem_id_to - problem_id_from)
        matrix += adj_matrix * weight
    # Laplace smoothing to obtain probabilities
    for row_index in range(matrix.shape[0]):
        matrix[row_index, :] = (matrix[row_index, :] + LAMBDA) / (
                np.sum(matrix[row_index, :]) + matrix.shape[0] * LAMBDA)
    # Take the logarithm and the negative value
    if logarithm:
        matrix = np.log(matrix) * (-1)
    return matrix


def get_weight(t, T):
    if WEIGHING_SCHEME == 'uniform':
        return 1
    elif WEIGHING_SCHEME == 'time_linear':
        return t / T
    elif WEIGHING_SCHEME == 'time_squared':
        return pow(t / T, 2)
    elif WEIGHING_SCHEME == 'time_exp':
        return ALPHA * pow(1 - ALPHA, T - t)
    else:
        print('Invalid weighing scheme: ', WEIGHING_SCHEME, '. Uniform (default) weight applied')
        return 1

