from gurobipy import Model, quicksum, GRB
from likelihood.Arc_Probability_Model import get_transition_probability_matrix
from Problem import Problem


def solve_arcs(stops, vehicle_capacity, num_of_routes, cost_matrix):
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
    return completed_tours

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


def solve_arcs_clustered(clusters, vehicle_capacity, cost_matrix):
    completed_tours = []
    for cluster in clusters:
        completed_tours.append(solve_arcs(cluster, vehicle_capacity, 1, cost_matrix)[0])
    return completed_tours