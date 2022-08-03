import copy
import math
import random
import time

import numpy as np

from Problem import Problem
from Solution import Solution
from likelihood.MIP_solver import get_clusters


def solve_cluster_ils(problem_id_from, problem_id, model):
    print('---------------------', problem_id, '-----------------------')
    start_time = time.time()
    solution = Solution(problem_id)
    solution_clusters = solution.clusters
    solution_score = model.evaluate_clusters(solution_clusters, context=solution.stops, problem_id=problem_id)
    solution_violates = solution.violates_cvrp or solution.violates_capacity
    if solution.nb_vehicles < len(solution_clusters):
        print("INVALID DATA")
        return [problem_id, None, None, None, None, None, None, None]
    print('Solution:', solution_clusters)
    print('Solution score:', solution_score)
    print('Invalid solution (CAP) (CVRP):', solution.violates_capacity, solution.violates_cvrp)

    problem = Problem(problem_id)
    # Step 1: FIND INTITIAL SOLUTION
    predicted_clusters = get_clusters(problem_id_from, problem_id)
    predicted_score = model.evaluate_clusters(predicted_clusters, context=problem.stops, problem_id=problem_id)
    RD = solution.eval_sd(predicted_clusters)
    print('Prediction:', predicted_clusters)
    print('Predicted score:', predicted_score)
    print('RD:', RD)
    # Step 2: ITERATED LOCAL SEARCH
    # Execute 100 Repeats
    for _ in range(100):
        predicted_clusters, improved = perform_ils(predicted_clusters, model, problem.stops, problem_id)
        if improved:
            print('Predicted score:', predicted_score)
            print('RD:', RD)
    end_time = time.time()
    return [problem_id, int(solution.violates_cvrp), int(solution.violates_capacity), solution_score, predicted_score, RD[0], RD[1], end_time-start_time]


def random_greedy_initialization(problem, model):
    stops_to_add = copy.copy(problem.stops)
    random.shuffle(stops_to_add)
    clusters = [[] for _ in range(problem.vehicles)]
    # Add first stops randomly
    for i in range(problem.vehicles):
        clusters.append([stops_to_add[i]])
    stops_to_add = stops_to_add[problem.vehicles:]
    # Add next stops greedily
    while len(stops_to_add) > 0:
        best_score = math.inf
        best_stop_index = None
        best_cluster_index = None
        for i in range(problem.vehicles, len(stops_to_add)):
            for j in range(problem.vehicles):
                score = model.evaluate(clusters[j] + [stops_to_add[i]], context=problem.stops, problem_id=problem.problem_id)
                if score < best_score:
                    best_score = score
                    best_stop_index = i
                    best_cluster_index = j
        best_stop = stops_to_add.pop(best_stop_index)
        clusters[best_cluster_index].append(best_stop)
    return clusters


def perform_ils(clusters_input, model, stops, problem_id):
    clusters = copy.copy(clusters_input)
    # Choose two random clusters
    c1, c2 = np.random.choice(list(range(len(clusters))), 2, replace=False)
    original_score_c1 = model.evaluate(clusters[c1], context=stops, problem_id=problem_id)
    original_score_c2 = model.evaluate(clusters[c2], context=stops, problem_id=problem_id)
    original_score = original_score_c1 + original_score_c2
    # STEP 1: Mutation
    # One for two
    if len(clusters[c1]) > 1:
        i11, i12 = sorted(np.random.choice(list(range(len(clusters[c1]))), 2, replace=False))
        i2 = random.randint(0, len(clusters[c2]) - 1)
        new_c1 = clusters[c1][:i11] + clusters[c1][i11+1:i12] + clusters[c1][i12+1:] + [clusters[c2][i2]]
        new_c2 = clusters[c2][:i2] + clusters[c2][i2 + 1:] + [clusters[c1][i11], clusters[c1][i12]]
        clusters[c1], clusters[c2] = new_c1, new_c2
    # Two for one
    elif len(clusters[c2]) > 1:
        i1 = random.randint(0, len(clusters[c1]) - 1)
        i21, i22 = sorted(np.random.choice(list(range(len(clusters[c2]))), 2, replace=False))
        new_c1 = clusters[c1][:i1] + clusters[c1][i1+1:] + [clusters[c2][i21], clusters[c2][i22]]
        new_c2 = clusters[c2][:i21] + clusters[c2][i21+1:i22] + clusters[c2][i22+1:] + [clusters[c1][i1]]
        clusters[c1], clusters[c2] = new_c1, new_c2
    # Both clusters of one node
    else:
        return clusters_input, False


    # STEP 2: LOCAL SEARCH
    best_score = original_score
    best = [clusters[c1], clusters[c2]]
    improved = False
    # Move one node from c1 to c2
    if len(new_c1) > 1:
        for i1 in range(len(new_c1)):
            score_c1_without = model.evaluate(new_c1[:i1] + new_c1[i1+1:], context=stops, problem_id=problem_id)
            score_c2_with = model.evaluate(new_c2 + [new_c1[i1]], context=stops, problem_id=problem_id)
            if score_c1_without + score_c2_with < best_score:
                best_score = score_c1_without + score_c2_with
                best = [new_c1[:i1] + new_c1[i1+1:], new_c2 + [new_c1[i1]]]
                improved = True
    # Move one node from c1 to c2
    if len(new_c2) > 1:
        for i2 in range(len(new_c2)):
            score_c2_without = model.evaluate(new_c2[:i2] + new_c2[i2+1:], context=stops, problem_id=problem_id)
            score_c1_with = model.evaluate(new_c1 + [new_c2[i2]], context=stops, problem_id=problem_id)
            if score_c2_without + score_c1_with < best_score:
                best_score = score_c2_without + score_c1_with
                best = [new_c2[:i2] + new_c2[i2+1:], new_c1 + [new_c2[i2]]]
                improved = True
    # Swap nodes between c1 and c2
    """
    for i1 in range(len(new_c1)):
        for i2 in range(len(new_c2)):
    """
    if improved:
        print('Improved')
        clusters[c1], clusters[c2] = best[0], best[1]
        return clusters, True
    else:
        return clusters_input, False
