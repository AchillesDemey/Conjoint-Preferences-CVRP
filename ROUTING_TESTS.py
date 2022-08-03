import pickle
import time
import sys
from Solution import Solution
from RoutingModel import RoutingModel
from RoutingSolverGenetic import RoutingSolver
from RoutingSolverExhaustive import solve_exhaustive


def routing_test(train_from=130, train_to=165, test_to=200, peak_form='exp'):
    linear_test_data_solution = []
    linear_test_data_path = []
    preference_model = RoutingModel()
    # Add the 20 first instances
    for problem_id in range(train_from, train_to):
        print('TRAIN:', problem_id)
        preference_model.train(problem_id)
    for problem_id in range(train_to, test_to):
        print('---------TEST:', problem_id, '---------------')
        start_time = time.time()
        solution = Solution(problem_id)
        # With linear
        solution_score = sum(preference_model.evaluate(path, peak_form=peak_form) for path in solution.paths)
        predicted_solution = []
        predicted_score = 0
        for i, cluster in enumerate(solution.clusters):
            solution_path_score = preference_model.evaluate(solution.paths[i], peak_form=peak_form)
            print('Solution:', solution.paths[i], solution_path_score)
            # Genetic algorithm
            if len(cluster) > 5:
                solver = RoutingSolver(cluster, preference_model)
                prediction, prediction_path_score = solver.solve(peak_form=peak_form)
            # Exhaustive algorithm
            else:
                prediction, prediction_path_score = solve_exhaustive(cluster, preference_model, peak_form=peak_form)
            linear_test_data_path.append([problem_id, solution_path_score, solution.paths[i], prediction_path_score, prediction])
            predicted_solution.append(prediction)
            predicted_score += prediction_path_score
            print('Prediction:', prediction, prediction_path_score)
        end_time = time.time()
        solve_time = end_time - start_time
        AD = solution.get_arc_difference(predicted_solution)
        print(predicted_solution)
        print('Total solution score:', solution_score)
        print('Total prediction score:', predicted_score)
        print('Total AD:', AD)
        print('Time:', solve_time)
        linear_test_data_solution.append([problem_id, solution_score, predicted_score, AD, solve_time])
        with open('./data/routing_path_evall_' + peak_form + '.data', 'wb') as output_file:
            pickle.dump(linear_test_data_path, output_file)
        with open('./data/routing_sol_evall_' + peak_form + '.data', 'wb') as output_file:
            pickle.dump(linear_test_data_solution, output_file)
        # ADD DATA OF SOLUTION TO THE MODEL
        preference_model.train(problem_id)


def read_routing_test_data(peak_form='exp'):
    with open('./data/routing_path_evall_' + peak_form + '.data', 'rb') as input_file:
        ltdp = pickle.load(input_file)
    with open('./data/routing_sol_evall_' + peak_form + '.data', 'rb') as input_file:
        ltds = pickle.load(input_file)
    return ltdp, ltds

peak = str(sys.argv[1])
print("PEAK", peak)
routing_test(130, 165, 200, peak)
