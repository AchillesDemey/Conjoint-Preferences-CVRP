# https://eml.berkeley.edu/books/choice2.html
import itertools
import math
import pickle
from matplotlib import pyplot as plt
import numpy as np
from PARAMETERS import CVRP_INSTANCE, LOCAL_NEIGHBOURHOOD
from Solution import Solution


def save_routing_model(model):
    with open('./data/routing.model', 'wb') as output_file:
        pickle.dump(model, output_file)
    print("MODEL SAVED")
    return

def load_routing_model():
    with open('./data/routing.model', 'rb') as input_file:
        conjoint_dataset = pickle.load(input_file)
    print("MODEL LOADED")
    return conjoint_dataset

class RoutingModel:

    def __init__(self):
        self.node_models = [ConjointModel(node_id) for node_id in range(CVRP_INSTANCE.nb_nodes)]

    def train(self, problem_id):
        solution = Solution(problem_id)
        for path in solution.paths:
            # Create perturbation set
            perturbations = self.get_perturbations_randomly(path)
            # Update the model of each node in the path
            for node in path[:-1]:
                self.node_models[node].train(path, perturbations, context=path, problem_id=solution.problem_id)

    def get_perturbations_exhaustive(self, path):
        return list(itertools.permutations(path[1:-1]))

    def get_perturbations_randomly(self, path):
        NB_PERMUTATIONS = 200
        perturbations = []
        for _ in range(NB_PERMUTATIONS):
            perturbations.append(np.random.permutation(path[1:-1]))
        return perturbations

    def evaluate(self, path, peak_form='exp'):
        score = 0
        for index, node in enumerate(path[:-1]):
            context = list(filter(lambda x: x != 0, path))
            score += (-1) * math.log(self.node_models[node].get_probability(path, context=context, peak_form=peak_form))
        return score

class ConjointModel:

    def __init__(self, node_id):
        self.node_id = node_id
        self.attribute_probability_models = None

    def train(self, path, perturbations, context, problem_id):
        assert(self.node_id in path)
        attributes = self.get_attributes(path)
        perturbed_attributes = [self.get_attributes([0]+list(perturbed_path)+[0]) for perturbed_path in perturbations]
        if self.attribute_probability_models is None:
            self.attribute_probability_models = {name: AttributeProbabilityModel(name) for name in attributes.keys()}
        for name in attributes.keys():
            self.attribute_probability_models[name].add_predictor(attributes, perturbed_attributes, context, problem_id)

    def get_probability(self, path, context=None, peak_form='exp'):
        if self.attribute_probability_models is None:
            return 1
        attribute_scores = []
        attributes = self.get_attributes(path)
        for name, value in attributes.items():
            attribute_scores.append(self.attribute_probability_models[name].predict(value, context, peak_form))
        # Weigh each attribute score equally
        score = np.average(attribute_scores)
        # Smoothing for logarithm
        if score == 0:
            score = 10e-30
        return score

    def get_attributes(self, path):
        assert(self.node_id in path)
        index = path.index(self.node_id)
        # Global path
        P_global_before = path[0:index + 1]
        P_global_after = path[index:len(path)]
        # Local path
        N = LOCAL_NEIGHBOURHOOD
        looped_path = path[:-1] * 3
        looped_index = len(path[:-1]) + index
        P_local_before = looped_path[looped_index - N:looped_index + 1]
        P_local_after = looped_path[looped_index:looped_index + 1 + N]
        # Position
        position = index
        # Global distance before
        global_distance_before = sum([CVRP_INSTANCE.distance_matrix[P_global_before[i]][P_global_before[i + 1]]
                                      for i in range(len(P_global_before) - 1)])
        # Global distance after
        global_distance_after = sum([CVRP_INSTANCE.distance_matrix[P_global_after[i]][P_global_after[i + 1]]
                                     for i in range(len(P_global_after) - 1)])
        # Local center before
        local_center_before = np.average(CVRP_INSTANCE.customer_positions[P_local_before], axis=0)
        # Local center after
        local_center_after = np.average(CVRP_INSTANCE.customer_positions[P_local_after], axis=0)
        return {
            "p": position,
            "gdb": global_distance_before,
            "gda": global_distance_after,
            "lcbx": local_center_before[0],
            "lcby": local_center_before[1],
            "lcax": local_center_after[0],
            "lcay": local_center_after[1]
        }

class AttributeProbabilityModel:

    def __init__(self, name):
        self.name = name
        self.predictors = []
        self.attribute_range = [math.inf, (-1)*math.inf]

    def add_predictor(self, attributes, perturbed_attibutes, context, problem_id):
        x_opt = attributes[self.name]
        x_min = (-1) * math.inf
        x_max = math.inf
        for value in np.unique([atti[self.name] for atti in perturbed_attibutes]):
            if x_min < value < x_opt:
                x_min = value
            if x_opt < value < x_max:
                x_max = value
        # Change attribute borders
        if x_opt < self.attribute_range[0]:
            self.attribute_range[0] = math.floor(x_opt)
        if x_opt > self.attribute_range[1]:
            self.attribute_range[1] = math.ceil(x_opt)
        if not math.isinf(x_min) and x_min < self.attribute_range[0]:
            self.attribute_range[0] = math.floor(x_min)
        if not math.isinf(x_max) and x_max > self.attribute_range[1]:
            self.attribute_range[1] = math.ceil(x_max)

        self.predictors.append(BilateralDecreasingPredictor(x_opt, x_min, x_max, context, problem_id))

    def predict(self, value, context=None, peak_form='exp'):
        if len(self.predictors) == 0:
            return 1
        else:
            # Obtain the probability and the weight of every predictor
            probabilities = []
            similarities = []
            for predictor in self.predictors:
                if peak_form == 'exp':
                    probability, similarity = predictor.predict_exponential(value, context)
                elif peak_form == 'lin':
                    probability, similarity = predictor.predict_linear(value, context)
                elif peak_form == 'ell':
                    probability, similarity = predictor.predict_elliptoid(value, context)
                else:
                    probability, similarity = predictor.predict_exponential2(value, context)
                probabilities.append(probability)
                similarities.append(similarity)
            # Normalize the weights
            similarities_sum = sum(similarities)
            similarities = [s / similarities_sum for s in similarities]
            # Return the score
            return np.dot(probabilities, similarities)

    def plot_attribute_probability(self, context=None):
        fig, axs = plt.subplots(3)
        x = []
        y_lin = []
        y_ell = []
        y_exp = []
        print(self.attribute_range)
        for value in range(int(self.attribute_range[0]*100), int(self.attribute_range[1]*100)):
            atrv = value/100
            x.append(atrv)
            # Obtain the probability and the weight of every predictor
            sim = []
            p_lin = []
            p_ell = []
            p_exp = []
            for predictor in self.predictors:
                probability_lin, similarity = predictor.predict_linear(atrv, context)
                probability_ell, _ = predictor.predict_elliptoid(atrv, context)
                probability_exp, _ = predictor.predict_exponential(atrv, context)
                p_lin.append(probability_lin)
                p_ell.append(probability_ell)
                p_exp.append(probability_exp)
                sim.append(similarity)
            # Normalize the weights
            similarities_sum = sum(sim)
            similarities = [s / similarities_sum for s in sim]
            # Return the score
            y_lin.append(np.dot(p_lin, similarities))
            y_ell.append(np.dot(p_ell, similarities))
            y_exp.append(np.dot(p_exp, similarities))
        plt.cla()
        axs[0].plot(x, y_lin)
        axs[1].plot(x, y_ell)
        axs[2].plot(x, y_exp)
        plt.show()

class BilateralDecreasingPredictor:

    def __init__(self, x_opt, x_min, x_max, context=None, problem_id=None):
        self.x_opt = x_opt
        self.x_min = x_min
        self.x_max = x_max
        self.context = list(filter(lambda x: x != 0, context))
        self.problem_id = problem_id

    def predict_linear(self, x, context=None):
        if context is None:
            similarity = 1
        else:
            similarity = len(set(self.context).intersection(set(context))) / len(set(self.context).union(set(context)))
        if self.x_opt <= x <= self.x_max and not math.isinf(self.x_max):
            probability = (self.x_max - x) / (self.x_max - self.x_opt)
        elif self.x_min <= x <= self.x_opt and not math.isinf(self.x_min):
            probability = (x - self.x_min) / (self.x_opt - self.x_min)
        else:
            probability = 0
        return [probability, similarity]

    def predict_elliptoid(self, x, context=None):
        if context is None:
            similarity = 1
        else:
            similarity = len(set(self.context).intersection(set(context))) / len(set(self.context).union(set(context)))
        if self.x_opt <= x <= self.x_max and not math.isinf(self.x_max):
            probability = 1 - (x - self.x_opt)**2 / (self.x_max - self.x_opt)**2
        elif self.x_min <= x <= self.x_opt and not math.isinf(self.x_min):
            probability = 1 - (self.x_opt - x)**2 / (self.x_opt - self.x_min)**2
        else:
            probability = 0
        return [probability, similarity]

    def predict_exponential(self, x, context=None):
        P = 0.1
        if context is None:
            similarity = 1
        else:
            similarity = len(set(self.context).intersection(set(context))) / len(set(self.context).union(set(context)))
        if self.x_opt <= x and not math.isinf(self.x_max):
            l_max = -math.log(P) / (self.x_max - self.x_opt)
            probability = math.exp((-1) * l_max * (x - self.x_opt))
        elif x <= self.x_opt and not math.isinf(self.x_min):
            l_min = -math.log(P) / (self.x_opt - self.x_min)
            probability = math.exp((-1) * l_min * (self.x_opt - x))
        else:
            probability = 0
        return [probability, similarity]

    def predict_exponential2(self, x, context=None):
        P = 0.001
        if context is None:
            similarity = 1
        else:
            similarity = len(set(self.context).intersection(set(context))) / len(set(self.context).union(set(context)))
        if self.x_opt <= x and not math.isinf(self.x_max):
            l_max = -math.log(P) / (self.x_max - self.x_opt)
            probability = math.exp((-1) * l_max * (x - self.x_opt))
        elif x <= self.x_opt and not math.isinf(self.x_min):
            l_min = -math.log(P) / (self.x_opt - self.x_min)
            probability = math.exp((-1) * l_min * (self.x_opt - x))
        else:
            probability = 0
        return [probability, similarity]

