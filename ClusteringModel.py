# https://eml.berkeley.edu/books/choice2.html
import copy
import itertools
import math
import pickle
from matplotlib import pyplot as plt
import numpy as np
from src.TwoStageCVRP.PARAMETERS import CVRP_INSTANCE
from PARAMETERS import LOCAL_CLUSTER_NEIGHBOURHOOD
from scipy.spatial import ConvexHull, convex_hull_plot_2d


def save_clustering_model(model):
    with open('./data/clustering.model', 'wb') as output_file:
        pickle.dump(model, output_file)
    print("MODEL SAVED")
    return

def load_clustering_model():
    with open('./data/clustering.model', 'rb') as input_file:
        conjoint_dataset = pickle.load(input_file)
    print("MODEL LOADED")
    return conjoint_dataset

class ClusteringModel:

    def __init__(self):
        self.node_models = [ConjointModel(node_id) for node_id in range(CVRP_INSTANCE.nb_nodes)]

    def train(self, solution):
        for cluster in solution.clusters:
            # Create perturbation set
            perturbations = self.get_perturbations(cluster, solution.stops)
            # Update the model of each node in the path
            for node in cluster:
                self.node_models[node].train(cluster,
                                             perturbations,
                                             context=solution.stops,
                                             problem_id=solution.problem_id)

    def get_perturbations(self, cluster, stops):
        permutations = []
        # Remove one stop from the cluster
        for i in range(len(cluster)):
            permutations.append(cluster[:i] + cluster[i+1:])
        # Add one stop to each cluster
        for stop in stops:
            if stop != 0 and stop not in cluster:
                permutations.append(sorted(cluster + [stop]))
        return permutations

    def evaluate(self, cluster, context=None, problem_id=None):
        score = 0
        for node in cluster:
            score += (-1) * math.log(self.node_models[node].get_probability(cluster, context, problem_id))
        return score

    def get_cluster_attributes(self, cluster):
        return self.node_models[cluster[0]].get_attributes(cluster)

    def evaluate_clusters(self, clusters, context=None, problem_id=None):
        return sum([self.evaluate(cluster, context, problem_id) for cluster in clusters])

    def plot(self, node_id, context=None):
        self.node_models[node_id].plot(context)

class ConjointModel:

    def __init__(self, node_id):
        self.node_id = node_id
        self.attribute_probability_models = None

    def train(self, cluster, perturbations, context, problem_id):
        assert(self.node_id in cluster)
        attributes = self.get_attributes(cluster)
        perturbed_attributes = []
        for perturbation in perturbations:
            if self.node_id in cluster and len(perturbation) > 0:
                perturbed_attributes.append(self.get_attributes(perturbation))
        if self.attribute_probability_models is None:
            self.attribute_probability_models = {name: AttributeProbabilityModel(name) for name in attributes.keys()}
        for name in attributes.keys():
            self.attribute_probability_models[name].add_predictor(attributes, perturbed_attributes, context, problem_id)

    def get_probability(self, path, context=None, problem_id=None):
        if self.attribute_probability_models is None:
            return 1
        attribute_scores = []
        attributes = self.get_attributes(path)
        for name, value in attributes.items():
            attribute_scores.append(self.attribute_probability_models[name].predict(value, context, problem_id))
        # Weigh each attribute score equally
        score = np.average(attribute_scores)
        if score == 0:
            score = 10e-30
        return score

    def get_attributes(self, cluster):
        # Number of nodes
        number_nodes = len(cluster)
        # GLOBAL CLUSTER
        # Center of global cluster
        center_cluster = np.average(CVRP_INSTANCE.customer_positions[cluster], axis=0)
        # Global cluster width
        global_width = max(CVRP_INSTANCE.distance_matrix[self.node_id][cluster])
        # LOCAL CLUSTER
        closest_indices = np.argsort(CVRP_INSTANCE.distance_matrix[self.node_id][cluster])
        local_cluster = [cluster[i] for i in closest_indices[:min(len(closest_indices), LOCAL_CLUSTER_NEIGHBOURHOOD)]]
        # Center of local cluster
        center_local_cluster = np.average(CVRP_INSTANCE.customer_positions[local_cluster], axis=0)
        # Global cluster width
        local_cluster_width = max(CVRP_INSTANCE.distance_matrix[self.node_id][local_cluster])
        return {
            "n": number_nodes,
            "gccx": center_cluster[0],
            "gccy": center_cluster[1],
            "gcw" : global_width,
            "lccx": center_local_cluster[0],
            "lccy": center_local_cluster[1],
            "lcw" : local_cluster_width,
        }

    def plot(self, context=None):
        if self.attribute_probability_models is not None:
            for model in self.attribute_probability_models.values():
                model.plot(context)

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

    def predict(self, value, context=None, problem_id=None):
        if len(self.predictors) == 0:
            return 1
        else:
            # Obtain the probability and the weight of every predictor
            probabilities = []
            similarities = []
            for predictor in self.predictors:
                probability, similarity = predictor.predict_exponential(value, context, problem_id)
                probabilities.append(probability)
                similarities.append(similarity)
            # Normalize the weights
            similarities_sum = sum(similarities)
            similarities = [s / similarities_sum for s in similarities]
            # Return the score
            return np.dot(probabilities, similarities)

    def plot(self, context=None):
        fig = plt.figure(figsize=(12,4))
        x = []
        y = []
        for value in range(int(self.attribute_range[0]*100)-200, int(self.attribute_range[1]*100)+200):
            atrv = value/100
            x.append(atrv)
            # Obtain the probability and the weight of every predictor
            sim = []
            p = []
            for predictor in self.predictors:
                probability, similarity = predictor.predict_exponential(atrv, context)
                p.append(probability)
                sim.append(similarity)
            # Normalize the weights
            similarities_sum = sum(sim)
            similarities = [s / similarities_sum for s in sim]
            # Return the score
            y.append(np.dot(p, similarities))
        plt.cla()
        plt.xlabel('$x$')
        plt.ylabel('$Pr(x)$')
        plt.axhline(0, color='black', linewidth=1)
        plt.plot(x, y, color='black', linewidth=2.5)
        plt.xlim([int(self.attribute_range[0])-2, int(self.attribute_range[1])+2])
        #plt.show()
        plt.savefig('./figures/attrprob.pdf')


    def plot_smooth(self, context=None):
        fig = plt.figure(figsize=(12,4))
        x = []
        y = []
        for value in range(int(self.attribute_range[0]*100)-200, int(self.attribute_range[1]*100)+200):
            atrv = value/100
            x.append(atrv)
            # Obtain the probability and the weight of every predictor
            sim = []
            p = []
            for predictor in self.predictors:
                probability, similarity = predictor.predict_exponential(atrv, context)
                p.append(probability)
                sim.append(similarity)
            # Normalize the weights
            similarities_sum = sum(sim)
            similarities = [s / similarities_sum for s in sim]
            # Return the score
            y.append(np.dot(p, similarities))
        window = 500
        y_new = []
        for i in range(len(y)):
            y_new.append(np.average(y[max(0, i-window):min(i+window, len(y))]))

        plt.cla()
        plt.xlabel('$x$')
        plt.ylabel('$Pr(x)$')
        plt.axhline(0, color='black', linewidth=1)
        plt.plot(x, y_new, color='black', linewidth=2.5)
        plt.xlim([int(self.attribute_range[0])-2, int(self.attribute_range[1])+2])
        #plt.show()
        plt.savefig('./figures/attrprob_smooth.pdf')

class BilateralDecreasingPredictor:

    def __init__(self, x_opt, x_min, x_max, context, problem_id):
        self.x_opt = x_opt
        self.x_min = x_min
        self.x_max = x_max
        self.context = context
        self.problem_id = problem_id

    def predict_linear(self, x, context=None, problem_id=None):
        if context is None or problem_id is None:
            similarity = 1
        else:
            similarity = (self.problem_id / problem_id) * len(set(self.context).intersection(set(context))) / len(
                set(self.context).union(set(context)))
        if self.x_opt <= x <= self.x_max and not math.isinf(self.x_max):
            probability = (self.x_max - x) / (self.x_max - self.x_opt)
        elif self.x_min <= x <= self.x_opt and not math.isinf(self.x_min):
            probability = (x - self.x_min) / (self.x_opt - self.x_min)
        else:
            probability = 0
        return [probability, similarity]

    def predict_elliptoid(self, x, context=None, problem_id=None):
        if context is None or problem_id is None:
            similarity = 1
        else:
            similarity = (self.problem_id / problem_id) * len(set(self.context).intersection(set(context))) / len(
                set(self.context).union(set(context)))
        if self.x_opt <= x <= self.x_max and not math.isinf(self.x_max):
            probability = 1 - (x - self.x_opt)**2 / (self.x_max - self.x_opt)**2
        elif self.x_min <= x <= self.x_opt and not math.isinf(self.x_min):
            probability = 1 - (self.x_opt - x)**2 / (self.x_opt - self.x_min)**2
        else:
            probability = 0
        return [probability, similarity]

    def predict_exponential(self, x, context=None, problem_id=None):
        P = 0.1
        if context is None or problem_id is None:
            similarity = 1
        else:
            similarity = (self.problem_id / problem_id) * len(set(self.context).intersection(set(context))) / len(set(self.context).union(set(context)))
        if self.x_opt <= x and not math.isinf(self.x_max):
            l_max = -math.log(P) / (self.x_max - self.x_opt)
            probability = math.exp((-1) * l_max * (x - self.x_opt))
        elif x <= self.x_opt and not math.isinf(self.x_min):
            l_min = -math.log(P) / (self.x_opt - self.x_min)
            probability = math.exp((-1) * l_min * (self.x_opt - x))
        else:
            probability = 0
        return [probability, similarity]

    def plot_peak_linear(self):
        fig = plt.figure(figsize=(6,4))
        params = {'mathtext.default': 'regular'}
        plt.rcParams.update(params)
        x = []
        y = []
        for i in range(math.floor(self.x_min - 1/(self.x_max-self.x_min)) * 100,
                       math.ceil(self.x_max + 1/(self.x_max-self.x_min)) * 100):
            x.append(i / 100)
            y.append(self.predict_linear(i / 100)[0])
        plt.plot(x, y, color='black', linewidth=2.5)

        plt.plot([self.x_opt, self.x_opt], [0, 1], color='black', linestyle='--', linewidth=1)
        plt.axhline(0, color='black', linewidth=1)


        xt = np.array([self.x_min, self.x_opt, self.x_max])
        xticks = ['$x_{min}$', '$x_{opt}$', '$x_{max}$']
        yticks = [0, 1]
        plt.xticks(xt, xticks)
        plt.xlabel('$x$')
        plt.ylabel('$f(x)$')

        plt.xlim([self.x_min - 1/(self.x_max-self.x_min), self.x_max + 1/(self.x_max-self.x_min)])
        plt.savefig('./figures/linear.pdf')
        plt.clf()

    def plot_peak_elliptoid(self):
        fig = plt.figure(figsize=(6,4))
        params = {'mathtext.default': 'regular'}
        plt.rcParams.update(params)
        x = []
        y = []
        for i in range(math.floor(self.x_min - 1/(self.x_max-self.x_min)) * 100,
                       math.ceil(self.x_max + 1/(self.x_max-self.x_min)) * 100):
            x.append(i / 100)
            y.append(self.predict_elliptoid(i / 100)[0])
        plt.plot(x, y, color='black', linewidth=2.5)

        plt.plot([self.x_opt, self.x_opt], [0, 1], color='black', linestyle='--', linewidth=1)
        plt.axhline(0, color='black', linewidth=1)


        xt = np.array([self.x_min, self.x_opt, self.x_max])
        xticks = ['$x_{min}$', '$x_{opt}$', '$x_{max}$']
        yticks = [0, 1]
        plt.xticks(xt, xticks)
        plt.xlabel('$x$')
        plt.ylabel('$f(x)$')

        plt.xlim([self.x_min - 1/(self.x_max-self.x_min), self.x_max + 1/(self.x_max-self.x_min)])
        plt.savefig('./figures/elliptoid.pdf')
        plt.clf()

    def plot_peak_exponential(self):
        fig = plt.figure(figsize=(6,4))
        params = {'mathtext.default': 'regular'}
        plt.rcParams.update(params)
        x = []
        y = []
        for i in range(math.floor(self.x_min - 1/(self.x_max-self.x_min)) * 100,
                       math.ceil(self.x_max + 1/(self.x_max-self.x_min)) * 100):
            x.append(i / 100)
            y.append(self.predict_exponential(i / 100)[0])
        plt.plot(x, y, color='black', linewidth=2.5)

        plt.plot([self.x_opt, self.x_opt], [0, 1], color='black', linestyle='--', linewidth=1)
        plt.axhline(0, color='black', linewidth=1)


        xt = np.array([self.x_min, self.x_opt, self.x_max])
        xticks = ['$x_{min}$', '$x_{opt}$', '$x_{max}$']
        yticks = [0, 1]
        plt.xticks(xt, xticks)
        plt.xlabel('$x$')
        plt.ylabel('$f(x)$')

        plt.xlim([self.x_min - 1/(self.x_max-self.x_min), self.x_max + 1/(self.x_max-self.x_min)])
        plt.savefig('./figures/exponential.pdf')
        plt.clf()

