# Learning driver preferences through conjoint analysis

## Abstract
This work establishes a decision support system for route planning for small and medium-sized businesses. The system is able to learn implicit preferences from historical routes created by a human planner. This person often brings some personal preferences and goals of the company into the plan such as equal distribution of customers among drivers. We assume that the same customers often recur in solutions. The system is able to create new plans based on the learned preferences as the human planner would do. In practice, however, one sets up a separate vehicle routing problem for each company with its own set of objectives and constraints. The method proposed by this work is an alternative way of generating solutions that can please both the planner and the drivers. The underlying model that learns preferences and evaluates new solutions is based on conjoint analysis. This technique analyzes a number of attributes of the historical trajectories such as geometric aspects or the length of the routes. For each of these attributes, the model learns what the preferred values are. Based on the conjoint model, a metaheuristic algorithm searches for the solution with the most preferred attribute values. The work takes a two-phase approach where the clients are first distributed among the drivers and then the best path is sought for each driver.

## Entry points for running the code and executing experiments
### Set the parameters of the models
./src/PARAMETERS.py <br />

### Execute incremental evaluation for both the routing and clustering problem
./src/TEST_CLUSTERING.py <br />
./src/TEST_ROUTING.py <br />

### Convert test results to LaTex tables:
./src/TEST_CLUSTERING_ANALYSIS.py
./src/TEST_ROUTING_ANALYSIS.py

## Description of the classes
### Helper classes:
CVRPInstance: Reads and processes data from dataset
Solution: Converts dataset into a historical solution of a specific day (by number) with context
Problem: Extracts context of a problem in the dataset
DriverTour: Representation of a path in the CVRP solution

### Preference models:
Routingmodel: preference model for routing with its own attributes
ClusteringModel: preference model for clustering with its own attributes

### Solvers:
MIP_solver: for solving the CVRP with arc probabilities or objective costs
RoutingSolverExhaustive: Exhaustive search through solution space
RoutingSolverGenetic: Genetic algorithm for finding most preferred path
ClusteringSolverILS: Iterated local search algorithm for finding clusters


Link to original research in dataset: https://github.com/JayMan91/CP2021-Data-Driven-VRP
