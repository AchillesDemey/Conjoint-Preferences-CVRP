import numpy as np


class RoutingSolver:

    def __init__(self, cluster, model):
        self.nb_chromosomes = 25
        self.nb_offspring = 25
        self.model = model
        self.cluster = cluster
        self.parents = self.init_population(cluster)
        self.offspring = None
        self.scores = None
        self.average = None
        self.best = None
        self.peak_form = 'exp'

    def solve(self, peak_form='exp'):
        """
        peak_form can be 'exp' (exponential), 'lin' (linear) or 'ell' (elliptoid)
        """
        self.peak_form = peak_form
        self.scores = np.asarray([self.fitness(chromosome) for chromosome in self.parents])
        self.average = np.average(self.scores)
        self.best = np.min(self.scores)

        if len(self.cluster) == 1:
            return [0]+self.cluster+[0], self.fitness([0]+self.cluster+[0])
        count = 0
        while count < 5:
            # Create the next generation
            self.next_generation()
            # Update fitness values
            self.scores = np.asarray([self.fitness(chromosome) for chromosome in self.parents])
            if np.average(self.scores) < self.average:
                count = 0
            else:
                count += 1
            self.average = np.average(self.scores)
            self.best = self.scores[0]
        return [0] + list(self.parents[0]) + [0], self.best

    def next_generation(self):
        self.offspring = np.zeros((self.nb_offspring * 2, self.parents.shape[1]), dtype=np.uint8)
        for offspring_nb in range(self.nb_offspring):
            p1, p2 = self.select()
            c1, c2 = self.crossover(p1, p2)
            c1, c2 = self.mutate(c1), self.mutate(c2)
            self.offspring[offspring_nb * 2] = c1
            self.offspring[offspring_nb * 2 + 1] = c2
        self.eliminate()

    # Random initialization
    def init_population(self, cluster):
        return np.asarray([np.random.permutation(cluster) for _ in range(self.nb_chromosomes)], dtype=np.uint8)

    # Fitness based on model
    def fitness(self, chromosome):
        return self.model.evaluate([0] + list(chromosome) + [0], peak_form=self.peak_form)

    # K-Tournament Selection
    def select(self, k=2, tournament=4):
        tournament = min(len(self.parents), tournament)
        chosen = np.random.choice(list(range(len(self.parents))), tournament, replace=False)
        selection = np.argsort([self.scores[i] for i in chosen])[:k]
        return self.parents[chosen[selection]]

    # PMX crossover
    def crossover(self, p1, p2):
        c1, c2 = np.zeros(len(p1), dtype=np.int), np.zeros(len(p2), dtype=np.int)
        i1, i2 = sorted(np.random.choice(range(len(p1)), 2, replace=False))
        for index in range(len(p1)):
            if index in range(i1, i2):
                c1[index] = p2[index]
                c2[index] = p1[index]
            else:
                if p1[index] in p2[i1:i2]:
                    node_to = p1[np.where(p2 == p1[index])[0][0]]
                    while node_to in p2[i1:i2]:
                        node_to = p1[np.where(p2 == node_to)[0][0]]
                    c1[index] = node_to
                else:
                    c1[index] = p1[index]

                if p2[index] in p1[i1:i2]:
                    node_to = p2[np.where(p1 == p2[index])[0][0]]
                    while node_to in p1[i1:i2]:
                        node_to = p2[np.where(p1 == node_to)[0][0]]
                    c2[index] = node_to
                else:
                    c2[index] = p2[index]
        assert(len(np.unique(c1)) == self.parents.shape[1])
        assert(len(np.unique(c2)) == self.parents.shape[1])
        return c1, c2

    # Mutation
    def mutate(self, c, p=0.3):
        if np.random.rand() < p:
            a, b = np.random.choice(len(c), 2)
            c[a], c[b] = (c[b], c[a])
        return c

    # Elimination
    def eliminate(self):
        # Merge offpring and parents
        self.parents = np.concatenate((self.parents, self.offspring), axis=0)
        # Get the unique population
        self.parents = np.unique(self.parents, axis=0)
        # Sort population by fitness value
        self.parents = self.parents[np.argsort([self.fitness(chomosome) for chomosome in self.parents])]
        # Remove the worst-fit individuals
        self.parents = self.parents[:min(self.parents.shape[0], self.nb_chromosomes)]
