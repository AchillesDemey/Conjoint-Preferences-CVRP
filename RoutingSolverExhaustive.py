import itertools
import math

def solve_exhaustive(cluster, model, peak_form='exp'):
    best_score = math.inf
    best_path = None
    for permutation in itertools.permutations(cluster):
        path = [0] + list(permutation) + [0]
        score = model.evaluate(path, peak_form=peak_form)
        if score < best_score:
            best_score = score
            best_path = path
    return best_path, best_score

