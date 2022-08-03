import numpy as np
from matplotlib import pyplot as plt, patches

from PARAMETERS import CVRP_INSTANCE, LAMBDA, WEIGHING_SCHEME, ALPHA

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

def plot_probabilites(nodes, problem_id):
    # PLOT VARS
    ax = plt.gca()
    probabilities = get_transition_probability_matrix(0, problem_id, logarithm=False)
    probabilities = probabilities[nodes, :]
    probabilities = probabilities[:, nodes]
    for i in range(probabilities.shape[0]):
        probabilities[i, :] = probabilities[i, :] / np.sum(probabilities[i, :])
    style = 'Simple, tail_width=0.5, head_width=5, head_length=6'
    coords = np.asarray([CVRP_INSTANCE.customer_positions[node] for node in nodes])
    ax.scatter(coords[:, 0], coords[:, 1], color='black', s=0.3)
    for node in nodes:
        ax.annotate(node, CVRP_INSTANCE.customer_positions[node], fontsize=6.5)

    for i in range(len(nodes)):
        for j in range(len(nodes)):
            p = probabilities[i][j]
            ax.add_patch(patches.FancyArrowPatch(coords[i],
                                                 coords[j],
                                                 connectionstyle="arc3,rad=0.0",
                                                 arrowstyle=style,
                                                 color='black',
                                                 alpha=p))
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    #plt.show()
    plt.savefig('./figures/transprob.pdf')

