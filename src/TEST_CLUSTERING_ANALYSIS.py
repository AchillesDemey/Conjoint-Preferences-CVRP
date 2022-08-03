import pickle

import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate


def read_cluster_test_data(name='before'):
    with open('./data/cluster_eval_'+name+'.data', 'rb') as input_file:
        data = pickle.load(input_file)

    # Remove Null columns
    data = list(filter(lambda x: None not in x, data))
    return data


def cluster_data_to_tex(data, header=None):
    if header is None:
        header = ['Problem_Id', '$Violation_{CVRP}$', '$Violation_{Q}$', '$HIST_{score}$', '$PRED_{score}$', '$RD$', '$RD(\%)$', '$Time (s)$']
    data = [header] + data
    table = tabulate(data, headers='firstrow', tablefmt='latex')
    print(table)

def boxplot():
    data_before = np.asarray(read_cluster_test_data('before'))[:, 6]
    data_after = np.asarray(read_cluster_test_data('after'))[:, 6]
    plt.boxplot([data_before, data_after], labels=['before drift', 'after drift'])
    plt.ylabel('RD(%)')
    plt.savefig('./figures/boxplot_rd_percent.pdf')

def convergence_table():
    data_before_hist = np.asarray(read_cluster_test_data('before'))[:, 3]
    data_before_pred = np.asarray(read_cluster_test_data('before'))[:, 4]
    data_after_hist = np.asarray(read_cluster_test_data('after'))[:, 3]
    data_after_pred = np.asarray(read_cluster_test_data('after'))[:, 4]
    # Correct, Good model, bad model
    x = [[0, 0, 0], [0, 0, 0]]
    for i in range(len(data_before_pred)):
        if data_before_pred[i] == data_before_hist[i]:
            x[0][0] += 1
        elif data_before_pred[i] > data_before_hist[i]:
            x[0][1] += 1
        else :
            x[0][2] += 1
    for i in range(len(data_after_pred)):
        if data_after_pred[i] == data_after_hist[i]:
            x[1][0] += 1
        elif data_after_pred[i] > data_after_hist[i]:
            x[1][1] += 1
        else :
            x[1][2] += 1
    print(x)

b =read_cluster_test_data('before')
a =read_cluster_test_data('after')
print(a,b)
t = [x[-1] for x in a+b]
print(t)
print(np.average(t))
print(np.std(t))

