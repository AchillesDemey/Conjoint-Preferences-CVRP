import pickle

import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate


def read_routing_test_data(name='exp'):
    with open('./data/routing_sol_evall_' + name + '.data', 'rb') as input_file:
        data = pickle.load(input_file)
    for i in range(len(data)):
        data[i] = data[i][:3] + [data[i][3][0], data[i][3][1]] + data[i][4:]
    # Remove Null columns
    data = list(filter(lambda x: None not in x, data))
    return data

def read_routin_path_test_data(name='exp'):
    with open('./data/routing_path_evall_' + name + '.data', 'rb') as input_file:
        data = pickle.load(input_file)
    # Remove Null columns
    data = list(filter(lambda x: None not in x, data))
    return data



def routing_data_to_tex(data, header=None):
    if header is None:
        header = ['Problem_Id', 'HIST_score', 'PRED_score', 'AD', 'AD%', 'Time (s)']
    data = [header] + data
    table = tabulate(data, headers='firstrow', tablefmt='latex')
    print(table)


def boxplot(datasets, i):
    data = [[record[i] for record in dataset] for dataset in datasets]
    plt.boxplot(data, labels=['linear', 'elliptic', 'exp1', 'exp2'])
    plt.ylabel('AD')
    plt.savefig('./figures/boxplot_ad.pdf')


def convergence_table(datasets):
    # Correct, Good model, bad model
    x = [[0, 0, 0] for _ in range(len(datasets))]
    for i, dataset in enumerate(datasets):
        for j in range(len(dataset)):
            if dataset[j][1] == dataset[j][2]:
                x[i][0] += 1
            elif dataset[j][1] < dataset[j][2]:
                x[i][1] += 1
            else:
                x[i][2] += 1
    print(x)


def accuracy_table_length(datasets):
    results = [[] for _ in range(12)]
    for dataset in datasets:
        for record in dataset:
            print(record)
            length = len(record[2])
            if record[2] == record[4]:
                results[length].append(1)
            else:
                results[length].append(0)
    print([np.average(i) for i in results])



data_exp2 = read_routing_test_data('exp2')
data_exp = read_routing_test_data('exp')
data_ell = read_routing_test_data('ell')
data_lin = read_routing_test_data('lin')

alldata = [data_lin,data_ell,data_exp,data_exp2]
times = []
for d in alldata:
    for r in d:
        times.append(r[-1])
print(np.average(times))
print(np.var(times))

data_p_exp2 = read_routin_path_test_data('exp2')
data_p_exp = read_routin_path_test_data('exp')
data_p_ell = read_routin_path_test_data('ell')
data_p_lin = read_routin_path_test_data('lin')

#accuracy_table_length([data_p_exp2, data_p_exp, data_p_ell, data_p_lin])

"""
boxplot([data_lin, data_ell, data_exp, data_exp2], 3)
convergence_table([data_lin, data_ell, data_exp, data_exp2])
"""

