import math

from matplotlib import pyplot as plt


def idealpointplot():
    fig = plt.figure(figsize=(6, 4))
    params = {'mathtext.default': 'regular'}
    x = []
    y = []
    for i in range(-200, 200):
        i = i/100
        if i < -1 or i > 1:
            c = 0
        else:
            c = math.sqrt(1 - i**2)
        x.append(i)
        y.append(c)
    plt.plot(x, y, color='black', linewidth=2.5)

    plt.plot([0, 0], [0, 1], color='black', linestyle='--', linewidth=1)
    plt.axhline(0, color='black', linewidth=1)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('$x$')
    plt.ylabel('$U(x)$')

    plt.xlim([-2,2])
    plt.savefig('./figures/ideal_point_utility.pdf')
    plt.clf()

def linearplot():
    fig = plt.figure(figsize=(6, 4))
    params = {'mathtext.default': 'regular'}
    plt.plot([-1,0.5], [1, 0.8], color='black', linewidth=2.5)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('$x$')
    plt.ylabel('$U(x)$')
    #plt.show()
    plt.savefig('./figures/linear_utility.pdf')
    plt.clf()

def part_worth_plot():
    fig = plt.figure(figsize=(6, 4))
    params = {'mathtext.default': 'regular'}
    x = []
    y = []
    for i in range(-50, 600):
        i = i/100
        c = 0
        if 0 < i < 1:
            c = 2
        elif 1.5 < i < 2.5:
            c = 3
        elif 3 < i < 4:
            c = 1
        elif 4.5 < i < 5.5:
            c = 0.5
        x.append(i)
        y.append(c)
    plt.plot(x, y, color='black', linewidth=2.5)

    plt.axhline(0, color='black', linewidth=1)
    plt.xticks([0.5, 2, 3.5, 5],['$x_1$','$x_2$','$x_3$','$x_4$'])
    plt.yticks([])
    plt.xlabel('$x$', fontsize=16)
    plt.ylabel('$U(x)$')

    plt.xlim([-0.5, 6])
    #plt.show()
    plt.savefig('./figures/part_worth_utility.pdf')
    plt.clf()

#idealpointplot()
part_worth_plot()