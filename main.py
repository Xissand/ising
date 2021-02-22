# MC simulation of an Ising 2d model

import mc
import ising
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
import seaborn as sns


def test(t):
    res = []
    for i in range(20):
        a = ising.Lattice(8)
        a.shuffle()
        mc.sim(a, 100, t)
        res.append([t, mc.sim(a, 100, t)])
        del a
    print(t)
    return res


def magnetization_map():
    with Pool(8) as p:
        a = p.map(test, np.arange(0.1, 20.1, 0.5))
    T = []
    m = []
    for res in a:
        for r in res:
            T.append(r[0])
            m.append(r[1])
    plt.scatter(T, m, s=40, facecolor='none', edgecolors='blue')
    plt.show()


def coolplot(n, t):
    a = ising.Lattice(n)
    a.shuffle()
    mc.sim(a, 2000, t)
    a.visualize("cool", filename="")


if __name__ == '__main__':
    magnetization_map()
    # coolplot(8, 1)
