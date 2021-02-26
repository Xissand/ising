# MC simulation of an Ising 2d model

import mc
import ising
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
import seaborn as sns
from timeit import default_timer as timer


def c(t):
    res = []
    for run in range(20):
        en = []
        a = ising.Lattice(16)
        a.shuffle()
        mc.sim(a, 20000, t)
        for i in range(50000):
            mc.sim(a, 1, t)
            e = a.energy()
            en.append(e)
        res.append([t, np.mean(en), np.mean(np.square(en))])
    del a
    print(t)
    return res


def specific_map():
    with Pool(8) as p:
        a = p.map(c, np.arange(1.0, 4.0, 0.1))
    t = []
    e = []
    de = []
    for res in a:
        for r in res:
            t.append(r[0])
            e.append(r[1])
            de.append(r[2])
    tt = np.arange(1.0, 4.0, 0.05)
    e2 = np.zeros(len(tt))
    ee2 = np.zeros(len(tt))
    for num, temp in enumerate(tt):
        for i, j, k in zip(t, e, de):
            if np.abs(i - temp) < 0.01:
                e2[num] += j / 10
                ee2[num] += k / 10
    ch = (ee2 - e2 ** 2) / tt
    plt.scatter(tt, ch, s=40, facecolor='none', edgecolors='blue')
    plt.show()


def test(t):
    res = []
    for i in range(20):
        a = ising.Lattice(20)
        a.shuffle()
        mc.sim(a, 100000, t, freq=0)
        b = mc.sim(a, 100000, t, freq=0, ave=True)
        m = float(b[2])
        res.append([t, m])
        del a
    print(t)
    return res


def magnetization_map():
    with Pool(8) as p:
        a = p.map(test, np.arange(0.1, 10.1, 0.1))
    t = []
    m = []
    for res in a:
        for r in res:
            t.append(r[0])
            m.append(r[1])
    plt.scatter(t, m, s=40, facecolor='none', edgecolors='blue')
    plt.show()


def coolplot(n, t):
    a = ising.Lattice(n)
    a.shuffle()
    b = mc.sim(a, int(1e5), t, ave=True)

    sns.heatmap(a.visualize(), vmin=-1, vmax=1)
    plt.show()

    print(b[0])
    # for name, value in zip(a.observables.split(" "),b):
    #    print(f"{name:6} {value:.4f}")


if __name__ == '__main__':
    magnetization_map()
    # coolplot(50, 0.1)

    # specific_map()
