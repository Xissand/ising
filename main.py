# MC simulation of an Ising 2d model

import mc
import ising
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
import seaborn as sns
import matplotlib.animation as animation


def en(t):
    res = []
    for i in range(1):
        a = ising.Lattice(20)
        a.shuffle()
        mc.sim(a, 100000, t, freq=0)
        b, bb = mc.sim(a, 100000, t, freq=0, ave=True)
        m = float(b[2])
        res.append([t, m])
        del a
    print(t)
    return res


def energy_map():
    with Pool(8) as p:
        a = p.map(en, np.arange(0.1, 10.1, 0.1))
    t = []
    e = []
    for res in a:
        for r in res:
            t.append(r[0])
            e.append(r[1])
    plt.scatter(t, e, s=40, facecolor="none", edgecolors="blue")
    plt.xlabel("Temperature")
    plt.ylabel("Energy per spin")
    plt.savefig("heat.png")
    plt.show()


def c(t):
    res = []
    for run in range(10):
        a = ising.Lattice(16)
        a.shuffle()
        mc.sim(a, 50000, t, freq=0)
        v, dv = mc.sim(a, 100000, t, ave=True, freq=0)
        res.append([t, v[1], v[0]])
    del a
    print(t)
    return res


def heat_map():
    tt = []
    e2 = []
    e = []
    with Pool(8) as p:
        a = p.map(c, np.arange(1.0, 4.0, 0.05))
    for aa in a:
        for v in aa:
            tt.append(v[0])
            e2.append(v[1])
            e.append(v[2])
    tt = np.array(tt)
    e2 = np.array(e2)
    e = np.array(e)
    ch = (e2 - e ** 2) / tt ** 2
    plt.scatter(tt, ch, s=40, facecolor="none", edgecolors="blue")
    plt.xlabel("Temperature")
    plt.ylabel("Heat capacity")
    plt.savefig("heat.png")
    plt.show()


def test(t):
    res = []
    for i in range(20):
        a = ising.Lattice(20)
        a.shuffle()
        mc.sim(a, 100000, t, freq=0)
        b, bb = mc.sim(a, 100000, t, freq=0, ave=True)
        m = float(b[3])
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
    plt.scatter(t, m, s=40, facecolor="none", edgecolors="blue")
    plt.xlabel("Temperature")
    plt.ylabel("Average magnetization")
    plt.savefig("magnetization.png")
    plt.show()


def coolplot(n, t):
    a = ising.Lattice(n)
    a.shuffle()
    b, bb = mc.sim(a, int(1e6), t, ave=True)

    sns.heatmap(a.visualize(), vmin=-1, vmax=1)
    plt.show()

    print(b[0])
    # for name, value in zip(a.observables.split(" "),b):
    #    print(f"{name:6} {value:.4f}")


def anime(n, t, steps=int(1e5), freq=1):
    a = ising.Lattice(n)
    a.shuffle()
    fig = plt.figure()
    frames = []
    for nsteps in range(steps // freq):
        mc.sim(a, freq, t, freq=False)
        frame = plt.imshow(a.visualize(), animated=True, vmin=-1, vmax=1, cmap="afmhot")
        frames.append([frame])
    ani = animation.ArtistAnimation(fig, frames, interval=16.67, blit=True, repeat_delay=1000)
    ani.save("results/animation.mp4", fps=60)
    plt.show()


if __name__ == "__main__":
    # magnetization_map()
    # coolplot(50, 0.1)
    # anime(100, 0.1, freq=1000, steps=int(2e6))
    # heat_map()
    energy_map()
