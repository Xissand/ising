import random
import numpy as np


def function():
    print("this is a start")


def step(model):
    model.states[0, 0] += 1


def update(model, T):
    states = model.states.ravel()
    size = len(states)
    i = random.randint(0, size - 1)
    #print("Trying spin: " + str(i))
    e0 = model.energy()
    states[i] *= -1
    e1 = model.energy()
    if e1 - e0 < 0:
        # print("Accepted")
        return
    else:
        acc = random.random()
        if np.exp(-(e1 - e0) / T) > acc:
            #    print("Accepted")
            return
        else:
            states[i] *= -1


def sim(model, nsteps, T):
    random.seed()
    d = model.states.flatten()
    for i in range(nsteps):
        update(model, T)
        #print(i)
        #model.visualize("cool")
        d += model.states.flatten()
        # print(f"Energy: {model.energy(): .4}")
    d /= nsteps
    #print("AVE:  " + str(d.mean()))
    return d.mean()
