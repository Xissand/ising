import random
import numpy as np


# Note: maybe this should be provided by the model class
def update(model: object, t: float) -> None:
    """Performs a MC move
    
    Chooses a random state to change, measures energy before and after. If energy change is negative, the move is 
    accepted, otherwise a move's probability exp(-dE/kT) is compared to a uniformly distributed float.
    
    Args:
        model: model object
        t: temperature
    """
    states = model.states.ravel()
    size = len(states)
    i = random.randint(0, size - 1)
    e0 = model.spin_energy(i)
    states[i] *= -1
    e1 = model.spin_energy(i)
    if e1 - e0 < 0:
        # print("Accepted")
        return
    else:
        acc = random.random()
        if np.exp(-(e1 - e0) / t) > acc:
            # print("Accepted")
            return
        else:
            states[i] *= -1


def sim(model: object, nsteps: int, t: float):
    """Performs Monte Carlo simulations

    Args:
        model: the model to be simulated
        nsteps: number of steps
        t: temperature

    Returns:
        nothing good
    """
    random.seed()
    # TODO: move magnetism calculation to main
    d = model.states.flatten()
    for i in range(nsteps):
        update(model, t)
        d += model.states.flatten()
        # print(f"Energy: {model.energy(): .4}")
    d /= nsteps
    # print("AVE:  " + str(d.mean()))
    return d.mean()
