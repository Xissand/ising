import random
import numpy as np
from scipy.ndimage.filters import uniform_filter1d


# Note: maybe this should be provided by the model class
def _update(model: object, t: float) -> None:
    """Performs a MC move

    Chooses a random state to change, measures energy before and after. If energy change is negative, the move is
    accepted, otherwise a move's probability exp(-dE/kT) is compared to a uniformly distributed float.

    Args:
        model: model object
        t: temperature
    """
    size = model.n ** 2
    i = random.randint(0, size - 1)
    e0 = model.energy
    model.move(i)
    e1 = model.energy
    if e1 - e0 < 0:
        # print("Accepted")
        return
    else:
        acc = random.random()
        if np.exp(-(e1 - e0) / t) > acc:
            # print("Accepted")
            return
        else:
            model.move(i)


def sim(model: object, nsteps: int, t: float, freq: int = 10000, ave: bool = False):
    """Performs Monte Carlo simulations

    Args:
        model: the model to be simulated
        nsteps: number of steps
        t: temperature
        freq: number of steps between console output. Set to 0 to disable output.

    Returns:
        nothing good
    """
    random.seed()
    # print(nsteps)
    if freq:
        print("Step " + model.observables)
    if ave:
        log = []
    for i in range(nsteps):
        _update(model, t)
        if freq:
            if i % freq == 0:
                v = model.observe()
                print(f"{i:>12}", end="")
                for value in v:
                    print(f"{value:>12}", end="")
                print()
        if ave:
            v = model.observe()
            log.append(v)

    if ave:
        averages = np.mean(log, axis=0)
        stds = np.std(log, axis=0)
        # TODO: return a dictionary instead
        return averages, stds
