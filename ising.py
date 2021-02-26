import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from numba import jit


class Lattice:
    """Implements a 2d Ising lattice

    Implements an Ising model in a field for Monte Carlo with a Hamiltonian H = -J sum_{ij}(s_i,s_j) - h sum_{j}(s_j)
    where s_i is an individual spin on the lattice, h is the field, and J is the interaction coefficient

    Note: external field not implemented

    Attributes:
        n: size of the nxn lattice
        observables: names of values that the model can report
    """

    def __init__(self, n: int = 2, h: float = 0.0, j: float = 1.0) -> None:
        """Initializes the lattice with all spins being 1

        Args:
            n: lattice size (linear)
            h: external field
            j: spin-spin interaction constant
        """
        if h != 0:
            raise NotImplementedError("External field not yet implemented")
        self.n = n
        self.h = h
        self.j = j
        self.energy = 0
        self.magnetism = n ** 2
        self.observables = "TotEng AveEng Mgnt"
        self.states = np.ones((self.n, self.n))
        self._update_energy("full")

    def shuffle(self) -> None:
        """Randomizes spin states"""
        for i in range(self.n):
            for j in range(self.n):
                self.states[i, j] = np.random.choice([-1, 1])
        self._update_magnetism()
        self._update_energy("full")

    def bc(self, x: int) -> int:
        """Apply periodic boundaries to a coordinate

        Args:
            x: Integer spin position

        Returns:
            Position with respect to periodic boundaries
        """
        if x >= self.n:
            return x - self.n
        if x < 0:
            return x + self.n
        else:
            return x

    def _update_energy(self, kind="full", i=0, j=0) -> float:
        """Updates lattice energy

        Recalculates lattice energy based on nearest neighbours interactinos either from zero, of update the previous
        value based on the change made

        Args:
            kind: "full" to recalculate energy. "single" to update it
            i, j: position of the flipped spin. used when kind is "single"

        Returns:
            Energy of the lattice
        """
        if kind == "full":
            h = 0
            for i in range(self.n):
                for j in range(self.n):
                    left = (i - 1 if i > 0 else self.n - 1)
                    right = (i + 1 if i < self.n - 1 else 0)
                    up = (j + 1 if j < self.n - 1 else 0)
                    down = (j - 1 if j > 0 else self.n - 1)

                    h -= 0.5 * self.j * self.states[i, j] * (self.states[left, j] + self.states[right, j] +
                                                             self.states[i, up] + self.states[i, down])
        elif kind == "single":
            h = self.energy
            delta = 2 * self._spin_energy(i, j)
            h += delta

        self.energy = h
        return h

    def _update_magnetism(self) -> float:
        """Updates average magnetism of the lattice"""
        self.magnetism = self.states.mean()
        return self.magnetism

    def _spin_energy(self, i: int, j: int) -> float:
        """Calculate energy of one spin

        Args:
            i,j : spin position

        Returns:
            Energy of the spin
        """
        delta = -self.j * self.states[i, j] * (self.states[self.bc(i - 1), j] + self.states[self.bc(i + 1), j] +
                                               self.states[i, self.bc(j + 1)] + self.states[i, self.bc(j - 1)])

        return delta

    def move(self, spin) -> None:
        """Performs a mc move for the lattice

        Flips the specified spin, update lattice energy and magnetism

        Args:
            spin: number of the spin in a flattened array of states
        """
        i = spin // self.n
        j = spin % self.n
        self.states[i, j] *= -1
        self._update_energy("single", i, j)
        self._update_magnetism()

    def observe(self):
        return self.energy, self.energy / self.n ** 2, self.magnetism

    def visualize(self, kind: str, filename: str = "") -> None:
        """Visualizes the lattice

        Visualizes the lattice state either in terminal or graphically via seaborn heatmap.

        Args:
            kind: "basic" to print states to console, "cool" to use seaborn.heatmap
            filename: Optional; filename to save the plot to. Leave empty to use interactive plot.

        Raises:
            ValueError: incorrect plot kind
        """
        if kind == "basic":
            for i in range(self.n):
                for j in range(self.n):
                    print(str(int(self.states[i, j])) + " ", end='')
                print()

        elif kind == "cool":
            sns.heatmap(self.states, vmin=-1, vmax=1)
            if filename:
                plt.savefig(filename)
            else:
                plt.show()

        else:
            raise ValueError("Wrong plot type")
