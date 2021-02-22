import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class Lattice:
    """Implements a 2d Ising lattice

    Implements an Ising model in a field for Monte Carlo with a Hamiltonian H = -J sum_{ij}(s_i,s_j) - h sum_{j}(s_j)
    where s_i is an individual spin on the lattice, h is the field, and J is the interaction coefficient

    Attributes:
        n: size of the nxn lattice
        states: an nxn ndarray of spins
    """

    def __init__(self, n: int = 2, h: float = 0.0, j: float = 1.0) -> None:
        """Initializes the lattice with all spins being 1

        Args:
            n: lattice size (linear)
            h: external field
            j: spin-spin interaction constant
        """
        self.n = n
        self.h = h
        self.j = j
        self.states = np.ones((self.n, self.n))

    def shuffle(self) -> None:
        """Randomizes spin states"""
        for i in range(self.n):
            for j in range(self.n):
                self.states[i, j] = np.random.choice([-1, 1])

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

    def energy(self) -> float:
        """Calculates lattice energy

        Calculates energy of the lattice according to either the full hamiltonian, nearest neighbours or a cutoff.
        Deprecated in favor of spin_energy

        Returns: Energy of the lattice

        """
        h = 0
        """ Full energy
        s = self.states.ravel()
        for i in s:
            h -= 0.5 * self.j * np.sum(np.multiply(i, s))
        """
        """ Cutoff
        # FIX: This doesn't work
        cutoff = 1
        for i in range(self.n):
            for j in range(self.n):
                for k in range(0, cutoff + 1):
                    for kk in range(0, cutoff - k + 1):
                        if (k + kk) == 0:
                            continue
                        h -= 0.5 * self.j * self.states[i, j] * self.states[self.bc(i + k), self.bc(j + kk)]
                        h -= 0.5 * self.j * self.states[i, j] * self.states[self.bc(i + k), self.bc(j - kk)]
                        h -= 0.5 * self.j * self.states[i, j] * self.states[self.bc(i - k), self.bc(j + kk)]
                        h -= 0.5 * self.j * self.states[i, j] * self.states[self.bc(i - k), self.bc(j - kk)]
        """  # Nearest neighbour
        for i in range(self.n):
            for j in range(self.n):
                left = (i - 1 if i > 0 else self.n - 1)
                right = (i + 1 if i < self.n - 1 else 0)
                up = (j + 1 if j < self.n - 1 else 0)
                down = (j - 1 if j > 0 else self.n - 1)

                h -= 0.5 * self.j * self.states[i, j] * (self.states[left, j] + self.states[right, j] +
                                                         self.states[i, up] + self.states[i, down])

        return h

    # TODO: rename this to something more generalized
    # TODO: create a general model class
    def spin_energy(self, k: int) -> float:
        """Calculate energy of one spin

        Args:
            k: 1d number of spin from a flattened array

        Returns:
            Energy of the spin, considering only nearest neighbour interactions.
        """
        i = k // self.n
        j = k % self.n

        delta = -self.j * self.states[i, j] * (self.states[self.bc(i - 1), j] + self.states[self.bc(i + 1), j] +
                                               self.states[i, self.bc(j + 1)] + self.states[i, self.bc(j - 1)])

        return delta

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
