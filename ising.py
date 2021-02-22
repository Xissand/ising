import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class Lattice:
    """Implements a 2d Ising lattice

    Implements an Ising model in a field for Monte Carlo
    with a Hamiltonian H = -J sum_{ij}(s_i,s_j) - h sum_{j}(s_j)
    where s_i is an individual spin on the lattice,
    h is the field, and J is the interaction coefficient

    Attributes:
        n: size of the nxn lattice
        h: external field
        j: spin-spin interaction constant
        states: an nxn ndarray of spins
     """

    def __init__(self, n=2, h=0, j=1):
        """Initializes the lattice with all spins being 1"""
        self.n = n
        self.h = h
        self.j = j
        self.states = np.ones((self.n, self.n))

    def shuffle(self):
        for i in range(self.n):
            for j in range(self.n):
                self.states[i, j] = np.random.choice([-1, 1])

    def check(self):
        a = (0, 0)
        print(self.states[a])

    def bc(self, x):
        if x >= self.n:
            return x - self.n
        if x < 0:
            return x + self.n
        else:
            return x

    def energy(self):
        h = 0
        """ Full energy
        s = self.states.ravel()
        for i in s:
            h -= 0.5 * self.j * np.sum(np.multiply(i, s))
        """
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
        """ Nearest neighbour
        for i in range(self.n):
            for j in range(self.n):
                left = (i - 1 if i > 0 else self.n - 1)
                right = (i + 1 if i < self.n - 1 else 0)
                up = (j + 1 if j < self.n - 1 else 0)
                down = (j - 1 if j > 0 else self.n - 1)

                h -= 0.5 * self.j * self.states[i, j] * (
                        self.states[left, j] + self.states[right, j] +
                        self.states[i, up] + self.states[i, down]
                )
        """
        return h

    def visualize(self, kind, filename=""):
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
            sns.heatmap(self.states, cbar=False)
            if filename:
                plt.savefig(filename)
            else:
                plt.show()
        else:
            raise ValueError("Wrong plot kind")
