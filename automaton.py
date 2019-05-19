# ===================================================================================== #
# Automaton simulation for forest fire model (Drossel & Schwable 1992)
# Author: Eddie Lee, edl56@cornell.edu
# ===================================================================================== #
import numpy as np
from numba import jit


class FF1D():
    """1D version of forest fire."""
    def __init__(self, n, f, p, rng=None, initial_config=None):
        """
        Parameters
        ----------
        n : int
            system size
        f : float
            lightning rate
        p : float
            regrowth rate
        rng : np.random.RandomState
            Random number generator.
        initial_config : ndarray, None
            Initial state of forest.
        """

        assert n>1
        assert 0<f<=1
        assert 0<p<=1

        self.n = n
        self.f = f
        self.p = p
        
        if rng is None:
            self.rng = np.random
        else:
            self.rng = rng

        # 0 represents empty, 1 represents burning, 2 represents tree
        if not initial_config is None:
            assert initial_config.size==n
            assert initial_config.ndim==1
            assert initial_config.dtype==np.uint8
            self.forest = initial_config
        else:
            self.forest = np.zeros(n, dtype=np.uint8) + 2

    def sweep_once(self):
        """Run a single sweep over the system.
        
        In order to avoid, multiple iterations through system, create a new copy of the
        system.
        """

        newForest = self.forest.copy()
        nTrees = 0
        nFires = 0
        nEmpty = 0

        for i in range(self.n):
            # if empty (0), can grow (2)
            if self.forest[i]==0:
                if self.rng.rand()<self.p:
                    newForest[i] = 2
                    nTrees += 1
                else:
                    nEmpty += 1
            # if on fire, it can spread and extinguishes (0)
            elif self.forest[i]==1:
                newForest[i] = 0
                nEmpty += 1
                if self.forest[(i-1)%self.n]==2 and newForest[(i-1)%self.n]==2:
                    nTrees -= 1
                    nFires += 1
                    newForest[(i-1)%self.n] = 1
                if self.forest[(i+1)%self.n]==2:
                    newForest[(i+1)%self.n] = 1
                    nFires += 1
            # if it hasn't already started burning from adjacent fire
            elif not newForest[i]==1:
                # if a tree (2), it can spontaneously combust (1)
                if self.rng.rand()<self.f:
                    newForest[i] = 1
                    nFires += 1
                else:
                    nTrees += 1
        
        # tree at i=0 will be double-counted with periodic boundary conditions
        if self.forest[-1]==1 and self.forest[0]==2:
            nTrees -= 1

        self.forest = newForest
        return nEmpty, nFires, nTrees

    def sweep(self, n_iters, record_every=0):
        """Sweep through entire lattice n_iters times. Options to record various
        quantities.

        Parameters
        ----------
        n_iters : int,
        record_every : int, 0
            Record every n steps. If 0, no records are taken.

        Returns
        -------
        ndarray
            nEmpty as measured during simulation.
        ndarray
            nFires
        ndarray
            nTrees
        ndarray
            (n_records, n) history of the forest state.
        """

        if record_every==0:
            for i in range(n_iters):
                self.sweep_once();
            return

        nEmpty = np.zeros(n_iters//record_every+1, dtype=int)
        nFires = np.zeros(n_iters//record_every+1, dtype=int)
        nTrees = np.zeros(n_iters//record_every+1, dtype=int)
        forestHistory = np.zeros((n_iters//record_every+1, self.n))

        counter = 0
        forestHistory[0] = self.forest[:]
        for i in range(n_iters//record_every):
            for j in range(record_every):
                nEmpty[counter+1], nFires[counter+1], nTrees[counter+1] = self.sweep_once()
                forestHistory[counter+1] = self.forest[:]
            counter += 1

        return nEmpty, nFires, nTrees, forestHistory
#end FF1D
