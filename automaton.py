# ===================================================================================== #
# Automaton simulation for forest fire model (Drossel & Schwable 1992)
# Author: Eddie Lee, edl56@cornell.edu
# ===================================================================================== #
import numpy as np
from numba import jit, njit


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
                # if left adjacent tile is tree and hasn't been burned yet
                if self.forest[(i-1)%self.n]==2 and newForest[(i-1)%self.n]==2:
                    if i!=0:
                        # since it's already been counted
                        nTrees -= 1
                    nFires += 1
                    newForest[(i-1)%self.n] = 1
                if self.forest[(i+1)%self.n]==2 and newForest[(i+1)%self.n]==2:
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
        if self.forest[-1]==1 and self.forest[0]==2 and self.forest[1]!=1:
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

        counter = 1
        forestHistory[0] = self.forest[:]
        nEmpty[0] = (self.forest==0).sum()
        nFires[0] = (self.forest==1).sum()
        nTrees[0] = (self.forest==2).sum()
        for i in range(n_iters//record_every):
            for j in range(record_every):
                nEmpty[counter], nFires[counter], nTrees[counter] = self.sweep_once()
                forestHistory[counter] = self.forest[:]
            counter += 1

        return nEmpty, nFires, nTrees, forestHistory
#end FF1D


class FF2D():
    """2D version of forest fire."""
    def __init__(self, n, f, p, rng=None, initial_config=None):
        """
        Parameters
        ----------
        n : int or tuple
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
        
        if not hasattr(n, '__len__'):
            self.n = (n,n)
        else:
            self.n = n
        self.f = f
        self.p = p
        
        if rng is None:
            self.rng = np.random
        else:
            self.rng = rng

        # 0 represents empty, 1 represents burning, 2 represents tree
        if not initial_config is None:
            assert initial_config.shape==self.n
            assert initial_config.ndim==2
            assert initial_config.dtype==np.uint8
            self.forest = initial_config
        else:
            # initialize with trees
            self.forest = np.zeros(self.n, dtype=np.uint8) + 2

    def get_neighbors(self, i, j):
        return (i, (j-1)%self.n[1]), (i, (j+1)%self.n[1]), ((i+1)%self.n[0], j), ((i-1)%self.n[0], j)

    def sweep_once(self):
        """Run a single sweep over the system.
        
        In order to avoid, multiple iterations through system, create a new copy of the
        system.
        """

        newForest = self.forest.copy()

        for i in range(self.n[0]):
            for j in range(self.n[1]):
                # if empty (0), can grow (2)
                if self.forest[i,j]==0 and self.rng.rand()<self.p:
                        newForest[i,j] = 2
                # if on fire, it can spread and extinguishes (0)
                elif self.forest[i,j]==1:
                    newForest[i,j] = 0
                    
                    # if neighbors are tree and haven't been burned yet
                    for ni,nj in self.get_neighbors(i,j):
                        if self.forest[ni,nj]==2:
                            newForest[ni,nj] = 1
                # if it hasn't already started burning from adjacent fire
                elif not newForest[i,j]==1:
                    # if a tree (2), it can spontaneously combust (1)
                    if self.rng.rand()<self.f:
                        newForest[i,j] = 1
        
        self.forest = newForest

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
        forestHistory = np.zeros((n_iters//record_every+1, self.n[0], self.n[1]))

        counter = 1
        forestHistory[0] = self.forest[:]
        nEmpty[0] = (self.forest==0).sum()
        nFires[0] = (self.forest==1).sum()
        nTrees[0] = (self.forest==2).sum()
        for i in range(n_iters//record_every):
            for j in range(record_every):
                self.sweep_once()
                stateCount = np.bincount(self.forest.ravel(), minlength=3)
                nEmpty[counter] = stateCount[0]
                nFires[counter] = stateCount[1]
                nTrees[counter] = stateCount[2]
                forestHistory[counter] = self.forest
            counter += 1

        return nEmpty, nFires, nTrees, forestHistory

    def clusters(self, forest=None, value=2):
        """Identify all connected clusters in the forest.
        
        Parameters
        ----------
        forest : ndarray, None
        value : int, 2

        Returns
        -------
        list of lists tuples
            Each internal list holds sets of coordinates that belong to single clusters.
        """
        
        if forest is None:
            forest = self.forest

        clusteredxy = set()
        clusters = []
        for i in range(self.n[0]):
            for j in range(self.n[1]):
                if forest[i,j]==value and not (i,j) in clusteredxy:
                    clusters.append([(i,j)])
                    clusteredxy.add((i,j))
                    toSearch = list(self.get_neighbors(i,j))
                    while toSearch:
                        thisi, thisj = toSearch.pop(0)
                        if forest[thisi, thisj]==value and not (thisi,thisj) in clusteredxy:
                            toSearch += list(self.get_neighbors(thisi, thisj))
                            clusteredxy.add((thisi,thisj))
                            clusters[-1].append((thisi,thisj))
        return clusters

    def grow_cluster(self, i, j, forest=None, value=2):
        """Identify connected cluster starting with (i,j).
        
        Parameters
        ----------
        i : int
        j : int
        forest : ndarray, None
        value : int, 2

        Returns
        -------
        list of tuples
            Each internal list holds sets of coordinates that belong to single clusters.
        """
        
        if forest is None:
            forest = self.forest
        assert forest[i,j]==value

        clusteredxy = set()
        cluster = [(i,j)]
        clusteredxy.add((i,j))
        toSearch = list(self.get_neighbors(i,j))
        while toSearch:
            thisi, thisj = toSearch.pop(0)
            if forest[thisi, thisj]==value and not (thisi,thisj) in clusteredxy:
                toSearch += list(self.get_neighbors(thisi, thisj))
                clusteredxy.add((thisi,thisj))
                cluster.append((thisi,thisj))
        return cluster
#end FF2D


class FastFF2D(FF2D):
    """2D version of forest fire with rapidly propagating fires and continuous time
    approximation between fires."""
    def __init__(self, n, f, p, rng=None, initial_config=None):
        """
        Parameters
        ----------
        n : int or tuple
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
        
        if not hasattr(n, '__len__'):
            self.n = (n,n)
        else:
            self.n = n
        assert (1/p)>=(100*self.n[0]) and (1/p)>=(100*self.n[1])
        self.f = f
        self.p = p
        
        if rng is None:
            self.rng = np.random
        else:
            self.rng = rng

        # 0 represents empty, 1 represents burning, 2 represents tree
        if not initial_config is None:
            assert initial_config.shape==self.n
            assert initial_config.ndim==2
            assert initial_config.dtype==np.uint8
            self.forest = initial_config
        else:
            # initialize with trees
            self.forest = np.zeros(self.n, dtype=np.uint8) + 2

    def sweep(self, n_iters):
        """Record n_iters fires. Ends with a successful lightning strike.

        Parameters
        ----------
        n_iters : int

        Returns
        -------
        list of lists of tuples
            fires coordinates
        ndarray
            (n_records, n) history of the forest state.
        """

        fires = []
        forestHistory = np.zeros((n_iters+1, self.n[0], self.n[1]))

        forestHistory[0] = self.forest[:]
        tf = 1/self.f/np.prod(self.n)
        counter = 1
        while len(fires)<n_iters:
            # fast forward time til next lightning strike
            t = self.rng.exponential(scale=tf)
            # fast forward concomitant tree growth
            ix = self.forest==0
            self.forest[ix] = (self.rng.exponential(scale=1/self.p, size=ix.sum())<t)*2.

            # lightning strikes a random site
            siteix = self.rng.randint(self.n[0]), self.rng.randint(self.n[1])
            
            if self.forest[siteix[0],siteix[1]]==2:
                burntCluster = self.grow_cluster(siteix[0], siteix[1])
                ix = list(zip(*burntCluster))
                self.forest[ix[0],ix[1]] = 0
                fires.append(burntCluster)
                forestHistory[counter] = self.forest[:,:]
                counter += 1

        return fires, forestHistory

    def clusters(self, forest=None, value=2):
        """Identify all connected clusters in the forest.
        
        Parameters
        ----------
        forest : ndarray, None
        value : int, 2

        Returns
        -------
        list of lists tuples
            Each internal list holds sets of coordinates that belong to single clusters.
        """
        
        if forest is None:
            forest = self.forest

        clusters = []
        clusteredxy = set()
        for i in range(self.n[0]):
            for j in range(self.n[1]):
                if forest[i,j]==value and not (i,j) in clusteredxy:
                    c, cxy = self.grow_cluster(i, j, forest, value, return_clustered=True)
                    clusters.append(c)
                    clusteredxy.union(cxy)
        return clusters

    def grow_cluster(self, i, j, forest=None, value=2, return_clustered=False):
        """Identify connected cluster starting with (i,j).

        Parameters
        ----------
        i : int
        j : int
        forest : ndarray, None
        value : int, 2
        return_clustered : bool, False

        Returns
        -------
        list of tuples
            Each internal list holds sets of coordinates that belong to single clusters.
        set, optional
        """

        if forest is None:
            forest = self.forest
        
        if return_clustered:
            return jit_grow_2dcluster(i, j, forest, value)
        return jit_grow_2dcluster(i, j, forest, value)[0]

    def _grow_cluster(self, i, j, forest=None, value=2):
        """Identify connected cluster starting with (i,j).

        SLOW, DEPRECATED VERSION.
        
        Parameters
        ----------
        i : int
        j : int
        forest : ndarray, None
        value : int, 2

        Returns
        -------
        list of tuples
            Each internal list holds sets of coordinates that belong to single clusters.
        """
        
        if forest is None:
            forest = self.forest
        assert forest[i,j]==value

        clusteredxy = set()
        cluster = [(i,j)]
        clusteredxy.add((i,j))
        toSearch = list(self.get_neighbors(i,j))
        while toSearch:
            thisi, thisj = toSearch.pop(0)
            if forest[thisi, thisj]==value and not (thisi,thisj) in clusteredxy:
                toSearch += list(self.get_neighbors(thisi, thisj))
                clusteredxy.add((thisi,thisj))
                cluster.append((thisi,thisj))
        return cluster
#end FastFF2D


@njit
def jit_grow_2dcluster(i, j, forest, value):
    """Identify connected cluster starting with (i,j).
    
    Parameters
    ----------
    i : int
    j : int
    forest : ndarray
    value : int

    Returns
    -------
    list of tuples
        Each internal list holds sets of coordinates that belong to single clusters.
    """
    
    assert forest[i,j]==value

    clusteredxy = set()  # use set to speed up search
    cluster = [(i,j)]
    clusteredxy.add((i,j))
    toSearch = jit_get_2dneighbors(forest.shape[0], forest.shape[1], i, j)
    while toSearch:
        thisi, thisj = toSearch.pop(0)
        if forest[thisi, thisj]==value and not (thisi,thisj) in clusteredxy:
            toSearch += jit_get_2dneighbors(forest.shape[0], forest.shape[1], thisi, thisj)
            clusteredxy.add((thisi,thisj))
            cluster.append((thisi,thisj))
    return cluster, clusteredxy

@njit
def jit_get_2dneighbors(h, w, i, j):
        return [(i, (j-1)%w), (i, (j+1)%w), ((i+1)%h, j), ((i-1)%h, j)]
