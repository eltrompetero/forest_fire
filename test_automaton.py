# ===================================================================================== #
# Automaton simulation for forest fire model (Drossel & Schwable 1992)
# Author: Eddie Lee, edl56@cornell.edu
# ===================================================================================== #
from .automaton import *


def test_FF1D():
    n = 10
    ff = FF1D(n, .01, .1)
    nEmpty, nFires, nTrees, _ = ff.sweep(1000, record_every=1)
    assert ((nEmpty+nFires+nTrees)==n).all()
