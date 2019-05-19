# ===================================================================================== #
# Automaton simulation for forest fire model (Drossel & Schwable 1992)
# Author: Eddie Lee, edl56@cornell.edu
# ===================================================================================== #
from matplotlib import colors


def regular_cmap():
    """
    Brown : empty
    Red : fire
    Green : tree
    """

    return colors.ListedColormap(['C5','C3','C2'])
