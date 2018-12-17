import numpy as np
import scipy

class ota(object):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.n = self.X.shape[0]
        self.p = self.X.shape[1]

    def align_bs(self, yb, n_boostrep):
        """
        Align bootstrap samples
        :return:
        """

        return 0

    def get_repre(self):
        """
        Find representative samples
        :return:
        """
        return 0

    def statforclst(self):
        """
        Statistics for stability of a result of clustering
        :return:
        """
        return 0

    def entropy(self):
        """
        Overall clustering stability
        :return:
        """
        return 0

    def confset(self):
        """
        Confident set
        :return:
        """
        return 0



