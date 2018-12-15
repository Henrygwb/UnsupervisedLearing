import numpy as np
import scipy

class ota(object):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.n = self.X.shape[0]
        self.p = self.X.shape[1]

    def genbssampls(self, n_bootstrep):
        """
        generate bootstrap sample with replacement.
        :param n_bootstrap: number of bootstrap sample
        :return: bootstap sample
        """
        idx = np.zeros((n_bootstrep, self.n))
        for i in xrange(n_bootstrep):
            idx[i,:] = np.random.choice(self.n, self.n, replace=True)
        return idx

    def align_bs(self):
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



