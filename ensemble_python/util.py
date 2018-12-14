import numpy as np

def gendata(n, p, c, k):
    """
    :param n: number of data
    :param p: dimension
    :param c: number of clusters
    :param k: number of small clusters
    :return:
    """
    pp = np.sort(np.random.uniform(0,c-1,3))
    pp = np.append(pp,[1]) - np.insert(pp,0,0)
    mu = np.zeros((p, c))
    mu

