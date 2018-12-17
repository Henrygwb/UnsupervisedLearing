import numpy as np
import scipy
import sklearn
from sklearn import datasets
from sklearn import cluster
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.mixture import GaussianMixture as Mclust
from scipy.stats import multivariate_normal as mvnorm

def gendata(n, p, c):
    """
    :param n: number of data
    :param p: dimension
    :param c: number of clusters
    :return:
    """
    pp = np.sort(np.random.uniform(0,1,c-1))
    pp = np.append(pp,[1]) - np.insert(pp,0,0)
    mu = np.random.uniform(-5,5,c*p).reshape((p, c))
    sigma = np.zeros((p,p,c))
    for i in xrange(c):
        scale = datasets.make_spd_matrix(n_dim = p)
        sigma[:,:,i] = scipy.stats.invwishart.rvs(df = 5*p, scale = scale)
    rsample = np.random.multinomial(1, pp, size=n)
    y = np.array([np.where(i == 1)[0][0] for i in rsample])
    X = np.array([mvnorm.rvs(mean=mu[:,i], cov=sigma[:,:,i]) for i in y])

    return X, y

def genbssamples(n_bootstrep, num_samples):
    """
    generate bootstrap sample with replacement.
    :param n_bootstrap: number of bootstrap sample
    :return: bootstap sample
    """
    idx = np.zeros((n_bootstrep, num_samples))
    for i in xrange(n_bootstrep):
        idx[i,:] = np.random.choice(num_samples, num_samples, replace=True).astype('int32')
    return idx


##### clustering approach
class Cluster(object):
    def __init__(self, X_train, X_test):
        self.X_train = X_train
        self.X_test = X_test

    ## k-means
    def kmeans(self, n_clusters):
        km = cluster.k_means(n_clusters=n_clusters).fix(self.X_train)
        self.y_pred = km.predict(self.X_test)
        return self.y_pred

    ## Mclust for GMM
    def mclust(self, n_clusters):
        gmm = Mclust(n_components=n_clusters).fit(self.X_train)
        self.y_pred = gmm.predict(self.X_test)
        return self.y_pred

    ## Hierarchical clustering with knn
    def knn(self, n_clusters):
        y_train = cluster.AgglomerativeClustering(n_clusters=n_clusters).fit_predict(self.X_train)
        kn = KNN.fit(self.X_train, y_train)
        self.y_pred = kn.predict(self.X_test)
        return self.y_pred

    ## dbscan with knn
    def mclust(self):
        y_train = cluster.dbscan(algorithm='auto', eps=3, leaf_size=30, metric='euclidean',
                                 metric_params=None, min_samples=2, n_jobs=None, p=None).fix_predict(self.X_train)
        kn = KNN.fit(self.X_train, y_train)
        self.y_pred = kn.predict(self.X_test)
        return self.y_pred

### Test gendata
# n = 10
# p = 2
# c = 4
# X, y = gendata(n,p,c)
