from __future__ import print_function
import sys
import re
import scipy
from sklearn import datasets
from sklearn import cluster
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.mixture import GaussianMixture as Mclust
from scipy.stats import multivariate_normal as mvnorm
import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, jaccard_similarity_score
from keras.preprocessing.image import ImageDataGenerator

class metrics(object):
    def nmi(self, y_true, y_pred):
        return normalized_mutual_info_score(y_true, y_pred)
    def ari(self, y_true, y_pred):
        return adjusted_rand_score(y_true, y_pred)
    def jac(self, y_true, y_pred):
        return jaccard_similarity_score(y_true, y_pred)
    def acc(self, y_true, y_pred):
        """
        Calculate clustering accuracy. Require scikit-learn installed

        # Arguments
            y: true labels, numpy.array with shape `(n_samples,)`
            y_pred: predicted labels, numpy.array with shape `(n_samples,)`

        # Return
            accuracy, in [0,1]
        """
        y_true = y_true.astype(np.int64)
        assert y_pred.size == y_true.size
        D = max(y_pred.max(), y_true.max()) + 1
        w = np.zeros((D, D), dtype=np.int64)
        for i in range(y_pred.size):
            w[y_pred[i], y_true[i]] += 1
        from sklearn.utils.linear_assignment_ import linear_assignment
        ind = linear_assignment(w.max() - w)
        return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

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
        idx[i,:] = np.random.choice(num_samples, num_samples, replace=True)
    return idx.astype('int32')

def genaugbs(X, y, augment_size=10000):
    image_generator = ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.05,
        width_shift_range=0.05,
        height_shift_range=0.05,
        horizontal_flip=False,
        vertical_flip=False,
        data_format="channels_last",
        zca_whitening=True)

    randidx = np.random.choice(X.shape[0], X.shape[0], replace=True)
    y = y[randidx]
    X = X[randidx]
    X = X.reshape(X.shape[0], 28, 28)
    X = np.expand_dims(X, -1)
    # fit data for zca whitening
    image_generator.fit(X, augment=True)

    # get transformed images
    randidx = np.random.randint(X.shape[0], size=augment_size)
    x_augmented = X[randidx].copy()
    y_augmented = y[randidx].copy()
    x_augmented = image_generator.flow(x_augmented, np.zeros(augment_size),
                                       batch_size=augment_size, shuffle=False).next()[0]
    # append augmented data to trainset
    X = np.concatenate((X, x_augmented))
    y = np.concatenate((y, y_augmented))
    X = X.reshape(X.shape[0], 784)
    return X, y


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

class ProgressBar(object):
    DEFAULT = 'Progress: %(bar)s %(percent)3d%%'
    FULL = '%(bar)s %(current)d/%(total)d (%(percent)3d%%) %(remaining)d to go'
    def __init__(self, total, width=40, fmt=DEFAULT, symbol='=',
                 output=sys.stderr):
        assert len(symbol) == 1

        self.total = total
        self.width = width
        self.symbol = symbol
        self.output = output
        self.fmt = re.sub(r'(?P<name>%\(.+?\))d',
            r'\g<name>%dd' % len(str(total)), fmt)

        self.current = 0

    def __call__(self):
        percent = self.current / float(self.total)
        size = int(self.width * percent)
        remaining = self.total - self.current
        bar = '[' + self.symbol * size + ' ' * (self.width - size) + ']'

        args = {
            'total': self.total,
            'bar': bar,
            'current': self.current,
            'percent': percent * 100,
            'remaining': remaining
        }
        print('\r' + self.fmt % args, file= self.output, end='')

    def done(self):
        self.current = self.total
        self()
        print('', file=self.output)

### Test gendata
# n = 10
# p = 2
# c = 4
# X, y = gendata(n,p,c)
