from __future__ import print_function
import sys
import os
import re
import scipy
from scipy import io
from sklearn.decomposition import PCA
from sklearn import datasets
from sklearn import cluster
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.mixture import GaussianMixture as Mclust
from scipy.stats import multivariate_normal as mvnorm
import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, jaccard_similarity_score
from keras.preprocessing.image import ImageDataGenerator
import Cluster_Ensembles as CE
import collections
from sklearn.metrics import precision_recall_fscore_support
from bhtsne import tsne
from sklearn.preprocessing import Normalizer

def genaugbs(X, y, augment_size=10000, augment = False):
    randidx = np.random.choice(X.shape[0], X.shape[0], replace=True)
    y = y[randidx]
    X = X[randidx]

    if augment == True:
        image_generator = ImageDataGenerator(
            rotation_range=10,
            zoom_range=0.05,
            width_shift_range=0.05,
            height_shift_range=0.05,
            horizontal_flip=False,
            vertical_flip=False,
            data_format="channels_last",
            zca_whitening=True)

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

def load_data(path, dataset):
    if dataset == 'mnist':
        path = os.path.join(path, 'mnist')
        file = os.path.join(path, 'train')
        X = io.loadmat(file)['train_x']
        y = io.loadmat(file)['train_y']
        file = os.path.join(path, 'testall')
        X = np.concatenate((X, io.loadmat(file)['test_x']))
        y = np.concatenate((y, io.loadmat(file)['test_y']))
        n_clusters = y.shape[1]
        y = np.array([np.where(r == 1)[0][0] for r in y])
    elif dataset == 'rcv':
        path = os.path.join(path, 'rcv')
        file = os.path.join(path, 'rcv1_4')
        X = io.loadmat(file)['X']
        y = io.loadmat(file)['Y']
        n_clusters = y.shape[1]
        y = np.array([np.where(r == 1)[0][0] for r in y])
    elif dataset == 'malware':
        path = os.path.join(path, 'malware')
        file = os.path.join(path, 'malware_data')
        X = io.loadmat(file)['X'][0:1000]
        y = io.loadmat(file)['y'].flatten()[0:1000]
        n_clusters = 5
    return X, y, path


class DimReduce(object):
    def __init__(self, X):
        self.X = X
    def pca(self):
        x_low = PCA(n_components=2).fit_transform(self.X)
        return x_low
    def bh_tsne(self):
        return tsne(self.X.astype('float64'))

class ensemble(object):
    def __init__(self, yb, n_bootstrap, num_sample):
        self.yb = yb
        self.n_bootstrap = n_bootstrap
        self.num_sample = num_sample

    def CSPA(self):
        yb = self.yb.reshape(self.n_bootstrap, self.num_sample)
        n_class = np.max(yb)+1
        return CE.cluster_ensembles(yb, N_clusters_max = n_class, verbose=0)

    def MCLA(self):
        yb = self.yb.reshape(self.n_bootstrap, self.num_sample)
        n_class = np.max(yb)+1
        return CE.MCLA('Cluster_Ensembles.h5' ,yb, N_clusters_max = n_class, verbose=0)

    def voting(self):
        yb = self.yb.reshape(self.n_bootstrap, self.num_sample)
        y_mean = np.zeros((self.num_sample))
        for i in xrange(self.num_sample):
            tmp = yb[:, i]
            a = collections.Counter(tmp)
            max = a.most_common()[0]
            y_mean[i] = max[0]
        return y_mean.astype('int')

class metrics(object):

    def nmi(self, y_true, y_pred):
        return normalized_mutual_info_score(y_true, y_pred)

    def ari(self, y_true, y_pred):
        return adjusted_rand_score(y_true, y_pred)

    def jac(self, y_tmp, y_ref):
        inst = np.intersect1d(y_tmp, y_ref)
        unin = np.union1d(y_tmp, y_ref)
        jaccard_dist = float(inst.shape[0])/float(unin.shape[0])
        return jaccard_dist

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
    def presion_recall_fscore(self, y_true, y_pred):
        prec, recall, fscore, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
        return prec, recall, fscore

##### clustering approach
class Cluster(object):
    def __init__(self, X_train, X_test):
        self.X_train = X_train
        self.X_test = X_test

    ## k-means
    def kmeans(self, n_clusters):
        km = cluster.KMeans(n_clusters=n_clusters).fit(self.X_train)
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
    def dbscan(self):
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

def normalize(X_train, X_test):
    transformer = Normalizer().fit(X_train)
    X_train = transformer.transform(X_train)
    X_test = transformer.transform(X_test)
    return X_train, X_test