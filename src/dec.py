import os
import sys
import numpy as np
import keras.backend as K
from keras.engine.topology import Layer, InputSpec
from keras.models import Model
from keras.layers import Dense, Dropout, Input
from sklearn.cluster import KMeans
from sklearn import metrics
import pickle
from util import ProgressBar
from time import sleep


def build_ae(hidden_neurons, rate = 0.5, act='relu'):
    n_layers = len(hidden_neurons) - 1
    X = Input(shape=(hidden_neurons[0],), name='input')

    for i in range(n_layers):
        if i == 0:
            H = Dense(hidden_neurons[i + 1], activation=act, name='encoder_%d' % (i+1))(X)
            H = Dropout(rate)(H)
        else:
            H = Dense(hidden_neurons[i + 1], activation=act, name='encoder_%d' % (i+1))(H)
            if i != 3:
                H = Dropout(rate)(H)

    for i in range(n_layers-1, -1, -1):
        if i == n_layers-1:
            Y = Dense(hidden_neurons[i], activation=act, name='decoder_%d' % i)(H)
            Y = Dropout(rate)(Y)

        else:
            Y = Dense(hidden_neurons[i], activation=act, name='decoder_%d' % i)(Y)
            if i != 0:
                Y = Dropout(rate)(Y)

    return Model(inputs=X, outputs=Y, name='AE'), Model(inputs=X, outputs=H, name='encoder')


class ClusteringLayer(Layer):
    def __init__(self, n_clusters, input_dim=None, weights=None, alpha=1.0, **kwargs):
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.input_dim = input_dim
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = [InputSpec(ndim=2)]

        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = [InputSpec(dtype=K.floatx(),
                                     shape=(None, input_dim))]
        self.clusters = self.add_weight((self.n_clusters, input_dim), initializer='glorot_uniform', name='clusters')
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True
        
    def call(self, x, mask=None):
        q = 1.0/(1.0 + (K.sum(K.square(K.expand_dims(x, 1) - self.clusters), axis=2) / self.alpha))
        q = q**((self.alpha+1.0)/2.0)
        q = K.transpose(K.transpose(q)/K.sum(q, axis=1))
        return q

    def get_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return (input_shape[0], self.clusters)

    def get_config(self):
        config = {'output_dim': self.clusters,
                  'input_dim': self.input_dim}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class DeepEmbeddingClustering(object):
    def __init__(self, X, y, hidden_neurons, n_clusters, alpha=1.0):
        self.X = X
        self.y = y
        self.n_clusters = n_clusters
        self.input_dim = hidden_neurons[0]
        self.ae, self.encoder = build_ae(hidden_neurons)
        print self.ae.summary()
        print self.encoder.summary()
        cluster_layer = ClusteringLayer(self.n_clusters,name='clustering')(self.encoder.output)
        self.model = Model(inputs = self.encoder.input, outputs = cluster_layer)

    def pretrain(self, batch_size, epochs, save_dir):
        self.ae.compile(optimizer='adam', loss = 'mse')
        self.ae.fit(self.X, self.X, batch_size = batch_size, epochs = epochs)
        self.ae.save(os.path.join(save_dir, 'pretrained_ae.h5'))
        print ('Finish pretraining and save the model to %s' % save_dir)

    def load_model(self, weights):
        self.model.load_weights(weights)

    def hidden_representations(self, X):
        return self.encoder.predict(X)

    def predict_classes(self, X):
        q = self.model.predict(X, verbose = 0)
        return q.argmax(1)

    def auxiliary_distribution(self, q):
        p = q ** 2 / q.sum(0)
        return (p.T / p.sum(1)).T

    def train(self, optimizer, batch_size, epochs, tol, update_interval, save_dir, shuffle):
        self.model.compile(optimizer = optimizer, loss = 'kld')
        print '================================================='
        print 'Initializing the cluster centers with k-means...'
        print '================================================='
        kmeans = KMeans(n_clusters = self.n_clusters, n_init = 20)
        y_pred = kmeans.fit_predict(self.hidden_representations(self.X))
        y_pred_last = np.copy(y_pred)
        self.model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])
        #print kmeans.cluster_centers_
        loss = 0
        index = 0
        index_array = np.arange(self.X.shape[0])

        print '================================================='
        print 'Start training ...'
        print '================================================='

        for ite in range(int(epochs)):
            #print self.model.layers[-1].clusters.get_value()
            if ite % update_interval == 0:
                if ite != 0 :
                    bar.done()
                bar = ProgressBar(update_interval, fmt=ProgressBar.FULL)

                q = self.model.predict(self.X, verbose=0)
                p = self.auxiliary_distribution(q)  # update the auxiliary target distribution p

                # evaluate the clustering performance
                y_pred = q.argmax(1)
                acc = np.round(metrics.accuracy_score(self.y, y_pred), 5)
                nmi = np.round(metrics.normalized_mutual_info_score(self.y, y_pred), 5)
                ari = np.round(metrics.adjusted_rand_score(self.y, y_pred), 5)
                loss = np.round(loss, 5)
                print '****************************************'
                print('Iter %d: acc = %.5f, nmi = %.5f, ari = %.5f' % (ite, acc, nmi, ari), ' ; loss=', loss)

                # check stop criterion
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                y_pred_last = np.copy(y_pred)
                if ite > 0 and delta_label < tol:
                    print '****************************************'
                    print('delta_label ', delta_label, '< tol ', tol)
                    print('Reached tolerance threshold. Stopping training.')
                    break

            if shuffle == True:
                np.random.shuffle(index_array)

            idx = index_array[index * batch_size: min((index+1) * batch_size, self.X.shape[0])]
            loss = self.model.train_on_batch(x=self.X[idx], y=p[idx])
            index = index + 1 if (index + 1) * batch_size <= self.X.shape[0] else 0
            bar.current += 1
            bar()
            sleep(0.1)

        # save the trained model
        print '****************************************'
        print('saving model to:', save_dir + '/DEC_model_final.h5')
        print '****************************************'
        #print self.model.layers[-1].clusters.get_value()
        self.model.save_weights(save_dir + '/DEC_model_final.h5')

        return y_pred

    def evaluate(self):
        y_pred = self.predict_classes(self.X)
        acc = np.round(metrics.accuracy_score(self.y, y_pred), 5)
        nmi = np.round(metrics.normalized_mutual_info_score(self.y, y_pred), 5)
        ari = np.round(metrics.adjusted_rand_score(self.y, y_pred), 5)
        print '================================================='
        print 'Start evaluate ...'
        print '================================================='
        print('acc = %.5f, nmi = %.5f, ari = %.5f.' % (acc, nmi, ari))
        return 0

    def test(self, X_test, y_test):
        print '================================================='
        print 'Start evaluate ...'
        print '================================================='
        y_pred = self.predict_classes(X_test)
        acc = np.round(metrics.accuracy_score(y_test, y_pred), 5)
        nmi = np.round(metrics.normalized_mutual_info_score(y_test, y_pred), 5)
        ari = np.round(metrics.adjusted_rand_score(y_test, y_pred), 5)
        print('acc = %.5f, nmi = %.5f, ari = %.5f.' % (acc, nmi, ari))
        return y_pred
