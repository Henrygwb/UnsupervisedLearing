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


def build_ae(hidden_neurons, rate = 0.5, act='relu'):
    n_layers = len(hidden_neurons) - 1
    X = Input(shape=(hidden_neurons[0],), name='input')

    for i in range(n_layers):
        if i == 0:
            H = Dense(hidden_neurons[i + 1], activation=act, name='encoder_%d' % (i+1))(X)
            H = Dropout(rate)(H)
        else:
            H = Dense(hidden_neurons[i + 1], activation=act, name='encoder_%d' % (i+1))(H)
            H = Dropout(rate)(H)

    for i in range(n_layers-1, -1, -1):
        if i == n_layers-1:
            Y = Dense(hidden_neurons[i], activation=act, name='decoder_%d' % i)(H)
            Y = Dropout(rate)(Y)

        else:
            Y = Dense(hidden_neurons[0], kernel_initializer=init, name='decoder_%d' % i)(Y)
            Y = Dropout(rate)(Y)

    return Model(inputs=X, outputs=Y, name='AE'), Model(inputs=X, outputs=H, name='encoder')


class ClusteringLayer(Layer):
    def __init__(self, n_cluster, input_dim=None, weights=None, alpha=1.0, **kwargs):
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_cluster = n_cluster
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

    def get_output_shape_for(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return (input_shape[0], self.output_dim)

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return (input_shape[0], self.output_dim)

    def get_config(self):
        config = {'output_dim': self.output_dim,
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
        cluster_layer = ClusteringLayer(self.n_clusters, alpha=alpha, name='clustering')(self.encoder)
        self.model = Model(inputs = self.encoder.input, outputs = cluster_layer)

    def pretrain(self, batch_size, epochs, save_dir):
        self.ae.compile(optimizer='adam', loss = 'mse')
        self.ae.fit(self.X, self.X, batch_size = batch_size, epochs = epochs)
        self.ae.save(os.path.join('save_dir', 'pretrained_ae.h5'))
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

    def train(self, optimizer, batch_size, epochs, tol, update_interval, save_dir):
        self.model.complie(optimizer = optimizer, loss = 'kld')
        print 'Initializing the cluster centers with k-means.'
        kmeans = KMeans(n_clusters = self.n_clusters, n_init = 20)
        y_pred = kmeans.fit_predict(self.encoder.predict(self.X))
        y_pred_last = np.copy(y_pred)
        self.model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])
        index_array = np.arange(self.X.shape[0])
        for ite in range(int(epochs)):
            if ite % update_interval == 0:
                q = self.model.predict(self.X, verbose=0)
                p = self.target_distribution(q)  # update the auxiliary target distribution p

                # evaluate the clustering performance
                y_pred = q.argmax(1)
                if y is not None:
                    acc = np.round(metrics.acc(self.y, y_pred), 5)
                    nmi = np.round(metrics.nmi(self.y, y_pred), 5)
                    ari = np.round(metrics.ari(self.y, y_pred), 5)
                    loss = np.round(loss, 5)
                    print('Iter %d: acc = %.5f, nmi = %.5f, ari = %.5f' % (ite, acc, nmi, ari), ' ; loss=', loss)

                # check stop criterion
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                y_pred_last = np.copy(y_pred)
                if ite > 0 and delta_label < tol:
                    print('delta_label ', delta_label, '< tol ', tol)
                    print('Reached tolerance threshold. Stopping training.')
                    break

            idx = index_array[index * batch_size: min((index+1) * batch_size, x.shape[0])]
            loss = self.model.train_on_batch(x=self.X[idx], y=p[idx])
            index = index + 1 if (index + 1) * batch_size <= x.shape[0] else 0
            ite += 1

        # save the trained model
        print('saving model to:', save_dir + '/DEC_model_final.h5')
        self.model.save_weights(save_dir + '/DEC_model_final.h5')

        return y_pred

    def evaluate(self):
        y_pred = self.predict_classes(self.X)
        acc = np.round(metrics.acc(self.y, y_pred), 5)
        nmi = np.round(metrics.nmi(self.y, y_pred), 5)
        ari = np.round(metrics.ari(self.y, y_pred), 5)
        print('Iter %d: acc = %.5f, nmi = %.5f, ari = %.5f' % (ite, acc, nmi, ari), ' ; loss=', loss)
        return 0

"""
if __name__ == "__main__":
    # setting the hyper parameters
    x, y = load_data(args.dataset)
    n_clusters = len(np.unique(y))

    update_interval = 140
    pretrain_epochs = 300
    init = VarianceScaling(scale=1. / 3., mode='fan_in',
                           distribution='uniform')  # [-limit, limit], limit=sqrt(1./fan_in)
    pretrain_optimizer = SGD(lr=1, momentum=0.9)
    update_interval = args.update_interval

    # prepare the DEC model
    dec = DEC(dims=[x.shape[-1], 500, 500, 2000, 10], n_clusters=n_clusters, init=init)

    if args.ae_weights is None:
        dec.pretrain(x=x, y=y, optimizer=pretrain_optimizer,
                     epochs=pretrain_epochs, batch_size=args.batch_size,
                     save_dir=args.save_dir)
    else:
        dec.autoencoder.load_weights(args.ae_weights)

    dec.model.summary()
    dec.compile(optimizer=SGD(0.01, 0.9), loss='kld')
    y_pred = dec.fit(x, y=y, tol=args.tol, maxiter=args.maxiter, batch_size=args.batch_size,
                     update_interval=update_interval, save_dir=args.save_dir)
    print('acc:', metrics.acc(y, y_pred))
    print('clustering time: ', (time() - t0))
"""