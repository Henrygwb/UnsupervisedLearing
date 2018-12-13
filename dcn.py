import os
import sys
import tensorflow as tf 
import tensorflow.contrib.slim as slim
import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics
import scipy
from keras.models import Model
from keras.layers import Dense, Dropout, Input

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
    		Y = Dense(dims[0], kernel_initializer=init, name='decoder_%d' % i)(Y)
	    	Y = Dropout(rate)(Y)

    return Model(inputs=X, outputs=Y, name='AE'), Model(inputs=X, outputs=H, name='encoder')


class DeepClusteringNetwork(object):
    def __init__(self,
    			 X,
    			 y,
                 hidden_neurons,
                 n_clusters,
                 alpha=1.0):
    	self.X = X
    	self.y = y
        self.n_clusters = n_clusters
        self.input_dim = hidden_neurons[0]
        self.ae, self.model = build_ae(hidden_neurons)

    def pretrain(self, batch_size, epochs, save_dir):
    	self.ae.compile(optimizer='adam', loss = 'mse')
    	self.ae.fit(self.X, self.X, batch_size = batch_size, epochs = epochs)
    	self.ae.save(os.path.join('save_dir', 'pretrained_ae.h5'))
    	print ('Finish pretraining and save the model to %s' % save_dir)
 	   	self.pretrained = True

 	def load_model(self, weights):
 		self.model.load_weights(weights) 

 	def hidden_representations(self):
 		return self.model.predict(self.X)

 	def predict_classes(self, X):

 		return

 	def build_finetune_model(self):
 		self.inputs = tf.placeholder(tf.float32, [None, self.X.shape[1]], 'inputs')
 		self.centers = tf.placeholder(tf.float32, [None, self.X.shape[1]], 'centers')
 		


    def train(self, optimizer, batch_size, epochs, tol, update_interval, save_dir)


    	print 'Initializing the cluster centers with k-means.'
    	kmeans = KMeans(n_clusters = self.n_clusters, n_init = 20)
    	y_pred = kmeans.fit_predict(self.encoder.predict(self.X))
        y_pred_last = np.copy(y_pred)
    	self.model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])
    	
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
