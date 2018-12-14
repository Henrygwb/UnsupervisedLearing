import os
import sys
import keras.backend as K
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
        if i == n_layers-1:
            H = Dense(hidden_neurons[i + 1], activation='linear', name='encoder_%d' % (i+1))(X)
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

 	def hidden_representations(self, X):
 		return self.model.predict(X)

 	def predict_classes(self, X):
        km = KMeans(n_clusters=self.n_clusters, n_init=10)
        km.fit(X)
        idx = km.labels_
        centers = km.cluster_centers_
        centers = centers.astype(numpy.float32)
        idx = idx.astype(numpy.int32)
        return idx

 	# def build_finetune_model(self):
 	# 	self.inputs = tf.placeholder(tf.float32, [None, self.X.shape[1]], 'inputs')
 	# 	self.centers = tf.placeholder(tf.float32, [None, self.X.shape[1]], 'centers')

    def finetune_loss(self, X, center, beta, lbd):
        low_repre = hidden_representations(X)
        temp = K.pow(center - low_repre, 2)
        L = K.sum(temp, axis = 1)

 		# Add the network reconstruction error
        z = self.ae(X)
        reconst_err = K.sum(K.pow(X - z, 2), axis=1)
        L = beta*L + lbd*reconst_err

        cost = K.mean(L)
        # reconst_cost = lbd*K.mean(reconst_err)
        # cluster_cost = cost - reconst_cost
        return cost

    def init_cluster(data):
        km = KMeans(n_clusters=self.n_clusters, n_init=10)
        km.fit(data)
        idx = km.labels_
        centers = km.cluster_centers_
        centers = centers.astype(numpy.float32)
        idx = idx.astype(numpy.int32)
        return idx, centers

    def batch_km(data, center, count):
        N = data.shape[0]
        K = center.shape[0]

        # update assignment
        idx = numpy.zeros(N, dtype=numpy.int)
        for i in range(N):
            dist = numpy.inf
            ind = 0
            for j in range(K):
                temp_dist = numpy.linalg.norm(data[i] - center[j])
                if temp_dist < dist:
                    dist = temp_dist
                    ind = j
            idx[i] = ind

        # update centriod
        center_new = center
        for i in range(N):
            c = idx[i]
            count[c] += 1
            eta = 1.0/count[c]
            center_new[c] = (1 - eta) * center_new[c] + eta * data[i]
        center_new.astype(numpy.float32)
        return idx, center_new, count

    def train(self, optimizer, batch_size, epochs, beta, lbd, tol, update_interval, save_dir)

    	self.model.compile(optimizer = optimizer, loss = self.finetune_loss)

        print 'Initializing the cluster centers with k-means.'
        hidden_array = self.hidden_representations(self.X)
        y_pred, center = init_cluster(hidden_array)
        count = 100*numpy.ones(nClass, dtype=numpy.int)

    	n_batches = self.X.shape[0]/batch_size
    	index_array = np.arange(self.X.shape[0])
        for ite in range(int(epochs)):
            for minibatch_idx in xrange(n_batches):
                center_tmp = center[y_pred[minibatch_index * batch_size:
                                    (minibatch_index + 1) * batch_size]]

                idx_tmp = index_array[index * batch_size: min((index+1) * batch_size, x.shape[0])]
                loss = self.model.train_on_batch(x=self.X[idx_tmp], center=center_tmp, beta = beta, lbd = lbd)
                hidden_rep_tmp = self.hidden_representations(self.X[idx_tmp])
                temp_y_pred, centers, count = batch_km(hidden_val, centers, count)
                y_pred[minibatch_index * batch_size:
                    (minibatch_index + 1) * batch_size] = temp_y_pred
                
                ## check if empty cluster happen, if it does random initialize it
                #for i in range(nClass):
                #    if count_samples[i] == 0:
                #        rand_idx = numpy.random.randint(low = 0, high = n_train_samples)
                #        # modify the centroid
                #        centers[i] = out_single(rand_idx)

            if ite % update_interval == 0:
                hidden_array = self.hidden_representations(self.X)
                y_pred_new, center = init_cluster(hidden_array)
                for i in range(self.n_clusters):
                    count[i] += y_pred_new.shape[0] - numpy.count_nonzero(y_pred_new - i)

                # evaluate the clustering performance
                acc = np.round(metrics.acc(self.y, y_pred_new), 5)
                nmi = np.round(metrics.nmi(self.y, y_pred_new), 5)
                ari = np.round(metrics.ari(self.y, y_pred_new), 5)
                print('Iter %d: acc = %.5f, nmi = %.5f, ari = %.5f' % (ite, acc, nmi, ari))

                # check stop criterion
                delta_label = np.sum(y_pred != y_pred_new).astype(np.float32) / idx.shape[0]
                y_pred = np.copy(y_pred_new)
                if ite > 0 and delta_label < tol:
                    print('delta_label ', delta_label, '< tol ', tol)
                    print('Reached tolerance threshold. Stopping training.')
                    break

        # save the trained model
        print('saving model to:', save_dir + '/DCN_model_final.h5')
        self.model.save_weights(save_dir + '/DCN_model_final.h5')

        return y_pred

    def evaluate(self):
        y_pred = self.predict_classes(self.X)
        acc = np.round(metrics.acc(self.y, y_pred), 5)
        nmi = np.round(metrics.nmi(self.y, y_pred), 5)
        ari = np.round(metrics.ari(self.y, y_pred), 5)
        print('Iter %d: acc = %.5f, nmi = %.5f, ari = %.5f' % (ite, acc, nmi, ari), ' ; loss=', loss)
       	return 0