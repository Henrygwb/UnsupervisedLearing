import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from sklearn.cluster import KMeans
from util import ProgressBar
from time import sleep
from util import metrics
metrics = metrics()

class DeepClusteringNetwork(object):
    def __init__(self, X, y, hidden_neurons, n_clusters, lbd, mode = 'pretrain', config = tf.ConfigProto()):
        self.X = X
        self.y = y
        self.mode = mode
        self.hidden_neurons = hidden_neurons
        self.n_clusters = n_clusters
        self.input_dim = hidden_neurons[0]
        self.lbd = lbd
        self.config = config

    def build_ae(self, input, rate=0.5, act=tf.nn.relu, reuse = False):
        n_layers = len(self.hidden_neurons) - 1

        with tf.variable_scope('encoder', reuse=reuse):
            with slim.arg_scope([slim.fully_connected], activation_fn=act, trainable= (self.mode == 'pretrain' or self.mode == 'train')):
                # with slim.arg_scope([slim.dropout], keep_prob = rate, is_training=(self.mode == 'pretrain'or self.mode == 'train')):
                    for i in range(n_layers):
                        if i == 0:
                            H = slim.fully_connected(input, self.hidden_neurons[i + 1], scope='encoder_%d' % (i + 1))
                            # H = slim.dropout(H, scope = 'encoder_dropout_%d' % (i + 1))
                        else:
                            H = slim.fully_connected(H, self.hidden_neurons[i + 1], scope='encoder_%d' % (i + 1))
                            # if i != n_layers - 1:
                            #     H = slim.dropout(H, scope='encoder_dropout_%d' % (i + 1))

        with tf.variable_scope('decoder', reuse=reuse):
            with slim.arg_scope([slim.fully_connected], activation_fn=act, trainable= (self.mode == 'pretrain' or self.mode == 'train')):
                # with slim.arg_scope([slim.dropout], keep_prob = rate, is_training=(self.mode == 'pretrain' or self.mode == 'train')):

                    for i in range(n_layers - 1, -1, -1):
                        #print i
                        if i == n_layers - 1:
                            Y = slim.fully_connected(H, self.hidden_neurons[i], scope='decoder_%d' % i)
                            # Y = slim.dropout(Y, scope='decoder_dropout_%d' % (i + 1))
                        else:
                            Y = slim.fully_connected(Y, self.hidden_neurons[i], scope='decoder_%d' % i)
                            # if i != 0:
                            #     Y =  slim.dropout(Y, scope='decoder_dropout_%d' % (i + 1))
        return H, Y

    def build_model(self, lr = 0.005):
        if self.mode == 'pretrain':
            self._input = tf.placeholder(tf.float32, [None, self.X.shape[-1]], 'input_data')
            self._fx, self._z = self.build_ae(self._input)

            ## loss and train
            self.loss = tf.reduce_mean(tf.square(self._input - self._z))
            self.optimizer = tf.train.MomentumOptimizer(learning_rate=1, momentum=0.9)
            ae_vars = tf.trainable_variables()
            self.train_op = slim.learning.create_train_op(self.loss, self.optimizer, variables_to_train=ae_vars)

        elif self.mode == 'train':
            self._input = tf.placeholder(tf.float32, [None, self.X.shape[-1]], 'input_data')
            self._centroids = tf.placeholder(tf.float32, [None, self.hidden_neurons[-1]], 'centroids')
            #self._centroids_idx = tf.placeholder(tf.float32, [None, self.n_clusters], 'clustering_idx')

            self._fx, self._z = self.build_ae(self._input, reuse = False)

            ## loss
            # clustering loss
            # self._clustering_loss = tf.reduce_sum(tf.pow(self._fx - tf.matmul(self._centroids_idx, self._centroids), 2), axis = 1)
            # self._clustering_loss = tf.reduce_sum(tf.pow(self._fx - self._centroids, 2), axis=1)
            self._clustering_loss =  tf.reduce_mean(tf.square(self._fx - self._centroids))
            # reconstruction_loss
            self._recont_loss = tf.reduce_mean(tf.square(self._input - self._z))
            # total loss
            self.loss = self.lbd*self._clustering_loss + self._recont_loss

            ## Train
            ae_vars = tf.trainable_variables()
            self.optimizer = tf.train.AdamOptimizer(learning_rate = lr)
            self.train_op = slim.learning.create_train_op(self.loss, self.optimizer, variables_to_train=ae_vars)

        elif self.mode == 'test':
            self._input = tf.placeholder(tf.float32, [None, self.X.shape[-1]], 'input_data')
            self._fx, self._z = self.build_ae(self._input)

    def init_cluster(self, data):
        km = KMeans(n_clusters=self.n_clusters, n_init=10)
        km.fit(data)
        idx = km.labels_
        centers = km.cluster_centers_
        centers = centers.astype(np.float32)
        idx = idx.astype(np.int32)
        return idx, centers

    def batch_km(self, data, center, count):
        N = data.shape[0]
        K = center.shape[0]

        # update assignment
        idx = np.zeros(N, dtype=np.int)
        for i in range(N):
            dist = np.inf
            ind = 0
            for j in range(K):
                temp_dist = np.linalg.norm(data[i] - center[j])
                if temp_dist < dist:
                    dist = temp_dist
                    ind = j
            idx[i] = ind

        # update centriod
        center_new = center
        for i in range(N):
            c = idx[i]
            count[c] += 1 # control the learning rate for updating the centroids
            eta = 1.0/count[c]
            center_new[c] = (1 - eta) * center_new[c] + eta * data[i]
        center_new.astype(np.float32)
        return idx, center_new, count

    def train(self, batch_size, pre_epochs, finetune_epochs, update_interval, pre_save_dir, save_dir, pretrain, lr, cen_lr = 100, tol=1e-3):
        if pretrain == True:
            print '================================================='
            print 'Pretraining AE...'
            print '================================================='
            with tf.Graph().as_default() as pretrain_graph:
                self.build_model(lr)
                with tf.Session(graph=pretrain_graph, config=self.config) as sess:
                    tf.global_variables_initializer().run()
                    saver = tf.train.Saver()
                    for step in range(pre_epochs):
                        num_batch = int(self.X.shape[0] / batch_size)
                        if step == 0:
                            print '****************************************'
                            print 'Iteration: %d.' % step
                            bar = ProgressBar(10, fmt = ProgressBar.FULL)
                        for minibatch_idx in xrange(num_batch):
                                batch_X = self.X[minibatch_idx * batch_size:min((minibatch_idx + 1) * batch_size, self.X.shape[0])]
                                feed_dict = {self._input: batch_X}
                                sess.run(self.train_op, feed_dict)

                        if (step + 1) % 10 == 0:
                            l = sess.run(self.loss, feed_dict = {self._input:self.X})
                            print '****************************************'
                            print ('Step: [%d/%d] loss: %.6f' % (step + 1, pre_epochs, l))
                            bar.done()
                            print '****************************************'
                            print 'Iteration: %d.' % step
                            bar = ProgressBar(10, fmt = ProgressBar.FULL)

                        if (step + 1) % pre_epochs == 0:
                            self.pretrained_model =  os.path.join(pre_save_dir, 'pretrain_model')
                            saver.save(sess, self.pretrained_model)
                            print '****************************************'
                            print 'pretrained_model saved..!'
                        bar.current += 1
                        bar()
                        sleep(0.1)

        else:
            self.pretrained_model = os.path.join(pre_save_dir, 'pretrain_model')
        print 'Strat training...'
        print '================================================='
        with tf.Graph().as_default() as finetune_graph:
            self.mode = 'train'
            self.build_model()
            count =  cen_lr*np.ones(self.n_clusters, dtype=np.int) # learning rate for each centroid
            count_sample = np.zeros(self.n_clusters, dtype=np.int)
            with tf.Session(config=self.config, graph=finetune_graph) as sess:
                tf.global_variables_initializer().run()

                ## 'loading pretrained model...'
                variables_to_restore = slim.get_model_variables()
	
                restorer = tf.train.Saver(variables_to_restore)
		restorer.restore(sess, self.pretrained_model)
                saver = tf.train.Saver()
                #saver.restore(sess, self.pretrained_model)

                recon_l = sess.run(self._recont_loss, feed_dict={self._input: self.X})
		print recon_l 
                print '****************************************'
                print ('Reconstruction loss: %.6f' % (recon_l))

                print '================================================='
                print 'Initializing the cluster centroids with k-means...'
                print '================================================='
                fx = sess.run(self._fx, feed_dict={self._input: self.X})
                self.y_pred, self.centroids = self.init_cluster(fx)

                # start training...
                for step in range(finetune_epochs):
                    if step == 0:
                        print '*********************************************'
                        print 'Iteration: %d.' %step
                        bar = ProgressBar(update_interval, fmt=ProgressBar.FULL)

                    num_batch = int(self.X.shape[0] / batch_size)
                    for minibatch_idx in range(int(self.X.shape[0] / batch_size)):
                        batch_centroids = self.centroids[self.y_pred[minibatch_idx * batch_size:min((minibatch_idx + 1) * batch_size, self.X.shape[0])]]
                        batch_X = self.X[minibatch_idx * batch_size:min((minibatch_idx + 1) * batch_size, self.X.shape[0])]
                        feed_dict = {self._input: batch_X, self._centroids:batch_centroids}
                        sess.run(self.train_op, feed_dict)

                        hidden_rep_tmp = sess.run(self._fx, feed_dict)
                        temp_y_pred, self.centroids, count = self.batch_km(hidden_rep_tmp, self.centroids, count)
                        self.y_pred[minibatch_idx * batch_size:min((minibatch_idx + 1) * batch_size, self.X.shape[0])] = temp_y_pred

                        # check if empty cluster happen, if it does random initialize it
                                # for i in range(self.n_clusters):
                                #    if count[i] == 0:
                                #        rand_idx = numpy.random.randint(low = 0, high = n_train_samples)
                                #        # modify the centroid
                                #        centers[i] = out_single(rand_idx)

                    if step > 0 and (step+1) % update_interval == 0:
                        bar.done()
                        print '*********************************************'
                        print 'Iteration: %d.' %step
                        bar = ProgressBar(update_interval, fmt=ProgressBar.FULL)

                        hidden_array = sess.run(self._fx, feed_dict = {self._input: self.X, self._centroids:self.centroids[self.y_pred]})
                        y_pred_new, self.centroids = self.init_cluster(hidden_array)
                        # for i in range(self.n_clusters):
                        #     count_sample[i] += y_pred_new.shape[0] - np.count_nonzero(y_pred_new - i)

                        ## check stop criterion
                        # delta_label = np.sum(self.y_pred != y_pred_new).astype(np.float32) / index_array.shape[0]
                        # self.y_pred = np.copy(y_pred_new)
                        # if ite > 0 and delta_label < tol:
                        #     print('delta_label ', delta_label, '< tol ', tol)
                        #     print('Reached tolerance threshold. Stopping training.')
                        #     break

                        # evaluate the clustering performance
                        acc = np.round(metrics.acc(self.y, y_pred_new), 5)
                        nmi = np.round(metrics.nmi(self.y, y_pred_new), 5)
                        ari = np.round(metrics.ari(self.y, y_pred_new), 5)
                        print '*********************************************'
                        print('Iter %d: acc = %.5f, nmi = %.5f, ari = %.5f' % (step, acc, nmi, ari))

                    if (step + 1) % finetune_epochs == 0:
                        self.final_model = os.path.join(save_dir, 'final_model')
                        saver.save(sess, self.final_model)
                        print '****************************************'
                        print 'pretrained_model saved..!'
                    bar.current += 1
                    bar()
                    sleep(0.1)

        return 0

    def test(self, X_test, y_test):
        print '================================================='
        print 'Strat testing...'
        print '================================================='
        self.mode = 'test'
        self.build_model()
        with tf.Session(config=self.config) as sess:
            saver = tf.train.Saver()
            saver.restore(sess, self.final_model)
            hidden_array = sess.run(self._fx, feed_dict={self._input: X_test})
            y_pred, _ = self.init_cluster(hidden_array)
            acc = np.round(metrics.acc(y_test, y_pred), 5)
            nmi = np.round(metrics.nmi(y_test, y_pred), 5)
            ari = np.round(metrics.ari(y_test, y_pred), 5)
            print '****************************************'
            print('acc = %.5f, nmi = %.5f, ari = %.5f.' % (acc, nmi, ari))

        return y_pred

