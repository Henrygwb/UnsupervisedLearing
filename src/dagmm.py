import tensorflow as tf
import numpy as np
from util import ProgressBar
from time import sleep
from util import metrics
metrics = metrics()
from os import makedirs
from os.path import exists, join

class GAGMM(object):
    def __init__(self,
                 compress_hidden,
                 estimate_hidden,
                 lambda1,
                 lambda2,
                 optimizer,
                 lr,
                 compress_activation = tf.nn.tanh,
                 estimate_activation = tf.nn.tanh,
                 dropout_rate = 0.5,
                 use_dropout = 1
                 ):
        self.input_dim = compress_hidden[0]
        self.compress_hidden = compress_hidden
        self.estimate_hidden = estimate_hidden
        self.n_component = estimate_hidden[-1]
        self.optimizer = optimizer(lr)
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.compress_activation = compress_activation
        self.estimate_activation = estimate_activation
        self.dropout_rate = dropout_rate
        self.use_dropout = use_dropout

    def build_compress(self, x):
        n_layers = len(self.compress_hidden) - 1
        zc = x
        with tf.variable_scope('encoder'):
            for i in range(n_layers):
                zc = tf.layers.dense(zc, self.compress_hidden[i + 1], activation = self.compress_activation, name='encoder_%d' % (i + 1))
                # if i != n_layers - 1:
                    # zr = tf.layers.dropout(zr, rate = self.dropout_rate, name = 'encoder_dropout_%d' % (i + 1))

        with tf.variable_scope('decoder'):
            x_prime = zc
            for i in range(n_layers - 1, -1, -1):
                x_prime = tf.layers.dense(x_prime, self.compress_hidden[i], activation = self.compress_activation, name='decoder_%d' % i)
                # if i != 0:
                    # x_hat = tf.layers.dropout(x_hat, rate = self.dropout_rate, name = 'decoder_dropout_%d' % (i + 1))
        norm_x = tf.norm(x, ord=2, axis=1)
        norm_x_prime = tf.norm(x_prime, ord=2, axis=1)
        norm_diff = tf.norm((x - x_prime), ord=2, axis=1)
        euclidean = norm_diff / (norm_x + 1e-4)
        cosine = tf.reduce_sum(tf.multiply(x, x_prime), axis=1)/(tf.multiply(norm_x * norm_x_prime) + 1e-4)
        #cosine = 0.5 * (1 - tf.reduce_sum(tf.multiply(x, x_prime), axis=1) / (tf.multiply(norm_x * norm_x_prime) + 1e-4))

        zr = tf.concat([euclidean, cosine], axis=1)
        z = tf.concat([zc, zr])
        return z, x_prime

    def build_estimate(self, z):
        layer = 0
        with tf.variable_scope('estimator'):
            for i in self.estimate_hidden:
                layer += 1
                if i != self.estimate_hidden[:-1]:
                    z = tf.layers.dense(z, activation=self.estimate_activation, name='estimator_%d' % (layer))
                    if self.use_dropout == 1:
                        z = tf.layers.dropout(z, rate=self.dropout_rate, name='estimator_%d' % (layer))
                else:
                    logits = tf.layers.dense(z, name='estimator_logit')
            gamma = tf.nn.softmax(logits, name='estimator_softmax')
        return gamma

    def build_gmm(self, z, gamma):
        
        return energy, gmm_cov_diag_loss

    def build_model(self):

        ### input and outputs
        self.x = tf.placeholder(tf.float32, [None, self.input_dim], name = 'input')
        self.z, self.x_prime = self.build_compress(self.x)
        self.gamma = self.build_estimate(self.z)
        self.energy, self.gmm_cov_diag_loss = self.build_gmm(self.z, self.gamma)

        ### loss function
        loss_1 = tf.reduce_mean(tf.norm((self.x - self.x_prime), ord = 2, axis = 1))
        loss_2 = self.lambda1 * tf.reduce_mean(self.energy)
        loss_3 = self.lambda2 * self.gmm_cov_diag_loss
        self.loss = loss_1 + loss_2 + loss_3

        ### trainable variable
        self.var = tf.trainable_variables()

        ### training function
        self.train_op = self.optimizer.minimize(self.loss)
        #self.train_op = tf.contrib.training.create_train_op(loss, self.optimizer, self.var)
        return 0

    def fit(self, X, y, batch, epoch, display_interval):
        print 'Strat training...'
        print '================================================='
        with tf.Graph().as_default() as graph:
            self.graph = graph
            self.build_model()
            with tf.Session() as sess:
                tf.global_variables_initializer().run()
                self.saver = tf.train.saver()
                num_batch = X.shape[0] // batch
                idx = np.arange(X.shape[0])
                # start training...
                for step in range(epoch):
                    bar = ProgressBar(num_batch, fmt=ProgressBar.FULL)
                    for i in num_batch:
                        feed_dict = {self.x: X[idx[i*batch:min((i+1)*batch, X.shape[0])]]}
                        sess.run(self.train_op, feed_dict)
                        bar.current += 1
                        bar()
                        sleep(0.1)
                    if step % display_interval == 0:
                        loss = sess.run(self.loss, {self.x:X})
                        print("Epoch %d/%d: loss = %.5f" % (step+1, epoch, loss))
        return 0

    def evaluate(self, X, y, cutoff):
        cutoff = X.shape[0]*cutoff
        energy = self.sess.run(self.energy, feed_dict={self.x: X}).flatten()
        pred = np.zeros_like(energy)
        pred[np.argsort(energy)[::-1][0:cutoff]] = 1
        acc = np.round(metrics.acc(y, pred), 5)
        nmi = np.round(metrics.nmi(y, pred), 5)
        ari = np.round(metrics.ari(y, pred), 5)
        print '*********************************************'
        print('acc = %.5f, nmi = %.5f, ari = %.5f' % (acc, nmi, ari))
        return 0

    def predict(self, X, cutoff):
        cutoff = X.shape[0]*cutoff
        energy = self.sess.run(self.energy, feed_dict={self.x: X}).flatten()
        pred = np.zeros_like(energy)
        pred[np.argsort(energy)[::-1][0:cutoff]] = 1
        return energy, pred

    def save(self, dir, model_name):
        if not exists(dir):
            makedirs(dir)
        model_path = join(dir, model_name)
        self.saver.save(self.sess, model_path)

    def restore(self, model_path):
        with tf.Graph().as_default() as graph:
            self.graph = graph
            self.sess = tf.Session(graph=graph)
            self.saver = tf.train.import_meta_graph(model_path+'.meta')
            self.saver.restore(self.sess, model_path)