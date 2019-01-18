import tensorflow as tf
import numpy as np
from util import ProgressBar
from time import sleep
from util import metrics
metrics = metrics()
from os import makedirs
from os.path import exists, join
import math

class DAGMM(object):
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
                 use_dropout = 1,
                 use_diag_cov = 1
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
        self.use_diag_cov = use_diag_cov

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
        #cosine = tf.reduce_sum(tf.multiply(x, x_prime), axis=1)/(tf.multiply(norm_x, norm_x_prime) + 1e-4)
        cosine = 0.5 * (1 - tf.reduce_sum(tf.multiply(x, x_prime), axis=1) / (tf.multiply(norm_x, norm_x_prime) + 1e-4))

        zr = tf.concat([euclidean[:,None], cosine[:,None]], axis=1)
        z = tf.concat([zc, zr], axis=1)
        return z, x_prime

    def build_estimate(self, z):
        layer = 0
        with tf.variable_scope('estimator'):
            for i in self.estimate_hidden:
                layer += 1
                if i != self.estimate_hidden[-1]:
                    z = tf.layers.dense(z, i, activation=self.estimate_activation, name='estimator_%d' % (layer))
                    if self.use_dropout == 1:
                        z = tf.layers.dropout(z, rate=self.dropout_rate, name='estimator_%d' % (layer))
                else:
                    logits = tf.layers.dense(z, i, name='estimator_logit')
            gamma = tf.nn.softmax(logits, name='estimator_softmax')
        return gamma

    def build_gmm(self, z, gamma):
        with tf.variable_scope('gmm'):
            # phi: [num_mixture, ]
            phi = tf.reduce_mean(gamma, axis=0)
            phi_sum = tf.reduce_sum(gamma, axis=0)

            # mu: [num_mixture, num_dim] mu_ij = sum(gamma_ki, z_kj)
            mu = tf.einsum('ki,kj->ij', gamma, z) / phi_sum[:,None]

            # simga: [num_mixture, num_dim]
            if self.use_diag_cov == 1:
                # z - mu_1:K
                z_t = (z[:, None, :] - mu[None, :, :]) ** 2

                # only preserve the diag values of the sigma, each element equals to the raw sum of zt*gamma
                sigma = tf.reduce_sum(z_t * gamma[:,:,None], 0) / phi_sum[:,None]
                dev = sigma ** 0.5

                # diag values of (z-u_k)*sigma^{-1}
                z_norm = tf.reduce_sum(z_t / sigma, 2)

                t1 = tf.exp(-0.5 * z_norm)
                t2 = ((2 * math.pi) ** (0.5 * z.shape[1]._value)) * tf.reduce_prod(dev, 1)

                tmp = phi * t1 / t2
                logits = tf.reduce_sum(tmp, 1)

                energy = -tf.log(logits)
                gmm_cov_diag_loss = tf.reduce_sum(1.0 / (sigma+1e-8))

            else:
                z_centered = tf.sqrt(gamma[:, :, None]) * (z[:, None, :] - mu[None, :, :])
                sigma = tf.einsum('ikl,ikm->klm', z_centered, z_centered) / phi_sum[:, None, None]
                n_features = z.shape[1]._value
                min_vals = tf.diag(tf.ones(n_features, dtype=tf.float32)) * 1e-6
                L = tf.cholesky(sigma + min_vals[None, :, :])
                z_centered = z[:, None, :] - mu[None, :, :]

                # sigma = LL^{T} Z^{T}sigma^{-1}Z = Z^{T}L^{T}^{-1}L^{-1}Z = (L^{-1}Z)^{T}L^{-1}Z
                v = tf.matrix_triangular_solve(L, tf.transpose(z_centered, [1, 2, 0])) # L^{-1}Z

                # log(det(Sigma)) = 2 * sum[log(diag(L))]
                log_det_sigma = 2.0 * tf.reduce_sum(tf.log(tf.matrix_diag_part(L)), axis=1)

                d = z.get_shape().as_list()[1]
                # tf.square(v) =  (L^{-1}Z)^{T}L^{-1}Z
                logits = tf.log(phi[:, None]) - 0.5 * (tf.reduce_sum(tf.square(v), axis=1)+
                                                       d * tf.log(2.0 * np.pi) + log_det_sigma[:, None])
                energy = - tf.reduce_logsumexp(logits, axis=0)
                gmm_cov_diag_loss = tf.reduce_sum(tf.divide(1, tf.matrix_diag_part(sigma)))

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

    def fit(self, X, batch, epoch, display_interval):
        print 'Strat training...'
        print '================================================='
        with tf.Graph().as_default() as graph:
            self.graph = graph
            self.build_model()
            self.sess = tf.Session(graph=self.graph)
            self.sess.run(tf.global_variables_initializer())
            self.saver = tf.train.Saver()
            num_batch = X.shape[0] // batch
            idx = np.arange(X.shape[0])
            # start training...
            for step in xrange(epoch):
                for i in xrange(num_batch):
                    feed_dict = {self.x: X[idx[i*batch:min((i+1)*batch, X.shape[0])]]}
                    self.sess.run(self.train_op, feed_dict)
                if step % display_interval == 0:
                    bar = ProgressBar(display_interval, fmt=ProgressBar.FULL)
                    loss = self.sess.run(self.loss, {self.x:X})
                    print("Epoch %d/%d: loss = %.5f" % (step, epoch, loss))
                bar.current += 1
                bar()
                # sleep(0.1)
            # energy = self.sess.run(self.energy, feed_dict={self.x: X})
            # print energy
            self.saver = tf.train.Saver()
        return 0

    def evaluate(self, X, y, thred):
        energy = self.sess.run(self.energy, feed_dict={self.x: X}).flatten()
        anomaly_energy_threshold = np.percentile(energy, thred)
        print("Energy value at thleshold %d is %.3f" % (thred, anomaly_energy_threshold))
        pred = np.where(energy >= anomaly_energy_threshold, 1, 0)
        acc = np.round(metrics.acc(y, pred), 5)
        nmi = np.round(metrics.nmi(y, pred), 5)
        ari = np.round(metrics.ari(y, pred), 5)
        print '*********************************************'
        print('acc = %.5f, nmi = %.5f, ari = %.5f' % (acc, nmi, ari))
        print '*********************************************'
        prec, recall, fscore = metrics.presion_recall_fscore(y, pred)
        print('precision = %.5f, recall = %.5f, f1_score = %.5f' % (prec, recall, fscore))
        return 0

    def predict(self, X, thred):
        energy = self.sess.run(self.energy, feed_dict={self.x: X}).flatten()
        anomaly_energy_threshold = np.percentile(energy, thred)
        print("Energy value at thleshold %d is %.3f" % (thred, anomaly_energy_threshold))
        pred = np.where(energy >= anomaly_energy_threshold, 1, 0)
        return energy, pred

    def save(self, dir, model_name):
        if not exists(dir):
            makedirs(dir)
        model_path = join(dir, model_name)
        self.saver.save(self.sess, model_path)
        return 0

    def restore(self, model_path):
        with tf.Graph().as_default() as graph:
            self.graph = graph
            self.build_model()
            self.sess = tf.Session(graph=graph)
            self.sess.run(tf.global_variables_initializer())
            self.saver = tf.train.Saver(self.var)
            self.saver.restore(self.sess, model_path)
        return 0