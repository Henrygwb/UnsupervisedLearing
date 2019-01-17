import tensorflow as tf

class GAGMM(object):
    def __init__(self, compress_hidden, extimate_hidden):
        self.compress_hidden = compress_hidden
        self.extimate_hidden = extimate_hidden
        self.n_component = self.extimate_hidden[-1]

    def build_compress(self, x, act=tf.nn.tanh, rate=0.5):
        n_layers = len(self.compress_hidden) - 1
        zc = x
        with tf.variable_scope('encoder'):
            for i in range(n_layers):
                zc = tf.layers.dense(zc, self.compress_hidden[i + 1], activation = act, name='encoder_%d' % (i + 1))
                # if i != n_layers - 1:
                    # zr = tf.layers.dropout(zr, rate = rate, name = 'encoder_dropout_%d' % (i + 1))

        with tf.variable_scope('decoder'):
            x_hat = zc
            for i in range(n_layers - 1, -1, -1):
                x_hat = tf.layers.dense(x_hat, self.compress_hidden[i], activation = act, name='decoder_%d' % i)
                # if i != 0:
                    # x_hat = tf.layers.dropout(x_hat, rate = rate, name = 'decoder_dropout_%d' % (i + 1))
        return zc, x_hat

    def build_estimate(self, z, act = tf.nn.tanh, use_dropout = 1, rate = 0.5):
        layer = 0
        with tf.variable_scope('estimator'):
            for i in self.extimate_hidden:
                layer += 1
                if i != self.extimate_hidden[:-1]:
                    z = tf.layers.dense(z, activation=act, name='estimator_%d' % (layer))
                    if use_dropout == 1:
                        z = tf.layers.dropout(z, rate=rate, name='estimator_%d' % (layer))
                else:
                    logits = tf.layers.dense(z, name='estimator_logit')
            gamma = tf.nn.softmax(logits, name='estimator_softmax')
        return gamma

    def build_model(self):
        x = tf.variable()
        return l

    def fit(self):
        return 0

    def evaluate(self):
        return 0

    def predict(self):
        return pred