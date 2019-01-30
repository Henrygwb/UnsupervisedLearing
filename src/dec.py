import os
import sys
import numpy as np
import keras.backend as K
from keras.models import Model, load_model
from keras.layers import Dense, Input, concatenate, LSTM, Embedding, Conv1D, MaxPooling1D, Flatten, Dropout, Bidirectional, Layer
from keras.utils import to_categorical
from sklearn.cluster import KMeans
from util import metrics, Cluster, DimReduce
from util import ProgressBar
from time import sleep
from keras.optimizers import SGD

metrics = metrics()

class ClusteringLayer(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(ClusteringLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(input_shape[1], self.output_dim),
                                      initializer='glorot_uniform',
                                      name='kernel',
                                      trainable=True)
        super(ClusteringLayer, self).build(input_shape)

    def call(self, x):
        q1 = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(x, -1) - self.kernel), axis=1) / 1.0))
        q2 = q1 ** ((1.0 + 1.0) / 2.0)
        q = K.transpose(K.transpose(q2) / K.sum(q2, axis=1))
        return q

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)


def build_ae(hidden_neurons, rate = 0.5, act='relu'):
    n_layers = len(hidden_neurons) - 1
    X = Input(shape=(hidden_neurons[0],), name='input')

    for i in range(n_layers):
        if i == 0:
            H = Dense(hidden_neurons[i + 1], activation=act, name='encoder_%d' % (i+1))(X)
            # H = Dropout(rate)(H)
        else:
            H = Dense(hidden_neurons[i + 1], activation=act, name='encoder_%d' % (i+1))(H)
            # if i != 3:
            #     H = Dropout(rate)(H)

    for i in range(n_layers-1, -1, -1):
        if i == n_layers-1:
            Y = Dense(hidden_neurons[i], activation=act, name='decoder_%d' % i)(H)
            # Y = Dropout(rate)(Y)

        else:
            Y = Dense(hidden_neurons[i], activation=act, name='decoder_%d' % i)(Y)
            # if i != 0:
            #     Y = Dropout(rate)(Y)

    return Model(inputs=X, outputs=Y, name='AE')

class DEC(object):
    def __init__(self, X, y, hidden_neurons, n_clusters, alpha=1.0):
        self.X = X
        self.y = y
        self.n_clusters = n_clusters
        self.hidden_neurons = hidden_neurons
        self.input_dim = hidden_neurons[0]
        self.ae = build_ae(hidden_neurons)

    def pretrain(self, batch_size, epochs, save_dir):
        self.ae.compile(optimizer='adam', loss = 'mse')
        self.ae.fit(self.X, self.X, batch_size = batch_size, epochs = epochs, verbose=1)
        self.ae.save(os.path.join(save_dir, 'pretrained_ae.h5'))
        print ('Finish pretraining and save the model to %s' % save_dir)

    def hidden_representations(self, X):
        return self.encoder.predict(X)

    def predict_classes(self, X):
        q = self.model.predict(X, verbose = 0)
        return q.argmax(1)

    def auxiliary_distribution(self, q):
        p = q ** 2 / q.sum(0)
        return (p.T / p.sum(1)).T

    def fit(self, optimizer, batch_size, pre_epochs, epochs, tol, update_interval, pre_save_dir, save_dir, shuffle,
              pretrain = False):
        if pretrain == False:
            self.ae = load_model(os.path.join(pre_save_dir, 'pretrained_ae.h5'))
            # self.ae.load_weights('../results/mnist/dec_16/ae_weights.h5')
        else:
            self.pretrain(batch_size = batch_size, epochs = pre_epochs, save_dir = pre_save_dir)
        self.encoder = Model(self.ae.input, self.ae.get_layer('encoder_' + str(len(self.hidden_neurons) - 1)).output)
        self.encoder.compile(optimizer='adam', loss='mse')

        cluster_layer = ClusteringLayer(self.n_clusters,name='clustering')(self.encoder.output)
        self.model = Model(inputs = self.encoder.input, outputs = cluster_layer)

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
                acc = np.round(metrics.acc(self.y, y_pred), 5)
                nmi = np.round(metrics.nmi(self.y, y_pred), 5)
                ari = np.round(metrics.ari(self.y, y_pred), 5)
                loss = np.round(loss, 5)
                print '****************************************'
                print('Iter %d: acc = %.5f, nmi = %.5f, ari = %.5f, loss = %f' % (ite, acc, nmi, ari, loss))

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
        acc = np.round(metrics.acc(self.y, y_pred), 5)
        nmi = np.round(metrics.nmi(self.y, y_pred), 5)
        ari = np.round(metrics.ari(self.y, y_pred), 5)
        print '================================================='
        print 'Start evaluate ...'
        print '================================================='
        print('acc = %.5f, nmi = %.5f, ari = %.5f.' % (acc, nmi, ari))
        return 0

    def predict(self, X_test, y_test):
        print '================================================='
        print 'Start evaluate ...'
        print '================================================='
        y_pred = self.predict_classes(X_test)
        acc = np.round(metrics.acc(y_test, y_pred), 5)
        nmi = np.round(metrics.nmi(y_test, y_pred), 5)
        ari = np.round(metrics.ari(y_test, y_pred), 5)
        print('acc = %.5f, nmi = %.5f, ari = %.5f.' % (acc, nmi, ari))
        return y_pred


class DEC_MALWARE(object):
    def __init__(self, data_path_1, data_path_2, n_clusters, label_change):
        self.load_data(data_path_1, data_path_2, label_change)
        self.n_clusters = n_clusters
        self.model_inputs = {'input_dex_op': self.x_dex_op, 'input_dex_permission': self.x_dex_permission,
                        'input_sandbox': self.x_sandbox, 'input_sandbox_1': self.x_sandbox_1}

        self.build_model(dim_op=self.x_dex_op.shape[1], dim_per=self.x_dex_permission.shape[1],
                         dim_sand=self.x_sandbox.shape[1])
    def normalize(self, x):
        x = (x - np.min(x, axis=0)) / (np.max(x, axis=0) - np.min(x, axis=0))
        return x

    def auxiliary_distribution(self, q):
        p = q ** 2 / q.sum(0)
        return (p.T / p.sum(1)).T

    def predict_classes(self, X):
        q = self.model.predict(X, verbose = 0)
        return q.argmax(1)

    def gen_bootstarp(self):
        randidx = np.random.choice(self.y_fal.shape[0], self.y_fal.shape[0], replace=True)
        x_dex_op = self.x_dex_op[randidx]
        x_dex_permission = self.x_dex_permission[randidx]
        x_sandbox = self.x_sandbox[randidx]
        x_sandbox_1 = self.x_sandbox_1[randidx]
        model_inputs = {'input_dex_op': x_dex_op, 'input_dex_permission': x_dex_permission,
                        'input_sandbox': x_sandbox, 'input_sandbox_1': x_sandbox_1}
        y_fal_1 = self.y_fal_1[randidx]
        return model_inputs, y_fal_1

    def load_data(self, data_path_1, data_path_2, label_change={2:4}, use_two = 0, use_malware_only = 1):
        recover_files = np.load(data_path_1)
        dex = recover_files['x_train_dex']
        x_opcode_count = dex[:,0:11+256+1]
        self.x_dex_op = self.normalize(x_opcode_count)
        self.x_dex_permission = dex[:,11+256+1:]
        self.x_sandbox = recover_files['x_train__sandbox'].astype('float32')
        self.y_fal_1 = recover_files['y_train_family']

        if use_two == 1:
            recover_files_2 = np.load(data_path_2)
            dex_2 = recover_files_2['x_train_dex']
            x_opcode_count = dex_2[:,0:11+256+1]
            self.x_dex_op = np.vstack((self.x_dex_op,self.normalize(x_opcode_count)))
            self.x_dex_permission = np.vstack((self.x_dex_permission, dex_2[:,11+256+1:]))
            self.x_sandbox = np.vstack((self.x_sandbox, recover_files_2['x_train__sandbox'].astype('float32')))
            self.y_fal_1 = np.concatenate((self.y_fal_1, recover_files_2['y_train_family']))

        if use_malware_only == 1:
            nonzero_row = np.where(self.y_fal_1==0)[0]
            self.x_dex_op = np.delete(self.x_dex_op, nonzero_row, 0)
            self.x_dex_permission = np.delete(self.x_dex_permission, nonzero_row, 0)
            self.x_sandbox = np.delete(self.x_sandbox, nonzero_row, 0)
            self.y_fal_1 = np.delete(self.y_fal_1, nonzero_row, 0) - 1

        for l in label_change.keys():
            l_sub = label_change[l]
            self.y_fal_1[self.y_fal_1==l] = l_sub

        ii = 0
        for l in np.unique(self.y_fal_1):
            self.y_fal_1[self.y_fal_1==l] = ii
            print np.where(self.y_fal_1 == ii)[0].shape[0]
            ii = ii + 1

        self.x_dex_op = self.x_dex_op#[0:1000]
        self.x_sandbox = self.x_sandbox#[0:1000]
        self.y_fal_1 = self.y_fal_1#[0:1000]
        self.x_dex_permission = np.expand_dims(self.x_dex_permission, axis=-1)#[0:1000]
        self.y_fal = to_categorical(self.y_fal_1)#[0:1000]
        self.x_sandbox_1 = np.expand_dims(self.x_sandbox, axis=-1)#[0:1000]


    def build_model(self, dim_op, dim_per, dim_sand):
        x_dex_op = Input(shape = (dim_op,), name = 'input_dex_op')
        x_dex_permission = Input(shape=(dim_per,1), name='input_dex_permission')
        x_sandbox = Input(shape=(dim_sand,), name='input_sandbox')
        x_sandbox_1 = Input(shape=(dim_sand,1), name='input_sandbox_1')

        x_dex_op_1 = Dense(200, input_dim=dim_op, activation='relu')(x_dex_op)
        x_dex_op_1 = Dropout(0.25)(x_dex_op_1)
        x_dex_permission_1 = Conv1D(filters=16, kernel_size=2, activation='relu')(x_dex_permission)
        x_dex_permission_2 = MaxPooling1D(4)(x_dex_permission_1)
        x_dex_permission_embedded = Flatten()(x_dex_permission_2)
        x_embedded = concatenate([x_dex_op_1, x_dex_permission_embedded])
        hidden_1 = Dense(100, activation='relu')(x_embedded)
        hidden_1 = Dropout(0.25)(hidden_1)
        x_sandbox_embedded = Embedding(201, 16, input_length = dim_sand)(x_sandbox)
        hidden_sandbox_1 = Bidirectional(LSTM(units=10, activation='tanh', input_shape = (dim_sand, 16), return_sequences=1))(x_sandbox_embedded)
        hidden_sandbox_2 = Bidirectional(LSTM(units=10, activation='tanh', input_shape = (dim_sand, 16), return_sequences=0))(hidden_sandbox_1)
        hidden_1_merged = concatenate([hidden_1, hidden_sandbox_2])
        hidden_2 = Dense(100, activation='relu')(hidden_1_merged)
        hidden_2 = Dropout(0.25)(hidden_2)
        hidden_3 = Dense(50, activation='relu')(hidden_2)
        self.encoder = Model(inputs = [x_dex_op, x_dex_permission, x_sandbox, x_sandbox_1], outputs = hidden_3)
        cluster_layer = ClusteringLayer(self.n_clusters, name='clustering')(self.encoder.output)
        self.model = Model(inputs = [x_dex_op, x_dex_permission, x_sandbox, x_sandbox_1], outputs = cluster_layer)

    def pretrain(self, dim_op, dim_per, dim_sand, pre_epochs, batch_size):
        x_dex_op = Input(shape = (dim_op,), name = 'input_dex_op')
        x_dex_permission = Input(shape=(dim_per,1), name='input_dex_permission')
        x_sandbox = Input(shape=(dim_sand,), name='input_sandbox')
        x_sandbox_1 = Input(shape=(dim_sand,1), name='input_sandbox_1')

        x_dex_op_1 = Dense(200, input_dim=dim_op, activation='relu')(x_dex_op)
        x_dex_op_1 = Dropout(0.25)(x_dex_op_1)
        x_dex_permission_1 = Conv1D(filters=16, kernel_size=2, activation='relu')(x_dex_permission)
        x_dex_permission_2 = MaxPooling1D(4)(x_dex_permission_1)
        x_dex_permission_embedded = Flatten()(x_dex_permission_2)
        x_embedded = concatenate([x_dex_op_1, x_dex_permission_embedded])
        hidden_1 = Dense(100, activation='relu')(x_embedded)
        hidden_1 = Dropout(0.25)(hidden_1)
        x_sandbox_embedded = Embedding(201, 16, input_length = dim_sand)(x_sandbox)
        hidden_sandbox_1 = Bidirectional(LSTM(units=10, activation='tanh', input_shape = (dim_sand, 16), return_sequences=1))(x_sandbox_embedded)
        hidden_sandbox_2 = Bidirectional(LSTM(units=10, activation='tanh', input_shape = (dim_sand, 16), return_sequences=0))(hidden_sandbox_1)
        hidden_1_merged = concatenate([hidden_1, hidden_sandbox_2])
        hidden_2 = Dense(100, activation='relu')(hidden_1_merged)
        hidden_2 = Dropout(0.25)(hidden_2)
        hidden_3 = Dense(50, activation='relu')(hidden_2)
        output = Dense(self.y_fal.shape[1], activation='softmax')(hidden_3)
        model = Model(inputs=[x_dex_op, x_dex_permission, x_sandbox], outputs=output)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit({'input_dex_op': self.x_dex_op, 'input_dex_permission': self.x_dex_permission,
                   'input_sandbox': self.x_sandbox}, self.y_fal, batch_size=batch_size, epochs=pre_epochs)
        return model

    def fit(self, batch_size, epochs, optimizer, update_interval, tol, shuffle, save_dir,  pretrained_dir, pre_epochs, use_boostrap = 0, use_pretrained = 1):
        self.model.compile(optimizer = optimizer, loss = 'kld')
        if use_boostrap == 1:
            model_inputs, y_fal_1 = self.gen_bootstarp()
        else:
            model_inputs = self.model_inputs
            y_fal_1 = self.y_fal_1

        if use_pretrained == 1:
            print 'Loading pretrained model ...'
            pretrained_model = load_model(pretrained_dir)
            pretrained_model.layers.pop()
            pretrained_model.layers.pop()
            self.encoder.set_weights(pretrained_model.get_weights())
        else:
            print 'Pretraining ...'
            pretrained_model = self.pretrain(dim_op=self.x_dex_op.shape[1],
                                             dim_per=self.x_dex_permission.shape[1],dim_sand=self.x_sandbox.shape[1],
                                             pre_epochs=pre_epochs, batch_size=batch_size)
            pretrained_model.layers.pop()
            pretrained_model.layers.pop()
            self.encoder.set_weights(pretrained_model.get_weights())
            self.encoder.save(save_dir+'/DEC_model.h5')
        print '================================================='
        print 'Start training ...'
        print '================================================='

        if epochs == 0:
	    #low_func = K.function(pretrained_model.inputs+[K.learning_phase()], outputs=[pretrained_model.layers[-3].output])
            #x_low = low_func([self.x_dex_op, self.x_dex_permission, self.x_sandbox, 1.])[0]
	    #print x_low.shape
            x_low = self.encoder.predict(self.model_inputs, verbose = 1, batch_size = 3000)
	    print 'Computing low d ...'
            dr = DimReduce(x_low)
            x_low_2 = dr.cuda_tsne()
            cl = Cluster(x_low_2, x_low_2)
            y_pred = cl.mclust(self.n_clusters)
            print '================================================='
            print 'Start evaluate ...'
            print '================================================='
            acc = np.round(metrics.acc(self.y_fal_1, y_pred), 5)
            nmi = np.round(metrics.nmi(self.y_fal_1, y_pred), 5)
            ari = np.round(metrics.ari(self.y_fal_1, y_pred), 5)
            print('acc = %.5f, nmi = %.5f, ari = %.5f.' % (acc, nmi, ari))
            return y_pred, x_low, x_low_2
        else:
            print '================================================='
            print 'Initializing the cluster centers with k-means...'
            print '================================================='
            kmeans = KMeans(n_clusters=self.n_clusters)
            y_pred = kmeans.fit_predict(self.encoder.predict(model_inputs))
            y_pred_last = np.copy(y_pred)
            self.model.get_layer(name='clustering').set_weights([np.transpose(kmeans.cluster_centers_)])

            index_array = np.arange(self.y_fal.shape[0])
            if shuffle == True:
                np.random.shuffle(index_array)

            for ite in range(int(epochs/update_interval)):
                print str(ite) + ' of ' + str(int(epochs/update_interval))
                #print self.model.layers[-1].clusters.get_value()

                q = self.model.predict(model_inputs, verbose=0)
                p = self.auxiliary_distribution(q)  # update the auxiliary target distribution p

                # evaluate the clustering performance
                y_pred = q.argmax(1)
                acc = np.round(metrics.acc(y_fal_1, y_pred), 5)
                nmi = np.round(metrics.nmi(y_fal_1, y_pred), 5)
                ari = np.round(metrics.ari(y_fal_1, y_pred), 5)
                print '****************************************'
                print('Iter %d: acc = %.5f, nmi = %.5f, ari = %.5f' % (ite, acc, nmi, ari))

                # check stop criterion
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                y_pred_last = np.copy(y_pred)
                if ite > 0 and delta_label < tol:
                   print '****************************************'
                   print('delta_label ', delta_label, '< tol ', tol)
                   print('Reached tolerance threshold. Stopping training.')
                   break

                batch_inputs = {'input_dex_op': self.x_dex_op, 'input_dex_permission': self.x_dex_permission,
                               'input_sandbox': self.x_sandbox, 'input_sandbox_1': self.x_sandbox_1}
                self.model.fit(x=batch_inputs, y=p, batch_size = batch_size, epochs=update_interval)
            print '================================================='
            print 'Start evaluate ...'
            print '================================================='
            y_pred = self.predict_classes(self.model_inputs)
            acc = np.round(metrics.acc(self.y_fal_1, y_pred), 5)
            nmi = np.round(metrics.nmi(self.y_fal_1, y_pred), 5)
            ari = np.round(metrics.ari(self.y_fal_1, y_pred), 5)
            print('acc = %.5f, nmi = %.5f, ari = %.5f.' % (acc, nmi, ari))
            return y_pred
