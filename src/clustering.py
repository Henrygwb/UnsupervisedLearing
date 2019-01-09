import os
os.environ["THEANO_FLAGS"] = "module=FAST_RUN,floatX=float32"
#device=gpu0,
#os.environ["CUDA_VISIBLE_DEVICES"]="1"
import pandas as pd
import numpy as np
from scipy import io
from dec import DeepEmbeddingClustering
from dcn_tf import DeepClusteringNetwork
from dcn_theano import test_SdC

from util import genaugbs, metrics, load_data
from keras.optimizers import SGD
metrics = metrics()

def clustering(X, y, n_clusters, n_bootstrep, method, path, batch, pre_epochs, finetune_epochs, update_interval,
               using_own = False):
    batch = batch
    pre_epochs = pre_epochs
    finetune_epochs = finetune_epochs
    update_interval = update_interval

    if method == 'dec':
        print '================================'
        print '================================'
        print '================================'
        print 'Using ICML_16 dec...'
        print '================================'
        print '================================'
        print '================================'

        path_dec = os.path.join(path, 'dec_16')
        hidden_neurons = [X.shape[-1], 500, 500, 2000, n_clusters]
        tol = 1e-3

        if os.path.exists(path_dec) == False:
            os.system('mkdir ' + path_dec)

        for i in xrange(n_bootstrep):
            if i == 0:
                pretrain = True
                X_bs, y_bs = X, y

            else:
                pretrain = False
                X_bs, y_bs = genaugbs(X, y)
            print '********************************'
            print '********************************'
            print '********************************'
            print 'Bootstrap sample time %d.' % i
            print '********************************'
            print '********************************'
            print '********************************'
            dir_path = os.path.join(path_dec, str(i)+'_bs')
            if os.path.exists(dir_path) == False:
                os.system('mkdir '+dir_path)
            optimizer = SGD(0.01, 0.9)
            dec = DeepEmbeddingClustering(X_bs, y_bs, hidden_neurons, n_clusters)
            dec.train(optimizer=optimizer, batch_size=batch, pre_epochs = pre_epochs, epochs=finetune_epochs, tol= tol,
                           update_interval=update_interval, pre_save_dir= path_dec, save_dir=dir_path, shuffle= False, pretrain = pretrain)
            dec.evaluate()

            pred_test = dec.test(X_test = X, y_test = y)
            io.savemat(dir_path+'/'+str(i)+'_results', {'y_pred':pred_test})

    elif method == 'dcn':
        print '================================'
        print '================================'
        print '================================'
        print 'Using ICML_17 dcn...'
        print '================================'
        print '================================'
        print '================================'

        path_dcn = os.path.join(path, 'dcn_17')
        if os.path.exists(path_dcn) == False:
            os.system('mkdir ' + path_dcn)

        for i in xrange(n_bootstrep):
            if i == 0:
                pretrain = True
                X_bs, y_bs = X, y
            else:
                pretrain = False
                X_bs, y_bs = genaugbs(X, y)
            print '********************************'
            print '********************************'
            print '********************************'
            print 'Bootstrap sample time %d.' % i
            print '********************************'
            print '********************************'
            print '********************************'
            dir_path = os.path.join(path_dcn, str(i)+'_bs')
            if os.path.exists(dir_path) == False:
                os.system('mkdir '+dir_path)

            if using_own == True:
                print 'Using tensorflow model...'
                hidden_neurons = [X.shape[-1], 500, 500, 2000, n_clusters]
                lr = 0.001
                lbd = 4

                dcn_test = DeepClusteringNetwork(X=X_bs, y=y_bs, hidden_neurons=hidden_neurons, n_clusters=n_clusters, lbd=lbd)
                dcn_test.train(batch_size=batch, pre_epochs=pre_epochs, finetune_epochs=finetune_epochs,update_interval=update_interval,
                           pre_save_dir=path_dcn, save_dir=dir_path, lr=lr, pretrain = pretrain)
                pred_test = dcn_test.test(X, y)
            else:
                print 'Using theano model...'
                config = {'train_x': X_bs,
                          'train_y': y_bs,
                          'test_x': X,
                          'test_y': y,
                          'lbd': 1,  # reconstruction
                          'beta': 1,
                          'pretraining_epochs': pre_epochs,
                          'pretrain_lr_base': 0.0001,
                          'mu': 0.9,
                          'finetune_lr': 0.0001,
                          'training_epochs': finetune_epochs,
                          'batch_size': 256,
                          'nClass': n_clusters,
                          'hidden_dim': [500, 500, 2000, 10],
                          'diminishing': False}
                pred_test = test_SdC(**config)
            io.savemat(dir_path+'/'+str(i)+'_results', {'y_pred':pred_test})
    return 0

if __name__ == "__main__":
    dataset = 'mnist'
    n_bootstraps = 10
    method = 'dcn'
    test = 1
    X, y, n_clusters, path = load_data(path="../results", dataset=dataset)

    if test == 1:
        n_bootstraps = 2
        batch = 256
        pre_epochs = 0
        finetune_epochs = 0
        update_interval = 2
        using_own = False

    else:
        n_bootstraps = 10
        if method == 'dec':
            batch = 256
            pre_epochs = 300
            finetune_epochs = 2e4
            update_interval = 140

        elif method == 'dcn':
            batch = 256
            pre_epochs = 250
            finetune_epochs = 300
            update_interval = 20
            using_own = False

    clustering(X =X, y=y, n_clusters = n_clusters, n_bootstrep=n_bootstraps, method = method, path = path,
               batch=batch, pre_epochs = pre_epochs, finetune_epochs = finetune_epochs, update_interval = update_interval)

