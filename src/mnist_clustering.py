import os
#os.environ["THEANO_FLAGS"] = "module=FAST_RUN,device=gpu0,floatX=float32"
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import numpy as np
from scipy import io
from dec import DeepEmbeddingClustering
from dcn import DeepClusteringNetwork
from util import genbssamples
from keras.optimizers import SGD
import argparse


def load_data(path):
    file = os.path.join(path, 'train')
    X = io.loadmat(file)['train_x']
    y = io.loadmat(file)['train_y']
    file = os.path.join(path, 'testall')
    X = np.concatenate((X, io.loadmat(file)['test_x']))
    y = np.concatenate((y, io.loadmat(file)['test_y']))
    y = np.array([np.where(r == 1)[0][0] for r in y])
    return X, y

def clustering(X, y, n_clusters, n_bootstrep, bs_idx, method):
    if method == 'dec':
        print '================================'
        print '================================'
        print '================================'
        print 'Using ICML_16 dec...'
        print '================================'
        print '================================'
        print '================================'

        path_dec = os.path.join(path, 'dec_16')
        hidden_neurons = [X.shape[-1], 500, 500, 2000, 10]
        batch = 256
        pre_epochs = 300
        finetune_epoch = 2e4
        update_interval = 140
        tol = 1e-3
        if os.path.exists(path_dec) == False:
            os.system('mkdir ' + path_dec)

        for i in xrange(n_bootstrep):
            if i == 0:
                pretrain = True
            else:
                pretrain = False
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
            X_bs = X[bs_idx[i,:]]
            y_bs = y[bs_idx[i,:]]
            optimizer = SGD(0.01, 0.9)
            dec = DeepEmbeddingClustering(X_bs, y_bs, hidden_neurons, n_clusters)
            dec.train(optimizer=optimizer, batch_size=batch, pre_epochs = pre_epochs, epochs=finetune_epoch, tol= tol,
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

        hidden_neurons = [X.shape[-1], 500, 500, 2000, 10]
        batch = 256
        pre_epochs = 250
        finetune_epochs = 50
        update_interval = 10
        lr = 0.005
        lbd = 1

        for i in xrange(n_bootstrep):
            if i == 0:
                pretrain = True
            else:
                pretrain = False
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
            X_bs = X[bs_idx[i,:]]
            y_bs = y[bs_idx[i,:]]
            dcn_test = DeepClusteringNetwork(X=X_bs, y=y_bs, hidden_neurons=hidden_neurons, n_clusters=n_clusters, lbd=lbd)
            dcn_test.train(batch_size=batch, pre_epochs=pre_epochs, finetune_epochs=finetune_epochs,update_interval=update_interval,
                           pre_save_dir=path_dcn, save_dir=dir_path, lr=lr, pretrain = pretrain)
            pred_test = dcn_test.test(X, y)
            io.savemat(dir_path+'/'+str(i)+'_results', {'y_pred':pred_test})
    return 0


if __name__ == "__main__":

    path = "../results/mnist"
    X, y = load_data(path)
    n_clusters = len(np.unique(y))
    n_bootstrep = 10
    num_samples = X.shape[0]
    bs_idx = genbssamples(n_bootstrep=n_bootstrep, num_samples=num_samples)
    bs_idx = np.concatenate((np.arange(X.shape[0]).reshape(1, X.shape[0]), bs_idx))
    io.savemat(path+'/bs_idx', {'idx': bs_idx})

    parser = argparse.ArgumentParser(description='clustering_mnist',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--m', default='dec', choices=['dec', 'dcn'])
    args = parser.parse_args()
    clustering(X, y, n_clusters, n_bootstrep+1, bs_idx, args.m)
