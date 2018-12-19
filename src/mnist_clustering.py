import os
import sys
import numpy as np
from scipy import io
from dec import DeepEmbeddingClustering
from dcn import DeepClusteringNetwork
from util import genbssamples

def load_data(path, option = 0):
    if option == 0:
        file = os.path.join(path, 'train')
        X = io.loadmat(file)['train_x']
        y = io.loadmat(file)['train_y']
        y = np.array([np.where(r == 1)[0][0] for r in y])

    else:
        file = os.path.join(path, 'testall')
        X = io.loadmat(file)['test_x']
        y = io.loadmat(file)['test_y']
        y = np.array([np.where(r == 1)[0][0] for r in y])

    return X, y

if __name__ == "__main__":

    path = "../results/mnist"
    X, y = load_data(path, option=1)
    n_clusters = len(np.unique(y))
    num_approaches = 2
    n_bootstrep = 30
    num_samples = X.shape[0]
    bs_idx = genbssamples(n_bootstrep=n_bootstrep, num_samples=num_samples)
    io.savemat(path+'/bs_idx', {'idx': bs_idx})

    for method in xrange(num_approaches):
        if method == 0:
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
            pre_epoch = 300
            finetune_epoch = 2e4
            update_interval = 140
            tol = 1e-3
            if os.path.exists(path_dec) == False:
                os.system('mkdir ' + path_dec)

            for i in xrange(n_bootstrep):
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
                dec_test = DeepEmbeddingClustering(X_bs, y_bs, hidden_neurons, n_clusters)
                dec_test.pretrain(batch_size=batch, epochs=pre_epoch, save_dir=dir_path)
                dec_test.train(optimizer='adam', batch_size=batch, epochs=finetune_epoch, tol= tol,
                               update_interval=update_interval, save_dir=dir_path, shuffle= False)
                dec_test.evaluate()

                pred_test = dec_test.test(X_test = X, y_test = y)
                io.savemat(dir_path+'/'+str(i)+'_results', {'y_pred':pred_test})
        elif method == 1:
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
            lbd = 1

            for i in xrange(n_bootstrep):
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
                dcn_test = DeepClusteringNetwork(X=X, y=y, hidden_neurons=hidden_neurons, n_clusters=n_clusters, lbd=lbd)
                dcn_test.train(batch_size=batch, pre_epochs=pre_epochs, finetune_epochs=finetune_epochs,
                               update_interval=update_interval, save_dir=dir_path)
                pred_test = dcn_test.test(X, y)
                io.savemat(dir_path+'/'+str(i)+'_results', {'y_pred':pred_test})
