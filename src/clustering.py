import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
#os.environ["THEANO_FLAGS"] = "module=FAST_RUN,floatX=float32"
#device=gpu0,
import pandas as pd
import numpy as np
from scipy import io
from dec import DEC, DEC_MALWARE
from dcn_tf import DCN
from dcn_theano import test_SdC
from keras.utils import to_categorical
from util import genaugbs, metrics, load_data
from keras.optimizers import SGD
metrics = metrics()

def clustering(dataset,
               n_clusters,
               n_bootstrap,
               method,
               optimizer_malware,
               batch,
               hidden_neurons,
               tol,
               pre_epochs,
               finetune_epochs,
               update_interval,
               using_own = False):

    if dataset == 'mnist' or dataset == 'rcv':
        print '================================'
        print 'Training ' + dataset + ' ...'
        print '================================'

        X, y, path = load_data(path="../results", dataset=dataset)

        print '================================'
        print 'Using' + method + ' for cluster number ' + str(n_clusters) + ' ...'
        print '================================'
        dir = method+'_'+str(n_clusters)
        path_method = os.path.join(path, dir)

        if os.path.exists(path_method) == False:
            os.system('mkdir ' + path_method)

        for i in xrange(n_bootstrap):
            if i == 0:
                pretrain = True
                X_bs, y_bs = X, y

            else:
                pretrain = False
                X_bs, y_bs = genaugbs(X, y)
            print '********************************'
            print 'Bootstrap sample time %d.' % i
            print '********************************'
            dir_path = os.path.join(path_method, str(i)+'_bs')
            if os.path.exists(dir_path) == False:
                os.system('mkdir '+dir_path)
            if method == 'dec':
                optimizer = SGD(0.01, 0.9)
                dec = DEC(X_bs, y_bs, hidden_neurons, n_clusters)
                dec.fit(optimizer=optimizer,
                          batch_size=batch,
                          pre_epochs = pre_epochs,
                          epochs=finetune_epochs,
                          tol= tol,
                          update_interval=update_interval,
                          pre_save_dir= path_method,
                          save_dir=dir_path,
                          shuffle= True,
                          pretrain = pretrain)
                dec.evaluate()

                pred_test = dec.predict(X_test = X, y_test = y)
                io.savemat(dir_path+'/'+str(i)+'_results', {'y_pred':pred_test})

            elif method == 'dcn':
                if using_own == True:
                    print 'Using tensorflow model...'
                    lr = 0.001
                    lbd = 4
                    dcn = DCN(X=X_bs, y=y_bs, hidden_neurons=hidden_neurons, n_clusters=n_clusters, lbd=lbd)
                    dcn.fit(batch_size=batch,
                              pre_epochs=pre_epochs,
                              finetune_epochs=finetune_epochs,
                              update_interval=update_interval,
                              pre_save_dir=path_method,
                              save_dir=dir_path,
                              lr=lr,
                              pretrain = pretrain)
                    pred = dcn.predict(X, y)
                else:
                    print 'Using theano model...'
                    config = {'train_x': X_bs,
                              'train_y': y_bs,
                              'test_x': X,
                              'test_y': y,
                              'lbd': 1,  # reconstruction
                              'beta': 1,
                              'pretraining_epochs': pre_epochs,
                              'pretrain_lr_base': 0.01,
                              'mu': 0.9,
                              'finetune_lr': 0.0001,
                              'training_epochs': finetune_epochs,
                              'batch_size': 256,
                              'nClass': n_clusters,
                              'hidden_dim': [500, 300, 100, 5],
                              'diminishing': False}
                    pred = test_SdC(**config)
                io.savemat(dir_path+'/'+str(i)+'_results', {'y_pred':pred})

    elif dataset == 'malware':
        print '================================'
        print 'Training ' + dataset + ' for cluster number ' + str(n_clusters) + ' ...'
        print '================================'

        data_path_1 = '../results/malware/trace1_train1.npz'
        data_path_2 = '../results/malware/trace1_train2.npz'
        path_method = '../results/malware/cluster_'+str(n_clusters)
        pretrained_dir = '../results/malware/pretrained_malware.h5'
        if os.path.exists(path_method) == False:
            os.system('mkdir ' + path_method)

        for i in xrange(n_bootstrap):
            if i == 0:
                use_boostrap = 0
            else:
                use_boostrap = 1
            print '********************************'
            print 'Bootstrap sample time %d.' % i
            print '********************************'
            dir_path = os.path.join(path_method, str(i)+'_bs')
            if os.path.exists(dir_path) == False:
                os.system('mkdir '+dir_path)
                malware_model = DEC_MALWARE(data_path_1, data_path_2, n_clusters)
                malware_model.fit(batch_size = batch,
                                    epochs = finetune_epochs,
                                    optimizer = optimizer_malware,
                                    update_interval = update_interval,
                                    tol = tol,
                                    shuffle = True,
                                    save_dir = dir_path,
                                    pretrained_dir = pretrained_dir,
                                    use_boostrap = use_boostrap)
                y_pred = malware_model.predict()
                io.savemat(dir_path+'/results', {'y_pred':y_pred})
    return 0


if __name__ == "__main__":
    """
    dataset = 'mnist'
    hidden_neurons = [784, 500, 500, 2000, n_clusters]
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

    if test == 1:
        n_bootstraps = 1
        batch = 100
        pre_epochs = 500
        finetune_epochs = 300
        update_interval = 20
        using_own = True
    """

    dataset = 'malware'
    #n_clusters = [3,4,5,6]
    n_clusters = 4
    n_bootstrap = 3
    method = 'dec_malware'
    optimizer_malware = 'adam'
    batch = 1000
    hidden_neurons = []
    tol = 1e-6
    pre_epochs = 0
    finetune_epochs = 200
    update_interval = 10


    clustering(dataset,
                n_clusters,
                n_bootstrap,
                method,
                optimizer_malware,
                batch,
                hidden_neurons,
                tol,
                pre_epochs,
                finetune_epochs,
                update_interval,
                using_own=False)