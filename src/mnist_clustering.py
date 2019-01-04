import os
#os.environ["THEANO_FLAGS"] = "module=FAST_RUN,device=gpu0,floatX=float32"
#os.environ["CUDA_VISIBLE_DEVICES"]="1"
import numpy as np
from scipy import io
from dec import DeepEmbeddingClustering
from dcn import DeepClusteringNetwork
from util import genbssamples, genaugbs
from keras.optimizers import SGD
from MeanUncertaintyCluster import MeanClustering, ClusterAnalysis
import argparse
from util import metrics
from collections import Counter

metrics = metrics()


def load_data(path):
    file = os.path.join(path, 'train')
    X = io.loadmat(file)['train_x']
    y = io.loadmat(file)['train_y']
    file = os.path.join(path, 'testall')
    X = np.concatenate((X, io.loadmat(file)['test_x']))
    y = np.concatenate((y, io.loadmat(file)['test_y']))
    y = np.array([np.where(r == 1)[0][0] for r in y])
    return X, y

def clustering(X, y, n_clusters, n_bootstrep, method, path = '../results/mnist'):
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
            dcn_test = DeepClusteringNetwork(X=X_bs, y=y_bs, hidden_neurons=hidden_neurons, n_clusters=n_clusters, lbd=lbd)
            dcn_test.train(batch_size=batch, pre_epochs=pre_epochs, finetune_epochs=finetune_epochs,update_interval=update_interval,
                           pre_save_dir=path_dcn, save_dir=dir_path, lr=lr, pretrain = pretrain)
            pred_test = dcn_test.test(X, y)
            io.savemat(dir_path+'/'+str(i)+'_results', {'y_pred':pred_test})
    return 0

def mean_par(X, y, n_boostrap, option):
    num_sample = y.shape[0]
    if option == 'dec':
        yb = io.loadmat("../results/mnist/dec_16/0_bs/0_results")['y_pred']

        for i in xrange(n_boostrap):
            i_tmp = i + 1
            if i_tmp != 5 and i_tmp != 10:
                path = "../results/mnist/dec_16/" + str(i_tmp)+"_bs/" + str(i_tmp)+ "_results"
                yb_tmp = io.loadmat(path)['y_pred']
                print np.max(yb_tmp, )
                yb = np.hstack((yb, yb_tmp))
        yb = yb.flatten()[num_sample:]
        y_orin = yb.flatten()[0:num_sample]
        print 'Clustering result of the original dec model.'
        acc = np.round(metrics.acc(y, y_orin), 5)
        nmi = np.round(metrics.nmi(y, y_orin), 5)
        ari = np.round(metrics.ari(y, y_orin), 5)
        print '****************************************'
        print('acc = %.5f, nmi = %.5f, ari = %.5f.' % (acc, nmi, ari))
        save_path = "../results/mnist/dec_16/"
        mean_cluster = MeanClustering(y, yb, 8)
        y_mean = mean_cluster.ota()
        io.savemat(save_path+'/mean_partition',{"y_mean":y_mean})


        cluster_analy = ClusterAnalysis(yb, n_boostrap, y_mean, len = 5000)
        wt, clst = mean_cluster.align(yb, y_mean)
        codect, nfave, res = cluster_analy.matchsplit(wt=wt, clsct=clst)
        res = np.round(res, 4)
        cluster_id, sample_id = cluster_analy.matchcluster(res, clst)
        confidentset, S = cluster_analy.confset(sample_id, cluster_id)
        io.savemat(save_path+'/confident_set',{"confidentset":confidentset, 'S':S})

    elif option == 'dcn':
        yb = io.loadmat("../results/mnist/dcn_17/0_bs/0_results")['y_pred']
        for i in xrange(n_boostrap):
            i_tmp = i + 1
            path = "../results/mnist/dcn_17/" + str(i_tmp) + "_bs/" + str(i_tmp) + "_results"
            yb_tmp = io.loadmat(path)['y_pred']
            yb = np.hstack((yb, yb_tmp))
        yb = yb.flatten()[num_sample:]
        y_orin = yb.flatten()[0:num_sample]
        print 'Clustering result of the original dcn model.'
        acc = np.round(metrics.acc(y, y_orin), 5)
        nmi = np.round(metrics.nmi(y, y_orin), 5)
        ari = np.round(metrics.ari(y, y_orin), 5)
        print '****************************************'
        print('acc = %.5f, nmi = %.5f, ari = %.5f.' % (acc, nmi, ari))
        save_path = "../results/mnist/dcn_17/"

        mean_cluster = MeanClustering(y, yb, 10)
        y_mean = mean_cluster.ota()
        io.savemat(save_path+'/mean_partition',{"y_mean":y_mean})


        cluster_analy = ClusterAnalysis(yb, n_boostrap, y_mean, len = 70000)
        wt, clst, _ = mean_cluster.align(yb, y_mean)
        codect, nfave, res = cluster_analy.matchsplit(wt=wt, clsct=clst)
        res = np.round(res, 4)
        cluster_id, sample_id = cluster_analy.matchcluster(res, clst)
        confidentset, S = cluster_analy.confset(sample_id, cluster_id)
        io.savemat(save_path+'/confident_set',{"confidentset":confidentset, 'S':S})

    else:
        yb = io.loadmat("../results/mnist/dec_16/0_bs/0_results")['y_pred']
        for i in xrange(n_boostrap):
            i_tmp = i + 1
            path = "../results/mnist/dec_16/" + str(i_tmp) + "_bs/" + str(i_tmp) + "_results"
            yb_tmp = io.loadmat(path)['y_pred']
            yb = np.hstack((yb, yb_tmp))
        yb = yb.flatten()[num_sample:]
        y_orin = yb.flatten()[0:num_sample]
        print 'Clustering result of the original dec model.'
        acc = np.round(metrics.acc(y, y_orin), 5)
        nmi = np.round(metrics.nmi(y, y_orin), 5)
        ari = np.round(metrics.ari(y, y_orin), 5)
        print '****************************************'
        print('acc = %.5f, nmi = %.5f, ari = %.5f.' % (acc, nmi, ari))

        yb_2 = io.loadmat("../results/mnist/dcn_17/0_bs/0_results")['y_pred']
        for i in xrange(n_boostrap):
            i_tmp = i + 1
            path = "../results/mnist/dcn_17/" + str(i_tmp) + "_bs/" + str(i_tmp) + "_results"
            yb_tmp = io.loadmat(path)['y_pred']
            yb_2 = np.hstack((yb_2, yb_tmp))
        yb_2 = yb_2.flatten()[num_sample:]
        y_orin = yb_2.flatten()[0:num_sample]
        print 'Clustering result of the original dcn model.'
        acc = np.round(metrics.acc(y, y_orin), 5)
        nmi = np.round(metrics.nmi(y, y_orin), 5)
        ari = np.round(metrics.ari(y, y_orin), 5)
        print '****************************************'
        print('acc = %.5f, nmi = %.5f, ari = %.5f.' % (acc, nmi, ari))

        yb = np.hstack((yb[0:num_sample*(n_boostrap/2)], yb_2[0:num_sample*(n_boostrap/2)]))
        save_path = "../results/mnist/"
    acc = 0
    new_confidentset = {}
    for i in xrange(len(confidentset)):
        idx = confidentset[i]
        y_cls = y[idx]
        b = Counter(y_cls).most_common(1)
        print b
        new_confidentset[b[0][0]] = confidentset[i]
        acc_tmp = b[0][1] / float(idx.shape[0])
        acc += acc_tmp
    acc = acc / len(confidentset)
    print acc

    for i in xrange(len(new_confidentset)):
        idx = new_confidentset[i]
        y_cls = y[idx]
        b = Counter(y_cls).most_common(1)
        print b
        print idx.shape[0]
        print b[0][1] / float(idx.shape[0])

    with open('confit') as f:
        idx = f.readlines()

    for i in xrange(10):
        print '******************************' + str(i)
        idx_tmp = np.fromstring(idx[i][2:-1], sep = ' ')
        y_cls = y[idx_tmp.astype('int')]
        b = Counter(y_cls).most_common(1)
        print b
        print idx_tmp.shape[0]
        print b[0][1] / float(idx_tmp.shape[0])


if __name__ == "__main__":
    option = 'dec'
    X, y = load_data("../results/mnist")
    n_boostrap = 10
    option = 'dcn'
    mean_par(X, y, n_boostrap, option)