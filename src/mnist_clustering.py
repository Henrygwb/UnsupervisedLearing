import os
#os.environ["THEANO_FLAGS"] = "module=FAST_RUN,device=gpu0,floatX=float32"
#os.environ["CUDA_VISIBLE_DEVICES"]="1"
import pandas as pd
import numpy as np
from scipy import io
from dec import DeepEmbeddingClustering
from dcn import DeepClusteringNetwork
from util import genbssamples, genaugbs, ensemble, metrics, tsne, pca
from keras.optimizers import SGD
from MeanUncertaintyCluster import MeanClustering, ClusterAnalysis
from collections import Counter
metrics = metrics()
from ggplot import *

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
            path = "../results/mnist/dec_16/" + str(i_tmp)+"_bs/" + str(i_tmp)+ "_results"
            yb_tmp = io.loadmat(path)['y_pred']
            print np.unique(yb_tmp)
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

    ########### Mean partition #######################
    mean_cluster = MeanClustering(y, yb, 10)
    y_mean = mean_cluster.ota()
    io.savemat(save_path+'/mean_partition',{"y_mean":y_mean})
    en = ensemble(yb, n_boostrap, num_sample)
    y_vote = en.voting()
    print 'Clustering result of voting.'
    acc = np.round(metrics.acc(y, y_vote), 5)
    nmi = np.round(metrics.nmi(y, y_vote), 5)
    ari = np.round(metrics.ari(y, y_vote), 5)
    print '****************************************'
    print('acc = %.5f, nmi = %.5f, ari = %.5f.' % (acc, nmi, ari))
    y_cspa = en.CSPA()
    print 'Clustering result of CSPA.'
    acc = np.round(metrics.acc(y, y_cspa), 5)
    nmi = np.round(metrics.nmi(y, y_cspa), 5)
    ari = np.round(metrics.ari(y, y_cspa), 5)
    print '****************************************'
    print('acc = %.5f, nmi = %.5f, ari = %.5f.' % (acc, nmi, ari))
    y_mcla = en.MCLA()
    print 'Clustering result of MCLA.'
    acc = np.round(metrics.acc(y, y_mcla), 5)
    nmi = np.round(metrics.nmi(y, y_mcla), 5)
    ari = np.round(metrics.ari(y, y_mcla), 5)
    print '****************************************'
    print('acc = %.5f, nmi = %.5f, ari = %.5f.' % (acc, nmi, ari))

    ########### Confident point set #######################
    cluster_analy = ClusterAnalysis(yb, n_boostrap, y_mean, len = 70000)
    wt, clst, dist = mean_cluster.align(yb, y_mean)
    codect, nfave, res = cluster_analy.matchsplit(wt=wt, clsct=clst)
    cluster_id, sample_id = cluster_analy.matchcluster(res, clst)
    confidentset, S = cluster_analy.confset(sample_id, cluster_id)
    Interset = cluster_analy.interset(sample_id, cluster_id)
    io.savemat(save_path+'/confident_set',{"confidentset":confidentset, 'interset':Interset})

    print 'Relabeling....'
    new_cluster_id = np.zeros_like(cluster_id)
    new_confidentset = {}
    for i in xrange(len(confidentset)):
        idx = confidentset[i]
        y_cls = y[idx]
        b = Counter(y_cls).most_common(1)
        print b
        new_confidentset[b[0][0]] = confidentset[i]
        new_cluster_id[b[0][0]] = cluster_id[i]
    if option == 'dcn':
        new_confidentset[9] = confidentset[1]
        new_cluster_id[9] = cluster_id[1]
    print '************************************'
    for i in xrange(len(new_confidentset)):
        idx = new_confidentset[i]
        y_cls = y[idx]
        if option == 'dcn' and i == 9:
            b = 9
            print b
            print idx.shape[0]
            print float(np.where(y_cls==0)[0].shape[0]) / float(idx.shape[0])
        b = Counter(y_cls).most_common(1)
        print b
        print idx.shape[0]
        print b[0][1] / float(idx.shape[0])

    ########### Cluster stability #######################
    print '************ Cluster stability *****************'
    stability = np.zeros((len(new_confidentset), 2))
    yb = yb.reshape(n_boostrap, num_sample)
    yb = np.vstack((y_mean.reshape(1, 70000), yb))
    for i in xrange(len(new_confidentset)):
        confset = new_confidentset[i]
        S_idx = new_cluster_id[i, ].astype('int')
        SS = []
        for ii in xrange(S_idx.shape[0]):
            if S_idx[ii] != -1:
                SS.append(np.where(yb[ii]==S_idx[ii])[0])
        atr, acr = cluster_analy.clu_stablity(confset, SS)
        stability[i, 0] = atr
        stability[i, 1] = acr
    print np.round(stability, 4)[:,0]
    print np.round(stability, 4)[:,1]


    ########### Cluster distance #######################
    print '************ Cluster distance *****************'
    cls_dis = np.zeros((len(new_confidentset), len(new_confidentset)))
    for i in xrange(len(new_confidentset)):
        for j in xrange(len(new_confidentset)):
            cls_dis[i,j] = cluster_analy.clu_dist(new_confidentset, i, j)
    print np.around(cls_dis, decimals=3)

    ########### Partition stability #######################
    print '************ Partition stability *****************'
    p_s = cluster_analy.par_stablity(dist, metric='mean')
    print p_s
    """
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
    """
    return y_mean, confidentset, Interset

def draw_figure(X, y):
    feat_cols = ['tsne-x', 'tsne-y']
    df = pd.DataFrame(X, columns=feat_cols)
    df['label'] = y

    # ggplot(df, aes(x='tsne-x', y='tsne-y', color='label')) + \
    # geom_point() + \
    # scale_color_brewer(type='diverging', palette=4) + \
    # xlab("tsne-x") + ylab("tsne-y") + ggtitle("MNIST")


if __name__ == "__main__":
    option = '...'
#    option = 'dcn'
#    option = 'dec'

    X, y = load_data("../results/mnist")
    n_boostrap = 10
    #y_mean, confidentset, interset = mean_par(X, y, n_boostrap, option)
    x_low = pca(X)
    draw_figure(x_low, y)
