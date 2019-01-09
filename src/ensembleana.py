import os
import shutil
import pandas as pd
import numpy as np
from scipy import io
from sklearn.preprocessing import OneHotEncoder
from util import ensemble, metrics, t_sne, pca, load_data
from MeanUncertaintyCluster import MeanClustering, ClusterAnalysis, MeanUncertainty_c
from collections import Counter
metrics = metrics()

def load_pre(y, n_bootstrap, option, path):
    num_sample = y.shape[0]
    if option == 'dec':
        yb = io.loadmat(path+"/dec_16/0_bs/0_results")['y_pred']

        for i in xrange(n_bootstrap):
            i_tmp = i + 1
            path_1 = path+"/dec_16/" + str(i_tmp)+"_bs/" + str(i_tmp)+ "_results"
            yb_tmp = io.loadmat(path_1)['y_pred']
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
        save_path = path+"/dec_16/"

    elif option == 'dcn':
        yb = io.loadmat(path+"/dcn_17/"+"0_bs/0_results")['y_pred']
        for i in xrange(n_bootstrap):
            i_tmp = i + 1
            path_1 = path+"/dcn_17/" + str(i_tmp) + "_bs/" + str(i_tmp) + "_results"
            yb_tmp = io.loadmat(path_1)['y_pred']
            yb = np.hstack((yb, yb_tmp))
        yb = yb.flatten()[num_sample:]
        y_orin = yb.flatten()[0:num_sample]
        print 'Clustering result of the original dcn model.'
        acc = np.round(metrics.acc(y, y_orin), 5)
        nmi = np.round(metrics.nmi(y, y_orin), 5)
        ari = np.round(metrics.ari(y, y_orin), 5)
        print '****************************************'
        print('acc = %.5f, nmi = %.5f, ari = %.5f.' % (acc, nmi, ari))
        save_path = path+"/dcn_17/"

    elif option == 'dcn_dec':
        yb = io.loadmat(path+"/dec_16/"+"0_bs/0_results")['y_pred']
        for i in xrange(n_bootstrap):
            i_tmp = i + 1
            path_1 = path+"/dec_16/" + str(i_tmp) + "_bs/" + str(i_tmp) + "_results"
            yb_tmp = io.loadmat(path_1)['y_pred']
            yb = np.hstack((yb, yb_tmp))
        yb = yb.flatten()[num_sample:]
        y_orin = yb.flatten()[0:num_sample]
        print 'Clustering result of the original dec model.'
        acc = np.round(metrics.acc(y, y_orin), 5)
        nmi = np.round(metrics.nmi(y, y_orin), 5)
        ari = np.round(metrics.ari(y, y_orin), 5)
        print '****************************************'
        print('acc = %.5f, nmi = %.5f, ari = %.5f.' % (acc, nmi, ari))

        yb_2 = io.loadmat(path+"/dcn_17/"+"0_bs/0_results")['y_pred']
        for i in xrange(n_bootstrap-1):
            i_tmp = i + 1
            path_2 = path+"/dcn_17/" + str(i_tmp) + "_bs/" + str(i_tmp) + "_results"
            yb_tmp = io.loadmat(path_2)['y_pred']
            yb_2 = np.hstack((yb_2, yb_tmp))
        yb_2 = yb_2.flatten()[num_sample:]
        y_orin = yb_2.flatten()[0:num_sample]
        print 'Clustering result of the original dcn model.'
        acc = np.round(metrics.acc(y, y_orin), 5)
        nmi = np.round(metrics.nmi(y, y_orin), 5)
        ari = np.round(metrics.ari(y, y_orin), 5)
        print '****************************************'
        print('acc = %.5f, nmi = %.5f, ari = %.5f.' % (acc, nmi, ari))

        yb = np.hstack((yb[0:num_sample*(n_bootstrap/2)], yb_2[0:num_sample*(n_bootstrap/2)]))
        save_path = path
    return y_orin, yb, save_path

def ensemble_ana(y,
                 n_bootstrap,
                 option,
                 path,
                 using_c =0,
                 return_mean =1,
                 compare = 0,
                 threshold = 0.8,
                 alpha = 0.1):
    num_sample = y.shape[0]
    y_orin, yb, save_path = load_pre(y, n_bootstrap, option, path)
    ########### Mean partition #######################

    yb = np.hstack((y_orin, yb))
    if using_c == 1:
        y_mean, \
        confidentset, \
        Interset, \
        cluster_id, \
        dist, \
        cluster_analy = MeanUncertainty_c(yb, num_sample, n_bootstrap, threshold, alpha, return_mean)
    else:
        mean_cluster = MeanClustering(y, yb, 10)
        if return_mean ==1:
            y_mean = mean_cluster.ota()
        else:
            idx = MeanClustering.ref_idx(yb)
            y_mean = yb[(idx-1)*num_sample:idx*num_sample]
        en = ensemble(yb, n_bootstrap, num_sample)
        if compare == 1:
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
        cluster_analy = ClusterAnalysis(yb, n_bootstrap, y_mean, len = num_sample)
        wt, clst, dist = mean_cluster.align(yb, y_mean)
        codect, nfave, res = cluster_analy.matchsplit(wt=wt, clsct=clst)
        cluster_id, sample_id = cluster_analy.matchcluster(res, clst)
        confidentset, S = cluster_analy.confset(sample_id, cluster_id)
        Interset = cluster_analy.interset(sample_id, cluster_id)

    print 'Relabeling....'
    new_cluster_id = np.zeros_like(cluster_id)
    new_confidentset = {}
    new_interset = {}
    new_y_mean = np.zeros_like(y_mean)
    for i in xrange(len(confidentset)):
        idx = confidentset[i]
        y_cls = y[idx]
        b = Counter(y_cls).most_common(1)
        print b
        new_confidentset[b[0][0]] = confidentset[i]
        new_cluster_id[b[0][0]] = cluster_id[i]
        new_interset[b[0][0]] = Interset[i]
        new_y_mean[y_mean==i] = b[0][0]

    # if option == 'dcn' and path == '../results/mnist':
    #     new_confidentset[9] = confidentset[1]
    #     new_cluster_id[9] = cluster_id[1]
    #     new_interset[9] = Interset[1].astype('int')
    #     new_y_mean[y_mean == 1] = 9
    #     new_y_mean = new_y_mean.astype('int')

    io.savemat(save_path+'/mean_confident',{"y_mean":new_y_mean, "confidentset":new_confidentset, 'interset':new_interset})

    print '************************************'
    for i in xrange(len(new_confidentset)):
        idx = new_confidentset[i]
        y_cls = y[idx]
        if option == 'dcn' and i == 9 and path == '../results/mnist':
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
    yb = yb.reshape(n_bootstrap, num_sample)
    yb = np.vstack((y_mean.reshape(1, num_sample), yb))
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
    return new_y_mean, new_confidentset, new_interset

def draw_figure(X, y, y_mean, confidentset, Interset, option):
    from ggplot import *
    feat_cols = ['tsne-x', 'tsne-y']
    df = pd.DataFrame(X, columns=feat_cols)
    df['label'] = [str(i) for i in y]
    p = ggplot(df, aes(x='tsne-x', y='tsne-y', color='label')) + \
        geom_point() + \
        scale_color_brewer(type='diverging', palette=4)+ \
        xlab(" ") + ylab(" ") + \
        ggtitle("Ground Truth")
    p.save(option + '_Original_Clusters.png')
    #plt.clf()

    df['label'] = [str(i) for i in y_mean]
    p = ggplot(df, aes(x='tsne-x', y='tsne-y', color='label')) + \
        geom_point() + \
        scale_color_brewer(type='diverging', palette=4) + \
        ggtitle("Mean")+ xlab(" ") + ylab(" ")
    #   +theme(axis_text_x=ggplot.theme_blank(), axis_text_y=ggplot.theme_blank())
    p.save(option+'_Mean_Clusters.png')

    for i in xrange(len(confidentset)):
        y_conf = np.zeros_like(y)
        idx = confidentset[i]
        y_conf[idx] = 1
        df['label'] = [str(ii) for ii in y_conf]
        p = ggplot(df, aes(x='tsne-x', y='tsne-y', color='label')) + \
            geom_point() + scale_color_brewer(type='diverging', palette=4) \
            + ggtitle('Confset_'+str(i))+ xlab(" ") + ylab(" ")
        p.save(option+'_confset_'+str(i)+'.png')


    for i in xrange(len(Interset)):
        y_conf = np.zeros_like(y)
        idx = Interset[i]
        y_conf[idx] = 1
        df['label'] = [str(ii) for ii in y_conf]
        p = ggplot(df, aes(x='tsne-x', y='tsne-y', color='label')) + \
            geom_point() + \
            scale_color_brewer(type='diverging', palette=4) + \
            ggtitle('Interset_' + str(i))+ xlab(" ") + ylab(" ")
        p.save(option + '_interset_' + str(i) + '.png')

if __name__ == "__main__":
    dataset = 'rcv'
    n_bootstraps = 10
    method = 'dec'
    test = 0
    X, y, n_clusters, path = load_data(path="../results", dataset=dataset)

    option = 'dcn_dec'
    #option = 'dcn'
    #option = 'dec'
    y_mean, confidentset, Interset = ensemble_ana(X, y, n_bootstraps, option, path)

    print '.....'
    x_low = t_sne(X, file="../results/mnist/low_d")
    print x_low.shape
    draw_figure(x_low, y, y_mean, confidentset, Interset, option)
