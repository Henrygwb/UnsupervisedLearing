import os
import shutil
import numpy as np
import collections
from sklearn.preprocessing import OneHotEncoder
from mnist_clustering import load_data
from scipy import io
from scipy.optimize import linprog
from util import metrics
metrics = metrics()

class MeanClustering(object):
    def __init__(self, y, yb, n_boostrap):
        self.y = y.astype('int')
        self.n = y.shape[0]
        self.yb = yb.astype('int')
        self.n_boostrap = n_boostrap

    def match(self, jaccard_dist, q1, q2, nc_1, nc_2):
        """
        Compute the Wasserstein distance and the corresponding soft matching matrix in eq(1)
        :param jaccard_dist: jaccard distance between two partition
        :param q1: significance weight of each cluster in partition 1
        :param q2: significance weight of each cluster in partition 2
        :param nc_1: number of clusters in partition 1
        :param nc_2: number of clusters in partition 2
        :return: mdist: distance between two partitions, wt_tmp: matching matrix: num_tmp * num_ref
        """
        # form the input for linprog
        c = jaccard_dist.flatten().tolist()
        b1 = q1.tolist()
        b2 = q2.tolist()
        b = b1 + b2
        a = np.zeros(((nc_1 + nc_2), (nc_1 * nc_2)))
        for i in xrange(nc_1):
            a[i, i * nc_2:(i + 1) * nc_2] = 1
        for i in xrange(nc_2):
            for j in xrange(nc_1):
                a[i + nc_1, j * nc_2 + i] = 1
        a = a.tolist()

        res = linprog(c, A_eq=a, b_eq=b, bounds=(0, None), method='simplex')
        if res.success == False:
            print 'Simplx did not succeed, should use IRM distance.'
            # todo: add fast match.
            mdist = 0
            wt_tmp = np.zeros((nc_1*nc_2))

        else:
            mdist = np.sum(np.multiply(res.x, jaccard_dist.flatten()))
            wt_tmp = res.x
            #print wt_tmp
        return mdist, wt_tmp

    def alignclusters(self, y_tmp, y_ref, nc_tmp, nc_ref, normalized = True):
        """
        Compute wassertein distance between two partitions
        :param y_tmp: partition 1
        :param y_ref: pratition 2
        :param nc_tmp: number of clusters in partition 1
        :param nc_ref: number of clusters in partition 2
        :param normalized: normalizing the jaccard distance
        :return: mdist: distance between two partitions, wt_tmp: matching matrix: num_tmp * num_ref
        """

        ## compute the jaccard distances for all cluster
        jaccard_dist = np.zeros((nc_tmp, nc_ref))
        for i in xrange(nc_tmp):
            for j in xrange(nc_ref):
                y_tmp_i = np.zeros_like(y_tmp)
                y_tmp_i[y_tmp == i] = 1

                y_ref_j = np.zeros_like(y_ref)
                y_ref_j[y_ref == j] = 1

                y_i_j = y_tmp_i + y_ref_j
                v3 = np.where(y_i_j == 2)[0].shape[0] # in cluster 1 and in cluster 2.
                if normalized == True:
                    jaccard_dist[i, j] = 1 - v3 / float(np.count_nonzero(y_i_j))
                else:
                    jaccard_dist[i,j] = np.where(y_i_j == 1)[0].shape[0]

        ## compute the weight for each cluster
        q1 = np.zeros((nc_tmp))
        count_1 = collections.Counter(y_tmp)
        for i in xrange(nc_tmp):
            q1[i] = float(count_1[i])/y_tmp.shape[0]
        # print q1

        q2 = np.zeros((nc_ref))
        count_2 = collections.Counter(y_ref)
        for i in xrange(nc_ref):
            q2[i] = float(count_2[i]) / y_ref.shape[0]
        # print q2

        mdist, wt_tmp = self.match(jaccard_dist, q1, q2, nc_tmp, nc_ref)

        wt_tmp[wt_tmp<0] = 0
        return mdist, wt_tmp

    def align(self, yb, y_ref, equalcls = True):
        """
        conduct alignment between the input partitions
        :param yb: bootstrap partitions
        :param y_ref: reference partition
        :param equalcls: force to have same number of clusters for each partition
        :return: wt: matching matrix
        :return: clsct: number of clusters in each partition
        :return: dc: distance of between bootstrap partitions and reference partition
        """
        clsct = np.zeros((self.n_boostrap,))
        for i in xrange(self.n_boostrap):
            clsct[i] = np.max(yb[i*self.n:(i+1)*self.n,])+1

        if equalcls:
            max_clsct = np.max(clsct)
            for i in xrange(self.n_boostrap):
                clsct[i] = max_clsct
        clsct = clsct.astype('int')
        clsct_ref = np.max(y_ref)+1

        dc = np.zeros((self.n_boostrap,))
        m = np.sum(clsct) * clsct_ref
        wt = np.zeros((m,))
        m = 0
        for i in xrange(self.n_boostrap):
            y_tmp = yb[i*self.n:(i+1)*self.n,]
            dc[i], wt[m:m+clsct[i]*clsct_ref,]= self.alignclusters(y_tmp, y_ref, clsct[i], clsct_ref)
            m += clsct[i]*clsct_ref
        return wt, clsct, dc

    def ref_idx(self, yb):
        """
        Conduct the alignment between the boostrap samples
        selecting the partition the has the minimum average distance to all the other partitions
        :param yb: bootstrap partitions
        :return: the index of reference partition
        """
        advdist = np.zeros((self.n_boostrap, ))
        for i in xrange(self.n_boostrap):
            yb_tmp = np.copy(yb)
            y_tmp_ref = np.copy(yb[i*self.n:(i+1)*self.n])
            _, _, dist= self.align(yb_tmp, y_tmp_ref)
            advdist[i] = sum(dist) / float(self.n_boostrap)
        idx_ref = int(np.argmin(advdist))
        print 'Index of the reference partition: %d' %idx_ref
        return idx_ref

    def ota(self, idx_ref = None):
        """
        Compute the mean partition by ota.
        :param idx_ref: index of reference partition
        :return: mean partition
        """
        # 1. select a reference partition.
        if idx_ref == None:
            idx_ref  = self.ref_idx(self.yb)

        # 2. compute w between reference partition and each bootstrap partitions
        k_rf = max(self.yb[(self.n * idx_ref):(self.n * (idx_ref+1))])+1
        y_ref = self.yb[(self.n * idx_ref):(self.n * (idx_ref+1))]

        wt, clsct, _ = self.align(self.yb, y_ref)

        # 3.1 compute the cluster-posterior of each bootstrap partition
        p_raw = []
        for i in xrange(self.n_boostrap):
            p_raw.append(OneHotEncoder().fit_transform(self.yb[self.n*i:self.n*(i+1),].reshape(self.n, 1)))

        # 3.2 transform bootstrap partitions to reference partition and sum them up
        p_t_r_sum = np.zeros((self.n, k_rf))
        m = 0
        for i in xrange(self.n_boostrap):
            wt_sub = wt[m:m+k_rf*clsct[i]]
            m += k_rf*clsct[i]
            wt_sub = wt_sub.reshape((clsct[i], k_rf))
            wt_row_sums = wt_sub.sum(axis=1)
            # print wt_row_sums
            wt_row_sums[np.where(wt_row_sums) == 0] = 1
            wt_sub = wt_sub / wt_row_sums[:, np.newaxis]
            p_t_r = np.matmul(p_raw[i].toarray(), wt_sub)
            p_t_r_sum = p_t_r_sum + p_t_r

        # 4. compute mean partition
        p_mean = p_t_r_sum / self.n_boostrap
        y_mean = np.argmax(p_mean, axis=1)

        acc = np.round(metrics.acc(self.y, y_mean), 5)
        nmi = np.round(metrics.nmi(self.y, y_mean), 5)
        ari = np.round(metrics.ari(self.y, y_mean), 5)
        print '****************************************'
        print('acc = %.5f, nmi = %.5f, ari = %.5f.' % (acc, nmi, ari))

        return y_mean

    def ota_costly(self):
        """
        Compute mean partition in a computationally intensive way.
        :return: mean partition
        """
        print "Compute a set of mean partitions by treating each bootstrap partiton as reference partition and select " \
              "the one with the minimum average distance."
        y_mean_all = np.zeros((self.n*self.n_boostrap,))
        for i in xrange(self.n_boostrap):
            y_mean_all[self.n*i:self.n*(i+1)] = self.ota(i)
        y_mean_all = y_mean_all.astype('int')
        selected_idx = self.ref_idx(y_mean_all)
        y_mean = y_mean_all[self.n*selected_idx:self.n*(selected_idx+1)]

        acc = np.round(metrics.acc(self.y, y_mean), 5)
        nmi = np.round(metrics.nmi(self.y, y_mean), 5)
        ari = np.round(metrics.ari(self.y, y_mean), 5)
        print '****************************************'
        print('acc = %.5f, nmi = %.5f, ari = %.5f.' % (acc, nmi, ari))

        return y_mean

    def align_bs(self, folder="align"):
        """
        Align bootstrap samples
        :param: folder: working folder
        :return: idx: index of reference partition
        """
        os.chdir("package5")
        if os.path.exists(folder) == False:
            os.mkdir(folder)
        shutil.copy2("labelsw_bunch2", os.path.join(folder, "labelsw_bunch2"))
        os.chdir(folder)
        yb_stack = self.yb.flatten()
        rfcls = np.zeros(self.n)
        ybcls = np.hstack((rfcls, yb_stack))
        np.savetxt("zb.cls", ybcls, fmt='%d')
        cmd = './labelsw_bunch2 -i zb.cls -o zb.ls -p zb.par -w zb.wt -h zb.h -c zb.summary -b ' + str(self.n_boostrap + 1) + ' -2'
        os.system(cmd + ' > tmp')
        idx = int(open('tmp', 'r').read())
        os.remove('tmp')
        os.chdir("..")
        os.chdir("..")
        print os.getcwd()
        return idx

    def get_mean(self, idx, threshold = 0.8, alpha = 0.1):
        """
        Find representative samples
        :param: idx: index of reference partition
        :return: y_mean: mean_partition
        """
        yb_stack = self.yb.flatten()
        k_rf = max(yb_stack[(self.n * idx):(self.n * (idx+1))])+1
        os.chdir("package5")

        rfcls = yb_stack[(self.n * idx):(self.n * (idx+1))]
        ybcls = np.hstack((rfcls, yb_stack))
        np.savetxt("zb.cls", ybcls, fmt='%d')

        cmd = "./labelsw -i zb.cls -o zb.ls -p zb.par -w zb.wt -h zb.h -c zb.summary -b " + str(self.n_boostrap + 1) + \
              " -t "+ str(threshold) +  " -a " + str(alpha) + " -2"
        os.system(cmd)
        dist = np.loadtxt("zb.par")
        wt = np.loadtxt("zb.wt")

        os.chdir("..")
        m = dist[:,0].astype('int')
        p_raw = []
        for i in xrange(self.n_boostrap):
            p_raw.append(OneHotEncoder().fit_transform(self.yb[0, self.n*i:self.n*(i+1)].reshape(self.n, 1)))

        p_tild = np.zeros((self.n, k_rf*self.n_boostrap))
        p_tild_sum = np.zeros((self.n, k_rf))

        for i in xrange(self.n_boostrap):
            wt_sub = wt[(sum(m[0:i])):sum(m[0:i+1]),]
            wt_row_sums = wt_sub.sum(axis=1)
            wt_row_sums[np.where(wt_row_sums) == 0 ] = 1
            wt_sub = wt_sub / wt_row_sums[:, np.newaxis]
            p_tild = np.matmul(p_raw[i].toarray(), wt_sub)
            p_tild_sum = p_tild_sum + p_tild

        p_bar = p_tild_sum / self.n_boostrap
        y_mean = np.argmax(p_bar, axis=1)

        acc = np.round(metrics.acc(self.y, y_mean), 5)
        nmi = np.round(metrics.nmi(self.y, y_mean), 5)
        ari = np.round(metrics.ari(self.y, y_mean), 5)
        print '****************************************'
        print('acc = %.5f, nmi = %.5f, ari = %.5f.' % (acc, nmi, ari))

        return y_mean


class ClusterAnalysis(MeanClustering):
    def __init__(self, threshold = 0.8, alpha = 0.1, **kwargs):
        super(MeanClustering, self).__init__(**kwargs)
        self.thred = threshold
        self.alpha = alpha

    def entropy(self, dist, metric = 'mean'):
        """
        Overall clustering stability
        :return: p_mean, stability of the partition
        """
        if metric == 'mean':
            p_mean = np.mean(dist)
        elif metric == 'entropy':
            std = np.std(dist)
            cate = round(dist/std)
            idx, count = np.unique(cate, return_counts=True)
            px = count/float(sum(count))
            lpx = np.log2(px)
            p_mean = -sum(px*lpx)
        return p_mean

    def statforclst(self):
        """
        Statistics for stability of a result of clustering
        :return:
        """
        
        return 0



    def confset(self):
        """
        Confident set
        :return:
        """
        return 0


if __name__  == '__main__':
    n_bs = 10
    n_sample = 5000
    yb = np.genfromtxt('zb.cls')[5000:]
    y = yb[0*n_sample:1*n_sample]
    mean_test = MeanClustering(y, yb, n_bs)
    y_mean = mean_test.ota()
    y_mean = mean_test.ota_costly()

"""
    X, y = load_data("../results/mnist")
    n_boostrap = 10
    yb = io.loadmat("../results/mnist/dcn_17/0_bs/0_results")['y_pred']
    for i in xrange(n_boostrap):
        i_tmp = i + 1
        path = "../results/mnist/dcn_17/" + str(i_tmp)+"_bs/" + str(i_tmp)+ "_results"
        yb_tmp = io.loadmat(path)['y_pred']
        yb = np.hstack((yb, yb_tmp))

    yb_2 = io.loadmat("../results/mnist/dec_16/0_bs/0_results")['y_pred']
    for i in xrange(n_boostrap):
        i_tmp = i + 1
        path = "../results/mnist/dec_16/" + str(i_tmp) + "_bs/" + str(i_tmp) + "_results"
        yb_tmp = io.loadmat(path)['y_pred']
        yb_2 = np.hstack((yb_2, yb_tmp))

    yb = np.hstack((yb, yb_2))
    ota_test = ota(X, y, yb, 2*(n_boostrap+1))
    idx = ota_test.align_bs(folder='align_test')
    y_mean = ota_test.get_repre(idx)
"""
    ######## call c package
