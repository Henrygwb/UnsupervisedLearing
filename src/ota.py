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


class ota(object):
    def __init__(self, X, y, yb, n_bootstrap):
        self.X = X
        self.y = y
        self.n = self.X.shape[0]
        self.p = self.X.shape[1]
        self.yb = yb
        self.n_boostrap = n_bootstrap

    def match(self, jaccard_dist, q1, q2, nc_1, nc_2):
        """
        Compute the Wasserstein distance and the corresponding soft matching matrix in eq(1)
        :param jaccard_dist: jaccard distance between two partition
        :param q1: significance weight of each cluster in partition 1
        :param q2: significance weight of each cluster in partition 2
        :param nc_1: number of clusters in partition 1
        :param nc_2: number of clusters in partition 2
        :return: dis: distance between two partitions, wt_tmp: matching matrix: num_tmp * num_ref
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
        :return: dis: distance between two partitions, wt_tmp: matching matrix: num_tmp * num_ref
        """

        ## compute the jaccard distances for all cluster
        jaccard_dist = np.zeros((nc_tmp, nc_ref))
        for i in xrange(nc_tmp):
            for j in xrange(nc_ref):
                nc_tmp_i = np.zeros_like(nc_tmp)
                nc_tmp_i[y_tmp == i] = 1

                nc_ref_j = np.zeros_like(nc_ref)
                nc_ref_j[y_ref == j] = 1

                nc_i_j = nc_tmp_i + nc_ref_j
                v3 = np.where(nc_i_j == 2)[0].shape[0] # in cluster 1 and in cluster 2.
                if normalized == True:
                    jaccard_dist[i, j] = 1 - v3 / float(np.count_nonzero(nc_i_j))
                else:
                    jaccard_dist[i,j] = np.where(nc_i_j == 1)[0].shape[0]

        ## compute the weight for each cluster
        q1 = np.zeros((nc_tmp))
        count_1 = collections.Counter(nc_tmp)
        for i in xrange(nc_tmp):
            q1[i] = float(count_1[i])/nc_tmp.shape[0]

        q2 = np.zeros((y_tmp))
        count_2 = collections.Counter(y_ref)
        for i in xrange(nc_tmp):
            q2[i] = float(count_2[i]) / nc_ref.shape[0]

        mdist, wt_tmp = self.match(jaccard_dist, q1, q2, nc_tmp, nc_ref)

        wt_tmp[wt_tmp<0] = 0
        return mdist, wt_tmp

    def align(self, yb, equalcls = False):
        """
        conduct alignment between the input partitions
        :param yb: input partitions, with the first one as the reference partition
        :param equalcls: force to have same number of clusters for each partition
        :return:
        """
        clsct = np.zeros((self.n_boostrap,))
        for i in xrange(self.n_boostrap):
            clsct[i] = np.max(yb[i*self.n:(i+1)*self.n,])

        if equalcls:
            max_clsct = np.max(clsct)
            for i in xrange(self.n_boostrap):
                clsct[i] = max_clsct

        dc = np.zeros((self.n_boostrap,))
        m = np.sum(clsct) * clsct[0]
        wt = np.zeros((m,))
        m = 0
        y_ref = yb[0:self.n,]
        for i in xrange(self.n_boostrap):
            y_tmp = yb[i*self.n:(i+1)*self.n,]
            dc[i], wt[m:clsct[i]*clsct[0],]= self.alignclusters(y_tmp, y_ref, clsct[i], clsct[0])
            m = clsct[i]*clsct[0]
        return wt, dc,

    def get_ref_idx(self):
        """
        Conduct the alignment between the boostrap samples
        selecting the partition the has the minimum average distance to all the other partitions
        :return: the index of reference partition
        """
        advdist = np.zeros((self.n_boostrap-1, ))
        for i in xrange(self.n_boostrap):
            yb_tmp = np.copy(self.yb)
            yb_tmp[0:self.n, ] = np.copy(self.yb[i*self.n:(i+1)*self.n])
            wt, dist = self.align(yb_tmp)
            advdist[i] = sum(dist) / float(self.n_boostrap)
        idx_ref = np.argmin(advdist)
        return idx_ref

    def align_bs(self, folder="align"):
        """
        Align bootstrap samples
        :return:
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


    def get_repre(self, idx, threshold = 0.8, alpha = 0.1, double = False):
        """
        Find representative samples
        :return:
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
        p_bar_hrd_asgn = np.argmax(p_bar, axis=1)

        acc = np.round(metrics.acc(self.y, p_bar_hrd_asgn), 5)
        nmi = np.round(metrics.nmi(self.y, p_bar_hrd_asgn), 5)
        ari = np.round(metrics.ari(self.y, p_bar_hrd_asgn), 5)
        print '****************************************'
        print('acc = %.5f, nmi = %.5f, ari = %.5f.' % (acc, nmi, ari))

        return p_bar_hrd_asgn


    def statforclst(self):
        """
        Statistics for stability of a result of clustering
        :return:
        """
        return 0

    def entropy(self):
        """
        Overall clustering stability
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




    ######## call c package
    # def align_bs(self, folder="align"):
    #     """
    #     Align bootstrap samples
    #     :return:
    #     """
    #     os.chdir("package5")
    #     if os.path.exists(folder) == False:
    #         os.mkdir(folder)
    #     shutil.copy2("labelsw_bunch2", os.path.join(folder, "labelsw_bunch2"))
    #     os.chdir(folder)
    #     yb_stack = self.yb.flatten()
    #     rfcls = np.zeros(self.n)
    #     ybcls = np.hstack((rfcls, yb_stack))
    #     np.savetxt("zb.cls", ybcls, fmt='%d')
    #     cmd = './labelsw_bunch2 -i zb.cls -o zb.ls -p zb.par -w zb.wt -h zb.h -c zb.summary -b ' + str(self.n_bootstrap + 1) + ' -2'
    #     os.system(cmd + ' > tmp')
    #     idx = int(open('tmp', 'r').read())
    #     os.remove('tmp')
    #     os.chdir("..")
    #     os.chdir("..")
    #     print os.getcwd()
    #     return idx
    #
    #
    # def get_repre(self, idx, threshold = 0.8, alpha = 0.1, double = False):
    #     """
    #     Find representative samples
    #     :return:
    #     """
    #     yb_stack = self.yb.flatten()
    #     k_rf = max(yb_stack[(self.n * idx):(self.n * (idx+1))])+1
    #     os.chdir("package5")
    #
    #     rfcls = yb_stack[(self.n * idx):(self.n * (idx+1))]
    #     ybcls = np.hstack((rfcls, yb_stack))
    #     np.savetxt("zb.cls", ybcls, fmt='%d')
    #
    #     cmd = "./labelsw -i zb.cls -o zb.ls -p zb.par -w zb.wt -h zb.h -c zb.summary -b " + str(self.n_bootstrap + 1) + \
    #           " -t "+ str(threshold) +  " -a " + str(alpha) + " -2"
    #     os.system(cmd)
    #     dist = np.loadtxt("zb.par")
    #     wt = np.loadtxt("zb.wt")
    #
    #     os.chdir("..")
    #     m = dist[:,0].astype('int')
    #     p_raw = []
    #     for i in xrange(self.n_bootstrap):
    #         p_raw.append(OneHotEncoder().fit_transform(self.yb[0, self.n*i:self.n*(i+1)].reshape(self.n, 1)))
    #
    #     p_tild = np.zeros((self.n, k_rf*self.n_bootstrap))
    #     p_tild_sum = np.zeros((self.n, k_rf))
    #
    #     for i in xrange(self.n_bootstrap):
    #         wt_sub = wt[(sum(m[0:i])):sum(m[0:i+1]),]
    #         wt_row_sums = wt_sub.sum(axis=1)
    #         wt_row_sums[np.where(wt_row_sums) == 0 ] = 1
    #         wt_sub = wt_sub / wt_row_sums[:, np.newaxis]
    #         p_tild = np.matmul(p_raw[i].toarray(), wt_sub)
    #         p_tild_sum = p_tild_sum + p_tild
    #
    #     p_bar = p_tild_sum / self.n_bootstrap
    #     p_bar_hrd_asgn = np.argmax(p_bar, axis=1)
    #
    #     acc = np.round(metrics.acc(self.y, p_bar_hrd_asgn), 5)
    #     nmi = np.round(metrics.nmi(self.y, p_bar_hrd_asgn), 5)
    #     ari = np.round(metrics.ari(self.y, p_bar_hrd_asgn), 5)
    #     print '****************************************'
    #     print('acc = %.5f, nmi = %.5f, ari = %.5f.' % (acc, nmi, ari))
    #
    #     return p_bar_hrd_asgn
