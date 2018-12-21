import os
import shutil
import numpy as np
import csv
from sklearn.preprocessing import OneHotEncoder
from mnist_clustering import load_data
from scipy import io
from util import metrics
metrics = metrics()
class ota(object):
    def __init__(self, X, y, yb, n_bootstrep):
        self.X = X
        self.y = y
        self.n = self.X.shape[0]
        self.p = self.X.shape[1]
        self.yb = yb
        self.n_boostrep = n_bootstrep

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
        cmd = './labelsw_bunch2 -i zb.cls -o zb.ls -p zb.par -w zb.wt -h zb.h -c zb.summary -b ' + str(self.n_boostrep + 1) + ' -2'
        os.system(cmd + ' > tmp')
        idx = int(open('tmp', 'r').read())
        os.remove('tmp')
        os.chdir("..")
        os.chdir("..")
        print os.getcwd()
        return idx


    def get_repre(self, idx, threshold = 0.8, alpha = 0.1):
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

        cmd = "./labelsw -i zb.cls -o zb.ls -p zb.par -w zb.wt -h zb.h -c zb.summary -b " + str(self.n_boostrep + 1) + \
              " -t "+ str(threshold) +  " -a " + str(alpha) + " -2"
        os.system(cmd)
        dist = np.loadtxt("zb.par")
        wt = np.loadtxt("zb.wt")

        os.chdir("..")
        m = dist[:,0].astype('int')
        p_raw = []
        for i in xrange(self.n_boostrep):
            p_raw.append(OneHotEncoder().fit_transform(self.yb[0, self.n*i:self.n*(i+1)].reshape(self.n, 1)))

        p_tild = np.zeros((self.n, k_rf*self.n_boostrep))
        p_tild_sum = np.zeros((self.n, k_rf))

        for i in xrange(self.n_boostrep):
            wt_sub = wt[(sum(m[0:i])):sum(m[0:i+1]),]
            wt_row_sums = wt_sub.sum(axis=1)
            wt_row_sums[np.where(wt_row_sums) == 0 ] = 1
            wt_sub = wt_sub / wt_row_sums[:, np.newaxis]
            p_tild = np.matmul(p_raw[i].toarray(), wt_sub)
            p_tild_sum = p_tild_sum + p_tild

        p_bar = p_tild_sum / self.n_boostrep
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

