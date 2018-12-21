import os
import shutil
import numpy as np
import csv
from sklearn.preprocessing import OneHotEncoder
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
        os.chdir("package5")
        yb_stack = self.yb.flatten()
        self.n = yb_stack.shape[0]/self.n_boostrep
        rfcls = np.zeros(self.n)
        ybcls = np.hstack((rfcls, yb_stack))
        with open('zb.cls', mode='w') as f:
            f.write(ybcls)
        cmd = './labelsw_bunch2 -i zb.cls -o zb.ls -p zb.par -w zb.wt -h zb.h -c zb.summary -b ' + str(self.n_boostrep + 1) + ' -2'
        os.system(cmd + ' < tmp')
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
        k_rf = yb_stack[(self.n * (idx - 1) + 1):(self.n * idx)]
        os.chdir("package5")

        rfcls = yb_stack[(self.n * (idx - 1) + 1):(self.n * idx)]
        ybcls = np.hstack((rfcls, yb_stack))
        with open('zb.cls', mode='w') as f:
            f.write(ybcls)
        cmd = "./labelsw -i zb.cls -o zb.ls -p zb.par -w zb.wt -h zb.h -c zb.summary -b " + str(self.n_boostrep + 1) + \
              " -t "+ str(threshold) +  " -a " + str(alpha) + " -2"
        os.system(cmd)
        dist = np.loadtxt("zb.par")
        wt = np.loadtxt("zb.wt")

        os.chdir("..")
        m = dist[:,0]
        p_raw = []
        for i in xrange(self.n_boostrep):
            p_raw[i] = OneHotEncoder().fit_transform(self.yb[i,:])

        p_tild = np.zeros((self.n, k_rf*self.n_boostrep))
        p_tild_sum = np.zeros((self.n, k_rf))

        for i in xrange(self.n_boostrep):
            wt_sub = wt[(sum(m[0:(i - 1)])):sum(m[0:i]),]
            wt_row_sums = wt_sub.sum(axis=1)
            wt_row_sums[np.where(wt_row_sums) == 0 ] = 1
            wt_sub = wt_sub / wt_row_sums[:, np.newaxis]
            p_tild = np.matmul(p_raw[i], wt_sub)
            p_tild_sum = p_tild_sum + p_tild

        p_bar = p_tild_sum / self.n_boostrep
        p_bar_hrd_asgn = np.argmax(p_bar, axis=1)

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



