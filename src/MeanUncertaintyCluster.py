import os
import shutil
import numpy as np
import collections
from sklearn.preprocessing import OneHotEncoder
from scipy import io
from scipy.optimize import linprog
from util import metrics
metrics = metrics()
import math


class ParititionAlignment(object):
    def __init__(self, y, yb, n_bootstrap):
        self.y = y.astype('int')
        self.n = y.shape[0]
        self.yb = yb.astype('int')
        self.n_bootstrap = n_bootstrap

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
                if np.count_nonzero(y_i_j) == 0:
                    jaccard_dist[i,j] = 1
                else:
                    if normalized == True:
                        jaccard_dist[i, j] = 1 - v3 / float(np.count_nonzero(y_i_j))
                    else:
                        jaccard_dist[i,j] = np.where(y_i_j == 1)[0].shape[0]

        ## compute the weight for each cluster
        q1 = np.zeros((nc_tmp))
        count_1 = collections.Counter(y_tmp)
        for i in xrange(nc_tmp):
            if i in count_1.keys():
                q1[i] = float(count_1[i])/y_tmp.shape[0]
            else:
                q1[i] = 0
        # print q1

        q2 = np.zeros((nc_ref))
        count_2 = collections.Counter(y_ref)
        for i in xrange(nc_ref):
            if i in count_1.keys():
                q2[i] = float(count_2[i]) / y_ref.shape[0]
            else:
                q2[i] = 0
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
        clsct = np.zeros((self.n_bootstrap,))
        for i in xrange(self.n_bootstrap):
            clsct[i] = np.max(yb[i*self.n:(i+1)*self.n,])+1

        if equalcls:
            max_clsct = np.max(clsct)
            for i in xrange(self.n_bootstrap):
                clsct[i] = max_clsct
        clsct = clsct.astype('int')
        clsct_ref = np.max(y_ref)+1

        dc = np.zeros((self.n_bootstrap,))
        m = np.sum(clsct) * clsct_ref
        wt = np.zeros((m,))
        m = 0
        for i in xrange(self.n_bootstrap):
            y_tmp = yb[i*self.n:(i+1)*self.n,]
            dc[i], wt[m:m+clsct[i]*clsct_ref,]= self.alignclusters(y_tmp, y_ref, clsct[i], clsct_ref)
            m += clsct[i]*clsct_ref
        return wt, clsct, dc

    def ref_idx(self, yb, strategy):
        """
        Conduct the alignment between the bootstrap samples
        selecting the partition the has the minimum average distance to all the other partitions
        :param yb: bootstrap partitions
        :return: the index of reference partition
        """
        advdist = np.zeros((self.n_bootstrap, ))
        for i in xrange(self.n_bootstrap):
            yb_tmp = np.copy(yb)
            y_tmp_ref = np.copy(yb[i*self.n:(i+1)*self.n])
            _, _, dist= self.align(yb_tmp, y_tmp_ref)
            if strategy == 0:
                advdist[i] = sum(dist) / float(self.n_bootstrap)
            else:
                advdist[i] = min(np.delete(dist, i))

        idx_ref = int(np.argmin(advdist))
        _, _, dist = self.align(yb, yb[idx_ref*self.n:(idx_ref+1)*self.n])
        print 'Index of the reference partition: %d' %idx_ref

        return idx_ref, dist

    def ota(self, idx_ref = None, strategy = 0):
        """
        Compute the mean partition by ota.
        :param idx_ref: index of reference partition
        :return: mean partition
        """
        # 1. select a reference partition.
        if idx_ref == None:
            idx_ref, _  = self.ref_idx(self.yb, strategy)

        # 2. compute w between reference partition and each bootstrap partitions
        k_rf = max(self.yb[(self.n * idx_ref):(self.n * (idx_ref+1))])+1
        y_ref = self.yb[(self.n * idx_ref):(self.n * (idx_ref+1))]

        wt, clsct, dist = self.align(self.yb, y_ref)

        # 3.1 compute the cluster-posterior of each bootstrap partition
        p_raw = []
        for i in xrange(self.n_bootstrap):
            p_raw.append(OneHotEncoder().fit_transform(self.yb[self.n*i:self.n*(i+1),].reshape(self.n, 1)))

        # 3.2 transform bootstrap partitions to reference partition and sum them up
        p_t_r_sum = np.zeros((self.n, k_rf))
        m = 0
        for i in xrange(self.n_bootstrap):
            wt_sub = wt[m:m+k_rf*clsct[i]]
            m += k_rf*clsct[i]
            wt_sub = wt_sub.reshape((clsct[i], k_rf))
            wt_row_sums = wt_sub.sum(axis=1)
            # print wt_row_sums
            wt_row_sums[np.where(wt_row_sums == 0)] = 1
            wt_sub = wt_sub / wt_row_sums[:, np.newaxis]
            p_t_r = np.matmul(p_raw[i].toarray(), wt_sub)
            p_t_r_sum = p_t_r_sum + p_t_r

        # 4. compute mean partition
        p_mean = p_t_r_sum / self.n_bootstrap
        y_mean = np.argmax(p_mean, axis=1)

        acc = np.round(metrics.acc(self.y, y_mean), 5)
        nmi = np.round(metrics.nmi(self.y, y_mean), 5)
        ari = np.round(metrics.ari(self.y, y_mean), 5)
        print '****************************************'
        print('acc = %.5f, nmi = %.5f, ari = %.5f.' % (acc, nmi, ari))

        return y_mean, dist

    def ota_costly(self):
        """
        Compute mean partition in a computationally intensive way.
        :return: mean partition
        """
        print "Compute a set of mean partitions by treating each bootstrap partiton as reference partition and select " \
              "the one with the minimum average distance."
        y_mean_all = np.zeros((self.n*self.n_bootstrap,))
        for i in xrange(self.n_bootstrap):
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
        cmd = './labelsw_bunch2 -i zb.cls -o zb.ls -p zb.par -w zb.wt -h zb.h -c zb.summary -b ' + str(self.n_bootstrap + 1) + ' -2'
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

        cmd = "./labelsw -i zb.cls -o zb.ls -p zb.par -w zb.wt -h zb.h -c zb.summary -b " + str(self.n_bootstrap + 1) + \
              " -t "+ str(threshold) +  " -a " + str(alpha) + " -2"
        os.system(cmd)
        dist = np.loadtxt("zb.par")
        wt = np.loadtxt("zb.wt")

        os.chdir("..")
        m = dist[:,0].astype('int')
        p_raw = []
        for i in xrange(self.n_bootstrap):
            p_raw.append(OneHotEncoder().fit_transform(self.yb[0, self.n*i:self.n*(i+1)].reshape(self.n, 1)))

        p_tild = np.zeros((self.n, k_rf*self.n_bootstrap))
        p_tild_sum = np.zeros((self.n, k_rf))

        for i in xrange(self.n_bootstrap):
            wt_sub = wt[(sum(m[0:i])):sum(m[0:i+1]),]
            wt_row_sums = wt_sub.sum(axis=1)
            wt_row_sums[np.where(wt_row_sums == 0)] = 1
            wt_sub = wt_sub / wt_row_sums[:, np.newaxis]
            p_tild = np.matmul(p_raw[i].toarray(), wt_sub)
            p_tild_sum = p_tild_sum + p_tild

        p_bar = p_tild_sum / self.n_bootstrap
        y_mean = np.argmax(p_bar, axis=1)

        acc = np.round(metrics.acc(self.y, y_mean), 5)
        nmi = np.round(metrics.nmi(self.y, y_mean), 5)
        ari = np.round(metrics.ari(self.y, y_mean), 5)
        print '****************************************'
        print('acc = %.5f, nmi = %.5f, ari = %.5f.' % (acc, nmi, ari))
        return y_mean

    def par_stability(self, yb, strategy=0):
        """
        Overall clustering stability
        :return: p_mean, stability of the partition
        """
        advdist = np.zeros((self.n_bootstrap, ))
        for i in xrange(self.n_bootstrap):
            yb_tmp = np.copy(yb)
            y_tmp_ref = np.copy(yb[i*self.n:(i+1)*self.n])
            _, _, dist= self.align(yb_tmp, y_tmp_ref)
            if strategy == 0:
                advdist[i] = sum(dist) / float(self.n_bootstrap)
            else:
                advdist[i] = min(np.delete(dist, i))
        return advdist

class ClusterAnalysis(object):
    def __init__(self, yb, n_bootstrap, y_mean, len):
        self.y_mean = y_mean
        self.yb = yb.reshape(n_bootstrap, len)
        self.n_bootstrap = n_bootstrap

    def covercmp(self, wt_tmp, wt_ref, n_tmp, threshold):
        """
        :param wt_tmp: one column of the raw-normalized wt
        :param wt_ref: one column of the column-normalized wt
        :param n_tmp: num of raws in wt_tmp and wt_ref
        :param thredshold: threshold fro match
        :return: n: number of clusters(raw) of wt_tmp that > threshold
        :return: m: argmax(wt_ref[wt_tmp > threshold])
        :return: v2: max(wt_ref[wt_tmp > threshold])
        """
        """
        n = 0
        m = 0
        v1 = v2 = 0
        for i in xrange(n_tmp):
            if wt_tmp[i] > threshold:
                n += 1
                v1 += wt_ref[i]
                if wt_ref[i] > v2:
                    v2 = wt_ref[i]
                    m = i
        """
        n = 0
        m = 0
        v1 = 0
        v2 = 0
        cov = np.zeros((n_tmp,)) - 1
        n = np.where(wt_tmp>threshold)[0].shape[0]
        if n != 0:
            v1 = sum(wt_ref[wt_tmp>threshold])
            m = np.where(wt_tmp>threshold)[0][np.argmax(wt_ref[wt_tmp>threshold])]
            v2 = max(wt_ref[wt_tmp>threshold])

            cov[wt_tmp>threshold] = wt_ref[wt_tmp>threshold]
        return n, m, v1, v2, cov

    def access(self, wt_sub, n_tmp, n_rf, threshold):
        """
        :param wt_sub: weight
        :param n_tmp: num of cluster
        :param n_rf: num of cluster in reference
        :param threshold:
        :return: topological relationship
        """
        wt_sub_raw = np.copy(wt_sub.reshape((n_tmp, n_rf)))
        wt_sub_raw_sum = np.sum(wt_sub_raw, axis=1)
        for i in xrange(n_tmp):
            wt_sub_raw[i,] = wt_sub_raw[i,]/wt_sub_raw_sum[i]

        wt_sub_col = np.copy(wt_sub.reshape((n_tmp, n_rf)))
        wt_sub_col_sum = np.sum(wt_sub_col, axis=0)
        for j in xrange(n_rf):
            wt_sub_col[:,j] = wt_sub_col[:,j]/wt_sub_col_sum[j]


        code = np.zeros((n_rf, ))
        nf = np.zeros((n_rf, ))
        res_ = np.zeros((n_tmp,n_rf))
        coverage = np.zeros((max(n_tmp, n_rf), ))

        for i in xrange(n_rf):
            wtcmp = wt_sub_raw[:, i]
            wtref = wt_sub_col[:, i]
            n,m,v1,v2, cov = self.covercmp(wtcmp, wtref, n_tmp, threshold)
            res_[:,i] = cov
            nf[i] = n
            if v2 > threshold:
                code[i] = 0
            elif v1 > threshold:
                    code[i] = 1
            else:
                v3 = max(wtref)
                m = np.argmax(wtref)
                res_[:,i] = -1
                if v3 > threshold:
                    _, nf[i], v4, v5, cov= self.covercmp(wt_sub_col[m, :], wt_sub_raw[m,:], n_rf, threshold)
                    if v4 > threshold:
                        code[i] = 2
                        res_[m, i] = cov[i] + 2
                    else:
                        code[i] = 3
                        nf[i] = 0
                else:
                    code[i] = 3
                    nf[i] = 0
        return code, nf, res_.flatten()

    def matchsplit(self, wt = None, clsct = None, threshold = 0.8):
        """
        :param y_mean: mean partition
        :param wt: weight
        :param clsct: mun of clusters
        :param threshold:
        :return: topological relationship
        """
        k_rf = wt.shape[0]/sum(clsct)
        clsct = clsct.astype('int')

        codect = np.zeros((k_rf, 4))
        nfave = np.zeros((k_rf, 4))
        res = np.zeros((sum(clsct)*k_rf, ))

        m = 0
        for i in xrange(self.n_bootstrap):
            code, nf, res[m:m+k_rf*clsct[i],] = self.access(wt[m:m+k_rf*clsct[i],], clsct[i], k_rf, threshold)
            code = code.astype('int')
            for j in xrange(k_rf):
                codect[j, code[j]] += 1
                nfave[j, code[j]] += nf[j]
            m += k_rf*clsct[i]
        nfave = nfave/codect
        nfave[np.isnan(nfave)] = 0
        
        return codect, nfave, res

    def hardassign(self):
        """
        hard assignment of topological relationship
        :return:
        """

        return 0

    def matchcluster(self, res, clsct, threshold = 0.8, usesplit = False):
        k_rf = res.shape[0]/sum(clsct)
        n0 = min(clsct)
        matched_cluster_id = np.zeros((k_rf, (self.n_bootstrap+1)))-1
        matched_sample_id = [[None for i in range((self.n_bootstrap+1)) ] for j in range(k_rf)]

        for i in xrange(k_rf):
            m = 0
            id_ref = np.where(self.y_mean==i)[0]
            matched_cluster_id[i, 0] = i
            matched_sample_id[i][0] = id_ref
            for j in xrange(self.n_bootstrap):
                n1 = clsct[j]
                res_tmp = res[m:m+k_rf*n1].reshape(n1, k_rf)[:, i]
                if res_tmp[(res_tmp>=threshold)&(res_tmp<=1)].shape[0] != 0:
                    cluster_matched = np.where((res_tmp>=threshold)&(res_tmp<=1))[0][0] # 0.8 <= x <= 1
                    matched_cluster_id[i, j+1] = cluster_matched
                    sample_matched = np.where(self.yb[j]==cluster_matched)[0]
                    matched_sample_id[i][j+1] = sample_matched
                m += k_rf * n1
        return matched_cluster_id, matched_sample_id

    def interset(self, matched_sample_id, matched_cluster_id):
        k_rf = matched_cluster_id.shape[0]
        Interset=[None for i in xrange(k_rf)]
        for i in xrange(k_rf):
            Interset_tmp = matched_sample_id[i][0]
            for j in xrange(self.n_bootstrap):
                if matched_cluster_id[i][j+1] !=-1:
                    Interset_tmp = np.intersect1d(Interset_tmp, matched_sample_id[i][j+1])
            Interset[i] = Interset_tmp
        return Interset

    def confset(self, matched_sample_id, matched_cluster_id, alpha = 0.1):
        """
        Least impact first removal for computing Confident set for a set of matched clusters
        start with a union of all the clusters and remove one point in each iteration
        terminate when the consraint cannot hold if any extra point is taken out.
        :return:
        """
        k_rf = matched_cluster_id.shape[0]
        confidentset=[None for i in xrange(k_rf)]
        SS = np.zeros_like(matched_cluster_id) - 1
        for i in xrange(k_rf):
            #print i
            # 1. find a set of matched clusters and get the union set of the clusters: S active set, S* union set, H*: union sample id
            sample_matched = []
            I = []
            S = []
            m = 0
            for ii in xrange(len(matched_sample_id[i])):
                if matched_sample_id[i][ii] is not None:
                    I.append(m)
                    S.append(ii)
                    m = m+1
                    sample_matched.append(matched_sample_id[i][ii])
            sample_matched = np.array(sample_matched)
            H = np.unique(np.hstack(sample_matched))

            M = I[-1]+1

            flag = 1
            while flag == 1:
                # 2. get nl for each point and Il
                Nl = np.zeros_like(H)
                for id in xrange(H.shape[0]):
                    xl_id = H[id]
                    nl = 0
                    for cls in xrange(sample_matched.shape[0]):
                        s_tmp = sample_matched[cls,]
                        if xl_id in s_tmp:
                            nl += 1
                    Nl[id] = nl

                # 3. find the minimum nl and the corresponding xl_id

                min_l = np.min(Nl)
                min_xl_id = H[np.argmin(Nl)]
                if (M - min_l) > m*(1-alpha):
                    # update m*
                    M = M - min_l
                    # update I*
                    for cls in xrange(sample_matched.shape[0]):
                        s_tmp = sample_matched[cls,]
                        if min_xl_id in s_tmp:
                            I.remove(cls)
                            del S[cls]

                    matched_id = []
                    for iii in I:
                        matched_id.append(sample_matched[iii])
                    ##  update S*, that is sample_matched
                    sample_matched = np.array(matched_id)
                    ## update H*
                    H = np.unique(np.hstack(sample_matched))
                    I = [jj for jj in xrange(sample_matched.shape[0])]

                elif math.ceil(M - min_l) <= m*(1-alpha):
                    flag = 0
                    confidentset[i] = H
                    SS[i, S] = matched_cluster_id[i, S]
        
        return confidentset, SS

    def clu_stability(self, s_confset, s):
        """
        :param s_confset: confit set of the cluster
        :param s: a collection of mathched clusters
        :return: atr: average tightness ratio; acr: average coverage ratio.
        """
        atr = []
        acr = []
        conf_len = float(s_confset.shape[0])
        m1 = 0
        m2 = 0
        for i in xrange(len(s)):
            s_tmp = s[i]
            s_ist_tmp = np.intersect1d(s_tmp, s_confset)
            if np.array_equal(s_ist_tmp, s_tmp) == True:
                m1 += 1
                atr.append(s_tmp.shape[0] / conf_len)
            else:
                m2 += 1
                acr.append(s_ist_tmp.shape[0] / float(s_tmp.shape[0]))
        if m1!= 0:
            atr = sum(atr) / m1
        else:
            atr = 0
        if m2!= 0:
            acr = sum(acr)/m2
        else:
            acr = 0
        return atr, acr

    def clu_dist(self, confset, i, j):
        """
        :param y: cluster result
        :param i: ith cluster
        :param j: jth cluster
        :return: cap_i_j clusuter alignment and points based separability between i and j
        """
        s_i = confset[i]
        s_j = confset[j]
        cap_i_j = metrics.jac(s_i, s_j)
        return cap_i_j

def ota_c(yb, num_sample, n_bootstrap, return_mean):
    folder = 'tmp'
    if os.path.exists(folder) == False:
        os.mkdir(folder)
    shutil.copy2("labelsw_bunch2", os.path.join(folder, "labelsw_bunch2"))
    shutil.copy2("labelsw", os.path.join(folder, "labelsw"))
    os.chdir('tmp')
    yb_stack = yb.flatten()
    rfcls = np.zeros(num_sample)
    ybcls = np.hstack((rfcls, yb_stack)).astype('int')
    ybcls.tofile('zb.cls', sep='\n')

    cmd = './labelsw_bunch2 -i zb.cls -o zb.ls -p zb.par -w zb.wt -h zb.h -c zb.summary -b ' + str(n_bootstrap + 1) + ' -2'
    os.system(cmd + ' > tmp_res')
    idx = int(open('tmp_res', 'r').read())
    os.remove('tmp_res')

    rfcls = ybcls[(num_sample * idx):(num_sample * (idx+1))]
    k_rf = np.unique(rfcls).shape[0]
    ybcls = np.hstack((rfcls, yb_stack)).astype('int')
    ybcls.tofile('zb.cls', sep='\n')
    cmd = "./labelsw -i zb.cls -o zb.ls -p zb.par -w zb.wt -h zb.h -c zb.summary -b " + str(n_bootstrap + 1) + \
          " -t " + str(0.8) + " -a " + str(0.1) + " -2"
    os.system(cmd)
    clst = np.loadtxt("zb.par")[:, 0][1:].astype('int')
    wt = np.loadtxt("zb.wt")
    dist = np.loadtxt('zb.par')[:,1][1:]

    p_tild_sum = np.zeros((num_sample, k_rf))

    m = 0
    for i in xrange(n_bootstrap):
        p_raw_tmp = OneHotEncoder().fit_transform(
            np.expand_dims(yb_stack[i * num_sample: (i+1) * num_sample], axis=1)).toarray()
        wt_sub = wt[m:m+clst[i], ]
        wt_row_sums = wt_sub.sum(axis=1)
        wt_row_sums[np.where(wt_row_sums == 0)] = 1
        wt_sub = wt_sub / wt_row_sums[:, np.newaxis]

        p_tild = np.matmul(p_raw_tmp, wt_sub)
        p_tild_sum = p_tild_sum + p_tild
        m = m+clst[i]

    p_bar = p_tild_sum / n_bootstrap
    if return_mean == 1:
        y_mean = np.argmax(p_bar, axis=1)
    else:
        y_mean = rfcls
    os.chdir('..')
    os.system('rm -r tmp')
    return y_mean, dist

def par_stability_c(yb, n_bootstrap, num_sample):
    folder = 'tmp'
    if os.path.exists(folder) == False:
        os.mkdir(folder)
    shutil.copy2("labelsw_bunch2", os.path.join(folder, "labelsw_bunch2"))
    shutil.copy2("labelsw", os.path.join(folder, "labelsw"))
    os.chdir('tmp')
    advdist = np.zeros((n_bootstrap,))
    for idx in xrange(n_bootstrap):
        rfcls = yb[(num_sample * idx):(num_sample * (idx+1))]
        k_rf = np.unique(rfcls).shape[0]
        ybcls = np.hstack((rfcls, yb)).astype('int')
        ybcls.tofile('zb.cls', sep='\n')
        cmd = "./labelsw -i zb.cls -o zb.ls -p zb.par -w zb.wt -h zb.h -c zb.summary -b " + str(n_bootstrap + 1) + \
              " -t " + str(0.8) + " -a " + str(0.1) + " -2"
        os.system(cmd)
        dist = np.loadtxt('zb.par')[:,1][1:]
        advdist[idx] = sum(dist) / float(n_bootstrap)
    os.chdir('..')
    os.system('rm -r tmp')
    return advdist

def confset_c(yb, num_sample, n_bootstrap, y_ref, threshold, alpha):
    ########### Confident point set #######################
    folder = 'tmp'
    if os.path.exists(folder) == False:
        os.mkdir(folder)
    shutil.copy2("labelsw_bunch2", os.path.join(folder, "labelsw_bunch2"))
    shutil.copy2("labelsw", os.path.join(folder, "labelsw"))
    os.chdir('tmp')
    yb_stack = yb.flatten()

    rfcls = y_ref
    k_rf = np.unique(rfcls).shape[0]
    ybcls = np.hstack((rfcls, yb_stack)).astype('int')
    ybcls.tofile('zb.cls', sep='\n')
    cmd = "./labelsw -i zb.cls -o zb.ls -p zb.par -w zb.wt -h zb.h -c zb.summary -b " + str(n_bootstrap + 1) + \
          " -t " + str(threshold) + " -a " + str(alpha) + " -2"
    os.system(cmd)
    res = np.loadtxt("zb.ls").flatten()
    clst = np.loadtxt("zb.par")[:, 0][1:].astype('int')
    dist = np.loadtxt('zb.par')[:,1][1:]

    cluster_analy = ClusterAnalysis(yb, n_bootstrap, y_ref, len=num_sample)
    cluster_id, sample_id = cluster_analy.matchcluster(res, clst)
    Interset = cluster_analy.interset(sample_id, cluster_id)

    confidentset = {}
    with open('zb.summary') as f:
        lines = f.readlines()
    for i in xrange(len(lines) - 2*k_rf):
        print str(i)
        lines_tmp = np.fromstring(lines[i+2*k_rf][2:-1], sep = ' ')
        idx_tmp = int(lines[i + 2 * k_rf][0])
        confidentset[idx_tmp] = lines_tmp.astype('int')
    os.chdir('..')
    os.system('rm -r tmp')
    return confidentset, Interset, cluster_id, dist, cluster_analy

"""
test_code:

if __name__  == '__main__':
    n_bs = 10
    n_sample = 5000
    yb = np.genfromtxt('test_data/zb.cls')[5000:]
    y = yb[0*n_sample:1*n_sample]
    mean_test = MeanClustering(y, yb, n_bs)
    y_mean = mean_test.ota()
    #y_mean = mean_test.ota_costly()
    wt = np.genfromtxt('test_data/zb.wt').flatten()
    res_ref = np.genfromtxt('test_data/zb.ls')

    clst = np.array([4,4,4,4,4,4,4,4,4,4])
    ca_test = ClusterAnalysis(yb, n_bs, y_mean, len = 5000)
    codect, nfave, res = ca_test.matchsplit(wt=wt, clsct=clst)
    res = np.round(res, 4)
    cluster_id, sample_id = ca_test.matchcluster(res, clst)
    confidentset, S = ca_test.confset(sample_id, cluster_id)

    yb = yb[0:5000]
    yb[4999] = 0
    y_mean[4997] = 0
    clst = np.array([4])
    ca_test = ClusterAnalysis(yb[0:5000], 1, y_mean, len = 5000)
    codect, nfave, res = ca_test.matchsplit(wt=wt[0:16], clsct=clst)
    cluster_id, sample_id = ca_test.matchcluster(res, clst)
    confidentset, S = ca_test.confset(sample_id, cluster_id)

"""
