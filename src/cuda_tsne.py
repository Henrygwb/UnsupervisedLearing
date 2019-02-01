import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"
from bhtsne import tsne
from scipy import io
import numpy as np
from util import Cluster, load_data, metrics
metrics = metrics()
"""
n_clusters = 5
label_change = {}
n_bootstrap = 4
path_method = '../results/malware/cluster_' + str(n_clusters)
print(str(n_clusters)+'...')
for i in range(n_bootstrap):
    print(str(i)+'...')
    dir_path = os.path.join(path_method, str(i) + '_bs')
    print(dir_path)
    x_low = io.loadmat(dir_path+'/low_d_data')['x_low']
    print(x_low.shape)
    x_low_2 = TSNE(n_components=2).fit_transform(x_low.astype('float64'))
    print(x_low_2.shape)
    cl = Cluster(x_low_2, x_low_2)
    y_pred = cl.mclust(n_clusters)
    io.savemat(dir_path+'/low_2', {'x_low_2':x_low_2, 'y_pred':y_pred})
"""
dataset = 'mnist'
X, y_orin, path = load_data(path="../results", dataset=dataset)
print(X.shape)
print(y_orin.shape)
print(path)
x_low_2 = tsne(X.astype('float64'))
print(x_low_2.shape)
cl = Cluster(x_low_2, x_low_2)
y_pred = cl.mclust(10)
acc = np.round(metrics.acc(y_pred, y_orin), 5)
nmi = np.round(metrics.nmi(y_pred, y_orin), 5)
ari = np.round(metrics.ari(y_pred, y_orin), 5)

print('acc = %.5f, nmi = %.5f, ari = %.5f.' % (acc, nmi, ari))

io.savemat(path+'/low_2', {'x_low_2':x_low_2, 'y_pred':y_pred})
