import numpy as np
import tensorflow as tf
from sklearn.datasets import make_blobs
from dagmm import DAGMM
from sklearn.model_selection import train_test_split
import pandas as pd
from util import metrics
metrics = metrics()
from scipy import io

"""
### Testing on synthetic data.

data, _ = make_blobs(n_samples=1000, n_features=5, centers=5, random_state=123)
data[300] = [-1, -1, -1, -1, -1]
data[500] = [ 1,  0,  1,  1,  1]
ano_index = [300, 500]

model_dagmm = DAGMM(compress_hidden=[5,16,8,1],
                    estimate_hidden=[8,4],
                    lambda1 = 0.1,
                    lambda2 = 0.0001,
                    optimizer = tf.train.AdamOptimizer,
                    lr = 0.0001,
                    compress_activation = tf.nn.tanh,
                    estimate_activation = tf.nn.tanh,
                    dropout_rate = 0.25,
                    use_dropout = 1,
                    use_diag_cov = 1)

model_dagmm.fit(data, batch = 128, epoch = 100, display_interval=100)
energy,_ = model_dagmm.predict(data, 80)
print energy[0:10]

model_dagmm.restore('../results/test_dagmm')
energy1,_ = model_dagmm.predict(data, 80)
print energy1[0:10]
"""

### KDDCUP 99
url_base = "http://kdd.ics.uci.edu/databases/kddcup99"
url_data = url_base+"/kddcup.data_10_percent.gz"
url_info = url_base+"/kddcup.names"

df_info = pd.read_csv(url_info, sep=":", skiprows=1, index_col=False, names=["colname", "type"])
colnames = df_info.colname.values
coltypes = np.where(df_info["type"].str.contains("continuous"), "float", "str")
colnames = np.append(colnames, ["status"])
coltypes = np.append(coltypes, ["str"])

df = pd.read_csv(url_data, names=colnames, index_col=False,
                 dtype=dict(zip(colnames, coltypes)))

X = pd.get_dummies(df.iloc[:,:-1]).values
y = np.where(df.status == "normal.", 1, 0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50, random_state=123)
X_train, y_train = X_train[y_train == 0], y_train[y_train == 0]

io.savemat('kddcup', {'X_train':X_train, 'y_train':y_train, 'X_test':X_test, 'y_test':y_test})

model_dagmm = DAGMM(compress_hidden=[X_train.shape[1],60, 30, 10, 1],
                    estimate_hidden=[10, 4],
                    lambda1 = 0.1,
                    lambda2 = 0.0001,
                    optimizer = tf.train.AdamOptimizer,
                    lr = 0.0001,
                    compress_activation = tf.nn.tanh,
                    estimate_activation = tf.nn.tanh,
                    dropout_rate = 0.5,
                    use_dropout = 1,
                    use_diag_cov = 0)

model_dagmm.fit(X_train, batch = 1024, epoch = 200, display_interval=20)
energy, pred = model_dagmm.predict(X_test, 80)
prec, recall, fscore = metrics.presion_recall_fscore(y_test, pred)
print('precision = %.5f, recall = %.5f, f1_score = %.5f' % (prec, recall, fscore))

print energy[0:10]
