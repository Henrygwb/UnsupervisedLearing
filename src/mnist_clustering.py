import os
import numpy as np
from scipy import io
from dec import DeepEmbeddingClustering
from dcn import DeepClusteringNetwork
from util import genbssamples

def load_data(path, option = 0):
    if option == 0:
        file = os.path.join(path, 'train')
        X = io.loadmat(file)['train_x']
        y = io.loadmat(file)['train_y']
        y = np.array([np.where(r == 1)[0][0] for r in y])

    else:
        file = os.path.join(path, 'testall')
        X = io.loadmat(file)['test_x']
        y = io.loadmat(file)['test_y']
        y = np.array([np.where(r == 1)[0][0] for r in y])

    return X, y

if __name__ == "__main__":
    # setting the hyper parameters
    path = "../results/mnist"
    X, y = load_data(path, option=1)
    n_clusters = len(np.unique(y))
    n_bootstrep = 30
    num_samples = X.shape[0]
    bs_idx = genbssamples(n_bootstrep=n_bootstrep, num_samples=num_samples)
    io.savemat('bs_idx', {'idx':bs_idx})
    hidden_neurons = [X.shape[-1], 500, 500, 2000, 10]
    batch = 256
    pre_epoch = 300
    finetune_epoch = 2e4
    update_interval = 140
    tol = 1e-3
    for i in xrange(n_bootstrep):
        print '********************************'
        print '********************************'
        print '********************************'
        print 'Bootstrap sample time %d.' % i
        print '********************************'
        print '********************************'
        print '********************************'
        X_bs = X[bs_idx[i,:]]
        y_bs = y[bs_idx[i,:]]
        dec_test = DeepEmbeddingClustering(X_bs, y_bs, hidden_neurons, n_clusters)
        dec_test.pretrain(batch_size=batch, epochs=pre_epoch, save_dir=path)
        dec_test.train(optimizer='adam', batch_size=batch, epochs=finetune_epoch, tol= tol,
                       update_interval=update_interval, save_dir=path, shuffle= False)
        dec_test.evaluate()

        pred_test = dec_test.test(X_test = X, y_test = y)
        io.savemat('bs_'+str(i), {'y_pred':pred_test})