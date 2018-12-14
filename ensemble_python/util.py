import numpy as np

def gendata(n, p, c, k):
    """
    :param n: number of data
    :param p: dimension
    :param c: number of clusters
    :param k: number of small clusters
    :return:
    """
    pp = np.sort(np.random.uniform(0,1,c-1))
    pp = np.append(pp,[1]) - np.insert(pp,0,0)
    mu = np.array(np.random.uniform(-5,5,c*p) (p, c))
    sigma_true = np.zeros((p,p,c))
    for i in xrange(c):
        



V = list()
for (ii in 1:C){
    # Sigma_true[,,ii] = riwish(p+13, diag(5, p))
    V[[ii]] = genPositiveDefMat(dim=p)$Sigma
Sigma_true[, , ii] = riwish(5 * p, V[[ii]])
}

rsample < - rmultinom(n, 1, pp_true)  ## C by n, indicator of sample
z_true < - apply(rsample, 2, function(x)
which(x == 1))
X < - t(sapply(z_true, function(x)
{rmvnorm(1, mu_true[, x], Sigma_true[,, x])}))  # nxp

return (list(X=X, z=z_true))
}