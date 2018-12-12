
library(MCMCpack) # for riwish
library(mvtnorm)
library(clusterGeneration)
########################################################
#### simulate C clusters from GMM with one rare cluster
########################################################

#set.seed(1155)
#n =1000; # Sample size  
#p = 20; # dimension
#C = 8; # number of clusters
#k = 4; # number of small clusters

GenData <- function(n,p,C,k){
  #pp_true <- c(rep(0.025,k), rep((1-(0.025*k))/(C-k),C-k))   ## prob that cluster k is geneated
  pp = sort(runif(C-1,0,1))
  pp_true <- c(pp,1) - c(0,pp)   ## prob that cluster k is geneated
  mu_true = matrix(0, nrow = p, ncol =C)     ## centroid of each cluster
  #mu_true[,1] <- rep(5, p)
  #mu_true[,2] <- rep(-5,p)
  #mu_true[,3:C] <- matrix(runif(p*(C-2),-5,5), nrow = p, ncol = C-2)
  mu_true[,1:C] <- matrix(runif(p*C,-5,5), nrow = p, ncol = C)
  
  Sigma_true = array(0, dim = c(p,p,C))    ## var-cov matrix of x,y coordinates 
  #Sigma_true[,,1] <- diag(2,p)
  #Sigma_true[,,2] <- Sigma_true[,,1]
  #for(ii in 3:C){
  #  Sigma_true[,,ii] = riwish(p+13, diag(5, p))
  #}
  V = list()
  for(ii in 1:C){
    #Sigma_true[,,ii] = riwish(p+13, diag(5, p))
    V[[ii]] = genPositiveDefMat(dim=p)$Sigma
    Sigma_true[,,ii] = riwish(5*p, V[[ii]])
  }
    
  rsample <- rmultinom(n, 1, pp_true)      ## C by n, indicator of sample
  z_true <- apply(rsample, 2, function(x)which(x == 1))
  X <- t(sapply(z_true, function(x) {rmvnorm(1, mu_true[,x], Sigma_true[,,x])})) #nxp

  return(list(X=X,z=z_true))
}  
########################################################

  
