library(flexclust) # for kcca
library(mclust) # for Mclust
library(class) # for knn
library(dbscan) # for dbscan

library('pcaReduce') # for pcaReduce
library("HMMVB") #for HMMVB
library("mclust") #for HMMVB
library("pryr") #for HMMVB

library("pcaReduce") #for pcareduce


kclst <- function(X, newdata){
  kcl = kcca(X, k=C, kccaFamily("kmeans"))
  res = predict(kcl, newdata=newdata)
  return(res)
}

mclst <- function(X, newdata){
  mcl = Mclust(X, G=G)
  res = predict(mcl, newdata=newdata)$classification
  return(res)
}

hclst <- function(X, newdata){
  hcl = hclust(dist(X))
  res = as.numeric(knn(X,newdata,cutree(hcl,C)))
  return(res)
}  

dclst <- function(X, newdata){
  dcl = dbscan(X, eps=E)
  res = predict(dcl, newdata=newdata, data=X)
  return(res)
}

pclst <- function(X, newdata){
  pcl <- PCAreduce(X, nbt=1, q=30, method='S')
  col.ind = which(apply(pcl[[1]],2,function(x) length(unique(x))) == C)
  res = as.numeric(knn(X,newdata,pcl[[1]][,col.ind]))
  return(res)
}

hmmvb <- function(X, newdat){
  HMM_1 = new("HMMVB", ndim = dim(X)[2], nseq = NVB, ndim_seq = DSQ, ncom_seq = CSQ,
              c_path = '/Users/BSS/Documents/7.Research/Jan23_RD/HMMVB/Hmmvb_package_v1.3.2')
  estimate(HMM_1, data = X, diagonal_flag = TRUE)
  res = cluster(HMM_1, newdat)
  return(res)
}

hmmvb2 <- function(X, newdat){
  wd=getwd()
  setwd('/Users/BSS/Documents/7.Research/Jan23_RD/HMMVB/Hmmvb_package_v1.3.2')
  
  HMM_1 = new("HMMVB", ndim = dim(X)[2], nseq = NVB, ndim_seq = DSQ, ncom_seq = CSQ,
              c_path = '/Users/BSS/Documents/7.Research/Jan23_RD/HMMVB/Hmmvb_package_v1.3.2')
  
  write_hyper_file(HMM_1)
  WriteData(HMM_1,X,"Train_data.txt")
  WriteData(HMM_1,newdat,"Test_data.txt")
  system(paste("./trainmaster -i Train_data.txt -m model_binary.dat -b hyperparam_HMMVB.dat -v", sep=""))
  system(paste("./testsync -i Test_data.txt -m model_binary.dat -o refcls.dat -t 0.3 -u -l 11", sep=""))
  res = c(as.matrix(read.table("refcls.dat", skip=1)))
  
  setwd(wd)
  
  return(res)
}
