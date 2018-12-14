########################################################
# 1. Visualize clustering in two dimensions
########################################################
library(ggplot2)
draw.obs <- function(Y,K){

  X = as.data.frame(prcomp(Y)$x)
  
  mu_x = sapply(1:max(K),function(z){mean(X[which(K==z),1])})
  mu_y = sapply(1:max(K),function(z){mean(X[which(K==z),2])})
  
  ggplot(X[,1:2], aes(x=PC1, y=PC2)) +
    geom_point(color=K+1, shape=K, size=1, alpha = 0.5) +
    theme_void() +
    annotate("text", x=mu_x, y=mu_y, label=1:max(K), size=5, fontface = "bold")
}

########################################################
# 2. Generate bootstrap samples
########################################################

GenBsSamps <- function(X, nb, clustering, stack=T){
  # set.seed(123)
  n <- nrow(X)                      #n: number of observations
  bet <- array(0, dim = c(nb, n))   #nb: number of bootstrap samples
  Zb <- array(0, dim = c(nb,n))     
  
  Zb.stack <- NULL
  m <- rep(NA,nb)
  for (i in 1:nb){
    bet[i,] = sample(c(1:n),n,replace = T)	# pick random indices
    Xb = as.data.frame(X[bet[i,],])
    
    Zb[i,] = clustering(Xb, newdata=X)      # train with bootstraped sample
                                            # and predict using new cluster
    m[i] = length(unique(Zb[i,]))
    Zb.stack = c(Zb.stack, Zb[i,])
    print(i)
  }
  
  if(stack==T){
    return(list(Z=Zb.stack,m=m))
  } else {
    return(list(Z=Zb,m=m))
  }
}

########################################################
# 3. Align bootstrap samples
########################################################

align <- function(Zb.stack, nb, folder="align"){
  setwd("package5")
  if(!(dir.exists(folder))) dir.create(folder)
  file.copy("labelsw_bunch", folder)
  setwd(folder)
  
  n = length(Zb.stack)/nb
  rfcls = rep(1,n)
  zbcls = c(rfcls,Zb.stack)-1
  write(zbcls, file = "zb.cls", ncolumns = 1)
  system(paste("./labelsw_bunch -i zb.cls -o zb.ls -p zb.par -w zb.wt -h zb.h -c zb.summary -b ",nb+1," -2",sep=""))
  
  dist <- list()
  res <- list()
  wt <- list()
  smry <- list()
    
  for(i in 1:nb){
    nam = i
    
    dist[[i]] <- read.table(paste("zb_",nam,".par",sep="")) 
    res[[i]] <- read.table(paste("zb_",nam,".ls",sep="")) 
    wt[[i]] <- read.table(paste("zb_",nam,".wt",sep=""))
    smry[[i]] <- read.table(paste("zb_",nam,".summary",sep=""))  
  }
  
  setwd("..")
  setwd("..")
  return(list(dist=dist,res=res,wt=wt,summary=smry))
}

########################################################
# 3.1 Align bootstrap samples - 2nd method
########################################################

align2 <- function(Zb.stack, nb, folder="align"){
  setwd("package5")
  if(!(dir.exists(folder))) dir.create(folder)
  file.copy("labelsw_bunch2", folder)
  setwd(folder)
  
  n = length(Zb.stack)/nb
  rfcls = rep(1,n)
  zbcls = c(rfcls,Zb.stack)-1
  write(zbcls, file = "zb.cls", ncolumns = 1)
  idx = as.numeric(system(paste("./labelsw_bunch2 -i zb.cls -o zb.ls -p zb.par -w zb.wt -h zb.h -c zb.summary -b ",nb+1," -2",sep=""), intern=T))
  
  setwd("..")
  setwd("..")
  
  return(list(idx=idx))
  
}

########################################################
# 4. Find representative samples - quick version
########################################################

Repre.quick <- function(Zb.stack, dist, wt, method="median"){
  nb = length(dist)
  n = length(Zb.stack)/nb
  m = dist[[1]][-1,1]
  
  avg.dist = rep(NA,nb)
  cnt.nrst = rep(NA,nb)
  P.raw = list()
  
  for(i in 1:nb){
    avg.dist[i] = mean(dist[[i]][,2])
  }
  
  std = sqrt(var(avg.dist))
  
  for(i in 1:nb){
    P.raw[[i]] = matrix(0,n,m[i])
    for(k in 1:n){
      P.raw[[i]][k, Zb.stack[n*(i-1)+k]] = 1
    }
    cnt.nrst[i] = sum(dist[[i]][,2]<std)
  }  
  
  if(method=="median") idx = which.min(avg.dist)
  else if(method=="mode") idx = which.max(cnt.nrst)
  
  wttemp = as.matrix(wt[[idx]])
  K.rf = ncol(wttemp)
  
  P.tild = matrix(NA,n,K.rf*nb)
  P.tild.sum = matrix(0,n,K.rf)
  
  for(k in 1:nb){
    wtsub = wttemp[(sum(m[0:(k-1)])+1):sum(m[0:k]),]
    P.tild = P.raw[[k]]%*%(wtsub/rowSums(wtsub))
    P.tild.sum = P.tild.sum + P.tild
  }
  
  P.bar = P.tild.sum/nb
  P.bar.hrd.asgn = apply(P.bar,1,which.max)
  
  return(list(repre=P.bar.hrd.asgn,idx=idx))
}

########################################################
# 4.1 Find representative samples - 2nd version
########################################################

Repre2 <- function(Zb.stack, nb, idx, threshold=0.8, alpha=0.1){
  n = length(Zb.stack)/nb
  K.rf = max(Zb.stack[(n*(idx-1)+1):(n*idx)])
    
  setwd("package5")
  
  rfcls = Zb.stack[(n*(idx-1)+1):(n*idx)]
  zbcls = c(rfcls,Zb.stack)-1
  write(zbcls, file = "zb.cls", ncolumns = 1)
  system(paste("./labelsw -i zb.cls -o zb.ls -p zb.par -w zb.wt -h zb.h -c zb.summary -b ",nb+1," -t ",threshold," -a ",alpha," -2",sep="")
         ,ignore.stdout = T)
  
  dist <- read.table("zb.par")
  wt <- read.table("zb.wt")

  setwd("..")
  
  m = dist[-1,1]
  
  P.raw = list()
  
  for(i in 1:nb){
    P.raw[[i]] = matrix(0,n,m[i])
    for(k in 1:n){
      P.raw[[i]][k, Zb.stack[n*(i-1)+k]] = 1
    }
  }  
    
  wt = as.matrix(wt)
  
  P.tild = matrix(NA,n,K.rf*nb)
  P.tild.sum = matrix(0,n,K.rf)
    
    for(k in 1:nb){
      wtsub = wt[(sum(m[0:(k-1)])+1):sum(m[0:k]),]
      P.tild = P.raw[[k]]%*%(wtsub/ifelse(rowSums(wtsub)==0,1,rowSums(wtsub)))
      P.tild.sum = P.tild.sum + P.tild
    }
    
  P.bar = P.tild.sum/nb
  P.bar.hrd.asgn = apply(P.bar,1,which.max)
  K.rf = max(P.bar.hrd.asgn)
  
  setwd("package5")
  
  rfcls = P.bar.hrd.asgn
  zbcls = c(rfcls,Zb.stack)-1
  write(zbcls, file = "zb.cls", ncolumns = 1)
  system(paste("./labelsw -i zb.cls -o zb.ls -p zb.par -w zb.wt -h zb.h -c zb.summary -b ",nb+1," -t ",threshold," -a ",alpha," -2",sep=""))
  
  dist <- read.table("zb.par")
  res <- read.table("zb.ls") 
  wt <- read.table("zb.wt")
  smry <- read.table("zb.summary",nrows=K.rf)
  jaccard_mat <- read.table("zb.summary",skip=K.rf,nrows=K.rf)
  confset.idx <- list()
  confset <- list()
  for(i in 1:K.rf){
    confset.idx[[i]] <- as.integer(read.table("zb.summary",skip=(2*K.rf+i-1),nrows=1))[-c(1,2)]+1
    confset[[i]] = rep(0,n)
    confset[[i]][confset.idx[[i]]]=1
    
  }
  
  setwd("..")
  
  return(list(repre=P.bar.hrd.asgn,
              dist=dist,res=res,wt=wt,
              summary=smry,
              jaccard_mat = jaccard_mat,
              confset=confset))
}


########################################################
# 5. Find representative samples - double run version
########################################################

Repre <- function(Zb.stack, dist, wt, folder="align2", method="median"){
  nb = length(dist)
  n = length(Zb.stack)/nb
  m = dist[[1]][-1,1]
  
  P.raw = list()
  
  for(i in 1:nb){
    P.raw[[i]] = matrix(0,n,m[i])
    for(k in 1:n){
      P.raw[[i]][k, Zb.stack[n*(i-1)+k]] = 1
    }
  }  
  
  P.bar.hrd.asgn = list()
  Zb.stack2 = NULL
  
  for(i in 1:nb){
    
    wttemp = as.matrix(wt[[i]])
    K.rf = ncol(wttemp)
    
    P.tild = matrix(NA,n,K.rf*nb)
    P.tild.sum = matrix(0,n,K.rf)
    
    for(k in 1:nb){
      wtsub = wttemp[(sum(m[0:(k-1)])+1):sum(m[0:k]),]
      P.tild = P.raw[[k]]%*%(wtsub/rowSums(wtsub))
      P.tild.sum = P.tild.sum + P.tild
    }
    
    P.bar = P.tild.sum/nb
    P.bar.hrd.asgn[[i]] = apply(P.bar,1,which.max)
    
    Zb.stack2 = c(Zb.stack2, P.bar.hrd.asgn[[i]])
  }
  
  # call align function #
  Zb.align <- align(Zb.stack2, nb, folder=folder)
  #######################
  
  dist2 = Zb.align$dist
  wt2 = Zb.align$wt
  
  avg.dist = rep(NA,nb)
  cnt.nrst = rep(NA,nb)
  P.raw = list()

  for(i in 1:nb){
    avg.dist[i] = mean(dist2[[i]][,2])
  }
  
  std = sqrt(var(avg.dist)/n)
  
  for(i in 1:nb){
    P.raw[[i]] = matrix(0,n,m[i])
    for(k in 1:n){
      P.raw[[i]][k, Zb.stack2[n*(i-1)+k]] = 1
    }
    cnt.nrst[i] = sum(dist2[[i]][,2]<std)
  }  
  
  if(method=="median") idx = which.min(avg.dist)
  else if(method=="mode") idx = which.max(cnt.nrst)
  
  return(list(repre=P.bar.hrd.asgn[[idx]],idx=idx))
}

########################################################
# 6. Statistics for stability of a result of clustering
########################################################

StatForClst <- function(Zb.repre, Zb.stack, nb=(length(Zb.stack)/length(Zb.repre)), t=0.8) {
  setwd("package5")
  
  rfcls = Zb.repre
  zbcls = c(rfcls,Zb.stack)-1
  write(zbcls, file = "zb.cls", ncolumns = 1)
  system(paste("./labelsw -i zb.cls -o zb.ls -p zb.par -w zb.wt -h zb.h -c zb.summary -b ",nb+1," -t ",t," -2",sep=""))
    
  dist <- read.table("zb.par")
  res <- read.table("zb.ls") 
  wt <- read.table("zb.wt")
  smry <- read.table("zb.summary")
  
  setwd("..")

  # Calculate 5 statistics #
  K.rf = ncol(res)
  P = I = S = M = O = rep(0,K.rf)

  m = dist[,1]
  for(k in 1:nb){
    wttemp = round(as.matrix(wt[sum(m[0:(k-1)],1):sum(m[0:k]),]),5)
    ind = which(wttemp>0)
    rowind = ifelse(ind%%m[k]==0,m[k],ind%%m[k])
    for(j in 1:K.rf){
      colcnt = (1+m[k]*(j-1) <= ind)&(ind <=m[k]*j)
      if(sum(colcnt)==1){
        rowcnt = (rowind == rowind[which(colcnt)])
        if(sum(rowcnt)==1) P[j] = P[j]+1
        else if(sum(rowcnt)>1) M[j] = M[j]+1
      } else if (sum(colcnt)>1){
        if(prod(wttemp[rowind[which(colcnt)],j] == rowSums(wttemp[rowind[which(colcnt)],]))){
          if(max(wttemp[ind[which(colcnt)]]/sum(wttemp[ind[which(colcnt)]]))>=t) I[j] = I[j] +1
          else S[j] = S[j]+1
        } else O[j] = O[j]+1
      }
    }
  }
  stat = rbind(P,I,S,M,O)
  colnames(stat) = colnames(wt)
  
  return(list(dist=dist,res=res,wt=wt,summary=smry,statistics=stat))
}

########################################################
# 7. Overall clustering stability
########################################################

Entropy <- function(dist){
  std = sqrt(var(dist))
  cate = round(dist/std)
  px  = table(cate)/length(cate)
  lpx = log(px, base=2)
  ent = -sum(px*lpx)
  return(ent)
}

########################################################
# 8. Confidence Set
########################################################

ConfSet <- function(res, Zb.stack, m, alpha=0.1, n=length(Zb.stack)/length(m), nb=length(m), j=1:ncol(res) ){
  setwd("package5")
  
  confset = list()
  cfset.stat = list()
  
  m.cum = apply(as.array(1:nb),1,function(x) sum(m[1:x]))
  
  j = j[-which(apply(res,2,function(x) prod(x==-1))==1)]
  for(k in j){
    idx.cls.mtc = which(res[,k]!=-1 & res[,k]<2)
    #val.cls.mtc = res[which(res[,k]!=-1 & res[,k]<2),k]
    
    idx.bt = apply(as.array(idx.cls.mtc),1,function(x) which(m.cum>=x)[1])
    idx.cls.bt = m[idx.bt]-(m.cum[idx.bt]-idx.cls.mtc)
    
    out = list()
    for(kk in 1:length(idx.cls.bt)){
      pred = Zb.stack[((idx.bt[kk]-1)*n+1):(idx.bt[kk]*n)]
      id = which(pred == idx.cls.bt[kk])
      out[[kk]] = c(length(id),id)
    }
    
    file.create(paste("cluster",k,".txt",sep=""))
    lapply(out, write, paste("cluster",k,".txt",sep=""), append=TRUE, ncolumns=n)
    
    system(paste("./confset -i cluster",k,".txt -o confset",k,".txt -a ", alpha, sep=""))
    
    cfset.idx = as.matrix(read.table(paste("confset",k,".txt",sep=""), nrows=1))
    cfset.stat[[k]] = as.matrix(read.table(paste("confset",k,".txt",sep=""), skip=1, nrows=1, col.names=1:6))
    confset[[k]] = rep(0,n)
    confset[[k]][cfset.idx[-1]]=1
  }
  
  setwd("..")
  return(list(confset=confset,cfset.stat=cfset.stat))
}
