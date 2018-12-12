library(ggplot2)
library(dplyr)
draw.obs2 <- function(X,true_cell_cls){
  X = as.data.frame(prcomp(X)$x[,1:2])
  X = as.data.frame(X)
  K = true_cell_cls
  A = as.character(true_cell_cls)
  
  s = cbind(X,A) %>% split(K)
  ch = s %>% lapply(., function(el) chull(el$PC1, el$PC2))
  ch = lapply(names(ch), function(x) s[[x]][ch[[x]],]) %>% do.call(rbind, .) 
  
  mu_x = sapply(1:max(K),function(z){mean(X[which(K==z),1])})
  mu_y = sapply(1:max(K),function(z){mean(X[which(K==z),2])})
  
  ggplot(as.data.frame(X[,1:2]), aes(x=PC1, y=PC2, color = A)) +
    geom_point(shape=19, size=1, alpha=0.5) +
    geom_polygon(data = ch, aes(fill = A), alpha = 0.2) +
    #guides(fill=guide_legend(title="Cell Type"), color=guide_legend(title="Cell Type")) +
    theme_classic() +
    annotate("text", x=mu_x, y=mu_y, label=1:max(K), size=4, fontface = "bold")
}

draw.obs3 <- function(X,true_cell_cls){
  X = as.data.frame(prcomp(X)$x[,1:2])
  X = as.data.frame(X)
  K = true_cell_cls
  A = as.character(true_cell_cls)
  
  s = cbind(X,A) %>% split(K)
  ch = s %>% lapply(., function(el) chull(el$PC1, el$PC2))
  ch = lapply(names(ch), function(x) s[[x]][ch[[x]],]) %>% do.call(rbind, .) 
  
  #mu_x = sapply(1:max(K),function(z){mean(X[which(K==z),1])})
  #mu_y = sapply(1:max(K),function(z){mean(X[which(K==z),2])})
  
  ggplot(as.data.frame(X[,1:2]), aes(x=PC1, y=PC2, color = A)) +
    geom_point(shape=19, size=1) +
    geom_polygon(data = ch[which(ch$A==1),], aes(fill = ch$A[which(ch$A==1)]), alpha = 0.2) +
    guides(fill=FALSE, color=FALSE) +
    scale_color_manual(values=c("black","red")) +
    scale_fill_manual(values=c("red")) +
    theme_classic()
  #annotate("text", x=mu_x, y=mu_y, label=1:max(K), size=4, fontface = "bold")
}

draw.obs4 <- function(X,true_cell_cls){
  X = as.data.frame(X)
  K = true_cell_cls
  A = as.character(true_cell_cls)
  
  s = cbind(X,A) %>% split(K)
  ch = s %>% lapply(., function(el) chull(el$PC1, el$PC2))
  ch = lapply(names(ch), function(x) s[[x]][ch[[x]],]) %>% do.call(rbind, .) 
  
  mu_x = sapply(unique(K),function(z){mean(X[which(K==z),1])})
  mu_y = sapply(unique(K),function(z){mean(X[which(K==z),2])})
  
  ggplot(as.data.frame(X[,1:2]), aes(x=PC1, y=PC2, color = A)) +
    geom_point(shape=19, size=1, alpha=0.5) +
    geom_polygon(data = ch, aes(fill = A), alpha = 0.2) +
    guides(color = guide_legend(title="Cluster"), fill=guide_legend(title="Cluster")) +
    theme_classic() +
    annotate("text", x=mu_x, y=mu_y, label=unique(K), size=2, fontface = "bold")
}

draw.obs5 <- function(Y,K,title="",xlab="",ylab=""){
  
  X = data.frame(PC1=Y[,1],PC2=Y[,2])
  
  mu_x = sapply(1:max(K),function(z){mean(X[which(K==z),1])})
  mu_y = sapply(1:max(K),function(z){mean(X[which(K==z),2])})
  
  #if(title==""){
  #  ggplot(X[,1:2], aes(x=PC1, y=PC2)) +
  #    geom_point(color=K+1, shape=K, size=1, alpha = 0.5) +
  #    theme_classic() +
  #    labs(x="",y="")  
  #} else{
    ggplot(X[,1:2], aes(x=PC1, y=PC2)) +
      geom_point(color=K+1, shape=K, size=1, alpha = 0.5) +
      theme_classic() +
      labs(x=xlab,y=ylab) +
      ggtitle(title) +
      theme(plot.title = element_text(face="bold", hjust = 0.5))
    #annotate("text", x=mu_x, y=mu_y, label=1:max(K), size=5, fontface = "bold")
  #}
}
