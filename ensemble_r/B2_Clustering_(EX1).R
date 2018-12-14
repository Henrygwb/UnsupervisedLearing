rm(list=ls())
setwd("/Users/BSS/Documents/7.Research/Cluster_Stability_2/Jan10_CS_(EX1)")

source("B11_DataGenerate.R")

source("A0_Functions.R")
source("A1_ClustMethods.R")
source("E0_draw.obs.R")

########################################################
#### Generate Data
########################################################
#set.seed(1200)
#set.seed(2018)
#set.seed(2050)
set.seed(2070)

dat = GenData(n=5000,p=2,C=4,k=2)
img.org = draw.obs5(dat$X,dat$z,title="Ground Truth",xlab="(a)")
#ggsave("0r_original.jpeg",width=60,height=60,units="mm")

########################################################
#### Draw whole sample plots
########################################################
C=4;G=4
X = dat$X
#set.seed(21)
K = kclst(X,X)
M = mclst(X,X)

img.wk = draw.obs5(X,K,title="K-means",xlab="(b)",ylab="Original")
#ggsave("0w_kclt.jpeg",width=60,height=60,units="mm")
img.wm = draw.obs5(X,M,title="Mclust",xlab="(c)")
#ggsave("0w_mclt.jpeg",width=60,height=60,units="mm")
########################################################
#### Generate Bootstrap samples
########################################################

zb.k = GenBsSamps(X,100,kclst)
zb.m = GenBsSamps(X,100,mclst)

#save(zb.k,zb.m, file="zbdata.Rdata")
load("zbdata.Rdata")

########################################################
#### Alignment
########################################################

#zb.align.k = align(zb.k$Z,100,folder="Kcls_jan08")
#zb.align.m = align(zb.m$Z,100,folder="Mcls_jan08")

id.k = align2(zb.k$Z,100,folder="Kcls_jan08")
id.m = align2(zb.m$Z,100,folder="Mcls_jan08")
########################################################
#### Find representative
########################################################

repre.k = Repre2(zb.k$Z, 100, id.k$idx)
repre.m = Repre2(zb.m$Z, 100, id.m$idx)

img.rk = draw.obs5(X,repre.k$repre)
img.rm = draw.obs5(X,repre.m$repre)

library(gridExtra)
g = grid.arrange(
  grobs = list(img.org,img.wk,img.wm,img.rk,img.rm,bar2),
  widths = c(1, 1, 1),
  layout_matrix = rbind(c(1, 2, 3),
                        c(6, 4, 5))
)

ggsave("EX1.jpeg",width=190,height=120,units="mm",g)

aa = repre.k$repre
aa[which(aa==1)]=5
aa[which(aa==3)]=1
aa[which(aa==4)]=3
aa[which(aa==5)]=4
img.rk = draw.obs5(X,aa,xlab="(e)",ylab="Optimal Transport Alignment")
#ggsave("0r_kclt.jpeg",width=60,height=60,units="mm")

bb = repre.m$repre
bb[which(bb==1)]=5
bb[which(bb==3)]=1
bb[which(bb==4)]=6
bb[which(bb==2)]=4
bb[which(bb==6)]=2
bb[which(bb==5)]=3
img.rm = draw.obs5(X,bb,xlab="(f)")
#ggsave("0r_mclt.jpeg",width=60,height=60,units="mm")

#save(dat,K,M,repre.k,repre.m, file="result0327.RD")
load("result0327.RD")

n.zb = c(dat$z,K,repre.k$repre,M,repre.m$repre)
setwd("package5")
write(n.zb, file = "zb.cls", ncolumns = 1)
system(paste("./labelsw -i zb.cls -o zb.ls -p zb.par -w zb.wt -h zb.h -c zb.summary -b ",5," -t ",0.8," -2",sep=""))
n.dist  <- read.table("zb.par")
setwd("..")

round(n.dist,4)

r.dist = c(1-adjustedRandIndex(dat$z,dat$z),
           1-adjustedRandIndex(dat$z,K),
           1-adjustedRandIndex(dat$z,repre.k$repre),
           1-adjustedRandIndex(dat$z,M),
           1-adjustedRandIndex(dat$z,repre.m$repre))

round(r.dist,4)

n.dist.4 = data.frame(dist=c(n.dist[2:5,2],r.dist[2:5]),
                      method=c("K-means","K-means","Mclust","Mclust","K-means","K-means","Mclust","Mclust"),
                      a=c("Orig.","OTA","Orig.","OTA",
                          "Orig.","OTA","Orig.","OTA"),
                      measure=c("Wasserstein dist","Wasserstein dist","Wasserstein dist","Wasserstein dist",
                                "1-Adj Rand ind","1-Adj Rand ind","1-Adj Rand ind","1-Adj Rand ind"))

round(n.dist.4[,1],4)
library(wesanderson)
library(latex2exp)
library(grid)
bar = ggplot(n.dist.4, aes(x=interaction(method,measure),y=dist,fill=factor(a, levels=c("Orig.","OTA")))) +
  geom_bar(colour="black", stat="identity", position=position_dodge()) +
  
  theme_classic() +
  labs(x="",y=TeX('Distance to the truth')) +
  
  theme(legend.title=element_blank()) +
  theme(legend.position = c(0.9,0.9)) +
  
  coord_cartesian(ylim = c(0, 0.5)) +
  theme(plot.margin = unit(c(1, 1, 2, 1), "lines"),
        axis.title.x = element_blank(),
        axis.text.x = element_blank())+
  
  annotate("text", x = 1:4 , y = -0.06,
           label = rep(c("K","M"), 2)) +
  annotate("text", x = c(1.5,3.5), y = - 0.10, label = c("1-ARI", "Wass")) +
  annotate("text", x = 2.5, y = - 0.14, label = "(d)") +
  #scale_fill_grey(start = .4, end = .9)
  scale_fill_manual(values=wes_palette(n=2, name="Darjeeling"))

bar2 <- ggplot_gtable(ggplot_build(bar))
bar2$layout$clip[bar2$layout$name == "panel"] <- "off"
grid.draw(bar2)

########################################################
#### Find statistics
########################################################

aaa = matrix(0,4,4)
aaa[1,3]=aaa[2,2]=aaa[3,4]=aaa[4,1]=1
t(aaa%*%as.matrix(repre.k$summary[,c(3,5,7,9)])/100)

bbb = matrix(0,4,4)
bbb[1,2]=bbb[2,4]=bbb[3,3]=bbb[4,1]=1
t(bbb%*%as.matrix(repre.m$summary[,c(3,5,7,9)])/100)

########################################################
#### Confidence set
########################################################

draw.obs5(X,repre.k$confset[[1]])
draw.obs5(X,repre.k$confset[[2]])
draw.obs5(X,repre.k$confset[[3]])
draw.obs5(X,repre.k$confset[[4]])

cf4 = draw.obs5(X,repre.m$confset[[1]],xlab="(d)")
cf1 = draw.obs5(X,repre.m$confset[[2]],xlab="(a)")
cf3 = draw.obs5(X,repre.m$confset[[3]],xlab="(c)")
cf2 = draw.obs5(X,repre.m$confset[[4]],xlab="(b)")

g2 = grid.arrange(
  grobs = list(cf1,cf2,cf3,cf4),
  widths = c(1, 1, 1, 1),
  layout_matrix = rbind(c(1, 2, 3, 4))
)
ggsave("EX1_cf.jpeg",width=190,height=50,units="mm",g2)

repre.m$summary
round(t(bbb%*%as.matrix(repre.m$summary[,c(11,12,13,14)])),3)

#--------------------------------------

ja = as.matrix(repre.m$jaccard_mat)
bbb
JA = matrix(0,4,4)
JA[1,2] = ja[2,4]
JA[1,3] = ja[2,3]
JA[1,4] = ja[2,1]
JA[2,3] = ja[4,3]
JA[2,4] = ja[4,1]
JA[3,4] = ja[3,1]
JA = JA+t(JA)
JA
########################################################
#### Find overall clustering stability - entropy
########################################################

round(mean(repre.k$dist[,2]),4)
round(mean(repre.m$dist[,2]),4)

