setwd("/Users/BSS/Documents/7.Research/Cluster_Stability_2/Jan10_CS_(EX1)")

setwd("sim_k")

dat = list()
zb = list()
zb.align = list()
repre.quick_med = list()
repre.quick_mod = list()
M = list()
stat = list()

for(i in 1:100){
  #load(paste("mclt_r1000_",i,".RD",sep=""))
  load(paste("kclt_r1000_",i,".RD",sep=""))
  
  zb[[i]] = izb
  zb.align[[i]] = izb.align
  repre.quick_med[[i]] = irepre.quick_med
  repre.quick_mod[[i]] = irepre.quick_mod
  M[[i]] = iM
  stat[[i]] = istat
}

setwd("..")

n=5000
ta=0.8
l.n.dist = list()

for(i in 1:100){
  n.zb = c(idat$z,
           M[[i]],
           repre.quick_med[[i]]$repre)-1
  setwd("package5")
  write(n.zb, file = "zb.cls", ncolumns = 1)
  system(paste("./labelsw -i zb.cls -o zb.ls -p zb.par -w zb.wt -h zb.h -c zb.summary -b ",length(n.zb)/n," -t ",ta," -2",sep=""))
  n.dist  <- read.table("zb.par")
  setwd("..")
  
  l.n.dist[[i]] = n.dist
}


d.dist = matrix(NA,100,3)
m.dist = matrix(NA,100,3)
r.dist = matrix(NA,100,3)
for(i in 1:100){
  m.dist[i,] = l.n.dist[[i]][,1]
  m.dist[i,2] = length(unique(M[[i]]))
  m.dist[i,3] = length(unique(repre.quick_med[[i]]$repre))
  d.dist[i,] = l.n.dist[[i]][,2]
  r.dist[i,] = c(1-adjustedRandIndex(idat$z,idat$z),
                 1-adjustedRandIndex(idat$z,M[[i]]),
                 1-adjustedRandIndex(idat$z,repre.quick_med[[i]]$repre))
}


m.dist
d.dist
r.dist

dd = d.dist[,2]>d.dist[,3]
rr = r.dist[,2]>r.dist[,3]
dd == rr
sum(!dd)
sum(!rr)
sum(dd!=rr)

a.dist=r.dist
apply(a.dist,2,function(x) sum(x==0))/100
c(0,table(apply(a.dist[,-1],1,which.min)+1))/100
apply(a.dist,2,function(x) sum(x<=a.dist[,2]))/100
apply(a.dist,2,function(x) sum(x<=a.dist[,3]))/100
round(apply(a.dist,2,mean),3)
round(apply(a.dist,2,median),3)
round(sqrt(apply(a.dist,2,var)),3)

table(a.dist[,2])
table(a.dist[,3])

i=7
d.dist[i,]
m.dist[i,]
draw.obs2(idat$X,idat$z)
draw.obs6(idat$X,M[[i]])
draw.obs6(idat$X,repre.quick_med[[i]]$repre)
#305
#20

indno = 1:100

plot.default(1:length(indno),a.dist[indno,3],type="l")
a.dist.p = rbind(data.frame(distance = a.dist[indno,2],M="W"),
                 data.frame(distance = a.dist[indno,3],M="R"))
ggplot(as.data.frame(a.dist.p), aes(distance, color = M)) +
  geom_histogram(alpha=0.1,position="identity", bins=50) +
  #geom_density(alpha=0.1) +
  #geom_point(data = data.frame(x=0.400126,y=0), aes(x, y), colour = gg_color_hue(4)[1], size = 2) +
  #geom_point(data = data.frame(x=0.236161,y=0), aes(x, y), colour = gg_color_hue(4)[2], size = 2) +
  #geom_point(data = data.frame(x=0.188783,y=0), aes(x, y), colour = "blue", size = 2) +
  #geom_point(data = data.frame(x=0.261105,y=0), aes(x, y), colour = gg_color_hue(4)[4], size = 2) +
  #geom_vline(xintercept=0.188783, linetype="dashed", color = "blue") +
  theme_minimal() +
  guides(color = guide_legend(title=""))

#ggsave("G0_simul_ri.jpeg",width=100,height=60,units="mm")


l.ovs = rep(NA,5)
ovs = list()
m.ovs=data.frame(y=NULL,x=NULL)
for(j in 1:3) {
  indno = eval(parse(text=paste("ind",j+10,sep="")))
  ovs[[j]] = rep(NA,length(indno))
  for(i in 1:length(indno)) ovs[[j]][i] = round(mean(stat[[indno[i]]]$dist[,2]),4)
  l.ovs[j] = mean(ovs[[j]])
  m.ovs = rbind(m.ovs,data.frame(y=j+8,x=ovs[[j]]))
}
m.ovs

plot.default(m.ovs$x,m.ovs$y)




plot.default(1:5,l.ovs,type="l")