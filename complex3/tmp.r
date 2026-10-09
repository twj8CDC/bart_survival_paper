library(tidyverse)
library(BART)

x = read_csv("new_runs/tmp_r.csv")
time= as.vector(ceiling(x$time/10))
status = as.vector(x$status)
xx = as.data.frame(x[2:11])
tmpmsk = (xx[,1]==1 & xx[,2]==1)
mean(status[tmpmsk])
mean(time[tmpmsk])
xx[xx[,1]==1 & xx[,2]==1, ]


pre=BART::surv.pre.bart(time, status, xx, xx)
pre$tx.train
length(time)
time
pre$tx.train[1:11,]
str(pre$tx.train)
sum(pre$y.train)


tmpmsk = (pre$tx.train[,2]==1 & pre$tx.train[,3]==1 & pre$tx.train[,5] >.95) 
tmpmsk
mean(pre$y.train[tmpmsk])


post = surv.bart(pre$tx.train, pre$y.train, x.test = pre$tx.test,)
str(post)
post$prob.train.mean
str(pre$tx.test)
sv = post$surv.test.mean
sv
sv = t(matrix(sv, nrow=10, ncol=1400))

plot(1:10,sv[1,], type="line", ylim=c(0,1))
for(i in 2:nrow(sv)){
    lines(1:10,sv[i,], type="line")
}

sv[1,]
sv[2,]

1:nrow(sv)

