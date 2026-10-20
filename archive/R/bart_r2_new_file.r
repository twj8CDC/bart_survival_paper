library(BART)
# get dat file name arg


library(dplyr)
library(tidyr)
#read in csv
df = read.csv("../data/exp2_tmp.csv")

x = matrix(df$x0)
x= df[,3:52]
N= nrow(df)
x_test = bind_rows(x,x)
x_test["x0"] = c(rep(1,N), rep(0,N))

# x_test = matrix(x_test[nrow(x_test):1,])
times <- df$t
delta <- df$s

# generate the pre df (only will be on times)
pre <- surv.pre.bart(times=times, delta=delta, x.train=x, x.test=x_test)##(, K=50)

# write.csv(cbind(pre$tx.train, pre$y.train), "../data/tmp_output2.csv")

post <- mc.surv.bart(x.train=pre$tx.train, y=pre$y.train,
                    x.test = pre$tx.test,
                     mc.cores=6, seed=99)

averager = function(x){
    apply(t(matrix(x,ncol=N, nrow=K)), 2, mean)
}

# p1 = predict(post, (pre$tx.test[1:N*K,]))
K = post$K
sv_mu1 = apply(t(matrix(post$surv.test.mean[1:(N*K)], ncol=N, nrow=K)), 2, mean)
sv_cil1 = apply(t(apply(post$surv.test[,1:(N*K)], 1, averager)), 2, quantile, probs=0.025)
sv_cih1 = apply(t(apply(post$surv.test[,1:(N*K)], 1, averager)), 2, quantile, probs=0.975)

sv_mu0 = apply(t(matrix(post$surv.test.mean[((N*K)+1):(N*K*2)], ncol=N, nrow=K)), 2, mean)
sv_cil0 = apply(t(apply(post$surv.test[,((N*K)+1):(N*K*2)], 1, averager)), 2, quantile, probs=0.025)
sv_cih0 = apply(t(apply(post$surv.test[,((N*K)+1):(N*K*2)], 1, averager)), 2, quantile, probs=0.975)



df_out = data.frame(
    sv_mu1, sv_cil1, sv_cih1,
    sv_mu0, sv_cil0, sv_cih0
)


# sv_cih = apply(post$surv.test, 2, quantile, probs=0.975)
# n = length(sv_mu)/2
# x_out = c(rep(x_test[1],n), rep(x_test[2],n))
# df_out = data.frame(x_out, sv_mu, sv_cil, sv_cih)


write.csv(df_out, "../data/exp2_tmp_out2.csv", row.names=F)
# write.csv(df_out, "data/exp2_tmp_out2.csv", row.names=F)

