w = rnorm(500,0,1) 
plot.ts(w, main="white noise")

# random walk (with drift) #
w = rnorm(500,0,1)
x = cumsum(w) 

wd = w + 0.1
xd = cumsum(wd)

plot.ts(xd, main="random walk")
plot.ts(xd, ylim=c(-10,130), main="random walk", col="red")
lines(x, col="blue")
lines(.2*(1:500), lty="dashed")
legend("topleft",c("RW-drift","random walk","mean function of RW-drift"),col=c("red","blue", "black"),lty=c(1,1,5), pt.cex=1, cex=0.5)





w = rnorm(500, 0, 1)
d = runif(500, -0.05, 0.05)
a = w + d
a1 = cumsum(a)
plot(a1)