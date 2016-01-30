up = read.table("upwelling.txt",header = TRUE)
up1 = up["Index"]
up2 = up1[up1 != -9999]
data = ts(up2/1000)

plot(data)

acf(data)
pacf(data)

acf(diff(data))
pacf(diff(data))

#######################################
w = rnorm(500,0,1) 
plot.ts(w, main="white noise")

# random walk (with drift) #
w = rnorm(500,0,1)
x = cumsum(w) 

delta = rnorm(500, mean(data), sd(data))
wd = w + delta
xd = cumsum(wd)

plot.ts(xd, main="random walk")
plot.ts(xd, ylim=c(-10,130), main="random walk", col="red")
lines(x, col="blue")
lines(.2*(1:500), lty="dashed")
legend("topleft",c("RW-drift","random walk","mean function of RW-drift"),col=c("red","blue", "black"),lty=c(1,1,5), pt.cex=1, cex=0.5)



