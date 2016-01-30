#3D heat equation
 
import numpy as np
from numpy import pi,sin,cos,sqrt, exp, cosh, arccos
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import matplotlib.animation as animation
import pylab		
import matplotlib as mpl
from scipy.integrate import *



 
fig = plt.figure()
fig.set_dpi(100)
ax1 = fig.gca(projection='3d')
 
x = np.linspace(-100,100,30)
y = np.linspace(-100,100,30)
#z = np.linspace(-5,5,30)
nx = 30
ny = 30
dl = 0.01
X,Y = np.meshgrid(x,y)
 
#Initial time
t0 = 0.001
 
#Time increment
dt = 0.05
 
#Initial temperature at (0,0) at t0=0
CONCENTRATION = 0.9999
 
#Sigma squared 
D = 51.68

#Wave speed
c = 9.4372*3600/100


#Try every combination
p = 3.0 #1 #5 #2 #5
q = 2.2 #1 #5 #3 #3

w = pi*c*sqrt((p/200.0)**2+(q/200.0)**2)

zv = 0.1488 #Ekman theory paper results
a = 200.0
b = 200.0
def integrand(x, y):
    return dblquad(lambda y,x: zv *sin(p*pi*x/200.0),0, b, lambda y: 0, lambda y: a)
    
A = integrand(a, b)

#Temperature function
def u(x,y,t):
    s = 2*D*t
    ans = (-10/sqrt(2*pi*s))*exp(-(x**2+y**2)/(2.0*s)) + 0.1263*(cos(w*t)+sin(w*t))*sin(pi*p*x/200)*sin(q*pi*y/200)
    return ans
      
a = []
for i in range(500):
    v = u(X,Y,t0)
    t0 = t0 + dt
    a.append(v)
     
m = plt.cm.ScalarMappable(cmap=plt.cm.jet)
m.set_array(a[0])
cbar = plt.colorbar(m)
 
k = 0
def animate(i):
    global k
    final = a[k]
    k += 1
    ax1.clear()
    ax1.plot_surface(X,Y,final,rstride=1, cstride=1,cmap=plt.cm.jet,linewidth=0,antialiased=False)
 
    ax1.contour(x,y,final)
    ax1.set_zlim(-5,5)
    ax1.set_xlim(-100,100)
    ax1.set_ylim(-100,100)
     
     
anim = animation.FuncAnimation(fig,animate,frames=220,interval=20)
plt.show()



def RandomWalk(N=10000):
    middle = np.cumsum(np.random.uniform(-0.05,0.05,(N,1)), axis = 0)+1
    for i in range(N):
        if middle[i] >= 0: 
            pass
        else:
            middle[i] = 0.00
    return middle
    
def PlotRandomWalkXT(N=10000):

    X = RandomWalk(N) 
    t = np.arange(0, N, 1)
    ax1 = fig.add_subplot(111)
    ax1.set_ylabel('Concentration Level')
    ax1.set_xlabel('Time')
    ax1.set_title('One realization of diffusion process')
    plt.plot(t,X)
    plt.show()

    
def Endpoints(W=10000, N=10000, d=1):
    all_W = []
    for w in range(W):
        e = float(RandomWalk(10000)[N-1])
        all_W.append(e)
    return all_W

def PlotEndpoints(W=10000, N=10000, d=1):
    t = np.arange(0, N, 1)
    X = Endpoints(W, N, 1)
    ax1 = fig.add_subplot(111)
    ax1.set_ylabel('Concentration Level')
    ax1.set_xlabel('Time')
    ax1.set_title('Endpoints of 10,000 realizations of diffusion process')
    plt.plot(t,X, "ro")
    plt.show()
'''
if __name__=="__main__":
    """Demonstrates solution"""
    print "Random Walk Demo"
    print "Random Walk X vs. t"
    PlotRandomWalkXT(10000)
    PlotEndpoints(W = 10000, N=10000)
'''