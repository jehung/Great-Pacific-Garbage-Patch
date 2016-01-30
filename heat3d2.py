#3D heat equation
 
import numpy as np
from numpy import pi,sin,cos,sqrt, exp, cosh, arccos
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import matplotlib.animation as animation
import pylab		
import matplotlib as mpl



 
fig = plt.figure()
fig.set_dpi(100)
ax1 = fig.gca(projection='3d')
 
x = np.linspace(-10,10,30)
y = np.linspace(-10,10,30)
#z = np.linspace(-111,111,30)
dl = 0.01
X,Y = np.meshgrid(x,y)
 
#Initial time
t0 = 0.001
 
#Time increment
dt = 1
 
#Initial temperature at (0,0) at t0=0
CONCENTRATION = 0.9999
 
#Sigma squared 
s = 2

#Wave speed
c = 9.4372*3600/100


#Try every combination
p = 4.25 #1 #5 #2 #5
q = 4.25 #1 #5 #3 #3

w = pi*c*sqrt(p**2+q**2)
  
#Temperature function
def u(x,y,t):
    #global Concentration
    #return np.array([x,y,(T/sqrt(1+4*t/s))*exp(-(x**2+y**2)/(s+4*t))])
    x = x+ np.random.uniform(-dl, dl, 1)
    y = y+ np.random.uniform(-dl, dl, 1)
    Concentration = CONCENTRATION + np.random.uniform(-0.5, 0.5, 1)    
    return CONCENTRATION + (-1/CONCENTRATION/sqrt(1+4*s/t)) *exp(-(x**2+y**2)/(s+4*t))+0.1*(cos(w*t)+sin(w*t))*sin(pi*p*x)*sin(q*pi*y)
    #return CONCENTRATION + (-1/CONCENTRATION/sqrt(1+4*s/t)) *exp(-(x**2+y**2)/(s+4*t)) + \
    #(1/pi)*(arccos((cos(pi*z) - (sin(pi*z))**2*(1-exp(-pi*t))/(cosh(pi*x)-cos(pi*z)))))
  
#def u1(x,y,t):
#    return (cos(w*t)+sin(w*t))*sin(pi*p*x)*sin(q*pi*y)
 

def RandomWalk(N=10000, d=2):
    """
    Use numpy.cumsum and numpy.random.uniform to generate
    a 2D random walk of length N, each of which has a random DeltaX and
    DeltaY between -1/2 and 1/2.  You'll want to generate an array of 
    shape (N,d), using (for example), random.uniform(min, max, shape).
    """
    Sim_T = []
    con = 1
    for n in range(N):
        if con >=0:
            con = con + np.random.uniform(-0.1,0.1,1) # TODO: use function call results
        else:
            con = 0 
        Sim_T.append(con)          
    return Sim_T
    
def PlotRandomWalkXT(N=10000):
    """
    Plot X(t) for one-dimensional random walk 
    """
    X = RandomWalk(N,1)
    pylab.plot(X)
    pylab.show()
    
      
a = []
#a1 = [] 
for i in range(500):
    v = u(X,Y,t0)
    #z1 = u1(X,Y,t0)
    t0 = t0 + dt
    a.append(v)
    #a1.append(z1)
    #A = a + a1
     
m = plt.cm.ScalarMappable(cmap=plt.cm.jet)
m.set_array(a[0])
cbar = plt.colorbar(m)
 
k = 0
def animate(i):
    global k, concentration
    final = a[k]
    k += 1
    ax1.clear()
    ax1.plot_surface(X,Y,final,rstride=1, cstride=1,cmap=plt.cm.jet,linewidth=0,antialiased=False)
 
    ax1.contour(x,y,final)
    ax1.set_zlim(0,CONCENTRATION)
    ax1.set_xlim(-10,10)
    ax1.set_ylim(-10,10)
     
     
anim = animation.FuncAnimation(fig,animate,frames=220,interval=20)
plt.show()

PlotRandomWalkXT(10000)
