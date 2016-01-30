import numpy as np
from numpy import pi,sin,cos,sqrt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import matplotlib.animation as animation
 
fig = plt.figure()
fig.set_dpi(100)
ax1 = fig.gca(projection='3d')
 
x = np.linspace(0,1,30)
y = np.linspace(0,1,30)
 
X,Y = np.meshgrid(x,y)
 
#Wave speed
c = 0.1
 
#Initial time
t0 = 0
 
#Time increment
dt = 0.1
 
#Try every combination
p = 2 #1 #5 #2 #5
q = 3 #1 #5 #3 #3
 
w = pi*c*sqrt(p**2+q**2)
 
#Wave
def u(x,y,t):
    return (cos(w*t)+sin(w*t))*sin(pi*p*x)*sin(q*pi*y)
 
#Building the datapoints
a = []
for i in range(500):
    z = u(X,Y,t0)
    t0 = t0 + dt
    a.append(z)
 
#Adding the colorbar 
m = plt.cm.ScalarMappable(cmap=plt.cm.jet)
m.set_array(a[0])
cbar = plt.colorbar(m)
 
k = 0
def animate(i):
    global k
    Z = a[k]
    k += 1
    ax1.clear()
    ax1.plot_surface(X,Y,Z,rstride=1, cstride=1,cmap=plt.cm.jet,linewidth=0,antialiased=False)
    #ax1.contour(X,Y,Z)
    ax1.set_zlim(0,5)
     
anim = animation.FuncAnimation(fig,animate,frames=220,interval=20)
plt.show()