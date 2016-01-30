#!/usr/bin/env python
"""
A program which uses an explicit finite difference
scheme to solve the diffusion equation with fixed
boundary values and a given initial value for the
density.

Two steps of the solution are stored: the current
solution, u, and the previous step, ui. At each time-
step, u is calculated from ui. u is moved to ui at the
end of each time-step to move forward in time.

"""
import time
import scipy as sp
import matplotlib
#matplotlib.use('GTKAgg') # Change this as desired.

#import gobject
from pylab import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import matplotlib.animation as animation


# Declare some variables:
x = np.linspace(-150,150,100)
y = np.linspace(-150,150,100)
z = np.linspace(-150,150,100)
X,Y = np.meshgrid(x,y)

dx=0.01        # Interval size in x-direction.
dy=0.01        # Interval size in y-direction.
dz=0.01
a=0.5          # Diffusion constant.
timesteps=2  # Number of time-steps to evolve system.

nx = int(1/dx)
ny = int(1/dy)
nz = int(1/dz)

dx2=dx**2 # To save CPU cycles, we'll compute Delta x^2
dy2=dy**2 # and Delta y^2 only once and store them.
dz2=dz**2 # and Delta y^2 only once and store them.

# For stability, this is the largest interval possible
# for the size of the time-step:
dt = dx2*dy2*dz2/( 2*a*(dx2+dy2+dz2) )

# Start u and ui off as zero matrices:
ui = sp.zeros([nx,ny,nz])
u = sp.zeros([nx,ny,nz])

xs = np.empty((timesteps + 1,))
ys = np.empty((timesteps + 1,))
zs = np.empty((timesteps + 1,))
# Now, set the initial conditions (ui).
for i in range(nx):
    for j in range(ny):
	if ( ( (i*dx-0.5)**2+(j*dy-0.5)**2 <= 4) & ((i*dx-0.5)**2+(j*dy-0.5)**2>=0.05) ):
            ui[i,j] = 100
            
#print "uni.shape", ui.shape
#print "initial", ui

def laplacian():
    uxx = ( ui[i+1,j,k] - 2*ui[i,j,k] + ui[i-1, j,k] )/dx2
    uyy = ( ui[i,j+1,k] - 2*ui[i,j,k] + ui[i, j-1,k] )/dy2
    uzz = ( ui[i,j,k+1] - 2*ui[i,j,k] + ui[i, j, k-1] )/dz2
    return uxx, uyy, uzz
    
'''
for i in range(1,nx-1):
    for j in range(1,ny-1):
        for k in range(1, nz-1):
            x_dd, y_dd, z_dd = laplacian()
            u[i,j,k] = ui[i,j,k] + dt*a*(x_dd+y_dd+z_dd)
            #u[i,j,k] = ui[i,j,k]+dt*a*(uxx+uyy+uzz)
'''    
	        
def fast_de(u,ui):
    u[1:-1,1:-1,1:-1] = ui[1:-1,1:-1.1:-1] + a*dt*( (ui[2:, 1:-1, -1:-1] - 2*ui[1:-1, 1:-1, 1:-1] + ui[:-2, 1:-1, 1:-1])/dx2 + (ui[1:-1, 2:, 1:-1] - 2*ui[1:-1, 1:-1,1:-1] + ui[1:-1, 1:-1:-2])/dy2  + 
                     (ui[1:-1,1:-1,2] - 2*ui[1:-1, 1:-1,1:-1] + ui[:-2, 1:-1,1:-1])/dz2 ) 
    ui = sp.copy(u)


m = plt.cm.ScalarMappable(cmap=plt.cm.jet)
m.set_array(u[0])
cbar = plt.colorbar(m)


for i in range(timesteps):
    fast_de
    

k = 0
def animate(i):
    global k
    final = ui[0]
    k += 1
    ax1.clear()
    ax1.plot_surface(X,Y,final,rstride=1, cstride=1,cmap=plt.cm.jet,linewidth=0,antialiased=False)
 
    #ax1.contour(x,y,final)
    #ax1.set_zlim(0,CONCENTRATION)
    ax1.set_xlim(-150,150)
    ax1.set_ylim(-150,150)
    #ax1.set_zlim(-150,150) 
    
fig = plt.figure()
fig.set_dpi(100)
ax1 = fig.gca(projection='3d')
     
anim = animation.FuncAnimation(fig,animate,frames=220,interval=5)

plt.show()
