#!/usr/bin/env python
"""
Author: Timothy A.V. Teatro <http://www.timteatro.net>
Date  : Oct 25, 2010
Lisence: Creative Commons BY-SA
(http://creativecommons.org/licenses/by-sa/2.0/)

Description:
	A program which uses an explicit finite difference
	scheme to solve the diffusion equation with fixed
	boundary values and a given initial value for the
	density u(x,y,t). This version uses a numpy
	expression which is evaluated in C, so the
	computation time is greatly reduced over plain
	Python code.

	This version also uses matplotlib to create an
	animation of the time evolution of the density.
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

dx=0.01        # Interval size in x-direction.
dy=0.01        # Interval size in y-direction.
dz=0.01        # Interval size in y-direction.
a=0.5          # Diffusion constant.
timesteps=10  # Number of time-steps to evolve system.

nx = int(1/dx)
ny = int(1/dy)

dx2=dx**2 # To save CPU cycles, we'll compute Delta x^2
dy2=dy**2 # and Delta y^2 only once and store them.
dz2=dz**2 # and Delta y^2 only once and store them.
# For stability, this is the largest interval possible
# for the size of the time-step:
dt = dx2*dy2*dz2/( 2*a*(dx2+dy2+dz2) )

# Start u and ui off as zero matrices:
ui = sp.zeros([nx,ny])
u = sp.zeros([nx,ny])

# Now, set the initial conditions (ui).
for i in range(nx):
	for j in range(ny):
		if ( ( (i*dx-0.5)**2+(j*dy-0.5)**2 <= 0.1)
			& ((i*dx-0.5)**2+(j*dy-0.5)**2>=.05) ):
				ui[i,j] = 1

fig = plt.figure()
ax1 = fig.add_subplot(111)

def de(u, ui):
	"""
	This function uses a numpy expression to
	evaluate the derivatives in the Laplacian, and
	calculates u[i,j] based on ui[i,j].
	"""
	u[1:-1, 1:-1] = ui[1:-1, 1:-1] + dt*( (ui[2:, 1:-1] - ui[1:-1, 1:-1] - ui[1:-1, 1:-1] + ui[:-2, 1:-1])/dx2 + (ui[1:-1, 2:] - ui[1:-1, 1:-1] - ui[1:-1, 1:-1] + ui[1:-1, :-2])/dy2 )
        return u
        
tstart = time.time()
for m in range(1, timesteps+1):
	a = de(u, ui)
	ui = sp.copy(u)
        
	
	

tfinish = time.time()

m = plt.cm.ScalarMappable(cmap=plt.cm.jet)
m.set_array(u[0])
cbar = plt.colorbar(m)

k = 0
def animate(i):
    global k
    final = u[k]
    k += 1
    ax1.clear()
    ax1.plot_surface(X,Y,final,rstride=1, cstride=1,cmap=plt.cm.jet,linewidth=0,antialiased=False)
 
    ax1.contour(x,y,final)
    ax1.set_zlim(0,CONCENTRATION)
    ax1.set_xlim(-10,10)
    ax1.set_ylim(-10,10)
     
     
anim = animation.FuncAnimation(fig,animate,frames=220,interval=20)
plt.show()