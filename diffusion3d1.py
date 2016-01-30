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
x = np.linspace(-5,5,30)
y = np.linspace(-5,5,30)
z = np.linspace(-5,5,30)


dx=0.01        # Interval size in x-direction.
dy=0.01        # Interval size in y-direction.
dz=0.01
a=0.5          # Diffusion constant.
timesteps=10  # Number of time-steps to evolve system.

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
ui = sp.zeros([nx,ny])
u = sp.zeros([nx,ny])

# Now, set the initial conditions (ui).
for i in range(nx):
    for j in range(ny):
	if ( ( (i*dx-0.5)**2+(j*dy-0.5)**2 <= 0.1) & ((i*dx-0.5)**2+(j*dy-0.5)**2>=0.05) ):
            ui[i,j] = 1000000
            
print "uni.shape", ui.shape
print "initial", ui

X = []
X0 = 0
Y = []
Y0 = 0

T0= 0
T=[]
X = []
X0 = 0
Y = []
Y0 = 0

def evolve_ts(x, y, t, u, ui):
    global nx, ny, T0, T, X0, X, Y0, Y
    #X = []
    #X0 = 0
    #Y = []
    #Y0 = 0
    for i in range(1,nx-1):
	for j in range(1,ny-1):
            uxx = ( ui[i+1,j] - 2*ui[i,j] + ui[i-1, j] )/dx2
	    uyy = ( ui[i,j+1] - 2*ui[i,j] + ui[i, j-1] )/dy2
	    uzz = ( ui[i,j+1] - 2*ui[i,j] + ui[i, j-1] )/dz2
	    u[i,j] = ui[i,j]+dt*a*(uxx+uyy+uzz)
	    Z = u
	    Y0 = Y0+dy    
	    Y.append(Y0)
        X0 = X0 + dx
        X.append(X0)
        T0 = T0 + dt
        T.append(T0)
    return X, Y, Z, u
    
# Now, start the time evolution calculation...
tstart = time.time()
for m in range(1, timesteps+1):
	evolve_ts(X0,Y0, T0,u, ui)
	print "Computing u for m =", m
	#ui = u
tfinish = time.time()




print "Done."
print "Total time: ", tfinish-tstart, "s"
print "Average time per time-step using numpy: ", ( tfinish - tstart )/timesteps

'''
def updatefig():
	global u, ui, m
	im.set_array(ui)
	manager.canvas.draw()
	# Uncomment the next two lines to save images as png
	# filename='diffusion_ts'+str(m)+'.png'
	# fig.savefig(filename)
	u[1:-1, 1:-1] = ui[1:-1, 1:-1] + a*dt*(
		(ui[2:, 1:-1] - 2*ui[1:-1, 1:-1] + ui[:-2, 1:-1])/dx2
		+ (ui[1:-1, 2:] - 2*ui[1:-1, 1:-1] + ui[1:-1, :-2])/dy2 )
	ui = u
        ui = sp.copy(u)
	m+=1
	print "Computing and rendering u for m =", m
	if m >= timesteps:
		return False
	return True


fig = plt.figure(1)
img = plt.subplot(111)
im = img.imshow( ui, cmap=cm.hot, interpolation='nearest', origin='lower')
manager = get_current_fig_manager()

m=1
fig.colorbar( im ) # Show the colorbar along the side

# once idle, call updatefig until it returns false.
#gobject.idle_add(updatefig)
show()
'''
k = 0
fig = plt.figure()
fig.set_dpi(100)
ax1 = fig.gca(projection='3d')
 

def animate(i):
    global k
    #concentration = ui[k]
    #print "concentration", concentration
    k += 1
    ax1.clear()
    ax1.plot_surface(X,Y,Z,rstride=1, cstride=1,cmap=plt.cm.jet,linewidth=0,antialiased=False)
 
    #ax1.contour(x,y,temp)
    ax1.set_zlim(0,T)
    ax1.set_xlim(-5,5)
    ax1.set_ylim(-5,5)
     
     
anim = animation.FuncAnimation(fig,animate,frames=220,interval=20)
plt.show()

