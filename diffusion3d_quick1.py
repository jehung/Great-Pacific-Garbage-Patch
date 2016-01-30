import time
import scipy as sp
import matplotlib
#matplotlib.use('GTKAgg') # Change this as desired.

#import gobject
from pylab import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import matplotlib.animation as animation



Lx=5000.0   # physical length x vector in micron
Ly=250.0   # physical length y vector in micron
Nx = 50     # number of point of mesh along x direction
Ny = 50     # number of point of mesh along y direction
Nz = 50
a = 0.0017 # diffusion coefficent
dx = 1.0/Nx
dy = 1.0/Ny
dz = 1.0/Nz
dt = (dx**2*dy**2)/(2*a*(dx**2 + dy**2)) # it is 0.04
x = linspace(0.1,Lx, Nx)[np.newaxis] # vector to create mesh
y = linspace(0.1,Ly, Ny)[np.newaxis] # vector to create mesh
I=sqrt(x*y.T) #initial data for heat equation
u=np.ones(([Nx,Ny,Nz])) # u is the matrix referred to heat function
steps=100
'''
for i in range(Nx):
    for j in range(Ny):
	if ( ( (i*dx-0.5)**2+(j*dy-0.5)**2 <= 4) & ((i*dx-0.5)**2+(j*dy-0.5)**2>=0.05) ):
            ui[i,j] = 10
'''

for m in range (0,steps):
    du=np.zeros(([Nx,Ny, Nz]))

    for i in range (1,Nx-1):

        for j in range(1,Ny-1):
            for k in range(1, Nz-1):
                dux = ( u[i+1,j,k] - 2*u[i,j,k] + u[i-1, j,k] ) / dx**2
                duy = ( u[i,j+1,k] - 2*u[i,j,k] + u[i, j-1,k] ) / dy**2            
                duz = ( u[i,j,k+1] - 2*u[i,j,k] + u[i, j,k-1] ) / dz**2            
                du[i,j] = dt*a*(dux+duy)


                if ( ( (i*dx-0.5)**2+(j*dy-0.5)**2 <= 4) & ((i*dx-0.5)**2+(j*dy-0.5)**2>=0.05) ):
                    u[i,j] = 100
    

'''
    # Boundary Conditions
    t1=(u[:,0]+u[:,1])/2
    u[:,0]=t1
    u[:,1]=t1
    t2=(u[0,:]+u[1,:])/2
    u[0,:]=t2
    u[1,:]=t2
    t3=(u[-1,:]+u[-2,:])/2
    u[-1,:]=t3
    u[-2,:]=t3
    u[:,-1]=1
'''



    #filename1='data_{:08d}.txt'

    #if m%100==0:
        #np.savetxt(filename1.format(m),u,delimiter='\t' )