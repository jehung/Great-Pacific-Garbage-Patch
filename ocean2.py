# For original tutorial, see (bottom of page): This tutorial came from here (bottom of page): 
# http://polar.ncep.noaa.gov/global/examples/usingpython.shtml

from mpl_toolkits.basemap import Basemap, shiftgrid
from mpl_toolkits.mplot3d import axes3d
import numpy as np
import matplotlib.pyplot as plt
from pylab import *
import netCDF4
 
plt.figure()
 
nc = 'C:/nco/outfile_trmm.nc'
nc1 =  'C:/nco/outfile_trmm_add.nc'
 
# In this example we will extract the surface temperature field from the model.
# Remember that indexing in Python starts at zero.
file = netCDF4.Dataset(nc)
file1 = netCDF4.Dataset(nc1)
lat  = file.variables['lat'][:]
lon = file.variables['lon'][:]
lev = file.variables['lev'][:]
u = file.variables['u'][0,0,:,:]#*100/86400 zonal velocity
v = file1.variables['v'][0,0,:,:]#*100/86400 meridional velocity
time = file.variables['time'][:]
file.close()
 
#There is a quirk to the global NetCDF files that isn't in the NOMADS data, namely 
#that there are junk values of longitude (lon>500) in the rightmost column of the 
#longitude array (they are ignored by the model itself). So we have to work around them a little with NaN substitution.
lon = np.where(np.greater_equal(lon,500),np.nan,lon)
 
#Plot the field using Basemap. Start with setting the map projection using the limits of the lat/lon data itself
m=Basemap(projection='mill',lat_ts=10, \
      llcrnrlon=np.nanmin(lon),urcrnrlon=np.nanmax(lon), \
      llcrnrlat=lat.min(),urcrnrlat=lat.max(), \
      resolution='c')
 
#Convert the lat/lon values to x/y projections.
x, y = m(lon,lat)

 
#Plot the field using the fast pcolormesh routine and set the colormap to jet.
cs = m.pcolormesh(x,y,u,shading='flat', cmap=plt.cm.jet)
 
cs1 = m.pcolormesh(x,y,v,shading='flat', cmap=plt.cm.jet)
#X, Y = np.meshgrid(np.arange(min(x), max(x), 100000), np.arange(min(y), max(y), 800000))
#rf = m.rotate_vector(u,v,x,y)     
#vf = plt.quiver(X, Y, u, v) 
Q = plt.quiver(x[::100], y[::100], u[::100], v[::100], units='width') 
 
#Add a coastline and axis values.
m.drawcoastlines()
m.fillcontinents()
m.drawmapboundary()
#m.drawparallels(np.arange(min(x),max(x),100000.), labels=[1,0,0,0])
#m.drawmeridians(np.arange(min(y),max(y),100000.),labels=[0,0,0,1])
 
#Add a colorbar and title, and then show the plot.
colorbar(cs)
plt.title('Surface Current Velocity from NetCDF')
plt.show()




###################################################
'''
fig=plt.figure(figsize=(20, 10))


map = Basemap(projection='hammer', 
              lat_0=0, lon_0=0)

#ugrid,newlons = shiftgrid(335.,u,lon,start=False)
#vgrid,newlons = shiftgrid(335.,v,lon,start=False)
#lons = newlons
#lats = lat

lons, lats = np.meshgrid(lon, lat)

v10 = np.ones((u.shape)) 
u10 = np.ones((v.shape))

u10_rot, v10_rot, x, y = map.rotate_vector(u10, v10, lons, lats, returnxy=True)

ax = fig.add_subplot(121)
ax.set_title('Without rotation')

map.drawmapboundary(fill_color='aqua')
map.fillcontinents(color='#cc9955', lake_color='aqua', zorder = 0)
map.drawcoastlines(color = '0.15')


map.barbs(x, y, u10, v10, 
    pivot='middle', barbcolor='#333333')

ax = fig.add_subplot(122)
ax.set_title('Rotated vectors')

map.drawmapboundary(fill_color='#9999FF')
map.fillcontinents(color='#ddaa66', lake_color='#9999FF', zorder = 0)
map.drawcoastlines(color = '0.15')

map.barbs(x, y, u10_rot, v10_rot, 
    pivot='middle', barbcolor='#ff7777')

plt.show()
'''

##############################################################
'''

file = netCDF4.Dataset(nc)
file1 = netCDF4.Dataset(nc1)
lat  = file.variables['lat'][:]
lon = file.variables['lon'][:]
lev = file.variables['lev'][:]
u = file.variables['u'][:,:,:,:]#*100/86400 zonal velocity
v = file1.variables['v'][:,:,:,:]#*100/86400 meridional velocity
time = file.variables['time'][:]
file.close()



fig = plt.figure()
ax = fig.gca(projection='3d')

x, y, z = np.meshgrid(np.arange(min(lon), max(lon), 1),
                      np.arange(min(lat), max(lat), 1),
                      np.arange(min(lev), max(lev), 1))

#u = np.sin(np.pi * x) * np.cos(np.pi * y) * np.cos(np.pi * z)
#v = -np.cos(np.pi * x) * np.sin(np.pi * y) * np.cos(np.pi * z)
#v = (np.sqrt(2.0 / 3.0) * np.cos(np.pi * x) * np.cos(np.pi * y) *np.random.normal(loc = 0.1, scale = 0.25))


ax.quiver(x, y, z, u, v, w, length=0.1)

#plt.show()
'''