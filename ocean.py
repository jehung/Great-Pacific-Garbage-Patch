# For original tutorial, see (bottom of page): This tutorial came from here (bottom of page): 
# http://polar.ncep.noaa.gov/global/examples/usingpython.shtml

from mpl_toolkits.basemap import Basemap, shiftgrid
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
lon  = file.variables['lon'][:]
u = file.variables['u'][0,0,:,:]
v = file1.variables['v'][0,0,:,:]
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
cs = m.pcolormesh(x,y,u,shading='flat', \
      cmap=plt.cm.jet)
 
#Add a coastline and axis values.
m.drawcoastlines()
m.fillcontinents()
m.drawmapboundary()
m.drawparallels(np.arange(-90.,120.,30.), \
      labels=[1,0,0,0])
m.drawmeridians(np.arange(-180.,180.,60.), \
      labels=[0,0,0,1])
 
#Add a colorbar and title, and then show the plot.
colorbar(cs)
plt.title('Global RTOFS SST from NetCDF')
plt.show()





parallels = np.arange(-80.,90,20.)
meridians = np.arange(0.,360.,20.)
clevs = np.arange(0,20,2)
CS1 = m.contour(x,y,u,clevs,linewidths=0.5,colors='k',animated=True)
CS2 = m.contourf(x,y,v,clevs,cmap=plt.cm.RdBu_r,animated=True)
ugrid,newlons = shiftgrid(315.,u,lon,start=False)
vgrid,newlons = shiftgrid(315.,v,lon,start=False)
uproj,vproj,xx,yy = m.transform_vector(ugrid,vgrid,newlons,lat,31,31,returnxy=True,masked=True)
# now plot.
Q = m.quiver(xx,yy,uproj,vproj,scale=700)
# make quiver key.
qk = plt.quiverkey(Q, 0.1, 0.1, 20, '20 m/s', labelpos='W')
# draw coastlines, parallels, meridians.
m.drawcoastlines(linewidth=1.5)
m.drawparallels(parallels)
m.drawmeridians(meridians)
# add colorbar
cb = m.colorbar(CS2,"bottom", size="5%", pad="2%")
cb.set_label('hPa')
# set plot title
ax.set_title('I am here '+str(time))
plt.show()