from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import matplotlib.pyplot as plt

#your index 
x = np.linspace(1, 200, 200);
y = np.linspace(1, 200, 200)

X, Y = np.meshgrid(x, y);     #making a grid from it

fig = plt.figure()
ax = fig.gca(projection='3d')
R = np.sqrt(X**2 + Y**2)      #make some calculations on the grid
Z = np.sin(R)                 #some more calculations
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
    linewidth=0, antialiased=False)
ax.set_zlim(-5, 5)

fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()