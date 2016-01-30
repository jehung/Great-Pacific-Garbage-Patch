import numpy
import pylab		# Plots; also imports array functions cumsum, transpose
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt


def RandomWalk(N=10000, d=3):
    """
    Use numpy.cumsum and numpy.random.uniform to generate
    a 2D random walk of length N, each of which has a random DeltaX and
    DeltaY between -1/2 and 1/2.  You'll want to generate an array of 
    shape (N,d), using (for example), random.uniform(min, max, shape).
    """
    return numpy.cumsum(numpy.random.uniform(-0.5,0.5,(N,d)), axis = 0)

def PlotRandomWalkXT(N=10000):
    """
    Plot X(t) for one-dimensional random walk 
    """
    X = RandomWalk(N,3) #plot 1
    pylab.plot(X)
    pylab.show()

def PlotRandomWalkXY(N=10000):
    """
    Plot X, Y coordinates of random walk where 
        X = numpy.transpose(walk)[0]
        Y = numpy.transpose(walk)[1]
    To make the X and Y axes the same length, 
    use pylab.figure(figsize=(8,8)) before pylab.plot(X,Y) and
    pylab.axis('equal') afterward.
    """
    walk = RandomWalk(N)
    X, Y, Z = numpy.transpose(walk)[0:3]
    pylab.figure(figsize=(8,8)) 
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(X, Y, Z, label = "Brownian Motion")
    ax.legend()
    pylab.axis('equal')
    pylab.show()

def Endpoints(W=10000, N=10000, d=3):
    """
    Returns a list of endpoints of W random walks of length N.
    (In one dimension, this should return an array of one-element arrays,
    to be consistent with higher dimensions.)
    One can generate the random walks and then peel off the final positions,
    or one can generate the steps and sum them directly, for example: 
        sum(numpy.random.uniform(-0.5,0.5,(10,100,2))
    """
    return numpy.cumsum((numpy.random.uniform(-0.5,0.5,(N,W,d)))[:,-1,:], axis = 0)

def PlotEndpoints(W=10000, N=10000, d=3):
    """
    Plot endpoints of random walks.
    Use numpy.transpose to pull out X, Y. 
    To plot black points not joined by lines use pylab.plot(X, Y, 'k.')
    Again, use pylab.figure(figsize=(8,8)) before and
    pylab.axis('equal') afterward.
    """
    X, Y, Z = numpy.transpose(Endpoints(W, N, d))
    pylab.figure(figsize=(8, 8))
    fig = plt.figure()
    ax = fig.gca(projection = "3d")
    ax.plot(X,Y,Z,"k.", label = "End points of drift")
    ax.legend()
    pylab.axis('equal')
    pylab.show()

def HistogramRandomWalk(W=10000, N=10000, d=1, bins=50):
    """
    Compares the histogram of random walks with the normal distribution
    predicted by the central limit theorem.
    #
    (1) Plots a histogram rho(x) of the probability that a random walk
    with N has endpoint X-coordinate at position x. 
    Uses pylab.hist(X, bins=bins, normed=1) to produce the histogram
    #
    (2) Calculates the RMS stepsize sigma for a random walk of length N
    (with each step uniform in [-1/2,1/2]
    Plots rho = (1/(sqrt(2 pi) sigma)) exp(-x**2/(2 sigma**2)) 
    for -3 sigma < x < 3 sigma on the same plot (i.e., before pylab.show).
    Hint: Create x using arange. Squaring, exponentials, and other operations
    can be performed on whole arrays, so typing in the formula for rho will
    work without looping over indices, except sqrt, pi, and exp need to be
    from the appropriate library (pylab, numpy, ...)
    """
    Z = Endpoints(W, N, d)[:,0] 
    pylab.hist(Z, bins=bins, normed=1)
    sigma = numpy.sqrt(N/12.)
    z = numpy.arange(-3*sigma,3*sigma,sigma/bins)
    rho = (1/(numpy.sqrt(2*numpy.pi)*sigma))*numpy.exp(-z**2/(2*sigma**2))
    pylab.plot(z, rho, "k-")
    pylab.show()
    
    
        
#PlotRandomWalkXT(10000)
#PlotRandomWalkXY(10000)
#PlotEndpoints(10000,10000,3)
#HistogramRandomWalk(10000,10000,5)


def yesno():
    response = raw_input('    Continue? (y/n) ')
    if len(response)==0:        # [CR] returns true
        return True
    elif response[0] == 'n' or response[0] == 'N':
        return False
    else:                       # Default
        return True

def demo():
    """Demonstrates solution for exercise: example of usage"""
    print "Random Walk Demo"
    print "Random Walk X vs. t"
    PlotRandomWalkXT(10000)
    if not yesno(): return
    print "Random Walk X vs. Y"
    PlotRandomWalkXY(10000)
    if not yesno(): return
    print "Endpoints of many random walks"
    print "N=1: square symmetry"
    PlotEndpoints(N=1000)
    if not yesno(): return
    print "N=10: emergent circular symmetry"
    PlotEndpoints(N=10000)
    if not yesno(): return
    print "Central Limit Theorem: Histogram N=10 steps"
    HistogramRandomWalk(N=1000)
    if not yesno(): return
    print "1 step"
    HistogramRandomWalk(N=2000)
    if not yesno(): return
    print "2 steps"
    HistogramRandomWalk(N=4000)
    if not yesno(): return
    print "4 steps"
    HistogramRandomWalk(N=6000)
    if not yesno(): return

if __name__=="__main__":
   demo()

