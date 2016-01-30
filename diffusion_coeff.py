import numpy

n_particles = 400
n_steps = 400
dl = 16.98696
D =  0
X = numpy.zeros(n_particles)

for i in range(n_steps):
    X += numpy.random.uniform(-dl, dl, n_particles)
    sigma = X.std()

    D = sigma**2 / ( 2 * i)
