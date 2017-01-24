#%%
import matplotlib.pyplot as plt
import numpy
import scipy

h = 0.7
g = 0.5

N = 50

xs = numpy.linspace(0, 1.0, N + 1)[:-1]

l = 1.0 / N

K = numpy.zeros((N, N))

for i in range(N):
    for j in range(N):
        if i == j:
            if i == 0:
                K[i, j] = 1.0 / l
            else:
                K[i, j] = 2.0 / l
        elif abs(i - j) == 1:
            K[i, j] = -1.0 / l

f = numpy.zeros(N)

f[0] = h
f[-1] = (1.0 / l) * g

u = numpy.linalg.solve(K, f)

plt.plot(xs, u)
plt.show()
#%%
g = 0.0

N = 50

xs = numpy.linspace(0, 1.0, N + 2)[1:-1]

l = 1.0 / N

K = numpy.zeros((N, N))
M = numpy.zeros((N, N))

for i in range(N):
    for j in range(N):
        if i == j:
            K[i, j] = 2.0 / l
        elif abs(i - j) == 1:
            K[i, j] = -1.0 / l

for i in range(N):
    for j in range(N):
        if i == j:
            M[i, j] = 2.0 * l / 6.0
        elif abs(i - j) == 1:
            M[i, j] = l / 6.0

eigs, evecs = scipy.linalg.eigh(K, M)

for i in range(5):
    plt.plot(xs, evecs[:, i])
    plt.show()

plt.plot(eigs[:10], numpy.array(range(1, 11))**2 * numpy.pi**2)

#%%
g = 0.0

N = 50

xs = numpy.linspace(0, 1.0, N + 2)[1:-1]

l = 1.0 / N

K = numpy.zeros((N, N))
M = numpy.zeros((N, N))

for i in range(N):
    for j in range(N):
        if i == j:
            K[i, j] = 2.0 / l
        elif abs(i - j) == 1:
            K[i, j] = -1.0 / l

for i in range(N):
    for j in range(N):
        if i == j:
            M[i, j] = 2.0 * l / 6.0
        elif abs(i - j) == 1:
            M[i, j] = l / 6.0

eigs, evecs = scipy.linalg.eigh(K, M)

for i in range(5):
    plt.plot(xs, evecs[:, i])
    plt.show()

plt.plot(eigs[:10], numpy.array(range(1, 11))**2 * numpy.pi**2)

