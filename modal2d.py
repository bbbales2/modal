#%%
import numpy
import scipy
import matplotlib.pyplot as plt

N = 15

p = 2700
young = 6.8
poisson = 0.36

mu0 = young / (2.0 * (1.0 + poisson))
lambda0 = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))

c11 = lambda0 + 2 * mu0
c12 = lambda0
c44 = mu0

X = 1.0
Y = 2.0

q = X / (N + 2)
w = Y / (N + 2)

ke = numpy.array([[[(c44 * q) / (3 * w) + (c11 * w) / (3 * q), (c12 + c44) / 4],
                   [(c12 + c44) / 4, (c11 * q) / (3 * w) + (c44 * w) / (3 * q)]],
                  [[-((c44 * q) / (3 * w)) + (c11 * w) / (6 * q), (c12 - c44) / 4],
                   [(-c12 + c44) / 4, -((c11 * q) / (3 * w)) + (c44 * w) / (6 * q)]],
                  [[-((c44 * q) / (6 * w)) - (c11 * w) / (6 *  q), (-c12 - c44) / 4],
                   [(-c12 - c44) / 4, -((c11 * q) / (6 * w)) - (c44 * w) / (6 * q)]],
                  [[(c44 * q) / (6 * w) - (c11 * w) / (3 * q), (-c12 + c44) / 4],
                   [(c12 - c44) / 4, (c11 * q) / (6 * w) - (c44 * w) / (3 * q)]]])

me = p * q * w * numpy.array([1.0 / 9.0, 1.0 / 18.0, 1.0 / 36.0, 1.0 / 18.0])

xs = range(N + 2)[1 : -1]
ys = range(N + 2)[1 : -1]

idxs = []

for i in range(N):
    for j in range(N):
        idxs.append((i, j))

vIds = dict((v, 2 * i) for i, v in enumerate(idxs))

u = numpy.zeros(2 * len(idxs))

K = numpy.zeros((len(u), len(u)))

for i in range(N):
    for j in range(N):
        vId = vIds[(i, j)]

        for ii in range(2):
            for jj in range(2):
                K[vId + ii, vId + jj] += 1.0 * ke[0, ii, jj]

        for oi, oj, k in zip([i, i - 1, i - 1], [j + 1, j + 1, j], [1, 2, 3]):
            if oi < 0 or oi >= N or oj < 0 or oj >= N:
                continue

            ovId = vIds[(oi, oj)]

            for ii in range(2):
                for jj in range(2):
                    K[vId + ii, ovId + jj] += ke[k, ii, jj]

K = (K.T + K)

M = numpy.zeros((len(u), len(u)))

for i in range(N):
    for j in range(N):
        vId = vIds[(i, j)]

        for ii, jj in zip([0, 1], [0, 1]):
            M[vId + ii, vId + jj] += 2.0 * me[0]

        for oi, oj, k in zip([i, i - 1, i - 1], [j + 1, j + 1, j], [1, 2, 3]):
            if oi < 0 or oi >= N or oj < 0 or oj >= N:
                continue

            ovId = vIds[(oi, oj)]

            for ii, jj in zip([0, 1], [0, 1]):
                M[vId + ii, ovId + jj] += me[k]

M = M.T + M

eigs, evecs = scipy.linalg.eigh(K, M)#)#
print eigs[0:10]

#%%

v = numpy.ones(len(u)) * 1.0

v[::2] = 0.0

us = numpy.linalg.solve(K, v)

#%%

us = evecs[:, 5].reshape((N, N, 2))

plt.imshow(us[:, :, 0], interpolation = 'NONE')
plt.colorbar()
plt.show()

plt.imshow(us[:, :, 1], interpolation = 'NONE')
plt.colorbar()
plt.show()