#%%
import numpy
import scipy
import matplotlib.pyplot as plt

N = 20

c11 = 1.0
c12 = 0.33
c44 = c11 - 2 * c12

keaa = numpy.array([[(c11 + c44) / 3.0, (c12 + c44) / 4.0],
                    [(c12 + c44) / 4.0, (c11 + c44) / 3.0]])

keab = numpy.array([[-(c11 + c44) / 6.0, -(c12 + c44) / 4.0],
                    [-(c12 + c44) / 4.0, -(c11 + c44) / 6.0]])

keac = numpy.array([[(-2 * c11 + c44) / 6.0, (-c12 + c44) / 4.0],
                    [(c12 - c44) / 4.0, (c11 - 2 * c44) / 6.0]])

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
                K[vId + ii, vId + jj] += 4.0 * keaa[ii, jj]

        for oi, oj in zip([i - 1, i - 1, i + 1, i + 1], [j - 1, j + 1, j - 1, j + 1]):
            if oi < 0 or oi >= N or oj < 0 or oj >= N:
                continue

            ovId = vIds[(oi, oj)]

            for ii in range(2):
                for jj in range(2):
                    K[vId + ii, ovId + jj] += keab[ii, jj]

        for oi, oj in zip([i - 1, i, i, i + 1], [j, j + 1, j - 1, j]):
            if oi < 0 or oi >= N or oj < 0 or oj >= N:
                continue

            ovId = vIds[(oi, oj)]

            for ii in range(2):
                for jj in range(2):
                    K[vId + ii, ovId + jj] += 2.0 * keac[ii, jj]

M = numpy.zeros((len(u), len(u)))

for i in range(N):
    for j in range(N):
        vId = vIds[(i, j)]

        for ii in range(2):
            for jj in range(2):
                M[vId + ii, vId + jj] += 4.0 / 4.0

        for oi, oj in zip([i - 1, i - 1, i + 1, i + 1], [j - 1, j + 1, j - 1, j + 1]):
            if oi < 0 or oi >= N or oj < 0 or oj >= N:
                continue

            ovId = vIds[(oi, oj)]

            for ii in range(2):
                for jj in range(2):
                    M[vId + ii, ovId + jj] += 1.0 / 4.0

        for oi, oj in zip([i - 1, i, i, i + 1], [j, j + 1, j - 1, j]):
            if oi < 0 or oi >= N or oj < 0 or oj >= N:
                continue

            ovId = vIds[(oi, oj)]

            for ii in range(2):
                for jj in range(2):
                    M[vId + ii, ovId + jj] += 2.0 / 4.0

eigs, evecs = scipy.linalg.eig(K, M)

#%%