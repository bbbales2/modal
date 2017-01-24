#%%

import numpy
import time
import scipy

p = 2700
young = 6.8
poisson = 0.36

mu0 = young / (2.0 * (1.0 + poisson))
lambda0 = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))

c11 = lambda0 + 2 * mu0
c12 = lambda0
c44 = mu0

X = 1.0
Y = 1.0

Dd = numpy.array([[c11, c12, 0],
                  [c12, c11, 0],
                  [0, 0, c44]])

N = 10

basis = []

tmp = time.time()
for i in range(0, N + 1):
    for j in range(0, N + 1):
        for k in range(0, N + 1):
            if i + j + k <= N:
                basis.append((i, j, k))
print "Building basis {0}!".format(time.time() - tmp)

basis = numpy.array(basis).astype('float')
dp = numpy.zeros((len(basis), len(basis), 3, 3))
pv = numpy.zeros((len(basis), len(basis)))

minx = -0.5
maxx = 0.5
miny = -0.5
maxy = 0.5
minz = -0.5
maxz = 0.5

tmp = time.time()
def polyint(n, m, l):
    if n < 0 or m < 0 or l < 0:
        return 0.0

    xtmp = numpy.power(maxx, n + 1) - numpy.power(minx, n + 1)
    ytmp = (numpy.power(maxy, m + 1) - numpy.power(miny, m + 1)) * xtmp
    return (numpy.power(maxz, l + 1) - numpy.power(minz, l + 1)) * ytmp / ((n + 1) * (m + 1) * (l + 1))

for i in range(basis.shape[0]):
    for j in range(basis.shape[0]):
        n0, m0, l0 = basis[i]
        n1, m1, l1 = basis[j]

        dp[i, j, 0, 0] = polyint(n1 + n0 - 2, m1 + m0, l1 + l0) * n0 * n1
        dp[i, j, 0, 1] = polyint(n1 + n0 - 1, m1 + m0 - 1, l1 + l0) * n0 * m1
        dp[i, j, 0, 2] = polyint(n1 + n0 - 1, m1 + m0, l1 + l0 - 1) * n0 * l1

        dp[i, j, 1, 0] = polyint(n1 + n0 - 1, m1 + m0 - 1, l1 + l0) * m0 * n1
        dp[i, j, 1, 1] = polyint(n1 + n0, m1 + m0 - 2, l1 + l0) * m0 * m1
        dp[i, j, 1, 2] = polyint(n1 + n0, m1 + m0 - 1, l1 + l0 - 1) * m0 * l1

        dp[i, j, 2, 0] = polyint(n1 + n0 - 1, m1 + m0, l1 + l0 - 1) * l0 * n1
        dp[i, j, 2, 1] = polyint(n1 + n0, m1 + m0 - 1, l1 + l0 - 1) * l0 * m1
        dp[i, j, 2, 2] = polyint(n1 + n0, m1 + m0, l1 + l0 - 2) * l0 * l1

        pv[i, j] = polyint(n1 + n0, m1 + m0, l1 + l0)
print "Building derivs {0}!".format(time.time() - tmp)

#%%
tmp = time.time()

print "Building stiffness {0}".format(time.time() - tmp)

import matplotlib.pyplot as plt

def buildKM(c11, c12, c44):
    K = numpy.array([[c11 * dp[:, :, 0, 0] + c44 * dp[:, :, 1, 1] + c44 * dp[:, :, 2, 2], c44 * dp[:, :, 1, 0] + c12 * dp[:, :, 0, 1], c44 * dp[:, :, 2, 0] + c12 * dp[:, :, 0, 2]],
                     [c12 * dp[:, :, 1, 0] + c44 * dp[:, :, 0, 1], c44 * dp[:, :, 0, 0] + c11 * dp[:, :, 1, 1] + c44 * dp[:, :, 2, 2], c44 * dp[:, :, 2, 1] + c12 * dp[:, :, 1, 2]],
                     [c12 * dp[:, :, 2, 0] + c44 * dp[:, :, 0, 2], c12 * dp[:, :, 2, 1] + c44 * dp[:, :, 1, 2], c44 * dp[:, :, 0, 0] + c44 * dp[:, :, 1, 1] + c11 * dp[:, :, 2, 2]]])

    K = numpy.rollaxis(K, 1, 3)
    K = K.reshape(3 * len(basis), 3 * len(basis), order = 'F')

    M = numpy.array([[p * pv[:, :], 0 * pv[:, :], 0 * pv[:, :]],
                     [0 * pv[:, :], p * pv[:, :], 0 * pv[:, :]],
                     [0 * pv[:, :], 0 * pv[:, :], p * pv[:, :]]])

    M = numpy.rollaxis(M, 1, 3)
    M = M.reshape(3 * len(basis), 3 * len(basis), order = 'F')

    return K, M

K, M = buildKM(c11, c12, c44)

tmp = time.time()

#M = numpy.zeros((3 * len(basis), 3 * len(basis)))

#for i in range(len(basis)):
#    for j in range(len(basis)):
#        M[3 * i, 3 * j] =
#        M[3 * i + 1, 3 * j + 1] = p * pv[i, j]
#        M[3 * i + 2, 3 * j + 2] = p * pv[i, j]
print "Building mass {0}".format(time.time() - tmp)

tmp = time.time()
eigs2, evecs = scipy.linalg.eigh(K, M, eigvals = (0, 36))
print "Solving for eigenvalues {0}!".format(time.time() - tmp)
print "\nEigs!"
for eig, eig2 in zip(eigs, eigs2[6:36]):
    print "{0:.4f}, {1:.4f}".format(eig, eig2)