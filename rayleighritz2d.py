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
        #for k in range(N + 1):
            if i + j <= N:# + k
                basis.append((i, j))#, k
print "Building basis {0}!".format(time.time() - tmp)

basis = numpy.array(basis).astype('float')
dp = numpy.zeros((len(basis), len(basis), 2, 2))
pv = numpy.zeros((len(basis), len(basis)))

minx = -0.5
maxx = 0.5
miny = -0.5
maxy = 0.5
tmp = time.time()
def polyint(n, m):
    if n < 0 or m < 0:
        return 0.0

    xtmp = numpy.power(maxx, n + 1) - numpy.power(minx, n + 1)
    return (numpy.power(maxy, m + 1) - numpy.power(miny, m + 1)) * xtmp / ((n + 1) * (m + 1))

for i in range(basis.shape[0]):
    for j in range(basis.shape[0]):
        n0, m0 = basis[i]
        n1, m1 = basis[j]

        dp[i, j, 0, 0] = polyint(n1 + n0 - 2, m1 + m0) * n0 * n1
        dp[i, j, 0, 1] = polyint(n1 + n0 - 1, m1 + m0 - 1) * n0 * m1
        dp[i, j, 1, 0] = polyint(n1 + n0 - 1, m1 + m0 - 1) * n1 * m0
        dp[i, j, 1, 1] = polyint(n1 + n0, m1 + m0 - 2) * m0 * m1

        pv[i, j] = polyint(n1 + n0, m1 + m0)
print "Building derivs {0}!".format(time.time() - tmp)


tmp = time.time()

K = numpy.zeros((2 * len(basis), 2 * len(basis)))
for i in range(len(basis)):
    for j in range(len(basis)):
        c = numpy.array([[c11 * dp[i, j, 0, 0] + c44 * dp[i, j, 1, 1], c44 * dp[i, j, 1, 0] + c12 * dp[i, j, 0, 1]],
                        [c12 * dp[i, j, 1, 0] + c44 * dp[i, j, 0, 1], c44 * dp[i, j, 0, 0] + c11 * dp[i, j, 1, 1]]])

        K[2 * i, 2 * j] += c[0, 0]
        K[2 * i + 1, 2 * j] += c[1, 0]
        K[2 * i, 2 * j + 1] += c[0, 1]
        K[2 * i + 1, 2 * j + 1] += c[1, 1]
print "Building stiffness {0}".format(time.time() - tmp)

import matplotlib.pyplot as plt

tmp = time.time()

M = numpy.zeros((2 * len(basis), 2 * len(basis)))

for i in range(len(basis)):
    for j in range(len(basis)):
        M[2 * i, 2 * j] = p * pv[i, j]
        M[2 * i + 1, 2 * j + 1] = p * pv[i, j]
print "Building mass {0}".format(time.time() - tmp)

tmp = time.time()
eigs, evecs = scipy.linalg.eigh(K, M)
print "Solving for eigenvalues {0}!".format(time.time() - tmp)
print "\nEigs!"
for eig in sorted(numpy.real(eigs))[0:30]:
    print "{0:.10f}".format(eig)