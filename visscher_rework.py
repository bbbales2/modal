#%%
import GPy
import pystan
import numpy
import time
import scipy
import os
os.chdir('/home/bbales2/modal')
import pyximport
pyximport.install(reload_support = True)

import polybasisqu
reload(polybasisqu)

# basis polynomials are x^n * y^m * z^l where n + m + l <= N
N = 10

## Dimensions for TF-2
X = 0.007753
Y = 0.009057
Z = 0.013199

#sample mass

#Sample density
density = 4401.695921

c11 = 2.0
anisotropic = 1.0
c44 = 1.0
c12 = -(c44 * 2.0 / anisotropic - c11)

def func(c11, anisotropic, c44):
    c12 = -(c44 * 2.0 / anisotropic - c11)

    C = numpy.array([[c11, c12, c12, 0, 0, 0],
                     [c12, c11, c12, 0, 0, 0],
                     [c12, c12, c11, 0, 0, 0],
                     [0, 0, 0, c44, 0, 0],
                     [0, 0, 0, 0, c44, 0],
                     [0, 0, 0, 0, 0, c44]])

    try:
        numpy.linalg.cholesky(C)
    except:
        return [numpy.nan]

    dp, pv, ddpdX, ddpdY, ddpdZ, dpvdX, dpvdY, dpvdZ = polybasisqu.build(N, X, Y, Z)

    K, M = polybasisqu.buildKM(C, dp, pv, density)
    eigs, evecs = scipy.linalg.eigh(K, M, eigvals = (6, 6 + len(data) - 1))

    return numpy.sqrt(eigs * 1e11) / (numpy.pi * 2000)

#%%

C = numpy.array([[c11, c12, c12, 0, 0, 0],
                 [c12, c11, c12, 0, 0, 0],
                 [c12, c12, c11, 0, 0, 0],
                 [0, 0, 0, c44, 0, 0],
                 [0, 0, 0, 0, c44, 0],
                 [0, 0, 0, 0, 0, c44]])

def Cvoigt(Ch):
    C = numpy.zeros((3, 3, 3, 3))

    voigt = [[(0, 0)], [(1, 1)], [(2, 2)], [(1, 2), (2, 1)], [(0, 2), (2, 0)], [(0, 1), (1, 0)]]

    for i in range(6):
        for j in range(6):
            for k, l in voigt[i]:
                for n, m in voigt[j]:
                    C[k, l, n, m] = Ch[i, j]
    return C

Cv = Cvoigt(C)

R = 3 * (N + 1) * (N + 2) * (N + 3) / 18

lmns = []
for l in range(0, N + 1):
    for m in range(0, N + 1):
        for n in range(0, N + 1):
            if l + m + n <= N:
                lmns.append((l, m, n))

M = numpy.zeros((R, 3, R, 3))
K = numpy.zeros((R, 3, R, 3))

def f(p, q, r):
    d1 = X
    d2 = Y
    d3 = Z

    return numpy.power(d1, p + 1) * numpy.power(d2, q + 1) * numpy.power(d3, r + 1) / ((p + 1) * (q + 1) * (r + 1))

for i in range(3):
    for k0, (l0, m0, n0) in enumerate(lmns):
        for k1, (l1, m1, n1) in enumerate(lmns):
            M[k0, i, k1, i] = density * f(l0 + l1, m0 + m1, n0 + n1)

for i in range(3):
    for ip in range(3):
        for k0, (l0, m0, n0) in enumerate(lmns):
            for k1, (l1, m1, n1) in enumerate(lmns):
                for j in range(3):
                    for jp in range(3):
                        K[k0, i, k1, ip] += Cv[i, j, ip, jp] * f(l0 + l1, m0 + m1, n0 + n1)
#%%

K = K.reshape((3 * R, 3 * R))
M = M.reshape((3 * R, 3 * R))

eigs, evecs = scipy.linalg.eigh(K, M, eigvals = (0, 6 + 30 - 1))