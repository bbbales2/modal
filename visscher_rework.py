import numpy
import time
import scipy.linalg
import os
os.chdir('/home/bbales2/modal')
import pyximport
pyximport.install(reload_support = True)

import polybasisqu
reload(polybasisqu)
from rotations import inv_rotations

# basis polynomials are x^n * y^m * z^l where n + m + l <= N
N = 4

## Dimensions for TF-2
X = 0.007753
Y = 0.009057
Z = 0.013199

#sample mass

#Sample density
density = 4401.695921

def func():
    M = numpy.random.rand(6, 6)

    C = M.transpose() * M

    emin = scipy.linalg.eigh(C)[0][0]

    C -= numpy.eye(6) * emin * 1.1
    
    print C
    
    numpy.linalg.cholesky(C)

    dp, pv, ddpdX, ddpdY, ddpdZ, dpvdX, dpvdY, dpvdZ = polybasisqu.build(N, X, Y, Z)

    cu = numpy.random.rand(3)
    print cu
    w, x, y, z = inv_rotations.cu2qu(list(cu))
    
    C, _, _, _, _, _ = polybasisqu.buildRot(C, w, x, y, z)

    K, M = polybasisqu.buildKM(C, dp, pv, density)
    eigs, evecs = scipy.linalg.eigh(K, M, eigvals = (6, 6 + 30 - 1))

    return numpy.sqrt(eigs * 1e11) / (numpy.pi * 2000), C, K, M

feigs, C, K_, M_ = func()

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

def f(l0, l1, d):
    p = l0 + l1
    
    return numpy.power(d, p + 1) / (p + 1)

def fd(l0, l1, d):
    p = l0 + l1 - 1

    if l1 == 0:
        return 0.0
    
    return l1 * numpy.power(d, p + 1) / (p + 1)

def df(l0, l1, d):
    return fd(l1, l0, d)

def dd(l0, l1, d):
    p = l0 + l1 - 2

    if l0 == 0:
        return 0.0

    if l1 == 0:
        return 0.0
    
    return l1 * l0 * numpy.power(d, p + 1) / (p + 1)

for i in range(3):
    for k0, (l0, m0, n0) in enumerate(lmns):
        for k1, (l1, m1, n1) in enumerate(lmns):
            M[k0, i, k1, i] = density * f(l0, l1, X) * f(m0, m1, Y) * f(n0, n1, Z)

dp = numpy.zeros((R, 3, R, 3))

for k0, (l0, m0, n0) in enumerate(lmns):
    for k1, (l1, m1, n1) in enumerate(lmns):
        dp[k0, 0, k1, 0] = dd(l0, l1, X) * f(m0, m1, Y) * f(n0, n1, Z)
        dp[k0, 1, k1, 0] = fd(l0, l1, X) * df(m0, m1, Y) * f(n0, n1, Z)
        dp[k0, 2, k1, 0] = fd(l0, l1, X) * f(m0, m1, Y) * df(n0, n1, Z)

        dp[k0, 0, k1, 1] = df(l0, l1, X) * fd(m0, m1, Y) * f(n0, n1, Z)
        dp[k0, 1, k1, 1] = f(l0, l1, X) * dd(m0, m1, Y) * f(n0, n1, Z)
        dp[k0, 2, k1, 1] = f(l0, l1, X) * fd(m0, m1, Y) * df(n0, n1, Z)

        dp[k0, 0, k1, 2] = df(l0, l1, X) * f(m0, m1, Y) * fd(n0, n1, Z)
        dp[k0, 1, k1, 2] = f(l0, l1, X) * df(m0, m1, Y) * fd(n0, n1, Z)
        dp[k0, 2, k1, 2] = f(l0, l1, X) * f(m0, m1, Y) * dd(n0, n1, Z)

for i in range(3):
    for ip in range(3):
        for k0, (l0, m0, n0) in enumerate(lmns):
            for k1, (l1, m1, n1) in enumerate(lmns):
                for j in range(3):
                    for jp in range(3):
                        K[k0, i, k1, ip] += Cv[i, j, ip, jp] * dp[k0, j, k1, jp]

K = K.reshape((3 * R, 3 * R))
M = M.reshape((3 * R, 3 * R))

eigs, evecs = scipy.linalg.eigh(K, M, eigvals = (6, 6 + 30 - 1))

fs = numpy.sqrt(eigs * 1e11) / (numpy.pi * 2000)
print feigs
print fs

