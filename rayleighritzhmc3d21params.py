#%%

import numpy
import time
import scipy
import os
os.chdir('/home/bbales2/modal')
import pyximport
pyximport.install(reload_support = True)

import polybasis
reload(polybasis)

N = 8

density = 4401.695921e-3#8700.0e-3

X = 0.007753e1#0.011959e1
Y = 0.009057e1#0.013953e1
Z = 0.013199e1#0.019976e1
#[ 1.61054549  0.9710114   0.44089017  0.46434645]
c11 = 1.7e-1
anisotropic = 1.0
c44 = 0.45e-1
c12 = -(c44 * 2.0 / anisotropic - c11)

a = 0.0
b = 0.0
y = 0.0

freqs = numpy.array([109.076,
136.503,
144.899,
184.926,
188.476,
195.562,
199.246,
208.460,
231.220,
232.630,
239.057,
241.684,
242.159,
249.891,
266.285,
272.672,
285.217,
285.670,
288.796,
296.976,
301.101,
303.024,
305.115,
305.827,
306.939,
310.428,
318.000,
319.457,
322.249,
323.464,
324.702,
334.687,
340.427,
344.087,
363.798,
364.862,
371.704,
373.248])

data = (freqs * numpy.pi * 2000) ** 2 / 1e11

#%%
reload(polybasis)

dp1, pv1, ddpdX, ddpdY, ddpdZ, dpvdX, dpvdY, dpvdZ = polybasis.build(N, X, Y, Z)

for i in range(3):
    factors = numpy.array([X, Y, Z])
    factors[i] *= 1.0001
    Xt, Yt, Zt = factors
    dp2, pv2, ddpdX, ddpdY, ddpdZ, dpvdX, dpvdY, dpvdZ = polybasis.build(N, Xt, Yt, Zt)

    ddps = [ddpdX, ddpdY, ddpdZ]
    dpvs = [dpvdX, dpvdY, dpvdZ]

    print (dp2[1, 1] - dp1[1, 1]) / (factors[i] - factors[i] / 1.0001)
    print ddps[i][1, 1]
    print (pv2[0, 0] - pv1[0, 0]) / (factors[i] - factors[i] / 1.0001)
    print dpvs[i][0, 0]
    print '----'

#%%
reload(polybasis)

tmp = time.time()

dp, pv, ddpdX, ddpdY, ddpdZ, dpvdX, dpvdY, dpvdZ = polybasis.build(N, X, Y, Z)

print X * Y * Z * density

print "Building stiffness {0}".format(time.time() - tmp)

import matplotlib.pyplot as plt

def buildKM(c11, c12, c44, dp, pv, density):
    dpe = numpy.zeros((dp.shape[0], dp.shape[1], dp.shape[2], dp.shape[3], 3, 3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    dpe[:, :, i, j, k, l] = dp[:, :, j, l]

    C = numpy.zeros((3, 3, 3, 3))

    Ch = numpy.array([[c11, c12, c12, 0, 0, 0],
                      [c12, c11, c12, 0, 0, 0],
                      [c12, c12, c11, 0, 0, 0],
                      [0, 0, 0, c44, 0, 0],
                      [0, 0, 0, 0, c44, 0],
                      [0, 0, 0, 0, 0, c44]])

    voigt = [[(0, 0)], [(1, 1)], [(2, 2)], [(1, 2), (2, 1)], [(0, 2), (2, 0)], [(0, 1), (1, 0)]]

    for i in range(6):
        for j in range(6):
            for k, l in voigt[i]:
                for n, m in voigt[j]:
                    C[k, l, n, m] = Ch[i, j]

    K = numpy.einsum('ijkl,nmijkl->nimk', C, dpe)
    K = K.reshape((K.shape[0] * K.shape[1], K.shape[2] * K.shape[3]))

    M = numpy.array([[density * pv[:, :], 0 * pv[:, :], 0 * pv[:, :]],
                     [0 * pv[:, :], density * pv[:, :], 0 * pv[:, :]],
                     [0 * pv[:, :], 0 * pv[:, :], density * pv[:, :]]])

    M = numpy.rollaxis(M, 1, 3)
    M = M.reshape(3 * pv.shape[0], 3 * pv.shape[0], order = 'F')

    return K, M

tmp = time.time()
C = numpy.array([[c11, c12, c12, 0, 0, 0],
                 [c12, c11, c12, 0, 0, 0],
                 [c12, c12, c11, 0, 0, 0],
                 [0, 0, 0, c44, 0, 0],
                 [0, 0, 0, 0, c44, 0],
                 [0, 0, 0, 0, 0, c44]])

C, dCda, dCdb, dCdy, K = polybasis.buildRot(C, a, b, y)

K, M = polybasis.buildKM(C, dp, pv, density)
print "Build KM: ", time.time() - tmp

eigs, evecs = scipy.linalg.eigh(K, M, eigvals = (6, 6 + 38))

for eig1, eigs2 in zip(eigs, data):
    print eig1, eigs2
#%%
dp1, pv1, ddpdX, ddpdY, ddpdZ, dpvdX, dpvdY, dpvdZ = polybasis.build(N, X, Y, Z)

C = numpy.array([[c11, c12, c12, 0, 0, 0],
                     [c12, c11, c12, 0, 0, 0],
                     [c12, c12, c11, 0, 0, 0],
                     [0, 0, 0, c44, 0, 0],
                     [0, 0, 0, 0, c44, 0],
                     [0, 0, 0, 0, 0, c44]])

C, dCda, dCdb, dCdy, K = polybasis.buildRot(C, a, b, y)

K1, M1 = polybasis.buildKM(C, dp1, pv1, density)

a, b, y = 0.1, 0.2, 0.3

for i in range(3):
    factors = numpy.array([X, Y, Z])
    factors[i] *= 1.0001
    Xt, Yt, Zt = factors
    dp2, pv2, ddpdX, ddpdY, ddpdZ, dpvdX, dpvdY, dpvdZ = polybasis.build(N, Xt, Yt, Zt)

    dKdX, dMdX = polybasis.buildKM(C, ddpdX, dpvdX, density)
    dKdY, dMdY = polybasis.buildKM(C, ddpdY, dpvdY, density)
    dKdZ, dMdZ = polybasis.buildKM(C, ddpdZ, dpvdZ, density)

    K2, M2 = polybasis.buildKM(C, dp2, pv2, density)

    ddps = [dKdX, dKdY, dKdZ]
    dpvs = [dMdX, dMdY, dMdZ]

    print (K2[494, 486] - K1[494, 486]) / (factors[i] - factors[i] / 1.0001)
    print ddps[i][494, 486]
    print (M2[0, 12] - M1[0, 12]) / (factors[i] - factors[i] / 1.0001)
    print dpvs[i][0, 12]
    print '----'
#%%
reload(polybasis)

dp, pv, ddpdX, ddpdY, ddpdZ, dpvdX, dpvdY, dpvdZ = polybasis.build(N, X, Y, Z)

C = numpy.array([[c11, c12, c12, 0, 0, 0],
                     [c12, c11, c12, 0, 0, 0],
                     [c12, c12, c11, 0, 0, 0],
                     [0, 0, 0, c44, 0, 0],
                     [0, 0, 0, 0, c44, 0],
                     [0, 0, 0, 0, 0, c44]])

C1, dCda1, dCdb1, dCdy1, K = polybasis.buildRot(C, a, b, y)

dCdc11 = K.dot(numpy.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).dot(K.T))

dCdc12 = K.dot(numpy.array([[0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                                [1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                                [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).dot(K.T))

dCdc44 = K.dot(numpy.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]).dot(K.T))

    #tmp = time.time()
dKda1, _ = polybasis.buildKM(dCda1, dp, pv, density)
dKdb1, _ = polybasis.buildKM(dCdb1, dp, pv, density)
dKdy1, _ = polybasis.buildKM(dCdy1, dp, pv, density)

dKdc111, _ = polybasis.buildKM(dCdc11, dp, pv, density)
dKdc121, _ = polybasis.buildKM(dCdc12, dp, pv, density)
dKdc441, _ = polybasis.buildKM(dCdc44, dp, pv, density)

dKdX1, dMdX1 = polybasis.buildKM(C1, ddpdX, dpvdX, density)
dKdY1, dMdY1 = polybasis.buildKM(C1, ddpdY, dpvdY, density)
dKdZ1, dMdZ1 = polybasis.buildKM(C1, ddpdZ, dpvdZ, density)

K1, M1 = polybasis.buildKM(C1, dp, pv, density)

for i in range(3):
    factors = numpy.array([c11, c12, c44])
    factors[i] *= 1.0001
    c11t, c12t, c44t = factors

    C = numpy.array([[c11t, c12t, c12t, 0, 0, 0],
                     [c12t, c11t, c12t, 0, 0, 0],
                     [c12t, c12t, c11t, 0, 0, 0],
                     [0, 0, 0, c44t, 0, 0],
                     [0, 0, 0, 0, c44t, 0],
                     [0, 0, 0, 0, 0, c44t]])

    C, dCda, dCdb, dCdy, K = polybasis.buildRot(C, a, b, y)

    dCdc11 = K.dot(numpy.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).dot(K.T))

    dCdc12 = K.dot(numpy.array([[0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                                    [1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                                    [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).dot(K.T))

    dCdc44 = K.dot(numpy.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]).dot(K.T))

        #tmp = time.time()
    dKda, _ = polybasis.buildKM(dCda, dp, pv, density)
    dKdb, _ = polybasis.buildKM(dCdb, dp, pv, density)
    dKdy, _ = polybasis.buildKM(dCdy, dp, pv, density)

    dKdc11, _ = polybasis.buildKM(dCdc11, dp, pv, density)
    dKdc12, _ = polybasis.buildKM(dCdc12, dp, pv, density)
    dKdc44, _ = polybasis.buildKM(dCdc44, dp, pv, density)

    dKdX, dMdX = polybasis.buildKM(C, ddpdX, dpvdX, density)
    dKdY, dMdY = polybasis.buildKM(C, ddpdY, dpvdY, density)
    dKdZ, dMdZ = polybasis.buildKM(C, ddpdZ, dpvdZ, density)

    K, M = polybasis.buildKM(C, dp, pv, density)

    ders = [dKdc111, dKdc121, dKdc441]

    print ((K - K1) / (factors[i] - factors[i] / 1.0001))[13:16, 13:16]
    print ders[i][13:16, 13:16]
    print '--'

#%%
c11, c12, c13, c22, c23, c33, c44, c55, c66
#%%
current_q = numpy.array([2e-1,  1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 2e-1, 1e-1, 1e-1, 1e-1, 1e-1, 2e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1,  1.0, 0.007753e1, 0.009057e1, 0.013199e1, 0.0, 0.0, 0.0])
#current_q = numpy.array([ 0.18699697,  0.07438127,  0.11770903,  0.18531491,  0.12539511,  0.23760396,
#  0.04348343,  0.04196064,  0.04293918,  0.2,  0.07753,     0.09057,
#  0.13199,     0.,          0.,          0.])
#current_q = numpy.array([ 0.16407578,  0.08022502,  0.07757862,  0.17259511,  0.07695353,
#        0.17368499,  0.04550905,  0.04358656,  0.04193578,  1.0       ,
#        0.07753   ,  0.09057   ,  0.13199   ,  0.        ,  0.        ,  0.        ])

current_q = numpy.array([ 0.20714866,  0.12739981,  0.12221891,  0.07361753,  0.0677342 ,
        0.06754691,  0.17164197,  0.13178536,  0.06855026,  0.08137753,
        0.08759628,  0.18721634,  0.09113698,  0.06745608,  0.09215318,
        0.31886459, -0.06353531, -0.08694748,  0.31201735, -0.06069056,
        0.29923206,  0.2,  0.07753   ,  0.09057   ,  0.13199   ,
        0.        ,  0.        ,  0.        ])
L = 100
epsilon = 0.00005

qs = []
logps = []
accepts = []
#%%
if True:
    c11, anisotropic, c44, std, X, Y, Z, a, b, y = current_q#[  2.51369533e-01,   2.68263998e+00 ,  1.36212875e-01,   2.77775173e-01, 1.19590000e-01,   1.39530000e-01,   1.99760000e-01,   6.22417941e-03,  -4.74030692e-04,   1.27670283e-03]
    c12 = -(c44 * 2.0 / anisotropic - c11)

    dp, pv, ddpdX, ddpdY, ddpdZ, dpvdX, dpvdY, dpvdZ = polybasis.build(N, X, Y, Z)

    C = numpy.array([[c11, c12, c12, 0, 0, 0],
                     [c12, c11, c12, 0, 0, 0],
                     [c12, c12, c11, 0, 0, 0],
                     [0, 0, 0, c44, 0, 0],
                     [0, 0, 0, 0, c44, 0],
                     [0, 0, 0, 0, 0, c44]])

    C, dCda, dCdb, dCdy, K = polybasis.buildRot(C, a, b, y)

    K, M = polybasis.buildKM(C, dp, pv, density)

    eigst, evecst = scipy.linalg.eigh(K, M)
    print eigst

#%%
print UgradU(q)
#%%
dp, pv, ddpdX, ddpdY, ddpdZ, dpvdX, dpvdY, dpvdZ = polybasis.build(N, X, Y, Z)

C = numpy.array([[1.0, 1.0, 1.0, 0, 0, 0],
                 [1.0, 1.0, 1.0, 0, 0, 0],
                 [1.0, 1.0, 1.0, 0, 0, 0],
                 [0, 0, 0, 1.0, 0, 0],
                 [0, 0, 0, 0, 1.0, 0],
                 [0, 0, 0, 0, 0, 1.0]])

C, dCda, dCdb, dCdy, K = polybasis.buildRot(C, 0, 0, 0)

dCdc11 = K.dot(numpy.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).dot(K.T))

dCdc12 = K.dot(numpy.array([[0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).dot(K.T))

dCdc13 = K.dot(numpy.array([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).dot(K.T))

dCdc14 = K.dot(numpy.array([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).dot(K.T))

dCdc15 = K.dot(numpy.array([[0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).dot(K.T))

dCdc16 = K.dot(numpy.array([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).dot(K.T))

dCdc22 = K.dot(numpy.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).dot(K.T))

dCdc23 = K.dot(numpy.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).dot(K.T))

dCdc24 = K.dot(numpy.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).dot(K.T))

dCdc25 = K.dot(numpy.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).dot(K.T))

dCdc26 = K.dot(numpy.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]]).dot(K.T))

dCdc33 = K.dot(numpy.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).dot(K.T))

dCdc34 = K.dot(numpy.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).dot(K.T))

dCdc35 = K.dot(numpy.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).dot(K.T))

dCdc36 = K.dot(numpy.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]).dot(K.T))

dCdc44 = K.dot(numpy.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).dot(K.T))

dCdc45 = K.dot(numpy.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).dot(K.T))

dCdc46 = K.dot(numpy.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]]).dot(K.T))

dCdc55 = K.dot(numpy.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).dot(K.T))

dCdc56 = K.dot(numpy.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]]).dot(K.T))

dCdc66 = K.dot(numpy.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]).dot(K.T))

dKdc11, _ = polybasis.buildKM(dCdc11, dp, pv, density)
dKdc12, _ = polybasis.buildKM(dCdc12, dp, pv, density)
dKdc13, _ = polybasis.buildKM(dCdc13, dp, pv, density)
dKdc14, _ = polybasis.buildKM(dCdc14, dp, pv, density)
dKdc15, _ = polybasis.buildKM(dCdc15, dp, pv, density)
dKdc16, _ = polybasis.buildKM(dCdc16, dp, pv, density)
dKdc22, _ = polybasis.buildKM(dCdc22, dp, pv, density)
dKdc23, _ = polybasis.buildKM(dCdc23, dp, pv, density)
dKdc24, _ = polybasis.buildKM(dCdc24, dp, pv, density)
dKdc25, _ = polybasis.buildKM(dCdc25, dp, pv, density)
dKdc26, _ = polybasis.buildKM(dCdc26, dp, pv, density)
dKdc33, _ = polybasis.buildKM(dCdc33, dp, pv, density)
dKdc34, _ = polybasis.buildKM(dCdc34, dp, pv, density)
dKdc35, _ = polybasis.buildKM(dCdc35, dp, pv, density)
dKdc36, _ = polybasis.buildKM(dCdc36, dp, pv, density)
dKdc44, _ = polybasis.buildKM(dCdc44, dp, pv, density)
dKdc45, _ = polybasis.buildKM(dCdc45, dp, pv, density)
dKdc46, _ = polybasis.buildKM(dCdc46, dp, pv, density)
dKdc55, _ = polybasis.buildKM(dCdc55, dp, pv, density)
dKdc56, _ = polybasis.buildKM(dCdc56, dp, pv, density)
dKdc66, _ = polybasis.buildKM(dCdc66, dp, pv, density)

def UgradU(q):
    c11, c12, c13, c14, c15, c16, c22, c23, c24, c25, c26, c33, c34, c35, c36, c44, c45, c46, c55, c56, c66, std, X, Y, Z, a, b, y = q

    C = numpy.array([[c11, c12, c13, c14, c15, c16],
                     [c12, c22, c23, c24, c25, c26],
                     [c13, c23, c33, c34, c35, c36],
                     [c14, c24, c34, c44, c45, c46],
                     [c15, c25, c35, c45, c55, c56],
                     [c16, c26, c36, c46, c56, c66]])

    K, M = polybasis.buildKM(C, dp, pv, density)
    #print 'Assemble: ', time.time() - tmp

    #tmp = time.time()
    #tmp = time.time()
    eigst, evecst = scipy.linalg.eigh(K, M, eigvals = (6, 43))
    #print 'Eigs: ', time.time() - tmp
    #print eigst[:] - data
    #print "Eigs: ", time.time() - tmp

    eigst = eigst[:]
    evecst = evecst[:, :]

    dldc11 = numpy.array([evecst[:, i].T.dot(dKdc11.dot(evecst[:, i])) for i in range(evecst.shape[1])])
    dldc12 = numpy.array([evecst[:, i].T.dot(dKdc12.dot(evecst[:, i])) for i in range(evecst.shape[1])])
    dldc13 = numpy.array([evecst[:, i].T.dot(dKdc13.dot(evecst[:, i])) for i in range(evecst.shape[1])])
    dldc14 = numpy.array([evecst[:, i].T.dot(dKdc14.dot(evecst[:, i])) for i in range(evecst.shape[1])])
    dldc15 = numpy.array([evecst[:, i].T.dot(dKdc15.dot(evecst[:, i])) for i in range(evecst.shape[1])])
    dldc16 = numpy.array([evecst[:, i].T.dot(dKdc16.dot(evecst[:, i])) for i in range(evecst.shape[1])])
    dldc22 = numpy.array([evecst[:, i].T.dot(dKdc22.dot(evecst[:, i])) for i in range(evecst.shape[1])])
    dldc23 = numpy.array([evecst[:, i].T.dot(dKdc23.dot(evecst[:, i])) for i in range(evecst.shape[1])])
    dldc24 = numpy.array([evecst[:, i].T.dot(dKdc24.dot(evecst[:, i])) for i in range(evecst.shape[1])])
    dldc25 = numpy.array([evecst[:, i].T.dot(dKdc25.dot(evecst[:, i])) for i in range(evecst.shape[1])])
    dldc26 = numpy.array([evecst[:, i].T.dot(dKdc26.dot(evecst[:, i])) for i in range(evecst.shape[1])])
    dldc33 = numpy.array([evecst[:, i].T.dot(dKdc33.dot(evecst[:, i])) for i in range(evecst.shape[1])])
    dldc34 = numpy.array([evecst[:, i].T.dot(dKdc34.dot(evecst[:, i])) for i in range(evecst.shape[1])])
    dldc35 = numpy.array([evecst[:, i].T.dot(dKdc35.dot(evecst[:, i])) for i in range(evecst.shape[1])])
    dldc36 = numpy.array([evecst[:, i].T.dot(dKdc36.dot(evecst[:, i])) for i in range(evecst.shape[1])])
    dldc44 = numpy.array([evecst[:, i].T.dot(dKdc44.dot(evecst[:, i])) for i in range(evecst.shape[1])])
    dldc45 = numpy.array([evecst[:, i].T.dot(dKdc45.dot(evecst[:, i])) for i in range(evecst.shape[1])])
    dldc46 = numpy.array([evecst[:, i].T.dot(dKdc46.dot(evecst[:, i])) for i in range(evecst.shape[1])])
    dldc55 = numpy.array([evecst[:, i].T.dot(dKdc55.dot(evecst[:, i])) for i in range(evecst.shape[1])])
    dldc56 = numpy.array([evecst[:, i].T.dot(dKdc56.dot(evecst[:, i])) for i in range(evecst.shape[1])])
    dldc66 = numpy.array([evecst[:, i].T.dot(dKdc66.dot(evecst[:, i])) for i in range(evecst.shape[1])])


    dlpdl = (data - eigst) / std ** 2
    dlpdstd = sum((-std ** 2 + (eigst - data) **2) / std ** 3)

    dlpdl = numpy.array(dlpdl)

    dlpdc11 = dlpdl.dot(dldc11)
    dlpdc12 = dlpdl.dot(dldc12)
    dlpdc13 = dlpdl.dot(dldc13)
    dlpdc14 = dlpdl.dot(dldc14)
    dlpdc15 = dlpdl.dot(dldc15)
    dlpdc16 = dlpdl.dot(dldc16)
    dlpdc22 = dlpdl.dot(dldc22)
    dlpdc23 = dlpdl.dot(dldc23)
    dlpdc24 = dlpdl.dot(dldc24)
    dlpdc25 = dlpdl.dot(dldc25)
    dlpdc26 = dlpdl.dot(dldc26)
    dlpdc33 = dlpdl.dot(dldc33)
    dlpdc34 = dlpdl.dot(dldc34)
    dlpdc35 = dlpdl.dot(dldc35)
    dlpdc36 = dlpdl.dot(dldc36)
    dlpdc44 = dlpdl.dot(dldc44)
    dlpdc45 = dlpdl.dot(dldc45)
    dlpdc46 = dlpdl.dot(dldc46)
    dlpdc55 = dlpdl.dot(dldc55)
    dlpdc56 = dlpdl.dot(dldc56)
    dlpdc66 = dlpdl.dot(dldc66)

    logp = sum(0.5 * (-((eigst - data) **2 / std**2) + numpy.log(1.0 / (2 * numpy.pi)) - 2 * numpy.log(std)))

    #dlpdc12tf = dlpdc12 * 2.0 * c44 / (anisotropic**2)c11, c12, c13, c22, c23, c33, c44, c55, c66
    return -logp, -numpy.array([dlpdc11,
    dlpdc12,
    dlpdc13,
    dlpdc14,
    dlpdc15,
    dlpdc16,
    dlpdc22,
    dlpdc23,
    dlpdc24,
    dlpdc25,
    dlpdc26,
    dlpdc33,
    dlpdc34,
    dlpdc35,
    dlpdc36,
    dlpdc44,
    dlpdc45,
    dlpdc46,
    dlpdc55,
    dlpdc56,
    dlpdc66, dlpdstd, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

while True:
    q = current_q
    p = numpy.random.randn(len(q)) # independent standard normal variates

    current_p = p
    # Make a half step for momentum at the beginning
    U, gradU = UgradU(q)
    p = p - epsilon * gradU / 2

    p[-7:] = 0

    # Alternate full steps for position and momentum
    for i in range(L):
        # Make a full step for the position
        q = q + epsilon * p

        # Make a full step for the momentum, except at end of trajectory
        if i != L - 1:
            U, gradU = UgradU(q)
            p = p - epsilon * gradU

            p[-7:] = 0

        #if numpy.isnan(U):
        #    1/0
        #print "New q, H: ", q, U + sum(p ** 2) / 2, U, sum(p ** 2) / 2
        #print gradU
        #print '---------'

    U, gradU = UgradU(q)
    # Make a half step for momentum at the end.
    p = p - epsilon * gradU / 2

    p[-7:] = 0
    # Negate momentum at end of trajectory to make the proposal symmetric
    p = -p
    # Evaluate potential and kinetic energies at start and end of trajectory
    UC, gradUC = UgradU(current_q)
    current_U = UC
    current_K = sum(current_p ** 2) / 2
    proposed_U = U
    proposed_K = sum(p ** 2) / 2

    # Accept or reject the state at end of trajectory, returning either
    # the position at the end of the trajectory or the initial position
    dQ = current_U - proposed_U + current_K - proposed_K

    qs.append(q)
    logps.append(UC)

    if numpy.random.rand() < min(1.0, numpy.exp(dQ)):
        current_q = q # accept

        accepts.append(len(qs) - 1)

        print "Accepted ({0} accepts so far): {1}".format(len(accepts), current_q)
    else:
        print "Rejected: ", current_q

    print "Energy change ({0} samples, {1} accepts): ".format(len(qs), len(accepts)), min(1.0, numpy.exp(dQ)), dQ, current_U, proposed_U, current_K, proposed_K
    print "Epsilon: ", epsilon
#%%
c11, c12, c13, c14, c15, c16, c22, c23, c24, c25, c26, c33, c34, c35, c36, c44, c45, c46, c55, c56, c66, stds, Xs, Ys, Zs, as_, bs_, cs_ = [numpy.array(a) for a in zip(*[qs[i] for i in accepts])]#
import matplotlib.pyplot as plt

for name, data in zip(['c11', 'c12', 'c13', 'c14', 'c15', 'c16', 'c22', 'c23', 'c24', 'c25', 'c26', 'c33', 'c34', 'c35', 'c36', 'c44', 'c45', 'c46', 'c55', 'c56', 'c66'],
                      [c11, c12, c13, c14, c15, c16, c22, c23, c24, c25, c26, c33, c34, c35, c36, c44, c45, c46, c55, c56, c66]):
    plt.plot(data)
    plt.title('{0} u = {1:.3e}, std = {2:.3e}'.format(name, numpy.mean(data), numpy.std(data)))
    plt.show()

#%%
plt.plot(Xs)
plt.title('Xs')
plt.show()
plt.plot(Ys)
plt.title('Ys')
plt.show()
plt.plot(Zs)
plt.title('Zs')
plt.show()
#%%

import seaborn
import pandas
import matplotlib.pyplot as plt

df = pandas.DataFrame({'c11' : c11s[-650:], 'c12' : c12s[-650:], 'c44' : c44s[-650:], 'y' : ys[-650:]})

seaborn.pairplot(df)
plt.gcf().set_size_inches((12, 8))
plt.show()

import scipy.stats

g = seaborn.PairGrid(df)
g.map_diag(plt.hist)
g.map_offdiag(seaborn.kdeplot, n_levels = 6);
plt.gcf().set_size_inches((12, 8))
plt.show()
#%%
for name, d in [('c11', c11s), ('c12', c12s), ('c44', c44s), ('y', ys)]:
    seaborn.distplot(d[-650:], kde = False, fit = scipy.stats.norm)
    plt.title("Dist. {0} w/ mean {1:0.4f} and std. {2:0.4f}".format(name, numpy.mean(d[-650:]), numpy.std(d[-650:])))
    plt.gcf().set_size_inches((5, 4))
    plt.show()

#%%
UgradU(qs[accepts[-1]])