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

density = 8700.0e-3#4401.695921e-3

X = 0.011959e1#0.007753e1
Y = 0.013953e1#0.009057e1
Z = 0.019976e1#0.013199e1
#[ 1.61054549  0.9710114   0.44089017  0.46434645]
c11 = 2.5e-1
anisotropic = 2.68
c44 = 1.31e-1
c12 = -(c44 * 2.0 / anisotropic - c11)

a = 0.0
b = 0.0
y = 0.0

freqs = numpy.array([71.25925,
75.75875,
86.478,
89.947375,
111.150125,
112.164125,
120.172125,
127.810375,
128.6755,
130.739875,
141.70025,
144.50375,
149.40075,
154.35075,
156.782125,
157.554625,
161.0875,
165.10325,
169.7615,
173.44925,
174.11675,
174.90625,
181.11975,
182.4585,
183.98625,
192.68125,
193.43575,
198.793625,
201.901625,
205.01475])

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

eigs, evecs = scipy.linalg.eigh(K, M, eigvals = (6, 35))

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

current_q = numpy.array([2.5e-1,  2.68,  1.25e-1,  0.28, 0.011959e1, 0.013953e1, 0.019976e1, 0.0, 0.0, 0.0])
L = 100
epsilon = 0.0001

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
def UgradU(q):
    c11, anisotropic, c44, std, X, Y, Z, a, b, y = q
    c12 = -(c44 * 2.0 / anisotropic - c11)
    #print q#X, Y, Z

    dp, pv, ddpdX, ddpdY, ddpdZ, dpvdX, dpvdY, dpvdZ = polybasis.build(N, X, Y, Z)

    C = numpy.array([[c11, c12, c12, 0, 0, 0],
                     [c12, c11, c12, 0, 0, 0],
                     [c12, c12, c11, 0, 0, 0],
                     [0, 0, 0, c44, 0, 0],
                     [0, 0, 0, 0, c44, 0],
                     [0, 0, 0, 0, 0, c44]])

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
    #print 'Assemble: ', time.time() - tmp

    #tmp = time.time()
    #tmp = time.time()
    eigst, evecst = scipy.linalg.eigh(K, M, eigvals = (6, 35))
    #print 'Eigs: ', time.time() - tmp
    #print eigst[6:]
    #print "Eigs: ", time.time() - tmp

    eigst = eigst[:]
    evecst = evecst[:, :]

    dlda = numpy.array([evecst[:, i].T.dot(dKda.dot(evecst[:, i])) for i in range(evecst.shape[1])])
    dldb = numpy.array([evecst[:, i].T.dot(dKdb.dot(evecst[:, i])) for i in range(evecst.shape[1])])
    dldy = numpy.array([evecst[:, i].T.dot(dKdy.dot(evecst[:, i])) for i in range(evecst.shape[1])])
    dldX = numpy.array([evecst[:, i].T.dot((dKdX - eigst[i] * dMdX).dot(evecst[:, i])) for i in range(evecst.shape[1])])
    dldY = numpy.array([evecst[:, i].T.dot((dKdY - eigst[i] * dMdY).dot(evecst[:, i])) for i in range(evecst.shape[1])])
    dldZ = numpy.array([evecst[:, i].T.dot((dKdZ - eigst[i] * dMdZ).dot(evecst[:, i])) for i in range(evecst.shape[1])])
    dldc11 = numpy.array([evecst[:, i].T.dot(dKdc11.dot(evecst[:, i])) for i in range(evecst.shape[1])])
    dldc12 = numpy.array([evecst[:, i].T.dot(dKdc12.dot(evecst[:, i])) for i in range(evecst.shape[1])])
    dldc44 = numpy.array([evecst[:, i].T.dot(dKdc44.dot(evecst[:, i])) for i in range(evecst.shape[1])])

    dlpdl = (data - eigst) / std ** 2
    dlpdstd = sum((-std ** 2 + (eigst - data) **2) / std ** 3)

    dlpdl = numpy.array(dlpdl)

    dlpda = dlpdl.dot(dlda)
    dlpdb = dlpdl.dot(dldb)
    dlpdy = dlpdl.dot(dldy)
    dlpdX = dlpdl.dot(dldX)
    dlpdY = dlpdl.dot(dldY)
    dlpdZ = dlpdl.dot(dldZ)
    dlpdc11 = dlpdl.dot(dldc11)
    dlpdc12 = dlpdl.dot(dldc12)
    dlpdc44 = dlpdl.dot(dldc44)

    logp = sum(0.5 * (-((eigst - data) **2 / std**2) + numpy.log(1.0 / (2 * numpy.pi)) - 2 * numpy.log(std)))

    dlpdc12tf = dlpdc12 * 2.0 * c44 / (anisotropic**2)
    return -logp, -numpy.array([dlpdc11 + dlpdc12, dlpdc12tf, dlpdc44 + dlpdc12 * -2 / anisotropic, dlpdstd, dlpdX, dlpdY, dlpdZ, dlpda, dlpdb, dlpdy])

while True:
    q = current_q
    p = numpy.random.randn(len(q)) # independent standard normal variates

    current_p = p
    # Make a half step for momentum at the beginning
    U, gradU = UgradU(q)
    p = p - epsilon * gradU / 2

    p[-6:-3] = 0

    # Alternate full steps for position and momentum
    for i in range(L):
        # Make a full step for the position
        q = q + epsilon * p

        # Make a full step for the momentum, except at end of trajectory
        if i != L - 1:
            U, gradU = UgradU(q)
            p = p - epsilon * gradU

            p[-6:-3] = 0

        #if numpy.isnan(U):
        #    1/0
        #print "New q, H: ", q, U + sum(p ** 2) / 2, U, sum(p ** 2) / 2

    U, gradU = UgradU(q)
    # Make a half step for momentum at the end.
    p = p - epsilon * gradU / 2

    p[-6:-3] = 0
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
c11s, anisotropics, c44s, stds, Xs, Ys, Zs, as_, bs_, ys_  = [numpy.array(a)[-2000:] for a in zip(*[qs[i] for i in accepts])]#
import matplotlib.pyplot as plt

for name, data in zip(['c11', 'anisotropics', 'c44', 'stds', 'Xs', 'Ys', 'Zs', 'eu[0]', 'eu[1]', 'eu[2]'],
                      [c11s, anisotropics, c44s, stds, Xs, Ys, Zs, as_, bs_, ys_]):
    plt.plot(data)
    plt.title('{0} u = {1:.3e}, std = {2:.3e}'.format(name, numpy.mean(data), numpy.std(data)))
    plt.show()

plt.plot(as_, ys_)
plt.ylabel('eu[2]s')
plt.xlabel('eu[0]s')
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
import math

def eu2ax(eu):
	t = math.tan(eu[1] / 2.0)
	sigma = (eu[0] + eu[2]) / 2.0
	tau = math.sqrt(t * t + math.sin(sigma) * math.sin(sigma))
	if abs(tau) <= 2.0 * epsilon:
		return [0.0, 0.0, 1.0, 0.0] # handle 0 rotation
	delta = (eu[0] - eu[2]) / 2.0
	alpha = math.pi if abs(sigma - math.pi / 2.0) <= epsilon else 2.0 * math.atan(tau / math.cos(sigma))
	n = [-1.0 / math.copysign(tau, alpha)] * 3
	n[0] *= t * math.cos(delta)
	n[1] *= t * math.sin(delta)
	n[2] *= math.sin(sigma)

	# normalize
	mag = math.sqrt(math.fsum([x * x for x in n]))
	n = [x / mag for x in n]

	# handle ambiguous case (rotation angle of pi)
	alpha = abs(alpha)
	if alpha + epsilon >= math.pi:
		return orientAxis(n) + [math.pi]
	return n + [alpha]

def eu2om(eu):
	s = [math.sin(x) for x in eu]
	c = [math.cos(x) for x in eu]
	s = [0.0 if abs(x) <= epsilon else x for x in s]
	c = [0.0 if abs(x) <= epsilon else x for x in c]

	om = [[0.0] * 3 for i in range(3)]
	om[0][0] =  c[0] * c[2] - s[0] * c[1] * s[2]
	om[0][1] =  s[0] * c[2] + c[0] * c[1] * s[2]
	om[0][2] =  s[1] * s[2]
	om[1][0] = -c[0] * s[2] - s[0] * c[1] * c[2]
	om[1][1] = -s[0] * s[2] + c[0] * c[1] * c[2]
	om[1][2] =  s[1] * c[2]
	om[2][0] =  s[0] * s[1]
	om[2][1] = -c[0] * s[1]
	om[2][2] =  c[1]
	return om

def getRotations(a, b, y):

    s = numpy.sin((a, b, y))
    c = numpy.cos((a, b, y))

    ds0da = numpy.cos(a)
    ds1db = numpy.cos(b)
    ds2dy = numpy.cos(y)

    dc0da = -numpy.sin(a)
    dc1db = -numpy.sin(b)
    dc2dy = -numpy.sin(y)

    Q = numpy.zeros((3, 3))

    Q = numpy.array([[c[0] * c[2] - s[0] * c[1] * s[2], s[0] * c[2] + c[0] * c[1] * s[2], s[1] * s[2]],
                     [-c[0] * s[2] - s[0] * c[1] * c[2], -s[0] * s[2] + c[0] * c[1] * c[2], s[1] * c[2]],
                     [s[0] * s[1], -c[0] * s[1], c[1]]])

    return Q

Qs = []
for a, b, y in zip(as_, bs_, ys_):
    Qs.append(numpy.linalg.solve(eu2om((a, b, y)), [1.0, 0.0, 0.0]))

Qs = numpy.array(Qs)
#%%
Qs = Qs.reshape((len(Qs), -1))
#%%
labels = []
for i in range(Qs.shape[1]):
    plt.plot(Qs[:, i])
    labels.append('vector {0}'.format(['x', 'y', 'z'][i]))
plt.legend(labels)
plt.show()
#%%
UgradU([1.6, 1.0, 0.45, 0.25])