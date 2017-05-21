#%%
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
from rotations.quaternion import Quaternion as Q
from rotations import symmetry

# basis polynomials are x^n * y^m * z^l where n + m + l <= N
N = 10

## Dimensions for TF-2
X = 0.012
Y = 0.020
Z = 0.014

c11 = 2.51
a = 2.83
c44 = 1.33
c12 = -(c44 * 2.0 / a - c11)
#sample mass

#Sample density
density = 8700.0#4401.695921

def func(X, Y, Z, w, x, y, z):
    q = numpy.array([w, x, y, z])
    q /= numpy.linalg.norm(q)
    w, x, y, z = q.flatten()

    C = numpy.array([[c11, c12, c12, 0, 0, 0],
                     [c12, c11, c12, 0, 0, 0],
                     [c12, c12, c11, 0, 0, 0],
                     [0, 0, 0, c44, 0, 0],
                     [0, 0, 0, 0, c44, 0],
                     [0, 0, 0, 0, 0, c44]])

    dp, pv, ddpdX, ddpdY, ddpdZ, dpvdX, dpvdY, dpvdZ = polybasisqu.build(N, X, Y, Z)

    C, _, _, _, _, _ = polybasisqu.buildRot(C, w, x, y, z)

    K, M = polybasisqu.buildKM(C, dp, pv, density)
    eigs, evecs = scipy.linalg.eigh(K, M, eigvals = (6, 6 + 50 - 1))

    return numpy.sqrt(eigs * 1e11) / (numpy.pi * 2000)

print "Computed: ", func(X, Y, Z, 0.703295, -0.111453, -0.101898, -0.69467299999999998)
print "Measured: ", func(X, Y, Z, 0.70376400000000006, -0.69423599999999996, -0.10509, 0.10820899999999999)
#%%
#71.25925, 75.75875, 86.478, 89.947375, 111.150125, 112.164125, 120.172125, 127.810375, 128.6755, 130.739875, 141.70025, 144.50375,
func(X, Y, Z, 0.69774412, 0.69817909, 0.12513392, 0.10020276)
#%%
q = Q([0.69774412,  0.69817909,  0.12513392,  0.10020276])

print q.conjugate() * Q([0, 1, 0, 0]) * q
print q.conjugate() * Q([0, 0, 1, 0]) * q

om = numpy.array(inv_rotations.qu2om(q.wxyz)).reshape(3, 3, order = 'C')#[-0.414888,-0.419461,0.575763,-0.566055]

print om.T.dot(numpy.array([1, 0, 0]))
print om.T.dot(numpy.array([0, 1, 0]))

#%%
cubicSym = symmetry.Symmetry.Cubic.quOperators()
orthoSym = symmetry.Symmetry.Orthorhombic.quOperators()

q1 = Q([0.703295, -0.111453, -0.101898, -0.69467299999999998])
q2 = Q([0.70376400000000006, -0.69423599999999996, -0.10509, 0.10820899999999999])

def ang(a1, a2):
    ax1 = numpy.array(a1.wxyz)[1:]

    angt = 360.0
    for cubic in cubicSym:
        ax2 = cubic.conjugate() * a2 * cubic
        ax2 = numpy.array(ax2.wxyz)[1:]
        angtn = 180 * numpy.arccos(ax1.dot(ax2) / (numpy.linalg.norm(ax1) * numpy.linalg.norm(ax2))) / numpy.pi
        if angtn < angt:
            angt = angtn
            print angt, ax2, cubic, ax1

    return angt

ax1 = (q1.conjugate() * Q([0, 1, 0, 0]) * q1)
ax2 = (q2.conjugate() * Q([0, 1, 0, 0]) * q2)
print ang(ax1, ax2)
#%%
ax1 = numpy.array((q1.conjugate() * Q([0, 0, 1, 0]) * q1).wxyz)[1:]
ax2 = numpy.array((q2.conjugate() * Q([0, 0, 1, 0]) * q2).wxyz)[1:]
print 180 * numpy.arccos(ax1.dot(ax2) / (numpy.linalg.norm(ax1) * numpy.linalg.norm(ax2))) / numpy.pi

ax1 = numpy.array((q1.conjugate() * Q([0, 0, 0, 1]) * q1).wxyz)[1:]
ax2 = numpy.array((q2.conjugate() * Q([0, 0, 0, 1]) * q2).wxyz)[1:]
print 180 * numpy.arccos(ax1.dot(ax2) / (numpy.linalg.norm(ax1) * numpy.linalg.norm(ax2))) / numpy.pi
