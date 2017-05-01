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

c11 = 2.5
a = 2.8
c44 = 1.33
c12 = -(c44 * 2.0 / a - c11)
#sample mass

#Sample density
density = 4401.695921

def func(w, x, y, z):
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
    eigs, evecs = scipy.linalg.eigh(K, M, eigvals = (6, 6 + 30 - 1))

    return numpy.sqrt(eigs * 1e11) / (numpy.pi * 2000)

print "Computed: ", func(-0.414888,-0.419461,0.575763,-0.566055)
print "Measured: ", func(0.69774412,  0.69817909,  0.12513392,  0.10020276)
