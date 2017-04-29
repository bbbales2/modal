#%%

import os
import numpy

os.chdir('/home/bbales2/modal')

from rotations import symmetry
from rotations import quaternion

from rotations import inv_rotations

import itertools

def miso(q1, q2):
    cubicSym = symmetry.Symmetry.Cubic.quOperators()
    orthoSym = symmetry.Symmetry.Orthorhombic.quOperators()

    misot = 180.0
    misoa = None


    def adj(q):
        if q.wxyz[0] < 0.0:
            q.wxyz[0] *= -1
            q.wxyz[1] *= -1
            q.wxyz[2] *= -1
            q.wxyz[3] *= -1

        return q

    for i in range(len(cubicSym)):
        for ii in range(len(orthoSym)):
            qa = orthoSym[ii] * q1 * cubicSym[i]

            for j in range(len(cubicSym)):
                #for jj in range(len(orthoSym)):
                    qb = q2 * cubicSym[j]

                    qasb1 = qa.conjugate() * qb

                    t1 = qasb1.wxyz / numpy.linalg.norm(qasb1.wxyz)

                    a1 = 2 * numpy.arccos(t1[0]) * 180 / numpy.pi

                    if a1 < misot:
                        misot = a1
                        misoa = qasb1

    return misot, misoa
#%%

from numpy import pi, cos, sin, array, concatenate
from rotations.quaternion import Quaternion

def quat(q):
    return Quaternion(array(q) / numpy.linalg.norm(q))

def quat2(ang, axis):
    tmp = sin(ang) * array(axis)
    return quat([cos(ang), tmp[0], tmp[1], tmp[2]])

inv_rotations.convention = inv_rotations.Convention.passive
qebsd = quat(inv_rotations.eu2qu(array([180.9, 89.3, 338.6]) * numpy.pi / 180.0))

phi = pi * -95.64 / 360.0
chi = pi * 18.266 / 360.0
q1 = quat2(chi, [0.0, 1.0, 0.0])
v1 = quat([0, 0, 0, 1])
v2 = q1 * v1 * q1.conjugate()

q2 = quat2(phi, v2.wxyz[1:])

q100 = q2 * q1

phi = -90.688
chi = 63.266
q1 = quat2(chi, [0.0, 1.0, 0.0])
v1 = quat([0, 0, 0, 1])
v2 = q1 * v1 * q1.conjugate()
q2 = quat2(phi, v2.wxyz[1:])

q110 = q2 * q1

phi = -90.688
chi = 63.266
q1 = quat2(chi, [0.0, 1.0, 0.0])
v1 = quat([0, 0, 0, 1])
v2 = q1 * v1 * q1.conjugate()
q2 = quat2(phi, v2.wxyz[1:])

q110 = q2 * q1
#%%
qcomp = quat([0.988, 0.0, 0.001, -0.151])
qcomp = quat([0.99, 0.0, -0.13, 0.056])
qcomp = quat([-0.41146,-0.425169,0.573716,-0.566376])
qwill = quat([0.70181185, 0.69437080, 0.12351422, 0.10026744])#[0.69774412,  0.69817909,  0.12513392,  0.10020276])
unit = quat([1.0,  0.0,  0.0,  0.0])

miso(qwill, qcomp)
#%%
import pickle

with open('paper/cmsx4/hmc_30_noprior.pkl') as f:
    hmc = pickle.load(f)

#%%
samples = dict(zip(*hmc.format_samples()))

misos = []
for i in range(1, 501):#len(samples['c11'])):
    qcomp = quat([samples['w_0'][-i], samples['x_0'][-i], samples['y_0'][-i], samples['z_0'][-i]])

    misos.append(miso(qwill, qcomp)[0])

    print "{0}/{1}".format(i, 500)

#%%
import matplotlib.pyplot as plt
import seaborn
plt.hist(misos)
import scipy.stats
#%%
seaborn.distplot(misos, fit = scipy.stats.chi2)
#%%
chi2 = scipy.stats.chi2(misos)

chi2.mean()
#%%
numpy.sqrt(numpy.mean(misos) / 2)