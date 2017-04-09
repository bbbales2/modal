#%%

import os
import pickle

os.chdir('/home/bbales2/modal')

with open('paper/cmsx4/hmc_30_noprior.pkl') as f:
    hmc = pickle.load(f)

from rotations import inv_rotations
from rotations import symmetry
from rotations import quaternion

cubicSym = symmetry.Symmetry.Cubic.quOperators()
orthoSym = symmetry.Symmetry.Orthorhombic.quOperators()

def adj(q):
    if q.wxyz[0] < 0.0:
        q.wxyz[0] *= -1
        q.wxyz[1] *= -1
        q.wxyz[2] *= -1
        q.wxyz[3] *= -1

    return q

def miso((q1, q2)):
    q1 = quaternion.Quaternion(numpy.array(q1) / numpy.linalg.norm(q1))
    q2 = quaternion.Quaternion(numpy.array(q2) / numpy.linalg.norm(q2))
    misot = 180.0
    misoa = None

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
qr = inv_rotations.eu2qu(numpy.array([180.9, 89.3, 338.6]) * numpy.pi / 180.0)
#%%

import numpy

v = quaternion.Quaternion([0, 0, 0, 1])

q = quaternion.Quaternion([numpy.cos(numpy.pi / 4), 0.0, 0.0, numpy.sin(numpy.pi / 4)])

print q * v * q.conjugate()

#%%
samples = dict(zip(*hmc.format_samples()))

#%%
angles = []
sampled = []
qs = []
for i in range(1, 1000):
    q1 = [samples['w_0'][-i], samples['x_0'][-i], samples['y_0'][-i], samples['z_0'][-i]]
    qs.append(q1)

    quaternion.Quaternion(numpy.array(q1) / numpy.linalg.norm(q1))

    a, b = miso((q1, qr))

    angles.append(a)
    sampled.append(b)
#%%
import sklearn.mixture

gmm = sklearn.mixture.GMM(2, tol = 1e-7, min_covar = 1e-10)
gmm.fit(qs)

print gmm.means_

miso(gmm.means_)
#%%
symmetry.Symmetry.Cubic.fzQuat(quaternion.Quaternion(numpy.array(q1) / numpy.linalg.norm(q1)))

#%%

cs = []

v = quaternion.Quaternion([0, 0, 0, 1])

for q in sampled:
    cs.append(q * v * q.conjugate())

_, xs, ys, zs = zip(*cs)

print "Should be zero: ", max(_)
#%%
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(xs, ys, zs)
ax.set_zlim3d(-1, 1)                    # viewrange for z-axis should be [-4,4]
ax.set_ylim3d(-1, 1)                    # viewrange for y-axis should be [-2,2]
ax.set_xlim3d(-1, 1)

plt.show()
#%%

plt.hist(zs)
#%%
#q3 = inv_rotations.eu2qu(numpy.array([180.9, 89.3, 338.6]) * numpy.pi / 180.0)
q3 = [0.9832, -0.00159, 0.1822, -0.00858]
q3 = quaternion.Quaternion(numpy.array(q3) / numpy.linalg.norm(q3))
q3 = symmetry.Symmetry.Cubic.fzQuat(q3)
_, x, y, z = q3 * v * q3.conjugate()
#%%

plt.plot(xs, ys, '*')
plt.axes().set_aspect('equal', 'datalim')