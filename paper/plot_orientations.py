#%%

import os
import pickle

os.chdir('/home/bbales2/modal')

with open('paper/cmsx4/hmc_30_noprior.pkl') as f:
    hmc = pickle.load(f)

from rotations import inv_rotations
from rotations import symmetry
from rotations import quaternion
#%%

import numpy

v = quaternion.Quaternion([0, 0, 0, 1])

q = quaternion.Quaternion([numpy.cos(numpy.pi / 4), 0.0, 0.0, numpy.sin(numpy.pi / 4)])

print q * v * q.conjugate()

#%%
samples = dict(zip(*hmc.format_samples()))

#%%
sampled = []
for i in range(1, 1000):
    q1 = [samples['w_0'][-i], samples['x_0'][-i], -samples['z_0'][-i], samples['y_0'][-i]]

    sampled.append(symmetry.Symmetry.Cubic.fzQuat(quaternion.Quaternion(numpy.array(q1) / numpy.linalg.norm(q1))))
#%%
symmetry.Symmetry.Cubic.fzQuat(quaternion.Quaternion(numpy.array(q1) / numpy.linalg.norm(q1)))

#%%

cs = []

for q in sampled:
    cs.append(q * v * q.conjugate())

_, xs, ys, zs = zip(*cs)

print "Should be zero: ", max(_)

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
plt.plot([x], [y], 'ro')
#plt.xlim(-0.1, 0.1)
#plt.ylim(-0.1, 0.1)
plt.axes().set_aspect('equal', 'datalim')