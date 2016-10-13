#%%
import os
import numpy

os.chdir('/home/bbales2/modal')

from rotations import symmetry
from rotations import quaternion
import itertools
# get the equivalent passive rotation in the fundamental zone (active post multiplies symmetry operator)

def fzQuat(qu):
    output = []
    for sym1 in symmetry.Symmetry.Cubic.quOperators():
        for sym2 in symmetry.Symmetry.Orthorhombic.quOperators():
            output.append((sym1 * qu * sym2.conjugate()).supplement().wxyz)

    return numpy.array(output)

    #return [(q * j.conjugate()).supplement() for q, j in itertools.product([i * qu for i in symmetry.Symmetry.Orthorhombic.quOperators()], symmetry.Symmetry.Orthorhombic.quOperators())]

w, x, y, z = (0.98849515554696088, -0.0026722254086214156, 0.015326741589590076, 0.15045024979640861)

o1 = fzQuat(quaternion.Quaternion([w, x, y, z]))
o2 = fzQuat(quaternion.Quaternion([w, x, -y, z]))

distances = []
idxs = []
for i in range(len(o1)):
    for j in range(len(o2)):
        distances.append(numpy.linalg.norm(o1[i] - o2[j]))
        idxs.append((i, j))

i, j = idxs[numpy.argsort(distances)[0]]

print o1[i], o2[j]
