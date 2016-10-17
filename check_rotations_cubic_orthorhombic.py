#%%
import os
import numpy

os.chdir('/home/pollock/modal')

import symmetry
import quaternion

import itertools
# get the equivalent passive rotation in the fundamental zone (active post multiplies symmetry operator)

q1 = quaternion.Quaternion([0.98849515554696088, -0.0026722254086214156, 0.015326741589590076, 0.15045024979640861])
q2 = quaternion.Quaternion([0.98849515554696088, -0.0026722254086214156, -0.015326741589590076, 0.15045024979640861])

# Get angle between two quaternions
deltaQ = q1 * q2.conjugate()

# Find all possible representations of this rotation in mix of Cubic + Orthorhombic symmetries
equivMiso = [(q * j.conjugate()).supplement() for q, j in itertools.product([i * deltaQ for i in symmetry.Symmetry.Cubic.quOperators()], symmetry.Symmetry.Orthorhombic.quOperators())] # compute all equivilent misorientation representations

# Pick the smallest angle one
d = max(max(equivMiso), max([q.conjugate() for q in equivMiso])) # select quaternion of smallest rotation with axis in most positive quadrant from equivilent rotations w/ switching symmetry

print "Quaternion rotation between q1 & q2", d
print "Difference in angle: {0} degrees".format(2 * numpy.arccos(d.wxyz[0]) * 180 / numpy.pi)

cubicSym = symmetry.Symmetry.Cubic.quOperators()
orthoSym = symmetry.Symmetry.Orthorhombic.quOperators()
miso = deltaQ
for i in range(len(cubicSym)):
    qa = cubicSym[i] * q1
    for j in range(len(orthoSym)):
        qas = qa * orthoSym[j]
        for k in range(len(cubicSym)):
            qasb = qas * cubicSym[k]
            miso = max([miso, qasb, qasb.conjugate()])
print(miso)
print "Difference in angle: {0} degrees".format(2 * numpy.arccos(miso.wxyz[0]) * 180 / numpy.pi)