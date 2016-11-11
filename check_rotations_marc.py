#%%
import os
import numpy

os.chdir('/home/bbales2/modal')

from rotations import symmetry
from rotations import quaternion

import itertools
# get the equivalent passive rotation in the fundamental zone (active post multiplies symmetry operator)

#q = [0.98849515554696088, -0.0026722254086214156, 0.015326741589590076, 0.15045024979640861]
q = (0.981, 0.0, 0.0, -0.1908)
#q = [0.594756, -0.202874, 0.640152, 0.441943]
q = q / numpy.linalg.norm(q)

q1 = quaternion.Quaternion(q)

#q = [0.98849515554696088, -0.0026722254086214156, -0.015326741589590076, 0.15045024979640861]
q = (0.988, 0.0, -0.01, 0.152)
#q = [0.928, -0.129, -0.315, -0.141]
q = q / numpy.linalg.norm(q)

q2 = quaternion.Quaternion(q)

cubicSym = symmetry.Symmetry.Cubic.quOperators()
orthoSym = symmetry.Symmetry.Orthorhombic.quOperators()

miso = q1 * q2.conjugate()
diso = q1 * q2.conjugate()

for i in range(len(cubicSym)):
    qa = cubicSym[i] * q1
    for j in range(len(orthoSym)):
        qas = qa * orthoSym[j]
        for k in range(len(cubicSym)):
            qasb = qas * q2 * cubicSym[k]

            miso = max([miso, qasb, qasb.conjugate()])
# to move q2 back to q1, use all the i, j, k rotations.... Not just miso
print(miso)
print "Difference in angle: {0} degrees".format(2 * numpy.arccos(miso.wxyz[0]) * 180 / numpy.pi)

#%%