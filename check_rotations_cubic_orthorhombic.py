#%%
import os
import numpy

os.chdir('/home/bbales2/modal')

from rotations import symmetry
from rotations import quaternion

import itertools
# get the equivalent passive rotation in the fundamental zone (active post multiplies symmetry operator)

q = [0.594756, -0.202874, 0.640152, 0.441943]
q = q / numpy.linalg.norm(q)

q1 = quaternion.Quaternion(q)#0.98849515554696088, -0.0026722254086214156, 0.015326741589590076, 0.15045024979640861])

q = [0.928, -0.129, -0.315, -0.141]
q = q / numpy.linalg.norm(q)

q2 = quaternion.Quaternion(q)#0.98849515554696088, -0.0026722254086214156, -0.015326741589590076, 0.15045024979640861])

# Get angle between two quaternions
deltaQ = q1 * q2.conjugate()

# Find all possible representations of this rotation in mix of Cubic + Orthorhombic symmetries
#equivMiso = [(q * j.conjugate()).supplement() for q, j in itertools.product([i * deltaQ for i in symmetry.Symmetry.Cubic.quOperators()], symmetry.Symmetry.Orthorhombic.quOperators())] # compute all equivilent misorientation representations

# Pick the smallest angle one
#d = max(max(equivMiso), max([q.conjugate() for q in equivMiso])) # select quaternion of smallest rotation with axis in most positive quadrant from equivilent rotations w/ switching symmetry

#print "Quaternion rotation between q1 & q2", d
#print "Difference in angle: {0} degrees".format(2 * numpy.arccos(d.wxyz[0]) * 180 / numpy.pi)

cubicSym = symmetry.Symmetry.Cubic.quOperators()
orthoSym = symmetry.Symmetry.Orthorhombic.quOperators()
miso = deltaQ
a = []
for i in range(len(cubicSym)):
    qt1 = cubicSym[i] * q1

    for j in range(len(orthoSym)):
        qt2 = qt1 * orthoSym[j]#cubicSym[j] * q2

        for i in range(len(cubicSym)):
            qz1 = cubicSym[i] * q2

            for j in range(len(orthoSym)):
                qz2 = qz1 * orthoSym[j]#cubicSym[j] * q2

                qasb = qt2 * qz2.conjugate()

                print max([miso, qasb, qasb.conjugate()])

                miso = max([miso, qasb, qasb.conjugate()])

                #if miso.wxyz[0] > 0.99:
                #    1/0

#for i in range(len(cubicSym)):
#    qa = cubicSym[i] * deltaQ
#    for j in range(len(orthoSym)):
#        qas = qa * orthoSym[j]
#        for k in range(len(cubicSym)):
#            qasb = qas * cubicSym[k]
#            #if numpy.abs(qasb.wxyz[0] - 0.59) < 0.05:
#            print qasb
#
#            miso = max([miso, qasb, qasb.conjugate()])
# to move q2 back to q1, use all the i, j, k rotations.... Not just miso
print(miso)
print "Difference in angle: {0} degrees".format(2 * numpy.arccos(miso.wxyz[0]) * 180 / numpy.pi)

#%%