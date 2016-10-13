#%%
import os
import numpy

os.chdir('/home/bbales2/modal')

from rotations import symmetry
from rotations import quaternion

import itertools
# get the equivalent passive rotation in the fundamental zone (active post multiplies symmetry operator)

q1 = quaternion.Quaternion([0.98849515554696088, -0.0026722254086214156, 0.015326741589590076, 0.15045024979640861])
q2 = quaternion.Quaternion([0.98849515554696088, -0.0026722254086214156, -0.015326741589590076, 0.15045024979640861])

deltaQ = q1 * q2.conjugate()
equivMiso = [(q * j.conjugate()).supplement() for q, j in itertools.product([i * deltaQ for i in symmetry.Symmetry.Cubic.quOperators()], symmetry.Symmetry.Orthorhombic.quOperators())] # compute all equivilent misorientation representations
d = max(max(equivMiso), max([q.conjugate() for q in equivMiso])) # select quaternion of smallest rotation with axis in most positive quadrant from equivilent rotations w/ switching symmetry

print "Quaternion rotation between q1 & q2", d
print "Difference in angle: {0} degrees".format(2 * numpy.arccos(d.wxyz[0]) * 180 / numpy.pi)

