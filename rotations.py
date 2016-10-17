# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                                                 #
# Copyright (c) 2016 William Lenthe                                               #
# All rights reserved.                                                            #
#                                                                                 #
# Redistribution and use in source and binary forms, with or without              #
# modification, are permitted provided that the following conditions are met:     #
#     * Redistributions of source code must retain the above copyright            #
#       notice, this list of conditions and the following disclaimer.             #
#     * Redistributions in binary form must reproduce the above copyright         #
#       notice, this list of conditions and the following disclaimer in the       #
#       documentation and/or other materials provided with the distribution.      #
#     * Neither the name of the <organization> nor the                            #
#       names of its contributors may be used to endorse or promote products      #
#       derived from this software without specific prior written permission.     #
#                                                                                 #
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND #
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED   #
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE          #
# DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY              #
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES      #
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;    #
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND     #
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT      #
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS   #
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.                    #
#                                                                                 #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# orientation transform routines based on
#  -Rowenhorst, David, et al. "Consistent Representations of and Conversions Between 3D Rotations." Model. Simul. Mater. Sci. Eng. 23.8 (2015): 083501.
#  -Rosca, D., et al. "A New Method of Constructing a Grid in the Space of 3D rotations and its Applications to Texture Analysis." Model. Simul. Mater. Sci. Eng. 22.7 (2014): 075013.
#  -fortran implementation of routines by Marc De Graef

# the following conventions are used:
#  -quaternions as [w, x, y, z]
#  -orientation matrices in row major order
#  -rotation angle <= pi
#  -rotation axis in positive hemisphere for rotations of pi
#  -rotation axis = [0, 0, 1] for rotations of 0

import sys
import math
import numpy
from enum import Enum

epsilon = sys.float_info.epsilon * 1e6 # limited by taylor expansion in ho2ax

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Passive / Active convention (passive by default, P / i*j*k in first reference)  #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
class Convention(Enum):
	passive = 1.0
	active = -1.0
convention = Convention.passive

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Helper Functions                                                                #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def orientAxis(n):
	if n[2] < 0.0:
		return [-x for x in n] # [_, _, +z]
	elif 0.0 == n[2]:
		if n[1] < 0.0:
			return [-x for x in n[:2]] + [0.0] # [_, +y, 0]
		elif 0.0 == n[1]:
			if n[0] < 0.0:
				return [-n[0]] + [0.0] * 2 # [+x, 0, 0]
	return [x for x in n]

def roundZeros(x):
	return [0.0 if abs(i) < epsilon else i for i in x]

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Direct Conversions                                                              #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#A.1
def eu2om(eu):
	c1 = math.cos(eu[0])
	c = math.cos(eu[1])
	c2 = math.cos(eu[2])
	s1 = math.sin(eu[0])
	s = math.sin(eu[1])
	s2 = math.sin(eu[2])

	om = [0] * 9
	om[0] = c1 * c2 - s1 * c * s2
	om[1] = s1 * c2 + c1 * c * s2
	om[2] = s * s2
	om[3] = -c1 * s2 - s1 * c * c2
	om[4] = -s1 * s2 + c1 * c * c2
	om[5] = s * c2
	om[6] = s1 * s
	om[7] = -c1 * s
	om[8] = c
	return roundZeros(om)

#A.2
def eu2ax(eu):
	t = math.tan(eu[1] / 2.0)
	sigma = (eu[0] + eu[2]) / 2.0
	tau = math.sqrt(t * t + math.sin(sigma) * math.sin(sigma))
	if abs(tau) < epsilon:
		return [0.0, 0.0, 1.0, 0.0] # handle 0 rotation
	delta = (eu[0] - eu[2]) / 2.0
	alpha = math.pi if abs(sigma - math.pi / 2.0) < epsilon else 2.0 * math.atan(tau / math.cos(sigma))
	n = [-convention.value / math.copysign(tau, alpha)] * 3
	n[0] *= t * math.cos(delta)
	n[1] *= t * math.sin(delta)
	n[2] *= math.sin(sigma)

	# normalize
	n = roundZeros(n)
	mag = math.sqrt(sum([x * x for x in n]))
	n = [x / mag for x in n]

	# handle ambiguous case (rotation angle of pi)
	if math.pi - abs(alpha) < epsilon:
		return orientAxis(n) + [math.pi]
	return n + [abs(alpha)]

#A.3
def eu2ro(eu):
	ax = eu2ax(eu)
	if abs(ax[3] - math.pi) < epsilon:
		ax[3] = float('inf')
	elif abs(ax[3]) < epsilon:
		ax = [0.0, 0.0, 1.0, 0.0]
	else:
		ax[3] = math.tan(ax[3] / 2.0)
	return ax

#A.4
def eu2qu(eu):
	eu = [x / 2.0 for x in eu]
	c = math.cos(eu[1])
	s = math.sin(eu[1])
	sigma = (eu[0] + eu[2])
	delta = (eu[0] - eu[2])
	qu = [-convention.value] * 4
	qu[0] = c * math.cos(sigma)
	qu[1] *= s * math.cos(delta)
	qu[2] *= s * math.sin(delta)
	qu[3] *= c * math.sin(sigma)
	if qu[0] < 0.0:
		qu = [-x for x in qu]

	# normalize
	qu = roundZeros(qu)
	mag = math.sqrt(sum([x * x for x in qu]))
	qu = [x / mag for x in qu]

	# handle ambiguous case (rotation angle of pi)
	if 0.0 == qu[0]:
		return [0.0] + orientAxis(qu[1:])
	return qu

#A.5
def om2eu(om):
	eu = [0.0] * 3
	if abs(abs(om[8]) - 1.0) < epsilon:
		if om[8] > 0.0:
			eu[0] = math.atan2(om[1], om[0]) # eu = [_, 0, _]
		else:
			eu[0] = -math.atan2(-om[1], om[0]) # eu = [_, pi, _]
			eu[1] = math.pi
	else:
		eu[1] = math.acos(om[8])
		zeta = 1.0 / math.sqrt(1.0 - om[8] * om[8])
		eu[0] = math.atan2(om[6] * zeta, -om[7] * zeta)
		eu[2] = math.atan2(om[2] * zeta, om[5] * zeta)
	eu = roundZeros(eu)
	return [x + 2.0 * math.pi if x < 0.0 else x for x in eu]

#A.6
def om2ax(om):
	omega = (om[0] + om[4] + om[8] - 1.0) / 2.0
	if 1.0 - abs(omega) < epsilon:
		omega = math.copysign(1.0, omega)
	if 1.0 == omega:
		return [0.0, 0.0, 1.0, 0.0]

	# compute eigenvector for eigenvalue of 1 (cross product of 2 adjacent columns of A-y*I)
	om0 = om[0] - 1.0
	om4 = om[4] - 1.0
	om8 = om[8] - 1.0
	vecs = [roundZeros([om[3]*om[7] - om[6]* om4 , om[6]*om[1] -  om0 *om[7],  om0 * om4  - om[3]*om[1]]),
	        roundZeros([ om4 * om8  - om[7]*om[5], om[7]*om[2] - om[1]* om8 , om[1]*om[5] -  om4 *om[2]]),
	        roundZeros([om[5]*om[6] -  om8 *om[3],  om8 * om0  - om[2]*om[6], om[2]*om[3] - om[5]* om0 ])]

	# select vector with largest magnitude
	mags = [math.sqrt(sum([x * x for x in v])) for v in vecs]
	i = mags.index(max(mags))
	if mags[i] < epsilon:
		return [0.0, 0.0, 1.0, 0.0]
	n = [x / mags[i] for x in vecs[i]]

	# check ambiguous case
	if -1.0 == omega:
		return orientAxis(n) + [math.pi]
	
	# check axis sign
	n[0] = math.copysign(n[0], convention.value * (om[7] - om[5]))
	n[1] = math.copysign(n[1], convention.value * (om[2] - om[6]))
	n[2] = math.copysign(n[2], convention.value * (om[3] - om[1]))
	return n + [math.acos(omega)]

#A.7
def om2qu(om):
	qu = [1.0 + om[0] + om[4] + om[8], 1.0 + om[0] - om[4] - om[8], 1.0 - om[0] + om[4] - om[8], 1.0 - om[0] - om[4] + om[8]]
	qu = [0.0 if x < epsilon else convention.value * math.sqrt(x) / 2.0 for x in qu]
	if convention.value * om[5] > convention.value * om[7]:
		qu[1] = -qu[1]
	if convention.value * om[6] > convention.value * om[2]:
		qu[2] = -qu[2]
	if convention.value * om[1] > convention.value * om[3]:
		qu[3] = -qu[3]
	qu = roundZeros(qu)

	# normalize
	mag = math.sqrt(sum([x * x for x in qu]))
	assert mag > epsilon
	qu = [x / mag for x in qu]

	# ensure rotation angle <= pi
	if qu[0] < 0.0:
		qu = [-x for x in qu]

	# handle ambiguous case (rotation of pi)
	if qu[0] < epsilon:
		ax = om2ax(om)
		return [0.0] + [math.copysign(q, n) for q, n in zip(qu[1:], ax)]
	return qu

#A.8
def ax2om(ax):
	c = math.cos(ax[3])
	s = math.sin(ax[3])
	omc = 1.0 - c
	om = [0.0] * 9
	om[0] = c + omc * ax[0] * ax[0]
	om[4] = c + omc * ax[1] * ax[1]
	om[8] = c + omc * ax[2] * ax[2]
	x = omc * ax[0] * ax[1]
	y = convention.value * s * ax[2]
	om[3] = x + y
	om[1] = x - y
	x = omc * ax[1] * ax[2]
	y = convention.value * s * ax[0]
	om[7] = x + y
	om[5] = x - y
	x = omc * ax[2] * ax[0]
	y = convention.value * s * ax[1]
	om[6] = x - y
	om[2] = x + y
	return roundZeros(om)

#A.9
def ax2ro(ax):
	if abs(ax[3]) < epsilon:
		return [0.0, 0.0, 1.0, 0.0]
	return ax[:3] + [float('inf') if abs(ax[3] - math.pi) < epsilon else math.tan(ax[3] / 2.0)]

#A.10
def ax2qu(ax):
	if abs(ax[3]) < epsilon:
		return [1.0, 0.0, 0.0, 0.0]
	s = math.sin(ax[3] / 2.0)
	qu = roundZeros([math.cos(ax[3] / 2.0)] + [x * s for x in ax[:3]])
	mag = math.sqrt(sum([x * x for x in qu]))
	return [x / mag for x in qu]

#A.11
def ax2ho(ax):
	k = math.pow(0.75 * ( ax[3] - math.sin(ax[3]) ), 1.0 / 3.0)
	return roundZeros([x * k for x in ax[:3]])

#A.12
def ro2ax(ro):
	if abs(ro[3]) < epsilon:
		return [0.0, 0.0, 1.0, 0.0]
	omega = 2.0 * math.atan(ro[3]) if ro[3] < float('inf') else math.pi
	return [x for x in ro[:3]] + [omega]

#A.13
def ro2ho(ro):
	t = 2.0 * math.atan(ro[3]) if ro[3] < float('inf') else math.pi
	f = math.pow(0.75 * (t - math.sin(t)), 1.0 / 3.0)
	return roundZeros([x * f for x in ro[:3]])

#A.14
def qu2eu(qu):
	eu = [0.0] * 3
	qu0 = qu[0] * convention.value
	q03 = qu0 * qu0 + qu[3] * qu[3]
	q12 = qu[1] * qu[1] + qu[2] * qu[2]
	chi = math.sqrt(q03 * q12)
	if chi < epsilon:
		if q12 < epsilon:
			eu[0] = math.atan2(-2.0 * qu0 * qu[3], qu0 * qu0 - qu[3] * qu[3])
		else:
			eu[0] = math.atan2(2.0 * qu[1] * qu[2], qu[1] * qu[1] - qu[2] * qu[2])
			eu[1] = math.pi
	else:
		eu[0] = math.atan2((qu[1] * qu[3] - qu0 * qu[2]) / chi, (-qu[2] * qu[3] - qu0 * qu[1]) / chi)
		eu[1] = math.atan2(2.0 * chi, q03 - q12)
		eu[2] = math.atan2((qu[1] * qu[3] + qu0 * qu[2]) / chi, (qu[2] * qu[3] - qu0 * qu[1]) / chi)
	eu = roundZeros(eu)
	return [x + 2.0 * math.pi if x < 0.0 else x for x in eu]

#A.15
def qu2om(qu):
	om = [0.0] * 9
	qbar = qu[0] * qu[0] - sum([x * x for x in qu[1:]])
	om[0] = qbar + 2.0 * qu[1] * qu[1]
	om[4] = qbar + 2.0 * qu[2] * qu[2]
	om[8] = qbar + 2.0 * qu[3] * qu[3]
	om[1] = 2.0 * (qu[1] * qu[2] - convention.value * qu[0] * qu[3]);
	om[3] = 2.0 * (qu[2] * qu[1] + convention.value * qu[0] * qu[3])
	om[2] = 2.0 * (qu[1] * qu[3] + convention.value * qu[0] * qu[2])
	om[6] = 2.0 * (qu[3] * qu[1] - convention.value * qu[0] * qu[2])
	om[5] = 2.0 * (qu[2] * qu[3] - convention.value * qu[0] * qu[1])
	om[7] = 2.0 * (qu[3] * qu[2] + convention.value * qu[0] * qu[1])
	return roundZeros(om)

#A.16
def qu2ax(qu):
	omega = 2.0 * math.acos(qu[0])
	if omega < epsilon:
		return [0.0, 0.0, 1.0, 0.0]
	s = math.copysign(1.0 / math.sqrt(sum([x * x for x in qu[1:]])), qu[0])
	return [s * n for n in qu[1:]] + [omega]

#A.17
def qu2ro(qu):
	if qu[0] < epsilon:
		return qu[1:] + [float('inf')]
	s = math.sqrt(sum([x * x for x in qu[1:]]))
	if s < epsilon:
		return [0.0, 0.0, 1.0, 0.0]
	return [x / s for x in qu[1:]] + roundZeros([math.tan(math.acos(qu[0]))])

#A.18
def qu2ho(qu):
	omega = 2.0 * math.acos(qu[0])
	if abs(omega) < epsilon:
		return [0.0] * 3
	s = 1.0 / math.sqrt(sum([x * x for x in qu[1:]]))
	f = math.pow(0.75 * (omega - math.sin(omega)), 1.0 / 3.0)
	return [s * x * f for x in qu[1:]]

#A.19
def ho2ax(ho):
	mag = sum([x * x for x in ho])
	if mag < epsilon:
		return [0.0, 0.0, 1.0, 0.0]
	ax = [x / math.sqrt(mag) for x in ho]
	tExpansion = [ 1.000000000001885, -0.500000000219485, -0.024999992127593, -0.003928701544781,
	              -0.000815270153545, -0.000200950042612, -0.000023979867761, -0.000082028689266,
	               0.000124487150421, -0.000174911421482,  0.000170348193414, -0.000120620650041,
	               0.000059719705869, -0.000019807567240,  0.000003953714684, -0.000000365550014]
	s = sum([tExpansion[i] * math.pow(mag, i) for i in range(len(tExpansion))])
	if abs(s) < epsilon:
		return ax + [math.pi]
	return ax + [2.0 * math.acos(s)]

# helper function for cubochoric <---> homochoric transformation symmetry
def pyramidType(x):
	maxX = max([abs(i) for i in x])
	if abs(x[2]) == maxX:
		return 0
	if abs(x[0]) == maxX:
		return 1
	if abs(x[1]) == maxX:
		return 2
	raise RuntimeError('failed to find pyramid type for %r' % str(x))

def ho2cu(ho):
	# check bounds, get pyramid, and shuffle coordinates to +z pyramid
	rs = math.sqrt(sum([x * x for x in ho])) # radius
	if rs - math.pow(0.75 * math.pi, 1.0 / 3.0) > epsilon:
		raise ValueError('%r lies outside the sphere of radius %r' % (str(ho), str(math.pow(0.75 * math.pi, 1.0 / 3.0))))
	p = pyramidType(ho)
	cu = ho[p:] + ho[:p]

	# handle origin
	if rs < epsilon:
		return [0.0] * 3

	# invert operation M3
	cu[:2] = [x * math.sqrt(2.0 * rs / (rs + abs(cu[2]))) for x in cu[:2]]
	cu[2] = -rs * math.sqrt(math.pi / 6.0) if cu[2] < 0.0 else rs / math.sqrt(6.0 / math.pi)

	# invert operation M2
	sq = sorted([x * x for x in cu[:2]])
	mag = sum(sq)
	if mag < epsilon:
		cu = [0.0, 0.0, cu[2]]
	else:
		swapped = False
		if abs(cu[0]) > abs(cu[1]):
			swapped = True
			cu[0], cu[1] = cu[1], cu[0]
		k = math.sqrt((mag + sq[1]) * sq[1]);
		sign = [-1 if x < 0.0 else 1 for x in cu[:2]]
		cu[:2] = [x * math.sqrt(math.pi / 3) * math.sqrt((mag + sq[1]) * mag / ((mag + sq[1]) - k)) / 2.0 for x in sign]
		k = (sq[0] + k) / (mag * math.sqrt(2.0))
		cu[0] *= 12.0 * math.acos(1.0 if 1.0 - k < epsilon else k) / math.pi
		if swapped:
			cu[0], cu[1] = cu[1], cu[0]

	#invert operation M1, unshuffle coordinates, and return
	cu = [x / math.pow(math.pi / 6.0, 1.0 / 6.0) for x in cu]
	return roundZeros(cu[-p:] + cu[:-p])

def cu2ho(cu):
	# check bounds, get pyramid, and shuffle coordinates to +z pyramid
	if max([abs(i) for i in cu]) - math.pow(math.pi, 2.0 / 3.0) / 2.0 > epsilon:
		raise ValueError('%r lies outside the cube of side length %r' % (str(cu), str(math.pow(math.pi, 2.0 / 3.0))))
	p = pyramidType(cu)

	ho = numpy.roll(cu, -p)

	# handle origin
	if abs(ho[2]) < epsilon:
		return [0.0] * 3

	# operation M1
	ho = [i * math.pow(math.pi / 6.0, 1.0 / 6.0) for i in ho]

	# operation M2
	if max([abs(i) for i in ho[:2]]) < epsilon:
		ho = [0.0, 0.0, ho[2]] # handle points along z axis (to avoid divide by zero)
	else:
		swapped = False
		if abs(ho[0]) > abs(ho[1]):
			swapped = True
			ho[0], ho[1] = ho[1], ho[0]
		theta = (math.pi * ho[0]) / (12.0 * ho[1])
		k = math.sqrt(3.0 / math.pi) * math.pow(2.0, 0.75) * ho[1] / math.sqrt(math.sqrt(2.0) - math.cos(theta))
		ho[0] = math.sqrt(2.0) * math.sin(theta) * k
		ho[1] = (math.sqrt(2.0) * math.cos(theta) - 1.0) * k
		if swapped:
			ho[0], ho[1] = ho[1], ho[0]

	# operation M3
	k = ho[0]*ho[0] + ho[1]*ho[1]
	ho[:2] = [i * math.sqrt(1.0 - math.pi * k / (24.0 * ho[2]*ho[2])) for i in ho[:2]]
	ho[2] = math.sqrt(6.0 / math.pi) * ho[2] - k * math.sqrt(math.pi / 24) / ho[2]

	# unshuffle coordinates
	return roundZeros(numpy.roll(ho, p))

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Indirect Conversions                                                            #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def ro2eu(ro):
	return om2eu(ro2om(ro))

def eu2ho(eu):
	return ax2ho(eu2ax(eu))

def om2ro(om):
	return eu2ro(om2eu(om))

def om2ho(om):
	return ax2ho(om2ax(om))

def ax2eu(ax):
	return om2eu(ax2om(ax))

def ro2om(ro):
	return ax2om(ro2ax(ro))

def ro2qu(ro):
	return ax2qu(ro2ax(ro))

def ho2eu(ho):
	return ax2eu(ho2ax(ho))

def ho2om(ho):
	return ax2om(ho2ax(ho))

def ho2ro(ho):
	return ax2ro(ho2ax(ho))

def ho2qu(ho):
	return ax2qu(ho2ax(ho))

def eu2cu(eu):
	return ho2cu(eu2ho(eu))

def om2cu(om):
	return ho2cu(om2ho(om))

def ax2cu(ax):
	return ho2cu(ax2ho(ax))

def ro2cu(ro):
	return ho2cu(ro2ho(ro))

def qu2cu(qu):
	return ho2cu(qu2ho(qu))

def cu2eu(cu):
	return ho2eu(cu2ho(cu))

def cu2om(cu):
	return ho2om(cu2ho(cu))

def cu2ax(cu):
	return ho2ax(cu2ho(cu))

def cu2ro(cu):
	return ho2ro(cu2ho(cu))

def cu2qu(cu):
	return ho2qu(cu2ho(cu))

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Testing                                                                         #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def defaultDist(a, b):
	return max([abs(i - j) for i, j in zip(a, b)])

def euDist(a, b):
	return defaultDist(eu2qu(a), eu2qu(b))

def test(n = 3, output = sys.stdout, verbose = False):
	# create test vector, total rotations = 4 * (n^3 - n^2) + n
	eulerList = []
	phi = [math.pi * i / (n - 1) for i in range((n - 1) * 2 + 1)]
	theta = [math.pi * i / (n - 1) for i in range(n)]
	for i in range(len(phi)):
		for j in range(len(theta)):
			for k in range(len(phi)):
				eulerList.append([phi[i], theta[j], phi[k]])

	# build matrix of functions
	names = ['eu', 'om', 'ax', 'ro', 'qu', 'ho', 'cu']
	representations = len(names)
	comparisons = [euDist, defaultDist, defaultDist, defaultDist, defaultDist, defaultDist, defaultDist]
	conversions = [[ None, eu2om, eu2ax, eu2ro, eu2qu, eu2ho, eu2cu],
	               [om2eu,  None, om2ax, om2ro, om2qu, om2ho, om2cu],
	               [ax2eu, ax2om,  None, ax2ro, ax2qu, ax2ho, ax2cu],
	               [ro2eu, ro2om, ro2ax,  None, ro2qu, ro2ho, ro2cu],
	               [qu2eu, qu2om, qu2ax, qu2ro,  None, qu2ho, qu2cu],
	               [ho2eu, ho2om, ho2ax, ho2ro, ho2qu,  None, ho2cu],
	               [cu2eu, cu2om, cu2ax, cu2ro, cu2qu, cu2ho,  None]]

	# check x = y2x(x2y(x)) 
	maxDiff = 0.0
	output.write('pairwise tests:\n')
	for i in range(representations):
		if verbose:
			output.write(names[i] + ' test\n') 
		for j in range(representations):
			if i == j:
				continue
			for eu in eulerList:
				try:
					base = eu if 0 == i else conversions[0][i](eu)
					conv = conversions[j][i](conversions[i][j](base))
					diff = comparisons[i](conv, base)
					if verbose or diff > 1e2 * epsilon:
						output.write(names[i] + '2' + names[j] + ' max difference(' + str(base) + ') = ' + str(diff) + '\n')
					if diff > maxDiff:
						maxDiff = diff
						maxIndex = [i, j, base, eulerList.index(eu)]
				except ValueError as err:
					output.write(names[i] + '2' + names[j] + '[' + str(eulerList.index(eu)) + ']: ' + str(err) + '\n')
	output.write('max diff pairwise: ' + names[maxIndex[0]] + '2' + names[maxIndex[1]] + '(' + str(maxIndex[2]) + ') = ' + str(maxDiff) + '\n')

	# check x = z2x(y2z(x2y(x)))
	maxDiff = 0.0
	output.write('triplet tests:\n')
	for i in range(representations):
		if verbose:
			output.write(names[i] + ' test\n')
		for j in range(representations):
			if i == j:
				continue
			for k in range(representations):
				if i == k or j == k:
					continue
				for eu in eulerList:
					try:
						base = eu if 0 == i else conversions[0][i](eu)
						conv = conversions[k][i](conversions[j][k](conversions[i][j](base)))
						diff = comparisons[i](conv, base)
						if verbose or diff > 1e2 * epsilon:
							output.write(names[k] + '2' + names[i] + '-' + names[j] + '2' + names[k] + '-' + names[i] + '2' + names[j] + ' max difference(' + str(base) + ') = ' + str(diff) + '\n')
						if diff > maxDiff:
							maxDiff = diff
							maxIndex = [i, j, k, base, eulerList.index(eu)]
					except ValueError as err:
						output.write(names[i] + '2' + names[j] + '[' + str(eulerList.index(eu)) + ']: ' + str(err) + '\n')
	output.write('max diff triplet: ' + names[maxIndex[2]] + '2' + names[maxIndex[0]] + '-' + names[maxIndex[1]] + '2' + names[maxIndex[2]] + '-' + names[maxIndex[0]] + '2' + names[maxIndex[1]] + '(' + str(maxIndex[3]) + ') = ' + str(maxDiff) + '\n')
