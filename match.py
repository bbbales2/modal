#%%

from __future__ import absolute_import
import sys
import six
from six.moves import range
sys.path.append('.')
sys.path.append('/home/bbales2/sfepy')
import os

os.chdir('/home/bbales2/modal')

import numpy as nm
import scipy.sparse.linalg as sla

from sfepy.base.base import assert_, output, Struct
from sfepy.discrete import (FieldVariable, Material, Integral, Integrals,
                            Equation, Equations, Problem)
from sfepy.discrete.fem import Mesh, FEDomain, Field
from sfepy.terms import Term
from sfepy.discrete.conditions import Conditions, EssentialBC
from sfepy.mechanics.matcoefs import stiffness_from_youngpoisson
from sfepy.mesh.mesh_generators import gen_block_mesh
from sfepy.solvers import Solver
import scipy
output.set_output(quiet=True)

import numpy

import time

def fder(C, a, b, y):
    # Code stolen from Will Lenthe
    K = numpy.zeros((6, 6))
    dKdQ = numpy.zeros((6, 6, 3, 3))

    Q = numpy.array([[numpy.cos(b) * numpy.cos(y), -numpy.cos(b) * numpy.sin(y), numpy.sin(b)],
                     [numpy.cos(y) * numpy.sin(a) * numpy.sin(b) + numpy.cos(a) * numpy.sin(y), numpy.cos(a) * numpy.cos(y) - numpy.sin(a) * numpy.sin(b) * numpy.sin(y), -numpy.cos(b) * numpy.sin(a)],
                     [-numpy.cos(a) * numpy.cos(y) * numpy.sin(b) + numpy.sin(a) * numpy.sin(y), numpy.cos(y) * numpy.sin(a) + numpy.cos(a) * numpy.sin(b) * numpy.sin(y), numpy.cos(a) * numpy.cos(b)]])

    dQda = numpy.array([[0, 0, 0],
            [numpy.cos(a) * numpy.cos(y) * numpy.sin(b) - numpy.sin(a) * numpy.sin(y), -numpy.cos(y) * numpy.sin(a) - numpy.cos(a) * numpy.sin(b) * numpy.sin(y), -numpy.cos(a) * numpy.cos(b)],
            [numpy.cos(y) * numpy.sin(a) * numpy.sin(b) + numpy.cos(a) * numpy.sin(y), numpy.cos(a) * numpy.cos(y) - numpy.sin(a) * numpy.sin(b) * numpy.sin(y), -numpy.cos(b) * numpy.sin(a)]])

    dQdb = numpy.array([[-numpy.cos(y) * numpy.sin(b), numpy.sin(b) * numpy.sin(y), numpy.cos(b)],
            [numpy.cos(b) * numpy.cos(y) * numpy.sin(a), -numpy.cos(b) * numpy.sin(a) * numpy.sin(y), numpy.sin(a) * numpy.sin(b)],
            [-numpy.cos(a) * numpy.cos(b) * numpy.cos(y), numpy.cos(a) * numpy.cos(b) * numpy.sin(y), -numpy.cos(a) * numpy.sin(b)]])

    dQdy = numpy.array([[-numpy.cos(b) * numpy.sin(y), -numpy.cos(b) * numpy.cos(y), 0],
            [numpy.cos(a) * numpy.cos(y) - numpy.sin(a) * numpy.sin(b) * numpy.sin(y), -numpy.cos(y) * numpy.sin(a) * numpy.sin(b) - numpy.cos(a) * numpy.sin(y), 0],
            [numpy.cos(y) * numpy.sin(a) + numpy.cos(a) * numpy.sin(b) * numpy.sin(y), numpy.cos(a) * numpy.cos(y) * numpy.sin(b) - numpy.sin(a) * numpy.sin(y), 0]])

    for i in range(3):
        for j in range(3):
            dKdQ[i, j, i, j] = 2.0 * Q[i, j]
            dKdQ[i, j + 3, i, (j + 1) % 3] = Q[i, (j + 2) % 3]
            dKdQ[i, j + 3, i, (j + 2) % 3] = Q[i, (j + 1) % 3]
            dKdQ[i + 3, j, (i + 1) % 3, j] = Q[(i + 2) % 3, j]
            dKdQ[i + 3, j, (i + 2) % 3, j] = Q[(i + 1) % 3, j]
            dKdQ[i + 3, j + 3, (i + 1) % 3, (j + 1) % 3] = Q[(i + 2) % 3, (j + 2) % 3]
            dKdQ[i + 3, j + 3, (i + 2) % 3, (j + 2) % 3] = Q[(i + 1) % 3, (j + 1) % 3]
            dKdQ[i + 3, j + 3, (i + 1) % 3, (j + 2) % 3] = Q[(i + 2) % 3, (j + 1) % 3]
            dKdQ[i + 3, j + 3, (i + 2) % 3, (j + 1) % 3] = Q[(i + 1) % 3, (j + 2) % 3]
            K[i][j] = Q[i][j] * Q[i][j]
            K[i][j + 3] = Q[i][(j + 1) % 3] * Q[i][(j + 2) % 3]
            K[i + 3][j] = Q[(i + 1) % 3][j] * Q[(i + 2) % 3][j]
            K[i + 3][j + 3] = Q[(i + 1) % 3][(j + 1) % 3] * Q[(i + 2) % 3][(j + 2) % 3] + Q[(i + 1) % 3][(j + 2) % 3] * Q[(i + 2) % 3][(j + 1) % 3]


    for i in range(3):
        for j in range(3):
            K[i][j + 3] *= 2.0
            dKdQ[i][j + 3] *= 2.0

    Crot = K.dot(C.dot(K.T))
    dCrotdQ = numpy.zeros((6, 6, 3, 3))
    for i in range(3):
        for j in range(3):
            dCrotdQ[:, :, i, j] = dKdQ[:, :, i, j].dot(C.dot(K.T)) + K.dot(C.dot(dKdQ[:, :, i, j].T))

    dCrotdas = numpy.zeros((6, 6, 3))

    for i in range(6):
        for j in range(6):
            dCrotdas[i, j, 0] = (dCrotdQ[i, j] * dQda).flatten().sum()
            dCrotdas[i, j, 1] = (dCrotdQ[i, j] * dQdb).flatten().sum()
            dCrotdas[i, j, 2] = (dCrotdQ[i, j] * dQdy).flatten().sum()

    return Crot, dCrotdas, K

D = numpy.array([[c11, c12, c13, 0, 0, 0],
                 [c12, c22, c23, 0, 0, 0],
                 [c13, c23, c33, 0, 0, 0],
                 [0, 0, 0, c44, 0, 0],
                 [0, 0, 0, 0, c55, 0],
                 [0, 0, 0, 0, 0, c66]])

Crot1, dCrotdas, K = fder(D, 0.4, 0.4, 0.0)

D1 = D.copy()

D1[0, 0] *= 1.0001
D1[1, 1] *= 1.0001
D1[2, 2] *= 1.0001

Crot2, dCrotdas, K = fder(D1, 0.4, 0.4, 0.0)

dCrotdc11, _, _ = fder(numpy.array([[1, 0, 0, 0, 0, 0],
                     [0, 1, 0, 0, 0, 0],
                     [0, 0, 1, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0]]), 0.4, 0.4, 0.0)

print (Crot2 - Crot1) / (D1[0, 0] - D[0, 0])
print dCrotdc11
#%%
Crot1, dCrotdas, K = fder(D, 0.1, 0.15, 0.25)
Crot2, dCrotdas, K = fder(D, -0.1, -0.15, 0.25)
#%%
D1 = numpy.linalg.solve(K, numpy.linalg.solve(K, Crot).T)
#%%

aux = "eig.scipy,method:'eigh',tol:1e-5,maxiter:1000".split(',')
kwargs = {}
for option in aux[1:]:
    key, val = option.split(':')
    kwargs[key.strip()] = eval(val)
eig_conf = Struct(name='evp', kind=aux[0], **kwargs)

output.level = -1

density = 8700.0#4401.6959210

dims = numpy.array([0.011959, 0.013953, 0.019976])#([0.007753, 0.009057, 0.013199])
dim = len(dims)

shape = [4, 4, 4]

centre = numpy.array([0.0, 0.0, 0.0])

order = 2

tmp = time.time()
mesh = gen_block_mesh(dims, shape, centre, name='mesh')
print "Mesh generation: ", time.time() - tmp

axis = -1

eig_solver = Solver.any_from_conf(eig_conf)

# Build the problem definition.
domain = FEDomain('domain', mesh)

bbox = domain.get_mesh_bounding_box()
min_coor, max_coor = bbox[:, axis]
eps = 1e-8 * (max_coor - min_coor)
ax = 'xyz'[:dim][axis]

omega = domain.create_region('Omega', 'all')

field = Field.from_args('fu', nm.float64, 'vector', omega, approx_order = order)

u = FieldVariable('u', 'unknown', field)
v = FieldVariable('v', 'test', field, primary_var_name = 'u')

#youngs = 200e9
#poisson = 0.3

#mtx_d = stiffness_from_youngpoisson(dim, youngs, poisson)
#, -0.08594789, -0.17563149, 0.03437309
#c11, c12, c44 = 2.30887372458, 0.778064244563, 0.757576236829
c11, c12, c44 = 2.82726684,  1.9424297 ,  1.28273717#1.24, .934, 0.4610#1.685, 0.7928, 0.4459#
#c11 = 3.00
#c12 = 1.5
#c44 = 0.75

c22 = c11
c33 = c11

c13 = c12
c23 = c12

c55 = c44
c66 = c44

a = 0.0
b = 0.0
c = 0.0

D = numpy.array([[c11, c12, c13, 0, 0, 0],
                     [c12, c22, c23, 0, 0, 0],
                     [c13, c23, c33, 0, 0, 0],
                     [0, 0, 0, c44, 0, 0],
                     [0, 0, 0, 0, c55, 0],
                     [0, 0, 0, 0, 0, c66]])

D, _, _ = fder(D, a, b, c)

dDdc11, _, _ = fder(numpy.array([[1, 0, 0, 0, 0, 0],
                     [0, 1, 0, 0, 0, 0],
                     [0, 0, 1, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0]]), a, b, c)

dDdc12, _, _ = fder(numpy.array([[0, 1, 1, 0, 0, 0],
                     [1, 0, 1, 0, 0, 0],
                     [1, 1, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0]]), a, b, c)

dDdc44, _, _ = fder(numpy.array([[0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 1, 0, 0],
                     [0, 0, 0, 0, 1, 0],
                     [0, 0, 0, 0, 0, 1]]), a, b, c)

def assemble(mtx_d):
    m = Material('m', D=mtx_d, rho=density)

    integral = Integral('i', order=2 * order)

    t1 = Term.new('dw_lin_elastic(m.D, v, u)', integral, omega, m=m, v=v, u=u)
    t2 = Term.new('dw_volume_dot(m.rho, v, u)', integral, omega, m=m, v=v, u=u)
    eq1 = Equation('stiffness', t1)
    eq2 = Equation('mass', t2)
    lhs_eqs = Equations([eq1, eq2])

    pb = Problem('modal', equations=lhs_eqs)

    pb.time_update()
    n_rbm = dim * (dim + 1) / 2

    pb.update_materials()

    tmp = time.time()
    # Assemble stiffness and mass matrices.
    mtx_k = eq1.evaluate(mode='weak', dw_mode='matrix', asm_obj=pb.mtx_a)
    mtx_m = mtx_k.copy()
    mtx_m.data[:] = 0.0
    mtx_m = eq2.evaluate(mode='weak', dw_mode='matrix', asm_obj=mtx_m)

    return mtx_k, mtx_m

dKdc11, _ = assemble(dDdc11)
dKdc12, _ = assemble(dDdc12)
dKdc44, _ = assemble(dDdc44)

tmp = time.time()
K, M = assemble(D)
print "Assemble matrices: ", time.time() - tmp

nrbm = 6

tmp = time.time()
try:
    #eigs, svecs = eig_solver(K, M, 30 + nrbm, eigenvectors=True)
    #eigs, svecs = scipy.sparse.linalg.eigsh(K, 30 + nrbm, M = M, which = 'SM')
    eigs, svecs = scipy.sparse.linalg.eigsh(K, 30 + nrbm, M = M, sigma = 1.0)
except sla.ArpackNoConvergence as ee:
    eigs2 = ee.eigenvalues
    svecs = ee.eigenvectors
    output('only %d eigenvalues converged!' % len(eigs))
print "Eigenvalue solve: ", time.time() - tmp

output('%d eigenvalues converged (%d ignored as rigid body modes)' %
    (len(eigs), nrbm))

eigs = eigs[nrbm:]
svecs = svecs[:, nrbm:]

omegas = nm.sqrt(eigs * 1e11)
freqs = omegas / (2 * nm.pi)

print 'number |         eigenvalue |  angular frequency |          frequency'
for ii, eig in enumerate(eigs):
    print '{0:6d} | {1:17.12e} | {2:17.12e} | {3:17.12e} | {4:17.12e}'.format(ii + 1, eig, eigs2[ii], omegas[ii], freqs[ii])
#%%
eigs2 = eigs.copy()
#%%
tmp = time.time()
scipy.sparse.linalg.spsolve(K, M)
print time.time() - tmp

sys.stdout.flush()

tmp = time.time()
scipy.sparse.linalg.spsolve(M, K)
print time.time() - tmp

#%%young = 5
import matplotlib.pyplot as plt

freqs = numpy.array([71.25925,
75.75875,
86.478,
89.947375,
111.150125,
112.164125,
120.172125,
127.810375,
128.6755,
130.739875,
141.70025,
144.50375,
149.40075,
154.35075,
156.782125,
157.554625,
161.0875,
165.10325,
169.7615,
173.44925,
174.11675,
174.90625,
181.11975,
182.4585,
183.98625,
192.68125,
193.43575,
198.793625,
201.901625,
205.01475])

eigs = (freqs * numpy.pi * 2000) ** 2 / 1e11


mu = eigs

llogp = []
youngs = []
poissons = []
#c11t, c12t, c44t =  2, 1, 0.5#1.685, 0.7928, 0.4459
#y = 1.0

#c11s = []
#c12s = []
#c44s = []
#ys = []

def UgradU(q):
    c11t, c12t, c44t, y, a, b, c = q#
    #y, a, b, c = 0.17362318, 0.19585363, -0.2803732, -0.09701904
    c22 = c11t
    c33 = c11t

    c13 = c12t
    c23 = c12t

    c55 = c44t
    c66 = c44t

    D = numpy.array([[c11t, c12t, c13, 0, 0, 0],
                     [c12t, c22, c23, 0, 0, 0],
                     [c13, c23, c33, 0, 0, 0],
                     [0, 0, 0, c44t, 0, 0],
                     [0, 0, 0, 0, c55, 0],
                     [0, 0, 0, 0, 0, c66]])

    D, Dd, K_ = fder(D, a, b, c)

    dDda = Dd[:, :, 0]
    dDdb = Dd[:, :, 1]
    dDdc = Dd[:, :, 2]

    tmp = time.time()
    dKda, _ = assemble(dDda)
    dKdb, _ = assemble(dDdb)
    dKdc, _ = assemble(dDdc)

    dKdc11, _ = assemble(K_.dot(dDdc11.dot(K_.T)))
    dKdc12, _ = assemble(K_.dot(dDdc12.dot(K_.T)))
    dKdc44, _ = assemble(K_.dot(dDdc44.dot(K_.T)))
    #print "Assembly: ", time.time() - tmp

    Kt, Mt = assemble(D)

    tmp = time.time()
    eigst, evecst = scipy.sparse.linalg.eigsh(Kt.tocsc(), 30 + nrbm, M = Mt.tocsc(), sigma = 1.0)
    print "Eigs: ", time.time() - tmp

    eigst = eigst[6:]
    evecst = evecst[:, 6:]

    #print eigst

    #print list(zip(eigst, eigs))

    t = 1 + (mu - eigst)**2 / y**2

    dlda = numpy.array([evecst[:, i].T.dot(dKda.dot(evecst[:, i])) for i in range(evecst.shape[1])])
    dldb = numpy.array([evecst[:, i].T.dot(dKdb.dot(evecst[:, i])) for i in range(evecst.shape[1])])
    dldc = numpy.array([evecst[:, i].T.dot(dKdc.dot(evecst[:, i])) for i in range(evecst.shape[1])])
    dldc11 = numpy.array([evecst[:, i].T.dot(dKdc11.dot(evecst[:, i])) for i in range(evecst.shape[1])])
    dldc12 = numpy.array([evecst[:, i].T.dot(dKdc12.dot(evecst[:, i])) for i in range(evecst.shape[1])])
    dldc44 = numpy.array([evecst[:, i].T.dot(dKdc44.dot(evecst[:, i])) for i in range(evecst.shape[1])])

    #dlpdl = (mu - eigst) / y ** 2
    #dlpdy = sum((-y ** 2 + (eigst - mu) **2) / y ** 3)
    dlpdl = 2 * (mu - eigst) / (t * y**2)
    dlpdy = sum(numpy.pi * t * ((2 * (mu - eigst) ** 2) / (numpy.pi * t**2 * y**4) - 1 / (numpy.pi * t * y**2)) * y)

    dlpdl = numpy.array(dlpdl)

    dlpdc11 = dlpdl.dot(dldc11)
    dlpdc12 = dlpdl.dot(dldc12)
    dlpdc44 = dlpdl.dot(dldc44)
    dlpda = dlpdl.dot(dlda)
    dlpdb = dlpdl.dot(dldb)
    dlpdc = dlpdl.dot(dldc)

    #logp = sum(0.5 * (-((eigst - mu) **2 / y**2) + numpy.log(1.0 / (2 * numpy.pi)) - 2 * numpy.log(y)))
    logp = sum(numpy.log(1 / (numpy.pi *  (1 + (mu - eigst)**2 / y**2) * y)))
    #E^(-((-u + x)^2/(2 s^2)))/(Sqrt[2 \[Pi]] Sqrt[s^2])

    return -logp, -numpy.array([dlpdc11, dlpdc12, dlpdc44, dlpdy, dlpda, dlpdb, dlpdc])# dlda, dldb, dldc])#, eigst#

q = numpy.array([ 3.05553881, 2.16139498, 1.17935285, 0.1479137, 0.28143762, -0.2803732, -0.09701904])
 #[3.05903756, 2.16126911, 1.18847201, 0.15285394, 0.24276501, -0.2803732, -0.09701904])
 #[3.06864815, 2.15470721, 1.16336802, 0.17362318, 0.19585363, -0.2803732, -0.09701904])
qs = []
logp, dlogpdq = UgradU(q)
print "Finite difference, analytical"
for i in range(7):
    q2 = q.copy()
    q2[i] *= 1.000001
    logp2, dlogpdq2 = UgradU(q2)
    print (logp2 - logp) / (q2[i] - q[i]), dlogpdq[i]
print dlogpdq
#%%
a, b, c = 0.28143762, 0.0, 0.0
dDrotdc11, Dd, K_ = fder(dDdc11, a, b, c)
print dDrotdc11 - K_.dot(dDdc11.dot(K_.T))
#%%
c11t, c12t, c44t, y, a, b, c = q#
a, b, c = 0.28143762, 0.0, 0.0
#y, a, b, c = 0.17362318, 0.19585363, -0.2803732, -0.09701904
c22 = c11t
c33 = c11t

c13 = c12t
c23 = c12t

c55 = c44t
c66 = c44t

D = numpy.array([[c11t, c12t, c13, 0, 0, 0],
                 [c12t, c22, c23, 0, 0, 0],
                 [c13, c23, c33, 0, 0, 0],
                 [0, 0, 0, c44t, 0, 0],
                 [0, 0, 0, 0, c55, 0],
                 [0, 0, 0, 0, 0, c66]])

D1, Dd, K_ = fder(D, a, b, c)

c11t *= 1.0001

c22 = c11t
c33 = c11t

c13 = c12t
c23 = c12t

c55 = c44t
c66 = c44t

D = numpy.array([[c11t, c12t, c13, 0, 0, 0],
                 [c12t, c22, c23, 0, 0, 0],
                 [c13, c23, c33, 0, 0, 0],
                 [0, 0, 0, c44t, 0, 0],
                 [0, 0, 0, 0, c55, 0],
                 [0, 0, 0, 0, 0, c66]])

D2, Dd, K_ = fder(D, a, b, c)

print (D2 - D1) / (c11t - c11t / 1.0001) - dDrotdc11
#%%
for i in range(2000):
    logp, dlogpdq = UgradU(q)
    llogp.append(logp)

    qs.append(q)

    print "log likelihood: ", logp, dlogpdq

    dlogpdq /= numpy.linalg.norm(dlogpdq)

    q -= dlogpdq * 0.001

    print "New parameters: ", q

    #c11t = min(5., max(1.0, c11t))
    #c12t = min(3.0, max(.5, c12t))
    #c44t = min(2.0, max(.25, c44t))

    #print "New parameters: ", c11t, c12t, c44t, y
    #print ""

#%%young = 5

mu = eigs

llogp = []
youngs = []
poissons = []
c11t, c12t, c44t = 2, 2.0, 2
#c11t = 2.00
#c12t = 1.0
#c44t = 1.0

c11s = []
c12s = []
c44s = []

for i in range(2000):
    c22 = c11t
    c33 = c11t

    c13 = c12t
    c23 = c12t

    c55 = c44t
    c66 = c44t

    D = numpy.array([[c11t, c12t, c13, 0, 0, 0],
                     [c12t, c22, c23, 0, 0, 0],
                     [c13, c23, c33, 0, 0, 0],
                     [0, 0, 0, c44t, 0, 0],
                     [0, 0, 0, 0, c55, 0],
                     [0, 0, 0, 0, 0, c66]])

    Kt, Mt = assemble(D)

    eigst, evecst = scipy.sparse.linalg.eigsh(Kt, 30 + nrbm, M = Mt, sigma = 1.0)

    eigst = eigst[6:]
    evecst = evecst[:, 6:]

    #print list(zip(eigst, eigs))

    dldc11 = numpy.array([evecst[:, i].T.dot(dKdc11.dot(evecst[:, i])) for i in range(evecst.shape[1])])
    dldc12 = numpy.array([evecst[:, i].T.dot(dKdc12.dot(evecst[:, i])) for i in range(evecst.shape[1])])
    dldc44 = numpy.array([evecst[:, i].T.dot(dKdc44.dot(evecst[:, i])) for i in range(evecst.shape[1])])

    #dlpdl = []
    y = 0.25
    dlpdl = ((2 * (eigst - mu)) / ((1 + (eigst - mu)**2 / y**2) * y**2))
    #for val in (eigst - mu):
    #    if abs(val) > eps:
    #        if val > 0.0:
    #            dlpdl.append(1.0)
    #        else:
    #            dlpdl.append(-1.0)
    #    else:
    #        dlpdl.append(10 * val)

        #if abs(val) < 0.001:
        #    dlpdl.append(0.0)
        #elif val < 0.0:
        #    dlpdl.append(-1.0)
        #else:
        #    dlpdl.append(1.0)

    dlpdl = numpy.array(dlpdl)#-(mu - eigst) / sigma**2

    dlpdc11 = dlpdl.dot(dldc11)
    dlpdc12 = dlpdl.dot(dldc12)
    dlpdc44 = dlpdl.dot(dldc44)

    logp = sum(numpy.log(1 / (numpy.pi *  (1 + (eigst - mu)**2 / y**2) * y)))

    if len(llogp) > 0:
        r = logp - llogp[-1]
    else:
        r = 1.0

    print "Difference in log likelihood: ", r, logp, llogp[-1] if len(llogp) > 0 else 0
    print "Proposing: ", c11t, c12t, c44t

    if r < 0.0:
        if numpy.random.rand() > numpy.exp(r):
            print "Rejecting"

            c11t = c11s[-1] + numpy.random.randn() / 15.0
            c12t = c12s[-1] + numpy.random.randn() / 15.0
            c44t = c44s[-1] + numpy.random.randn() / 15.0
            continue

    print "Accepting"

    llogp.append(logp)
    #llogp.append(-sum(0.5 * (mu - eigst)[:6]**2 / sigma**2))# - 0.5 * (0.3 - poisson)**2 / 0.1**2)

    c11s.append(c11t)
    c12s.append(c12t)
    c44s.append(c44t)

    c11t = c11s[-1] + numpy.random.randn() / 15.0
    c12t = c12s[-1] + numpy.random.randn() / 15.0
    c44t = c44s[-1] + numpy.random.randn() / 15.0

    print "log likelihood: ", llogp[-1]
    print "Parameters: ", c11t, c12t, c44t
    print ""

    continue

    #c11t -= dlpdc11 * 0.00001
    #c12t -= dlpdc12 * 0.00001
    #c44t -= dlpdc44 * 0.00001

    print "New parameters: ", c11t, c12t, c44t
    print ""

    c11t = min(5., max(1.0, c11t))
    c12t = min(3.0, max(.5, c12t))
    c44t = min(2.0, max(.25, c44t))

    print "New parameters: ", c11t, c12t, c44t
    print ""

#%%young = 5
freqs = numpy.array([109.076,
136.503,
144.899,
184.926,
188.476,
195.562,
199.246,
208.460,
231.220,
232.630,
239.057,
241.684,
242.159,
249.891,
266.285,
272.672,
285.217,
285.670,
288.796,
296.976,
301.101,
303.024,
305.115,
305.827,
306.939,
310.428,
318.000,
319.457,
322.249,
323.464])

freqs = numpy.array([71.25925,
75.75875,
86.478,
89.947375,
111.150125,
112.164125,
120.172125,
127.810375,
128.6755,
130.739875,
141.70025,
144.50375,
149.40075,
154.35075,
156.782125,
157.554625,
161.0875,
165.10325,
169.7615,
173.44925,
174.11675,
174.90625,
181.11975,
182.4585,
183.98625,
192.68125,
193.43575,
198.793625,
201.901625,
205.01475])

eigs = (freqs * numpy.pi * 2000) ** 2 / 1e11
#%%

mu = eigs

llogp = []
youngs = []
poissons = []

c11t, c12t, c44t = 2.2,  2.68,  1.20630875# 1.24, .934, 0.4610#1.685, 0.7928, 0.4459#2, 1.0, 1
y =  0.10176695
a = 0.19991014
b = -0.30910792
c = -0.21779237
#0.2, 0.1, 0.15
c11s = []
c12s = []
c44s = []
ys = []
as_ = []
bs_ = []
cs_ = []

#%%

current_q = numpy.array([c11t, c12t, c44t, y, a, b, c])#c11t, c12t, c44t, a, b, c
L = 50
epsilon = 0.002
#for ii in range(2000):
ii = 0
qs = []
accepts = []

#%%
def UgradU(q):
    c11t, c12t, c44t, y, a, b, c = q#
    #c12t = -(c12tf * c44t * 2.0 - c11t)
    #y, a, b, c = 0.17362318, 0.19585363, -0.2803732, -0.09701904
    c22 = c11t
    c33 = c11t

    c13 = c12t
    c23 = c12t

    c55 = c44t
    c66 = c44t

    D = numpy.array([[c11t, c12t, c13, 0, 0, 0],
                     [c12t, c22, c23, 0, 0, 0],
                     [c13, c23, c33, 0, 0, 0],
                     [0, 0, 0, c44t, 0, 0],
                     [0, 0, 0, 0, c55, 0],
                     [0, 0, 0, 0, 0, c66]])

    D, Dd, K_ = fder(D, a, b, c)

    dDda = Dd[:, :, 0]
    dDdb = Dd[:, :, 1]
    dDdc = Dd[:, :, 2]

    tmp = time.time()
    dKda, _ = assemble(dDda)
    dKdb, _ = assemble(dDdb)
    dKdc, _ = assemble(dDdc)

    dKdc11, _ = assemble(K_.dot(dDdc11.dot(K_.T)))
    dKdc12, _ = assemble(K_.dot(dDdc12.dot(K_.T)))
    dKdc44, _ = assemble(K_.dot(dDdc44.dot(K_.T)))
    #print "Assembly: ", time.time() - tmp

    Kt, Mt = assemble(D)

    #tmp = time.time()
    eigst, evecst = scipy.sparse.linalg.eigsh(Kt, 30 + nrbm, M = Mt, sigma = 0.0)
    #print "Eigs: ", time.time() - tmp

    eigst = eigst[6:]
    evecst = evecst[:, 6:]

    #print eigst

    #print list(zip(eigst, eigs))

    #t = 1 + (mu - eigst)**2 / y**2

    dlda = numpy.array([evecst[:, i].T.dot(dKda.dot(evecst[:, i])) for i in range(evecst.shape[1])])
    dldb = numpy.array([evecst[:, i].T.dot(dKdb.dot(evecst[:, i])) for i in range(evecst.shape[1])])
    dldc = numpy.array([evecst[:, i].T.dot(dKdc.dot(evecst[:, i])) for i in range(evecst.shape[1])])
    dldc11 = numpy.array([evecst[:, i].T.dot(dKdc11.dot(evecst[:, i])) for i in range(evecst.shape[1])])
    dldc12 = numpy.array([evecst[:, i].T.dot(dKdc12.dot(evecst[:, i])) for i in range(evecst.shape[1])])
    dldc44 = numpy.array([evecst[:, i].T.dot(dKdc44.dot(evecst[:, i])) for i in range(evecst.shape[1])])

    dlpdl = (mu - eigst) / y ** 2
    dlpdy = sum((-y ** 2 + (eigst - mu) **2) / y ** 3)
    #dlpdl = 2 * (mu - eigst) / (t * y**2)
    #dlpdy = sum(numpy.pi * t * ((2 * (mu - eigst) ** 2) / (numpy.pi * t**2 * y**4) - 1 / (numpy.pi * t * y**2)) * y)

    dlpdl = numpy.array(dlpdl)

    dlpdc11 = dlpdl.dot(dldc11)
    dlpdc12 = dlpdl.dot(dldc12)
    dlpdc44 = dlpdl.dot(dldc44)
    dlpda = dlpdl.dot(dlda)
    dlpdb = dlpdl.dot(dldb)
    dlpdc = dlpdl.dot(dldc)

    logp = sum(0.5 * (-((eigst - mu) **2 / y**2) + numpy.log(1.0 / (2 * numpy.pi)) - 2 * numpy.log(y)))
    #logp = sum(numpy.log(1 / (numpy.pi *  (1 + (mu - eigst)**2 / y**2) * y)))
    #E^(-((-u + x)^2/(2 s^2)))/(Sqrt[2 \[Pi]] Sqrt[s^2])

    return -logp, -numpy.array([dlpdc11, dlpdc12, dlpdc44, dlpdy, dlpda, dlpdb, dlpdc])## dlda, dldb, dldc])#

while True:#len(c11s) < 500:
    q = current_q
    p = numpy.random.randn(len(q)) # independent standard normal variates

    current_p = p
    # Make a half step for momentum at the beginning
    U, gradU = UgradU(q)
    p = p - epsilon * gradU / 2

    # Alternate full steps for position and momentum
    for i in range(L):
        # Make a full step for the position
        q = q + epsilon * p

        # Make a full step for the momentum, except at end of trajectory
        if i != L - 1:
            U, gradU = UgradU(q)
            p = p - epsilon * gradU

        #print "New q, H: ", q, U + sum(p ** 2) / 2, U, sum(p ** 2) / 2

    U, gradU = UgradU(q)
    # Make a half step for momentum at the end.
    p = p - epsilon * gradU / 2
    # Negate momentum at end of trajectory to make the proposal symmetric
    p = -p
    # Evaluate potential and kinetic energies at start and end of trajectory
    UC, gradUC = UgradU(current_q)
    current_U = UC
    current_K = sum(current_p ** 2) / 2
    proposed_U = U
    proposed_K = sum(p ** 2) / 2

    # Accept or reject the state at end of trajectory, returning either
    # the position at the end of the trajectory or the initial position
    dQ = current_U - proposed_U + current_K - proposed_K

    qs.append(q)

    if numpy.random.rand() < min(1.0, numpy.exp(dQ)):
        current_q = q # accept

        accepts.append(len(qs) - 1)
        #c11s.append(current_q[0])
        #c12s.append(current_q[1])
        #c44s.append(current_q[2])
        #ys.append(current_q[3])
        #as_.append(current_q[3])
        #bs_.append(current_q[4])
        #cs_.append(current_q[5])

        print "Accepted ({0} accepts so far): {1}".format(len(accepts), current_q)
        #epsilon *= 1.2
    else:
        print "Rejected: ", current_q
        #epsilon /= 1.4

    print "Energy change ({0} samples, {1} accepts): ".format(ii + 1, len(accepts)), min(1.0, numpy.exp(dQ)), dQ, current_U, proposed_U, current_K, proposed_K
    print "Epsilon: ", epsilon

    ii = ii + 1
#%%
s12 = -3e-12

s44 = (8e-12)

s11 = 8e-12

s44 = 1.8e-12

s44 = 7.8e-12

c11 = (s11 + s12) / ((s11 - s12) * (s11 + 2 * s12))

c12 = -s12 / ((s11 - s12) * (s11 + 2 * s12))

c44 = 1.0 / s44

c11 /= 1e11

c12 /= 1e11

c44 /= 1e11

c11
Out[26]: 2.272727272727273

c12
Out[27]: 1.363636363636364

c44
Out[28]: 1.282051282051282
#%%
import pickle

f = open('september_7_stuff.pkl', 'w')
pickle.dump((qs, accepts), f)
f.close()
#%%
c11s, c12s, c44s, ys, as_, bs_, cs_ = [numpy.array(a) for a in zip(*[qs[i] for i in accepts])]

#%%
qs2 = list(qs)
accepts2 = list(accepts)
#%%
logps = []
for i, q in enumerate(qs):
    logp, _ = UgradU(q)
    logps.append(logp)
    print "{0} / {1}, {2}, {3}".format(i + 1, len(qs), q, logp)
#%%
logp, _, eigst = UgradU(qs[-1])
#(freqs * numpy.pi * 2000) ** 2 / 1e11
print "Computed, Real, Difference (Khz)"
for real, computed in zip(eigs, eigst):
    computed = numpy.sqrt(computed * 1e11) / (2 * numpy.pi * 1000)
    real = numpy.sqrt(real * 1e11) / (2 * numpy.pi * 1000)
    print "{0:10.4f} {1:10.4f} {2:10.4f}".format(computed, real, computed - real)
#%%
import matplotlib.pyplot as plt

plt.plot(c11s)
plt.title('c11')
plt.show()
plt.plot(c12s)
plt.title('c12')
plt.show()
plt.plot(c44s)
plt.title('c44')
plt.show()
plt.plot(ys)
plt.title('ys')
plt.show()
plt.plot(as_)
plt.title('3rd rotation (about x)')
plt.show()
plt.plot(bs_)
plt.title('2nd rotation (about y)')
plt.show()
plt.plot(cs_)
plt.title('1st rotation (about z)')
plt.show()
plt.plot(logps)
plt.title('-log probability of model (smaller is more likely)')
plt.show()

#%%
for minv, value, maxv in zip(numpy.sqrt((eigs - 0.35357) * 1e11) / (numpy.pi * 2000), freqs, numpy.sqrt((eigs + 0.35357) * 1e11) / (numpy.pi * 2000)):
    print "[{0:4.2f} {1:4.2f} {2:4.2f}]".format(minv, value, maxv)

#%%

import seaborn
import pandas
import matplotlib.pyplot as plt

df = pandas.DataFrame({'c11' : c11s[-3000:], 'c12' : c12s[-3000:], 'c44' : c44s[-3000:], 'y' : ys[-3000:], 'a' : as_[-3000:], 'b' : bs_[-3000:], 'c' : cs_[-3000:]})

seaborn.pairplot(df)
plt.gcf().set_size_inches((12, 8))
plt.show()

import scipy.stats

g = seaborn.PairGrid(df)
g.map_diag(plt.hist)
g.map_offdiag(seaborn.kdeplot, n_levels = 6);
plt.gcf().set_size_inches((12, 8))
plt.show()
#%%
for name, d in [('c11', c11s), ('c12', c12s), ('c44', c44s), ('a', as_), ('b', bs_), ('c', cs_)]:
    seaborn.distplot(d[-3000:], kde = False, fit = scipy.stats.norm)
    plt.title("Dist. {0} w/ mean {1:0.4f} and std. {2:0.4f}".format(name, numpy.mean(d[-3000:]), numpy.std(d[-3000:])))
    plt.gcf().set_size_inches((5, 4))
    plt.show()
#%%
plt.plot(c11s, c12s)
plt.ylabel('c12')
plt.xlabel('c11')
plt.title('Full trajectory c11 vs. c12')
plt.show()
plt.plot(c11s, c44s)
plt.ylabel('c44')
plt.xlabel('c11')
plt.title('Full trajectory c11 vs. c44')
plt.show()
plt.plot(c12s, c44s)
plt.ylabel('c44')
plt.xlabel('c12')
plt.title('Full trajectory c12 vs. c44')
plt.show()
#%%
import pickle
f = open('posterior.pickle', 'w')
pickle.dump((c11s, c12s, c44s), f)
f.close()