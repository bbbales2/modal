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

import numpy

import time

aux = "eig.scipy,method:'eigh',tol:1e-5,maxiter:1000".split(',')
kwargs = {}
for option in aux[1:]:
    key, val = option.split(':')
    kwargs[key.strip()] = eval(val)
eig_conf = Struct(name='evp', kind=aux[0], **kwargs)

output.level = -1

density = 8500.0

dims = numpy.array([0.007, 0.008, 0.013])
dim = len(dims)

shape = [5, 5, 5]

centre = numpy.array([0.0, 0.0, 0.0])

order = 1

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

#c11, c12, c44 = 2.30887372458, 0.778064244563, 0.757576236829
c11, c12, c44 = 3.0, 1.5, 0.75
#c11 = 3.00
#c12 = 1.5
#c44 = 0.75

c22 = c11
c33 = c11

c13 = c12
c23 = c12

c55 = c44
c66 = c44

D = numpy.array([[c11, c12, c13, 0, 0, 0],
                     [c12, c22, c23, 0, 0, 0],
                     [c13, c23, c33, 0, 0, 0],
                     [0, 0, 0, c44, 0, 0],
                     [0, 0, 0, 0, c55, 0],
                     [0, 0, 0, 0, 0, c66]])

dDdc11 = numpy.array([[1, 0, 0, 0, 0, 0],
                     [0, 1, 0, 0, 0, 0],
                     [0, 0, 1, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0]])

dDdc12 = numpy.array([[0, 1, 1, 0, 0, 0],
                     [1, 0, 1, 0, 0, 0],
                     [1, 1, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0]])

dDdc44 = numpy.array([[0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 1, 0, 0],
                     [0, 0, 0, 0, 1, 0],
                     [0, 0, 0, 0, 0, 1]])

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
    eigs, svecs = eig_solver(K, M, 30 + nrbm, eigenvectors=True)
except sla.ArpackNoConvergence as ee:
    eigs2 = ee.eigenvalues
    svecs = ee.eigenvectors
    output('only %d eigenvalues converged!' % len(eigs))
print "Eigenvalue solve: ", time.time() - tmp

output('%d eigenvalues converged (%d ignored as rigid body modes)' %
    (len(eigs), nrbm))

eigs = eigs[nrbm:]
svecs = svecs[:, nrbm:]

omegas = nm.sqrt(eigs)
freqs = omegas / (2 * nm.pi)

output('number |         eigenvalue |  angular frequency '
           '|          frequency')
for ii, eig in enumerate(eigs):
    output('%6d | %17.12e | %17.12e | %17.12e'
               % (ii + 1, eig, omegas[ii], freqs[ii]))

#%%young = 5
import matplotlib.pyplot as plt

poisson = 0.4

mu = eigs#[:12]
sigma = 0.1

llogp = []
youngs = []
poissons = []
#2.30887372458 0.778064244563 0.757576236829
c11t, c12t, c44t = 2, 1, 1
#c11t, c12t, c44t = 2.82691758885, 1.2418749993, 0.74314183455
#2.98075401161 1.47430936292 0.756130056282
#c11t, c12t, c44t = 2.87594152737, 1.35677794317, 0.7530964877#2.72837807963, 1.21195198294, 0.746315594831
#2.28912477153, 0.794740110352, 0.849348201626
#c11t = 2.00
#c12t = 1.0
#c44t = 1.0

c11s = []
c12s = []
c44s = []

for i in range(1000):
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

    eigst, evecst = eig_solver(Kt, Mt, 30 + nrbm, eigenvectors=True)

    eigst = eigst[6:]
    evecst = evecst[:, 6:]

    #print list(zip(eigst, eigs))

    dldc11 = numpy.array([evecst[:, i].T.dot(dKdc11.dot(evecst[:, i])) for i in range(evecst.shape[1])])
    dldc12 = numpy.array([evecst[:, i].T.dot(dKdc12.dot(evecst[:, i])) for i in range(evecst.shape[1])])
    dldc44 = numpy.array([evecst[:, i].T.dot(dKdc44.dot(evecst[:, i])) for i in range(evecst.shape[1])])

    dlpdl = []
    eps = 0.5
    for val in (eigst - mu):
        if abs(val) > eps:
            if val > 0.0:
                dlpdl.append(1.0)
            else:
                dlpdl.append(-1.0)
        else:
            dlpdl.append(10 * val)

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

    llogp.append(sum(numpy.abs(eigst - mu)))
    #llogp.append(-sum(0.5 * (mu - eigst)[:6]**2 / sigma**2))# - 0.5 * (0.3 - poisson)**2 / 0.1**2)

    c11s.append(c11t)
    c12s.append(c12t)
    c44s.append(c44t)

    print "log likelihood: ", llogp[-1]
    print "Parameters: ", c11t, c12t, c44t
    print ""

    c11t -= dlpdc11 * 0.00001
    c12t -= dlpdc12 * 0.00001
    c44t -= dlpdc44 * 0.00001

    print "New parameters: ", c11t, c12t, c44t
    print ""

    c11t = min(5., max(1.0, c11t))
    c12t = min(3.0, max(.5, c12t))
    c44t = min(2.0, max(.25, c44t))

    print "New parameters: ", c11t, c12t, c44t
    print ""
