#%%
from __future__ import absolute_import
import sys
import six
from six.moves import range
sys.path.append('.')
from optparse import OptionParser

import numpy as nm
import scipy.sparse.linalg as sla
import matplotlib.pyplot as plt

sys.path.append('/home/bbales2/sfepy/')

from sfepy.base.base import assert_, output, Struct
from sfepy.discrete import (FieldVariable, Material, Integral, Integrals,
                            Equation, Equations, Problem)
from sfepy.discrete.fem import Mesh, FEDomain, Field
from sfepy.terms import Term
from sfepy.discrete.conditions import Conditions, EssentialBC
from sfepy.mechanics.matcoefs import stiffness_from_youngpoisson
from sfepy.mesh.mesh_generators import gen_block_mesh
from sfepy.solvers import Solver
from sfepy.base.base import output
output.set_output(quiet=True)
import numpy
import scipy

aux = "eig.scipy,method:'eigh',tol:1e-7,maxiter:10000".split(',')
kwargs = {}
for option in aux[1:]:
    key, val = option.split(':')
    kwargs[key.strip()] = eval(val)
eig_conf = Struct(name='evp', kind=aux[0], **kwargs)

dims = nm.array([1.0, 1.5], dtype=nm.float64)
dim = len(dims)

centre = nm.array([0.0, 0.0], dtype=nm.float64)[:dim]
shape = nm.array([11, 16], dtype=nm.int32)[:dim]

mesh = gen_block_mesh(dims, shape, centre, name='mesh')

eig_solver = Solver.any_from_conf(eig_conf)

# Build the problem definition.
domain = FEDomain('domain', mesh)

bbox = domain.get_mesh_bounding_box()
min_coor, max_coor = bbox[:, -1]
eps = 1e-8 * (max_coor - min_coor)
ax = 'xyz'[:dim][-1]

omega = domain.create_region('Omega', 'all')
bottom = domain.create_region('Bottom', 'vertices in (%s < %.10f)' % (ax, min_coor + eps), 'facet')
bottom_top = domain.create_region('BottomTop', 'r.Bottom +v vertices in (%s > %.10f)' % (ax, max_coor - eps), 'facet')

field = Field.from_args('fu', nm.float64, 'vector', omega, approx_order=1)

u = FieldVariable('u', 'unknown', field)
v = FieldVariable('v', 'test', field, primary_var_name = 'u')

mtx_d = stiffness_from_youngpoisson(dim, 6.80, 0.36)
young = 6.8
poisson = 0.36

mu0 = young / (2.0 * (1.0 + poisson))
mu1 = mu0 * 1.00001
lambda0 = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
lambda1 = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson)) * 1.00001

D0 = numpy.array([[lambda0 + 2 * mu0, lambda0, 0.0],
                 [lambda0, lambda0 + 2 * mu0, 0.0],
                 [0.0, 0.0, mu0]])

D1 = numpy.array([[lambda0 + 2 * mu1, lambda0, 0.0],
                 [lambda0, lambda0 + 2 * mu1, 0.0],
                 [0.0, 0.0, mu1]])

D2 = numpy.array([[lambda0 + 2 * mu0, lambda0, 0.0],
                 [lambda0, lambda0 + 2 * mu0, 0.0],
                 [0.0, 0.0, mu0]])

D3 = numpy.array([[lambda1 + 2 * mu0, lambda1, 0.0],
                 [lambda1, lambda1 + 2 * mu0, 0.0],
                 [0.0, 0.0, mu0]])

Ks = []
Ms = []
for D in [D0, D1, D2, D3]:
    m = Material('m', D = D, rho = 2700.0)

    integral = Integral('i', order=2)

    t1 = Term.new('dw_lin_elastic(m.D, v, u)', integral, omega, m=m, v=v, u=u)
    t2 = Term.new('dw_volume_dot(m.rho, v, u)', integral, omega, m=m, v=v, u=u)
    eq1 = Equation('stiffness', t1)
    eq2 = Equation('mass', t2)
    lhs_eqs = Equations([eq1, eq2])

    pb = Problem('modal', equations = lhs_eqs)

    pb.time_update()
    n_rbm = dim * (dim + 1) / 2

    pb.update_materials()

    # Assemble stiffness and mass matrices.
    mtx_k = eq1.evaluate(mode='weak', dw_mode='matrix', asm_obj=pb.mtx_a)
    mtx_m = mtx_k.copy()
    mtx_m.data[:] = 0.0
    mtx_m = eq2.evaluate(mode='weak', dw_mode='matrix', asm_obj=mtx_m)

    Ks.append(mtx_k)
    Ms.append(mtx_m)

dKdmu = (Ks[1] - Ks[0]) / (mu1 - mu0)
dKdlambda = (Ks[3] - Ks[2]) / (lambda1 - lambda0)
print dKdmu
#print Ks[-1]
#%%

m = Material('m', D = mtx_d, rho = 2700.0)

integral = Integral('i', order=2)

t1 = Term.new('dw_lin_elastic(m.D, v, u)', integral, omega, m=m, v=v, u=u)
t2 = Term.new('dw_volume_dot(m.rho, v, u)', integral, omega, m=m, v=v, u=u)
eq1 = Equation('stiffness', t1)
eq2 = Equation('mass', t2)
lhs_eqs = Equations([eq1, eq2])

pb = Problem('modal', equations = lhs_eqs)

pb.time_update()
n_rbm = dim * (dim + 1) / 2

pb.update_materials()

# Assemble stiffness and mass matrices.
mtx_k = eq1.evaluate(mode='weak', dw_mode='matrix', asm_obj=pb.mtx_a)
mtx_m = mtx_k.copy()
mtx_m.data[:] = 0.0
mtx_m = eq2.evaluate(mode='weak', dw_mode='matrix', asm_obj=mtx_m)

try:
    eigs, svecs = eig_solver(mtx_k, mtx_m, 8, eigenvectors = True)
except sla.ArpackNoConvergence as ee:
    eigs = ee.eigenvalues
    svecs = ee.eigenvectors
    print 'only %d eigenvalues converged!' % len(eigs)

print eigs
#%%
K0 = Ks[0]
K1 = Ks[1]

eigs0, evecs0 = scipy.sparse.linalg.eigsh(K0, k = 10, M = Ms[0], which = 'SM')
eigs1, evecs1 = scipy.sparse.linalg.eigsh(K1, k = 10, M = Ms[1], which = 'SM')

print eigs0[0:10]
print eigs1[0:10]

dydk = (eigs1[0:10] - eigs0[0:10]) / (mu1 - mu0)

print dydk[0:7]

dydk2 = numpy.array([evecs0[:, i].T.dot(dKdmu.dot(evecs0[:, i])) for i in range(10)])

print dydk2[0:7]

#dydk3 = numpy.array([evecs1[i][0]**2 for i in range(10)])

#print dydk3[0:7]
#%%
def evalValAndDeriv(D):
    m = Material('m', D = D, rho = 2700.0)

    integral = Integral('i', order=2)

    t1 = Term.new('dw_lin_elastic(m.D, v, u)', integral, omega, m=m, v=v, u=u)
    t2 = Term.new('dw_volume_dot(m.rho, v, u)', integral, omega, m=m, v=v, u=u)
    eq1 = Equation('stiffness', t1)
    eq2 = Equation('mass', t2)
    lhs_eqs = Equations([eq1, eq2])

    pb = Problem('modal', equations = lhs_eqs)

    pb.time_update()
    n_rbm = dim * (dim + 1) / 2

    pb.update_materials()

    # Assemble stiffness and mass matrices.
    mtx_k = eq1.evaluate(mode='weak', dw_mode='matrix', asm_obj=pb.mtx_a)
    mtx_m = mtx_k.copy()
    mtx_m.data[:] = 0.0
    mtx_m = eq2.evaluate(mode='weak', dw_mode='matrix', asm_obj=mtx_m)

    eigs0, evecs0 = scipy.sparse.linalg.eigsh(mtx_k, k = 10, M = mtx_m, which = 'SM')

    eigs = eigs0[3:]
    evecs = evecs0[:, 3:]

    dydmu = numpy.array([evecs[:, i].T.dot(dKdmu.dot(evecs[:, i])) for i in range(evecs.shape[1])])
    dydlambda = numpy.array([evecs[:, i].T.dot(dKdlambda.dot(evecs[:, i])) for i in range(evecs.shape[1])])

    return eigs, dydmu, dydlambda
#%%
N = 10

young = 6.8
poisson = 0.36

data = []
for i in range(N):
    young = min(10.0, max(2.0, numpy.random.randn() / 2.0 + 6.8))
    poisson = min(0.49, max(0.10, numpy.random.randn() / 10.0 + 0.36))

    mu0 = young / (2.0 * (1.0 + poisson))
    lambd = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
    D = numpy.array([[lambd + 2 * mu0, lambd, 0.0],
                                 [lambd, lambd + 2 * mu0, 0.0],
                                 [0.0, 0.0, mu0]])

    vals, derivmu, derivlambda = evalValAndDeriv(D)

    print D, vals

    data.append(vals)

data = numpy.array(data)
#%%
young = 5
poisson = 0.4

mu = numpy.mean(data, axis = 0)
sigma = numpy.std(data, axis = 0)

for i in range(50):
    mu0 = young / (2.0 * (1.0 + poisson))
    lambd = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))

    D = numpy.array([[lambd + 2 * mu0, lambd, 0.0],
                                 [lambd, lambd + 2 * mu0, 0.0],
                                 [0.0, 0.0, mu0]])

    vals, derivmu, derivlambda = evalValAndDeriv(D)

    dlpdmu = derivmu.dot((mu - vals) / sigma)
    dlpdlambda = derivlambda.dot((mu - vals) / sigma)

    dmudyoung = 1 / (2 * (1 + poisson))
    dmudpoisson = -young / (2 * (1 + poisson)**2)
    dlambdadyoung = poisson / ((1 + poisson) * (1 - 2 * poisson))
    dlambdadpoisson = (young * ((1 + poisson) * (1 - 2 * poisson)) + (1 + 4 * poisson) * young * poisson) / (((1 + poisson) * (1 - 2 * poisson))**2)

    dlpdyoung = dlpdmu * dmudyoung + dlpdlambda * dlambdadyoung
    dlpdpoisson = dlpdmu * dmudpoisson + dlpdlambda * dlambdadpoisson + (0.3 - poisson) / 1.0

    print "log likelihood: ", sum(0.5 * (mu - vals)**2 / sigma)
    print "young's modulus: ", young, dlpdyoung
    print "poisson: ", poisson, dlpdpoisson
    print ""

    young += dlpdyoung
    poisson += dlpdpoisson * 0.1

    print "New young's modulus: ", young
    print "new poisson ratio: ", poisson
    print ""

#%%
import matplotlib.pyplot as plt

eigs, eigvector = scipy.sparse.linalg.eigsh(mtx_k, M = mtx_m, k = 50, which = 'SM')

print eigs

#%%

A = scipy.sparse.linalg.spsolve(mtx_m, mtx_k)

#%%
eigs, eigvector = scipy.sparse.linalg.eigs(scipy.sparse.linalg.spsolve(mtx_m, mtx_k), k = 50, which = 'SM')

print eigs