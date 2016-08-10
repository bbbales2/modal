#%%
from __future__ import absolute_import
import sys
import six
from six.moves import range
sys.path.append('.')
from optparse import OptionParser

import numpy as nm
import scipy.sparse.linalg as sla

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
import matplotlib.pyplot as plt

eigs, eigvector = scipy.sparse.linalg.eigsh(mtx_k, M = mtx_m, k = 50, which = 'SM')

print eigs