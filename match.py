#%%
"""
Modal analysis of a linear elastic block in 2D or 3D.

The dimension of the problem is determined by the length of the vector
in ``--dims`` option.

Optionally, a mesh file name can be given as a positional argument. In that
case, the mesh generation options are ignored.

The default material properties correspond to aluminium in the following units:

- length: m
- mass: kg
- stiffness / stress: Pa
- density: kg / m^3

Examples
--------

- Run with the default arguments, show results (color = strain)::

    python examples/linear_elasticity/modal_analysis.py --show

- Fix bottom surface of the domain, show 9 eigen-shapes::

    python examples/linear_elasticity/modal_analysis.py -b cantilever -n 9 --show

- Increase mesh resolution::

    python examples/linear_elasticity/modal_analysis.py -s 31,31 -n 9 --show

- Use 3D domain::

    python examples/linear_elasticity/modal_analysis.py -d 1,1,1 -c 0,0,0 -s 8,8,8 --show

- Change the eigenvalue problem solver to LOBPCG::

    python examples/linear_elasticity/modal_analysis.py --solver="eig.scipy_lobpcg,i_max:100,largest:False" --show

  See :mod:`sfepy.solvers.eigen` for available solvers.
"""
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

usage = '%prog [options] [filename]\n' + __doc__.rstrip()

helps = {
    'dims' :
    'dimensions of the block [default: %default]',
    'centre' :
    'centre of the block [default: %default]',
    'shape' :
    'numbers of vertices along each axis [default: %default]',
    'bc_kind' :
    'kind of Dirichlet boundary conditions on the bottom and top surfaces,'
    ' one of: free, cantilever, fixed [default: %default]',
    'axis' :
    'the axis index of the block that the bottom and top surfaces are related'
    ' to [default: %default]',
    'young' : "the Young's modulus [default: %default]",
    'poisson' : "the Poisson's ratio [default: %default]",
    'density' : "the material density [default: %default]",
    'order' : 'displacement field approximation order [default: %default]',
    'n_eigs' : 'the number of eigenvalues to compute [default: %default]',
    'ignore' : 'if given, the number of eigenvalues to ignore (e.g. rigid'
    ' body modes); has precedence over the default setting determined by'
    ' --bc-kind [default: %default]',
    'solver' : 'the eigenvalue problem solver to use. It should be given'
    ' as a comma-separated list: solver_kind,option0:value0,option1:value1,...'
    ' [default: %default]',
    'show' : 'show the results figure',
}

if True:
    aux = "eig.scipy,method:'eigh',tol:1e-5,maxiter:1000".split(',')
    kwargs = {}
    for option in aux[1:]:
        key, val = option.split(':')
        kwargs[key.strip()] = eval(val)
    eig_conf = Struct(name='evp', kind=aux[0], **kwargs)

    #output.level += 1
    #for key, val in six.iteritems(kwargs):
    #    output('%s: %r' % (key, val))
    output.level = -1

    density = 8500.0

    youngs = 200e9
    poisson = 0.3

    dims = numpy.array([0.007, 0.008, 0.013])
    dim = len(dims)

    shape = [5, 5, 5]

    centre = numpy.array([0.0, 0.0, 0.0])

    order = 2

    tmp = time.time()
    mesh = gen_block_mesh(dims, shape, centre, name='mesh')
    print "Mesh generation: ", time.time() - tmp

    axis = -1

    tmp = time.time()
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

    mtx_d = stiffness_from_youngpoisson(dim, youngs, poisson)

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
    print "Set up problem: ", time.time() - tmp

    tmp = time.time()
    # Assemble stiffness and mass matrices.
    mtx_k = eq1.evaluate(mode='weak', dw_mode='matrix', asm_obj=pb.mtx_a)
    mtx_m = mtx_k.copy()
    mtx_m.data[:] = 0.0
    mtx_m = eq2.evaluate(mode='weak', dw_mode='matrix', asm_obj=mtx_m)
    print "Assemble matrices: ", time.time() - tmp

    tmp = time.time()
    try:
        eigs, svecs = eig_solver(mtx_k, mtx_m, 30 + n_rbm,
                                 eigenvectors=True)

    except sla.ArpackNoConvergence as ee:
        eigs = ee.eigenvalues
        svecs = ee.eigenvectors
        output('only %d eigenvalues converged!' % len(eigs))
    print "Eigenvalue solve: ", time.time() - tmp

    output('%d eigenvalues converged (%d ignored as rigid body modes)' %
           (len(eigs), n_rbm))

    eigs = eigs[n_rbm:]
    svecs = svecs[:, n_rbm:]

    omegas = nm.sqrt(eigs)
    freqs = omegas / (2 * nm.pi)

    output('number |         eigenvalue |  angular frequency '
           '|          frequency')
    for ii, eig in enumerate(eigs):
        output('%6d | %17.12e | %17.12e | %17.12e'
               % (ii + 1, eig, omegas[ii], freqs[ii]))
