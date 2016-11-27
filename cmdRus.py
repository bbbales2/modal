#!/usr/bin/env python

import argparse
import numpy
import sympy
import os
import pickle
from mpi4py import MPI

import rus

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

parser = argparse.ArgumentParser(description = 'Do resonance ultrasound spectroscopy inversion!')

parser.add_argument('density', type = float, help = 'Density of material (kg/m^3)')
parser.add_argument('X', type = float, help = 'X dimension of sample (m)')
parser.add_argument('Y', type = float, help = 'Y dimension of sample (m)')
parser.add_argument('Z', type = float, help = 'Z dimension of sample (m)')
#parser.add_argument('symmetry', choices = ['cubic'], default = 'cubic')
parser.add_argument('mode_file', type = str, help = 'File containing measured resonance modes one per line in khz')
parser.add_argument('--tol', type = float, default = 1e-3, help = 'Relative accuracy of Rayleigh-Ritz approximation')
parser.add_argument('--epsilon', type = float, default = 0.0001, help = 'Timestep of ODE integrator in Hamiltonian Monte Carlo solver')
parser.add_argument('-L', type = int, default = 50, help = 'Number of ODE timesteps in each HMC step')
parser.add_argument('--debug', help = "Print debug information", action = 'store_true')
parser.add_argument('--rotations', help = "Estimate orientation parameters", action = 'store_true')
parser.add_argument('--progress_folder', type = str, help = 'Folder to save progress in')
parser.add_argument('--output_file', type = str, help = 'File to save output in. Unlike the progress file this will be completely re-written each execution')
parser.add_argument('initial', type = float, nargs = 4, help = """Initial parameter guesses, 'c11 anisotropic_ratio c44 std' for cubic, ex: 2.0 1.0 1.0 5.0""")

args = parser.parse_args()

data = numpy.loadtxt(args.mode_file)

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

data = []
with open(args.mode_file, 'r') as f:
    for line in f.readlines():
        if is_number(line):
            data.append(float(line))

if rank == 0:
    if os.path.exists(args.progress_folder) and not os.path.isdir(args.progress_folder):
        raise Exception("{0} already exists but is not a directory".format(args.progress_folder))

    if not os.path.exists(args.progress_folder):
        print "No Checkpoint folder found, creating now"
        os.mkdir(args.progress_folder)
    else:
        print "Checkpoint folder found"

comm.barrier()

checkf = os.path.join(args.progress_folder, str(rank))

def checkpoint():
    with open(checkf, 'w') as f:
        pickle.dump(hmc, f)

    comm.barrier()

def sample():
    hmc.set_timestepping(epsilon = args.epsilon, L = args.L)

    hmc.sample(steps = 1, debug = args.debug, silent = True)

    print_output(1)

    comm.barrier()

def print_header():
    labels, values = hmc.format_samples(-1)

    out = ['rank'] + labels

    print ", ".join(out)

def print_output(lastN = -1):
    labels, values = hmc.format_samples(lastN)

    for i in range(max(len(values[0]), lastN)):
        out = [str(rank)]
        for idx in numpy.argsort(labels):
            out.append(str(values[idx][i]))

        print ", ".join(out)

if os.path.exists(checkf):
    print "{0} loading from file".format(rank)

    with open(checkf) as f:
        hmc = pickle.load(f)

    comm.barrier()

    if rank == 0:
        print_header()

    comm.barrier()

    print_output()
else:
    print "{0} starting new chain".format(rank)

    c11, anisotropic, c44 = sympy.symbols('c11 anisotropic c44')

    c12 = sympy.sympify("-(c44 * 2.0 / anisotropic - c11)") # The c11 and c44 and anisotropic are the same as above

    C = sympy.Matrix([[c11, c12, c12, 0, 0, 0],
                      [c12, c11, c12, 0, 0, 0],
                      [c12, c12, c11, 0, 0, 0],
                      [0, 0, 0, c44, 0, 0],
                      [0, 0, 0, 0, c44, 0],
                      [0, 0, 0, 0, 0, c44]])

    hmc = rus.HMC(tol = args.tol, # Accuracy of Rayleigh-Ritz approximation
                  density = args.density, X = args.X, Y = args.Y, Z = args.Z,
                  resonance_modes = data, # List of resonance modes
                  stiffness_matrix = C, # Stiffness matrix
                  parameters = { c11 : args.initial[0], anisotropic : args.initial[1], c44 : args.initial[2], 'std' : args.initial[3] }, # Parameters
                  rotations = args.rotations)

    hmc.set_labels({ c11 : 'c11', anisotropic : 'a', c44 : 'c44', 'std' : 'std' })
    hmc.set_timestepping(epsilon = args.epsilon / 10.0, L = args.L)

    sample()
    
    hmc.set_timestepping(epsilon = args.epsilon, L = args.L)
    print "Chain started, saving progress and starting regular sampling"

    comm.barrier()

    if rank == 0:
        print_header()

    comm.barrier()
    
checkpoint()

while 1:
    sample()

    checkpoint()
