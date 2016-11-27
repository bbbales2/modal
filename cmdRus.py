#!/usr/bin/env python

import argparse
import numpy
import sympy
import os
import pickle
import time
from mpi4py import MPI

import rus

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

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
parser.add_argument('--maxT', type = float, default = 1.5, help = "Maximum parallel tempering temperature")
parser.add_argument('--swap_prob', type = float, default = 0.1, help = "Probability of doing swap step instead of sampling")
parser.add_argument('--checkpoint', type = int, default = 10, help = "Number of iterations between checkpoint")
parser.add_argument('--progress_folder', type = str, help = 'Folder to save progress in')
parser.add_argument('--output_file', type = str, help = 'File to save output in. Unlike the progress file this will be completely re-written each execution')
parser.add_argument('initial', type = float, nargs = 4, help = """Initial parameter guesses, 'c11 anisotropic_ratio c44 std' for cubic, ex: 2.0 1.0 1.0 5.0""")

args = parser.parse_args()

Ts = numpy.linspace(1.0, args.maxT, size)
swap_prob = args.swap_prob

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
    with open(checkf + '.tmp', 'w') as f:
        pickle.dump(hmc, f)

    os.rename(checkf + '.tmp', checkf)

    comm.barrier()

def sample(steps = 1):
    tmp = time.time()
    #hmc.set_timestepping(epsilon = args.epsilon, L = args.L)

    hmc.sample(steps = steps, debug = args.debug, silent = True)

    print_output(1)

    print "Checkpoint (to barrier): ", time.time() - tmp
    comm.barrier()
    print "Checkpoint took: ", time.time() - tmp

def print_header():
    labels, values = hmc.format_samples(-1)

    out = ['rank'] + labels

    print ", ".join(out)

def print_output(lastN = -1):
    labels, values = hmc.format_samples(lastN)

    for i in range(len(values[0])):
        out = [str(rank)]

        for j in range(len(labels)):
            out.append(str(values[j][i]))

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
    print "{0} starting new chain with temperature {1}".format(rank, Ts[rank])

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
                  rotations = args.rotations,
                  T = Ts[rank])

    print args.epsilon, args.rotations, Ts[rank]

    hmc.set_labels({ c11 : 'c11', anisotropic : 'a', c44 : 'c44', 'std' : 'std' })
    hmc.set_timestepping(epsilon = args.epsilon, L = args.L)

    comm.barrier()

    if rank == 0:
        print_header()

    sample()
    sample()
    sample()
    sample()
    
    hmc.set_timestepping(epsilon = args.epsilon * 10, L = args.L)
    print "Chain started, saving progress and starting regular sampling"

    comm.barrier()

    if rank == 0:
        print_header()

    comm.barrier()
    
checkpoint()

loops = 0
while 1:
    comm.barrier()

    if rank == 0:
        r = numpy.random.rand()
    else:
        r = 0

    r = comm.bcast(r, root = 0)

    if r < 1.0 - swap_prob:
        sample()
    else:
        if rank == 0:
            i = numpy.random.randint(0, size - 1)
        else:
            i = 0

        i = comm.bcast(i, root = 0)

        if rank != 0 and (rank == i or rank == i + 1):
            comm.send(hmc.logps[-1], dest = 0)

        swapem = None
        if rank == 0:
            if i == 0:
                logpi = hmc.logps[-1]
            else:
                logpi = comm.recv(source = i)

            logpip = comm.recv(source = i + 1)

            logpi_p = logpi * Ts[i] / Ts[i + 1]
            logpip_p = logpip * Ts[i + 1] / Ts[i]

            ps = min(1.0, numpy.exp(-logpi_p - logpip_p + logpi + logpip))
            
            #print "logpi", logpi
            #print "logpip", logpip
            #print "logpi_p", logpi_p
            #print "logpip_p", logpip_p

            if numpy.random.rand() < ps:
                swapem = True
                #print "Swapem"
            else:
                swapem = False
                #print "Noswap"

            print "Debug: Swapping {0} and {1} with probability {2} (result: {3})".format(i, i + 1, ps, swapem)
            #print ""

        swapem = comm.bcast(swapem, root = 0)

        if swapem and rank in [i, i + 1]:
            if rank == i:
                comm.Send(hmc.current_q, dest = i + 1)
                comm.Send(hmc.current_qr.flatten(), dest = i + 1)
                comm.send(hmc.logps[-1], dest = i + 1)
                comm.send(hmc.accepts[-1], dest = i + 1)
                #print "Send from ", i
            elif rank == i + 1:
                oq = numpy.zeros(hmc.current_q.shape)
                oqr = numpy.zeros(hmc.current_qr.flatten().shape)
                comm.Recv(oq, source = i)
                comm.Recv(oqr, source = i)
                logp = comm.recv(source = i)
                accept = comm.recv(source = i)
                #print oq
                #print "Receive on ", i + 1

            if rank == i + 1:
                comm.Send(hmc.current_q, dest = i)
                comm.Send(hmc.current_qr.flatten(), dest = i)
                comm.send(hmc.logps[-1], dest = i)
                comm.send(hmc.accepts[-1], dest = i)
                #print "Send from ", i + 1
            elif rank == i:
                oq = numpy.zeros(hmc.current_q.shape)
                oqr = numpy.zeros(hmc.current_qr.flatten().shape)
                comm.Recv(oq, source = i + 1)
                comm.Recv(oqr, source = i + 1)
                logp = comm.recv(source = i + 1)
                accept = comm.recv(source = i + 1)
                #print "Receive on ", i

            hmc.current_q = oq
            hmc.current_qr = oqr.reshape(hmc.current_qr.shape)

            if rank == i:
                hmc.logps[-1] = logp * Ts[i + 1] / Ts[i]
            elif rank == i + 1:
                hmc.logps[-1] = logp * Ts[i] / Ts[i + 1]

            hmc.accepts[-1] = accept

        #if rank == 0:
        #    print "Done swapping"
        #comm.barrier()
        #print_output(1)

    loops += 1

    if loops % args.checkpoint == 0:
        checkpoint()
