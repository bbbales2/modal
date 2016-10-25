#%%

import numpy
import pyximport
pyximport.install(reload_support = True)
import polybasis
import scipy
#%%
import time
#%%

import contextlib

# Stolen from http://stackoverflow.com/a/2891805/3769360
@contextlib.contextmanager
def printoptions(*args, **kwargs):
    original = numpy.get_printoptions()
    numpy.set_printoptions(*args, **kwargs)
    yield
    numpy.set_printoptions(**original)

class HMC():
    def __init__(self, N, density, X, Y, Z, resonance_modes, stiffness_matrix, parameters, constrained_positive = None):
        self.C = stiffness_matrix
        self.dC = {}
        self.density = density
        self.X = X
        self.Y = Y
        self.Z = Z
        self.order = {}
        self.initial_conditions = parameters
        self.labels = {}

        self.constrained_positive = set()
        if constrained_positive != None:
            self.constrained_positive = set(constrained_positive)

        self.constrained_positive.add('std')

        # Choose an ordering of internal parameters
        for i, p in enumerate(self.initial_conditions):
            self.order[p] = i
            self.labels[p] = str(i)

        self.labels['std'] = 'std'

        self.data = resonance_modes

        for p in self.initial_conditions:
            if p != 'std':
                self.dC[p] = self.C.diff(p)

        self.dp, self.pv, _, _, _, _, _, _ = polybasis.build(N, X, Y, Z)

        self.reset(self.initial_conditions)

    def reset(self, initial_conditions = None):
        if initial_conditions == None:
            initial_conditions = self.initial_conditions
        else:
            self.initial_conditions = initial_conditions

        self.qs = []
        self.logps = []
        self.accepts = []

        self.current_q = numpy.zeros(len(self.order))

        for p, i in self.order.items():
            self.current_q[i] = numpy.log(initial_conditions[p]) if p in self.constrained_positive else initial_conditions[p]

        self.accepts.append(self.current_q)

    def qdict(self, q):
        result = {}

        for p, i in self.order.items():
            result[p] = q[i]

        return result

    def UgradU(self, q):
        qdict = self.qdict(q)

        for p in qdict:
            if p in self.constrained_positive:
                qdict[p] = numpy.exp(qdict[p])

        std = qdict['std']

        C = numpy.array(self.C.evalf(subs = qdict)).astype('float')

        logp_total = 0.0
        dlogp_total = numpy.zeros(len(self.order))

        # Likelihood p(data | params)
        K, M = polybasis.buildKM(C, self.dp, self.pv, self.density)

        eigs, evecs = scipy.linalg.eigh(K, M, eigvals = (6, 6 + len(self.data) - 1))

        freqs = numpy.sqrt(eigs * 1e11) / (numpy.pi * 2000)
        dfreqsdl = 0.5e11 / (numpy.sqrt(eigs * 1e11) * numpy.pi * 2000)

        dlpdfreqs = (self.data - freqs) / std ** 2
        dlpdstd = sum((-std ** 2 + (self.data - freqs) **2) / std ** 3) * qdict['std'] + 1
        logp = sum(0.5 * (-((self.data - freqs) **2 / std**2) + numpy.log(1.0 / (2 * numpy.pi)) - 2 * numpy.log(std)))

        logp_total += logp + sum([q[self.order[p]] for p in self.constrained_positive])

        # Param likelihood p(params | hyper)
        #logp = sum(0.5 * (-((q[self.params] - q[self.mus]) **2 / q[self.stds]**2) + numpy.log(1.0 / (2 * numpy.pi)) - 2 * numpy.log(q[self.stds])))
        #logp_total += logp

        dlpdl = dlpdfreqs * dfreqsdl

        for p, i in self.order.items():
            if p == 'std':
                continue

            dKdp, _ = polybasis.buildKM(numpy.array(self.dC[p].evalf(subs = qdict)).astype('float'), self.dp, self.pv, self.density)

            dldp = numpy.array([evecs[:, j].T.dot(dKdp.dot(evecs[:, j])) for j in range(evecs.shape[1])])

            dlpdp = dlpdl.dot(dldp)

            dlogp_total[i] = (dlpdp * qdict[p] + 1) if p in self.constrained_positive else dlpdp

        dlogp_total[self.order['std']] = dlpdstd

        return -logp_total, -dlogp_total

    def set_labels(self, labels):
        self.labels = labels

    def set_timestepping(self, epsilon = 0.0001, L = 50, param_scaling = None):
        self.epsilon = epsilon

        if param_scaling != None:
            self.epsilon = numpy.ones(len(self.order)) * self.epsilon

            for p in param_scaling:
                self.epsilon[self.order[p]] *= param_scaling[p]

        self.L = L

    def sample(self, steps = -1, debug = False):
        if not hasattr(self, 'epsilon'):
            raise Exception("Must call 'rus.HMC.set_timestepping' before 'rus.HMC.sample'")

        epsilon = self.epsilon
        L = self.L

        while True:
            q = self.current_q.copy()
            p = numpy.random.randn(len(q)) # independent standard normal variates

            current_p = p
            # Make a half step for momentum at the beginning
            U, gradU = self.UgradU(q)
            p = p - epsilon * gradU / 2

            # Alternate full steps for position and momentum
            for i in range(L):
                # Make a full step for the position
                q = q + epsilon * p

                #q[-3:] = inv_rotations.qu2eu(symmetry.Symmetry.Cubic.fzQuat(quaternion.Quaternion(inv_rotations.eu2qu(q[-3:]))))
                # Make a full step for the momentum, except at end of trajectory
                if i != L - 1:
                    U, gradU = self.UgradU(q)
                    p = p - epsilon * gradU

                if debug:
                    print "New q: {0}".format(self.print_q(q))
                    print "H (constant or decreasing): ", U + sum(p ** 2) / 2, U, sum(p **2) / 2.0
                    print ""

            U, gradU = self.UgradU(q)
            # Make a half step for momentum at the end.
            p = p - epsilon * gradU / 2

            # Negate momentum at end of trajectory to make the proposal symmetric
            p = -p
            # Evaluate potential and kinetic energies at start and end of trajectory
            UC, gradUC = self.UgradU(self.current_q)
            current_U = UC
            current_K = sum(current_p ** 2) / 2
            proposed_U = U
            proposed_K = sum(p ** 2) / 2

            # Accept or reject the state at end of trajectory, returning either
            # the position at the end of the trajectory or the initial position
            dQ = current_U - proposed_U + current_K - proposed_K

            self.logps.append(UC)

            with printoptions(precision = 5):
                if numpy.random.rand() < min(1.0, numpy.exp(dQ)):
                    self.current_q = q # accept

                    self.accepts.append(len(self.qs) - 1)

                    print "Accepted ({0} accepts so far): {1}".format(len(self.accepts), self.print_q(self.current_q))
                else:
                    print "Rejected: ", self.current_q

                self.qs.append(q.copy())
                print "Energy change ({0} samples, {1} accepts): ".format(len(self.qs), len(self.accepts)), min(1.0, numpy.exp(dQ)), dQ, current_U, proposed_U, current_K, proposed_K

    def print_q(self, q = None, precision = 5):
        if q == None:
            q = self.current_q

        out = []

        printed = set()

        for p, name in self.labels.items():
            if p in self.order:
                out.append("{0} : {1:0.{2}f}".format(name, numpy.exp(q[self.order[p]]) if p in self.constrained_positive else q[self.order[p]], precision))

                printed.add(self.order[p])

        return "{{ {0} }}".format(", ".join(out))

    def print_current(self, precision = 5):
        qdict = self.qdict(self.current_q)

        for p in qdict:
            qdict[p] = numpy.exp(qdict[p]) if p in self.constrained_positive else qdict[p]

        C = numpy.array(self.C.evalf(subs = qdict)).astype('float')

        K, M = polybasis.buildKM(C, self.dp, self.pv, self.density)

        eigs, evecs = scipy.linalg.eigh(K, M, eigvals = (6, 6 + len(self.data) - 1))

        freqs = numpy.sqrt(eigs * 1e11) / (numpy.pi * 2000)

        print "Mean error: ", numpy.mean(freqs - self.data)
        print "Std deviation in error: ", numpy.std(freqs - self.data)

        print "Computed, Measured"
        for freq, dat in zip(freqs, self.data):
            print "{0:0.{2}f}, {1:0.{2}f}".format(freq, dat, precision)

    def save(self, filename):
        f = open(filename, 'w')
        f.write(self.saves())
        f.close()

    def format_samples(self):
        header = []
        samples = []

        printOrder = []
        do_exp = []

        for p, name in self.labels.items():
            if p in self.order:
                header.append(name)
                printOrder.append(self.order[p])

                if p in self.constrained_positive:
                    do_exp.append(True)
                else:
                    do_exp.append(False)

        for i, do_exp_ in zip(printOrder, do_exp):
            tmp_samples = []

            for q in self.qs:
                tmp_samples.append(numpy.exp(q[i]) if do_exp_ else q[i])

            samples.append(tmp_samples)

        return header, samples

    def saves(self):
        output = []

        header, samples = self.format_samples()

        output.append("# {0}".format(", ".join(header)))

        for params in zip(*samples):
            output.append(", ".join([str(sample) for sample in params]))

        return "\n".join(output)

    def derivative_check(self):
        for i in range(len(self.current_q)):
            q = self.current_q.copy()

            q2 = q.copy()

            q2[i] *= 1.0000001

            U1, gradU = self.UgradU(q)
            U2, gradU = self.UgradU(q2)

            print "Computed: ", (U2 - U1) / (q2[i] - q[i])
            print "Analytical: ", gradU[i]
            print ""