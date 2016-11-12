#%%

import numpy
import time
import scipy
import sympy
import os
os.chdir('/home/bbales2/modal')

import rus
reload(rus)
#%%
#656 samples
#%%

# basis polynomials are x^n * y^m * z^l where n + m + l <= N
N = 10

## Dimensions for TF-2
X = 0.011959#0.007753
Y = 0.013953#0.009057
Z = 0.019976#0.013199

#Sample density
density = 8700.0#4401.695921 #Ti-64-TF2

c110 = 2.0
anisotropic0 = 1.5
c440 = 1.0
c120 = -(c440 * 2.0 / anisotropic0 - c110)

# Standard deviation around each mode prediction
std0 = 5.0

# Ti-64-TF2 Test Data
#data = numpy.array([109.076,
#136.503,
#144.899,
#184.926,
#188.476,
#195.562,
#199.246,
#208.460,
#231.220,
#232.630,
#239.057,
#241.684,
#242.159,
#249.891,
#266.285,
#272.672,
#285.217,
#285.670,
#288.796,
#296.976,
#301.101,
#303.024,
#305.115,
#305.827,
#306.939,
#310.428,
#318.000,
#319.457,
#322.249,
#323.464,
#324.702,
#334.687,
#340.427,
#344.087,
#363.798,
#364.862,
#371.704,
#373.248])

data = numpy.array([71.25925,
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
205.01475,
206.619,
208.513875,
208.83525,
212.22525,
212.464125,
221.169625,
225.01225,
227.74775,
228.31175,
231.4265,
235.792875,
235.992375,
236.73675,
238.157625,
246.431125,
246.797125,
248.3185,
251.69425,
252.97225,
253.9795,
256.869875,
258.23825,
259.39025])

#%%

# These are the two HMC parameters
#   L is the number of timesteps to take -- use this if samples in the traceplots don't look random
#   epsilon is the timestep -- make this small enough so that pretty much all the samples are being accepted, but you
#       want it large enough that you can keep L ~ 50 -> 100 and still get independent samples
L = 50
# start epsilon at .0001 and try larger values like .0005 after running for a while
# epsilon is timestep, we want to make as large as possibe, wihtout getting too many rejects
epsilon = 0.0001

# Set this to true to debug the L and eps values
debug = False

#%%

reload(rus)

c11, anisotropic, c44 = sympy.symbols('c11 anisotropic c44')

c12 = sympy.sympify("-(c44 * 2.0 / anisotropic - c11)") # The c11 and c44 and anisotropic are the same as above

C = sympy.Matrix([[c11, c12, c12, 0, 0, 0],
                  [c12, c11, c12, 0, 0, 0],
                  [c12, c12, c11, 0, 0, 0],
                  [0, 0, 0, c44, 0, 0],
                  [0, 0, 0, 0, c44, 0],
                  [0, 0, 0, 0, 0, c44]])

hmc = rus.HMC(N = N, # Order of Rayleigh-Ritz approximation
              density = density, X = X, Y = Y, Z = Z,
              resonance_modes = data, # List of resonance modes
              stiffness_matrix = C, # Stiffness matrix
              parameters = { c11 : 2.0, anisotropic : 2.0, c44 : 1.0, 'std' : 5.0 }, # Parameters
              rotations = True,
              T = 1.0)

hmc.set_labels({ c11 : 'c11', anisotropic : 'a', c44 : 'c44', 'std' : 'std' })
hmc.set_timestepping(epsilon = epsilon, L = 50)
hmc.sample(steps = 5, debug = True)
#%%
hmc.set_timestepping(epsilon = epsilon * 10, L = 50)
hmc.sample(debug = False)#True)#False)#True)
#%%
hmc.derivative_check()
#%%
hmc.set_timestepping(epsilon = epsilon * 20, L = 75)
hmc.sample(debug = True)#False)#True)
#%%
hmc.print_current()
#%%
hmc.posterior_predictive(plot = False)
#%%
hmc.save('/home/bbales2/modal/paper/cmsx4/qs.csv')
#%%
import polybasisqu
import pandas
import seaborn
import matplotlib.pyplot as plt

def posterior_predictive(self, lastN = 200, precision = 5, plot = True):
        lastN = min(lastN, len(self.qs))

        posterior_predictive = numpy.zeros((max(self.modes), lastN, max(self.R, 1)))

        for i, (q, qr) in enumerate(zip(self.qs[-lastN:], self.qrs[-lastN:])):
            for r in range(max(self.R, 1)):
                qdict = self.qdict(q)

                for p in qdict:
                    qdict[p] = numpy.exp(qdict[p]) if p in self.constrained_positive else qdict[p]

                C = numpy.array(self.C.evalf(subs = qdict)).astype('float')

                if self.rotations:
                    w, x, y, z = qr[self.rotations[r]]

                    C, _, _, _, _, _ = polybasisqu.buildRot(C, w, x, y, z)

                K, M = polybasisqu.buildKM(C, self.dp, self.pv, self.density)

                eigs, evecs = scipy.linalg.eigh(K, M, eigvals = (6, 6 + max(self.modes) - 1))

                posterior_predictive[:, i, r] = numpy.sqrt(eigs * 1e11) / (numpy.pi * 2000)
        #print l, r, posterior_predictive[0]

        for s in range(self.S):
            if self.rotations:
                r = self.rotations[s]
            else:
                r = 0

            ppl = numpy.percentile(posterior_predictive[:, :, r], 2.5, axis = 1)
            ppr = numpy.percentile(posterior_predictive[:, :, r], 97.5, axis = 1)

            if plot:
                data = []

                for l in range(len(self.data[s])):
                    tmp = []
                    for ln in range(lastN):
                        tmp.append(posterior_predictive[l, ln, r] - self.data[s][l])

                    data.append(tmp)
                    #data.append([l, self.data[s][l], 'Measured'])

                #df = pandas.DataFrame(data, columns = ['Modes', 'Frequency', 'Type'])

                #seaborn.boxplot(x = 'Modes', y = 'Frequency', data = df)
                data = numpy.array(data)
                plt.boxplot(numpy.array(data).transpose())

                #ax1 = plt.gca()

                #for ll, meas, rr, tick in zip(ppl, self.data[s], ppr, range(len(self.data[s]))):
                #    ax1.text(tick + 1, ax1.get_ylim()[1] * 0.90, '{0:10.{3}f} {1:10.{3}f} {2:10.{3}f}'.format(ll, meas, rr, precision),
                #             horizontalalignment='center', rotation=45, size='x-small')
                plt.xlabel('Mode')
                plt.ylabel('Computed - Measured')
            else:
                print "For dataset {0}".format(s)
                print "{0:8s} {1:10s} {2:10s} {3:10s}".format("Outside", "2.5th %", "measured", "97.5th %")
                for ll, meas, rr in zip(ppl, self.data[s], ppr):
                    print "{0:8s} {1:10.{4}f} {2:10.{4}f} {3:10.{4}f}".format("*" if (meas < ll or meas > rr) else " ", ll, meas, rr, precision)

posterior_predictive(hmc)
gcf = plt.gcf()
gcf.set_size_inches(12, 8)
#%%
reload(rus)

hmc.set_timestepping(epsilon = epsilon * 2, L = 50)

hmc.sample(debug = False)

##%%
# Working on this -- not ready yet
reload(rus)

print hmc.saves()

#%%
reload(rus)

c11mc12, c11p2c12, c44 = sympy.symbols('c11mc12 c11p2c12 c44')

V = sympy.Matrix([[-1, -1, 1, 0, 0, 0],
                  [0, 1, 1, 0, 0, 0],
                  [1, 0, 1, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1],
                  [0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 1, 0, 0]])

Vinv = sympy.Matrix([[-1, -1, 2, 0, 0, 0],
                     [-1, 2, -1, 0, 0, 0],
                     [1, 1, 1, 0, 0, 0],
                     [0, 0, 0, 0, 0, 3],
                     [0, 0, 0, 0, 3, 0],
                     [0, 0, 0, 3, 0, 0]]) / 3.0

C = sympy.Matrix([[c11mc12, 0, 0, 0, 0, 0],
                  [0, c11mc12, 0, 0, 0, 0],
                  [0, 0, c11p2c12, 0, 0, 0],
                  [0, 0, 0, c44, 0, 0],
                  [0, 0, 0, 0, c44, 0],
                  [0, 0, 0, 0, 0, c44]])

C = V * C * Vinv

hmc = rus.HMC(N, density, X, Y, Z, data, C, { c11mc12 : 1.0, c11p2c12 : 4.0, c44 : 1.0, 'std' : 5.0 }, constrained_positive = [c44, 'std'])

hmc.set_labels({ c11mc12 : 'c11 - c12', c11p2c12 : 'c11 + 2 * c12', c44 : 'c44', 'std' : 'std' })
hmc.set_timestepping(epsilon = epsilon, L = 50)

hmc.sample(debug = True)
#%%
reload(rus)

hmc.set_timestepping(epsilon = epsilon * 20, L = 50)

hmc.sample(debug = False)
#%%
import matplotlib.pyplot as plt
import seaborn

for name, data1 in zip(*hmc.format_samples()):
    plt.plot(data1)
    plt.title('{0}'.format(name, numpy.mean(data1), numpy.std(data1)), fontsize = 24)
    plt.tick_params(axis='y', which='major', labelsize=16)
    plt.show()
#%%
for name, data1 in zip(*hmc.format_samples()):
    data1 = data1[-400:]
    seaborn.distplot(data1, kde = False, fit = scipy.stats.norm)
    plt.title('{0}, $\mu$ = {1:0.3f}, $\sigma$ = {2:0.3f}'.format(name, numpy.mean(data1), numpy.std(data1)), fontsize = 36)
    plt.tick_params(axis='x', which='major', labelsize=16)
    plt.show()