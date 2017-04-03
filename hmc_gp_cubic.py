#%%
import GPy
import pystan
import numpy
import time
import scipy
import os
os.chdir('/home/bbales2/modal')
import pyximport
pyximport.install(reload_support = True)

import polybasisqu
reload(polybasisqu)

# basis polynomials are x^n * y^m * z^l where n + m + l <= N
N = 10

## Dimensions for TF-2
X = 0.007753#0.011959e1#
Y = 0.009057#0.013953e1#
Z = 0.013199#0.019976e1#

#sample mass

#Sample density
density = 4401.695921 #Ti-64-TF2
#density = 8700.0 #CMSX-4
#density = (mass / (X*Y*Z))

c11 = 2.0
anisotropic = 1.0
c44 = 1.0
c12 = -(c44 * 2.0 / anisotropic - c11)

# Standard deviation around each mode prediction
std = 1.0

# Rotations
a = 0.0
b = 0.0
y = 0.0

# These are the sampled modes in khz
# data for sample 2M-A

# Ti-64-TF2 Test Data
data = numpy.array([109.076,
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
323.464,
324.702,
334.687,
340.427,
344.087,
363.798,
364.862,
371.704,
373.248])

#data = (freqs * numpy.pi * 2000) ** 2 / 1e11

qs = []
logps = []
accepts = []

current_q = numpy.array([c11, anisotropic, c44, std])

accepts.append(current_q)
#%%

# These are the two HMC parameters
#   L is the number of timesteps to take -- use this if samples in the traceplots don't look random
#   epsilon is the timestep -- make this small enough so that pretty much all the samples are being accepted, but you
#       want it large enough that you can keep L ~ 50 -> 100 and still get independent samples
L = 50
# start epsilon at .0001 and try larger values like .0005 after running for a while
# epsilon is timestep, we want to make as large as possibe, wihtout getting too many rejects
epsilon = 0.0005

# Set this to true to debug the L and eps values
debug = False

#%%
S = 100

minc11 = 1.0
maxc11 = 4.0
mina = 0.5
maxa = 2.0
minc44 = 0.1
maxc44 = 2.0

def func(c11, anisotropic, c44):
    c12 = -(c44 * 2.0 / anisotropic - c11)

    C = numpy.array([[c11, c12, c12, 0, 0, 0],
                     [c12, c11, c12, 0, 0, 0],
                     [c12, c12, c11, 0, 0, 0],
                     [0, 0, 0, c44, 0, 0],
                     [0, 0, 0, 0, c44, 0],
                     [0, 0, 0, 0, 0, c44]])

    try:
        numpy.linalg.cholesky(C)
    except:
        return [numpy.nan]

    dp, pv, ddpdX, ddpdY, ddpdZ, dpvdX, dpvdY, dpvdZ = polybasisqu.build(N, X, Y, Z)

    K, M = polybasisqu.buildKM(C, dp, pv, density)
    eigs, evecs = scipy.linalg.eigh(K, M, eigvals = (6, 6 + len(data) - 1))

    return numpy.sqrt(eigs * 1e11) / (numpy.pi * 2000)

Xs = []
Ys = []
i = 0
while i < S:
    c11 = numpy.random.rand() * (maxc11 - minc11) + minc11
    a = numpy.random.rand() * (maxa - mina) + mina
    c44 = numpy.random.rand() * (maxc44 - minc44) + minc44

    y = func(c11, a, c44)

    if numpy.isnan(y).any():
        continue
    else:
        Xs.append([c11, a, c44])
        Ys.append(y)

        print i, " / ", S
        i += 1

Xs = numpy.array(Xs)
Ys = numpy.array(Ys)

#%%
kernel = GPy.kern.RBF(input_dim = 3, variance = 1.0, lengthscale = 1.0)
m = GPy.models.GPRegression(Xs, Ys[:, :1], kernel)
#m.optimize(messages = True)
m.optimize_restarts(num_restarts = 5)
#%%
print m
#%%
m.plot(visible_dims = [0])
#%%
m.predict(numpy.array([[1.7, 1.0, 0.46]]))
#%%
model_code = """
data {
  int<lower=1> N; // Number of single samples
  int<lower=1> L;
  matrix<lower=0.0>[N, 3] x;
  matrix<lower=0.0>[N, L] y;
  //real yd;
}

transformed data {
  vector[N] z;

  for(i in 1:N)
    z[i] = 0.0;
}

parameters {
  real<lower=0> eta_sq;
  vector<lower=0>[3] inv_rho_sq;
  real<lower=0> sigma_sq;

  /*real<lower = 0.0> dsigma;

  real c11;
  real a;
  real c44;*/
}

transformed parameters {
  vector<lower=0>[3] rho_sq;
  rho_sq = inv(inv_rho_sq);
}

model {
  matrix[N, N] Sigma;
  //row_vector[N] Ks;

  // off-diagonal elements
  for (i in 1:(N-1)) {
    for (j in (i+1):N) {
      Sigma[i, j] = eta_sq * exp(-rho_sq[1] * pow(x[i, 1] - x[j, 1], 2)
                                 -rho_sq[2] * pow(x[i, 2] - x[j, 2], 2)
                                 -rho_sq[3] * pow(x[i, 3] - x[j, 3], 2));
      Sigma[j, i] = Sigma[i, j];
    }
  }

  // diagonal elements
  for (k in 1:N)
    Sigma[k, k] = eta_sq + sigma_sq; // + jitter

  /*for(k in 1:N)
    Ks[k] = eta_sq * exp(-rho_sq[1] * pow(x[k, 1] - c11, 2)
                         -rho_sq[2] * pow(x[k, 2] - a, 2)
                         -rho_sq[3] * pow(x[k, 3] - c44, 2));*/

  eta_sq ~ cauchy(0, 1000.0);
  inv_rho_sq ~ cauchy(0, 10.0);
  sigma_sq ~ cauchy(0, 10.0);

  for(i in 1:L)
    y[:, i] ~ multi_normal(z, Sigma);

  //dsigma ~ cauchy(0.0, 10.0);

  //yd ~ normal(Ks * (Sigma \ y), dsigma);

  //y ~ normal(mu, );
}
"""

#generated quantities {
#  vector[N] yhat;
#
#  for(n in 1:N) {
#    yhat[n] <- normal_rng(a * x[n] + b, sigma);
#  }
#}
#"""

sm = pystan.StanModel(model_code = model_code)

#%%
fit = sm.sampling(data = {
    "N" : S,
    "L" : Ys.shape[1],
    "x" : Xs,
    "y" : Ys,
    "yd" : 109.0
})
#%%
def cov(Xs, Ys):
    Xs = numpy.array(Xs)
    Ys = numpy.array(Ys)

    K = numpy.zeros((Xs.shape[0], Ys.shape[0]))
    for i in range(Xs.shape[0]):
        for j in range(Ys.shape[0]):
            K[i, j] = 9303.9 * numpy.exp(-0.82 * pow(Xs[i, 0] - Ys[j, 0], 2)
                                        -5.38 * pow(Xs[i, 1] - Ys[j, 1], 2)
                                        -1.09 * pow(Xs[i, 2] - Ys[j, 2], 2));

    return K

K = cov(Xs, Xs)
b = numpy.linalg.solve(K, Ys[:, 0])# + 29.51 * numpy.eye(S)

#%%

s = []
i = 0
while i < S:
    c11 = numpy.random.rand() * (maxc11 - minc11) + minc11
    a = numpy.random.rand() * (maxa - mina) + mina
    c44 = numpy.random.rand() * (maxc44 - minc44) + minc44

    y = func(c11, a, c44)

    if numpy.isnan(y).any():
        continue
    else:
        Ks = cov(Xs, [[c11, a, c44]])

        print Ks.T.dot(b)[0], y[0]
        s.append(Ks.T.dot(b)[0] - y[0])
        #Xs.append([c11, a, c44])
        #Ys.append(y)

        print i, " / ", S
        #i += 1


#%%
#%%
model_code = """
data {
  int<lower=1> N; // Number of single samples
  int<lower=1> L;
  matrix<lower=0.0>[N, 3] x;
  matrix<lower=0.0>[N, L] y;
  vector<lower=0.0>[L] yd;
}

transformed data {
  vector[N] z;
  matrix[N, N] Sigma;
  matrix[N, L] Kiy;

  for(i in 1:N)
    z[i] = 0.0;

  //matrix[N, N] L;

  // off-diagonal elements
  for (i in 1:(N-1)) {
    for (j in (i+1):N) {
      Sigma[i, j] = 9303.9 * exp(-0.82 * pow(x[i, 1] - x[j, 1], 2)
                                 -5.38 * pow(x[i, 2] - x[j, 2], 2)
                                 -1.09 * pow(x[i, 3] - x[j, 3], 2));
      Sigma[j, i] = Sigma[i, j];
    }
  }

  for (k in 1:N)
    Sigma[k, k] = 9303.9 + 30.0; // + jitter

  Kiy = Sigma \ y;
  //L = cholesky_decompose(Sigma);
}

parameters {
  real<lower = 0.0> dsigma;

  real<lower = 0.1, upper = 4.0> c11;
  real<lower = 0.5, upper = 2.0> a;
  real<lower = 0.1, upper = 1.0> c44;
}

model {
  row_vector[N] Ks;
  real Kv;
  vector[L] tmp;

  // diagonal elements

  for(k in 1:N)
    Ks[k] = 9303.9 * exp(-0.82 * pow(x[k, 1] - c11, 2)
                         -5.38 * pow(x[k, 2] - a, 2)
                         -1.09 * pow(x[k, 3] - c44, 2));

  Kv = sqrt(9303.9 + 30.0 - Ks * (Sigma \ Ks'));

  dsigma ~ cauchy(0.0, 10.0);

  //for(i in 1:N)
  //  tmp[i] = ;

  yd ~ normal(Ks * Kiy, Kv + dsigma);

  //y ~ normal(mu, );
}

generated quantities {
    real Kv;

    {
      row_vector[N] Ks;

      // diagonal elements

      for(k in 1:N)
        Ks[k] = 9303.9 * exp(-0.82 * pow(x[k, 1] - c11, 2)
                             -5.38 * pow(x[k, 2] - a, 2)
                             -1.09 * pow(x[k, 3] - c44, 2));

      Kv = sqrt(9303.9 + 30.0 - Ks * (Sigma \ Ks'));
    }
}
"""

#generated quantities {
#  vector[N] yhat;
#
#  for(n in 1:N) {
#    yhat[n] <- normal_rng(a * x[n] + b, sigma);
#  }
#}
#"""

sm2 = pystan.StanModel(model_code = model_code)
#%%
fit = sm2.sampling(data = {
    "N" : S,
    "L" : Ys.shape[1],
    "x" : Xs,
    "y" : Ys,
    "yd" : data
})

print fit
#%%
import matplotlib.pyplot as plt
a = fit.extract()
plt.plot(a['c11'], a['a'], '*')
plt.xlabel('c11')
plt.ylabel('a')
plt.show()
plt.plot(a['c11'], a['c44'], '*')
plt.xlabel('c11')
plt.ylabel('c44')
plt.show()
plt.plot(a['a'], a['c44'], '*')
plt.xlabel('a')
plt.ylabel('c44')
plt.show()
#%%


import polybasisqu
import sys
sys.path.append('/home/bbales2/gpc')
import gpc

func2 = lambda c11, a, c44 : func(c11, 1.0, c44)


hd = gpc.GPC(5, func2, [('n', (2.0, 0.5), 3),
                       ('u', (mina, maxa), 5),
                       ('u', (minc44, maxc44), 5)])

#%%
#This block runs the HMC

def UgradU(q):
    c11, anisotropic, c44, std = q

    anisotropic = 1.0

    try:
        e = hd.approx(c11, anisotropic, c44)
        dedc11 = hd.approxd(0, c11, anisotropic, c44)
        deda = hd.approxd(1, c11, anisotropic, c44)
        dedc44 = hd.approxd(2, c11, anisotropic, c44)
    except Exception as e:
        e = numpy.nan
        dedc11 = [numpy.nan] * len(data)
        deda = [numpy.nan] * len(data)
        dedc44 = [numpy.nan] * len(data)

    dlpde = (data - e) / std ** 2
    dlpdstd = sum((-std ** 2 + (e - data) **2) / std ** 3)

    #dlpde = numpy.array(dlpde)

    dlpdc11 = dlpde.dot(dedc11)
    dlpda = dlpde.dot(deda)
    dlpdc44 = dlpde.dot(dedc44)

    logp = sum(0.5 * (-((e - data) **2 / std**2) + numpy.log(1.0 / (2 * numpy.pi)) - 2 * numpy.log(std)))

    return -logp, -numpy.array([dlpdc11, 0.0, dlpdc44, dlpdstd])

#%%
d = 0.00001
q = numpy.array([1.6, 1.0, 0.7, 1.0])

nlogp, dnlogp = UgradU(q)

for i in range(3):
    q_ = q.copy()

    q_[i] += d

    nlogp_, _ = UgradU(q_)

    print dnlogp[i], (nlogp_ - nlogp) / d
#%%
c11 = 1.7
anisotropic = 1.0
c44 = 0.5

qs = []
logps = []
accepts = []

current_q = numpy.array([c11, anisotropic, c44, std])

accepts.append(current_q)

debug = False

while True:
    q = current_q.copy()
    p = numpy.random.randn(len(q)) # independent standard normal variates

    current_p = p
    # Make a half step for momentum at the beginning
    U, gradU = UgradU(q)
    p = p - epsilon * gradU / 2

    # Alternate full steps for position and momentum
    for i in range(L):
        # Make a full step for the position
        q = q + epsilon * p

        #q[-3:] = inv_rotations.qu2eu(symmetry.Symmetry.Cubic.fzQuat(quaternion.Quaternion(inv_rotations.eu2qu(q[-3:]))))
        # Make a full step for the momentum, except at end of trajectory
        if i != L - 1:
            U, gradU = UgradU(q)
            p = p - epsilon * gradU

        #print 'hi'
        if debug:
            print "New q: ", q
            print "H (constant or decreasing): ", U + sum(p ** 2) / 2, U, sum(p **2) / 2.0
            print ""

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

    logps.append(UC)

    if numpy.random.rand() < min(1.0, numpy.exp(dQ)) and not numpy.isnan(proposed_U):
        current_q = q # accept

        accepts.append(len(qs) - 1)

        print "Accepted ({0} accepts so far): {1}".format(len(accepts), current_q)
    else:
        print "Rejected: ", current_q

    qs.append(q.copy())
    print "Energy change ({0} samples, {1} accepts): ".format(len(qs), len(accepts)), min(1.0, numpy.exp(dQ)), dQ, current_U, proposed_U, current_K, proposed_K


#%%
# Save samples (qs)
# First argument is filename

import os
import tempfile
import datetime

_, filename = tempfile.mkstemp(prefix = "data_{0}_".format(datetime.datetime.now().strftime("%Y-%m-%d")), suffix = ".txt", dir = os.getcwd())
numpy.savetxt(filename, qs, header = 'c11 anisotropic c44 std')
#%%
# This block does the plotting

c11s, anisotropics, c44s, stds = [numpy.array(a) for a in zip(*qs)]#
import matplotlib.pyplot as plt
import seaborn

for name, data1 in zip(['c11', 'anisotropic ratio', 'c44', 'std deviation', '-logp'],
                      [c11s, anisotropics, c44s, stds, logps]):
    plt.plot(data1)
    plt.title('{0}'.format(name, numpy.mean(data1), numpy.std(data1)), fontsize = 24)
    plt.tick_params(axis='y', which='major', labelsize=16)
    plt.show()
    #seaborn.distplot(d[-650:], kde = False, fit = scipy.stats.norm)
    #plt.title('{0} u = {1:.3e}, std = {2:.3e}'.format(name, numpy.mean(data1), numpy.std(data1)))
    #plt.show()

#plt.plot(as_, ys_)
#plt.ylabel('eu[2]s')
#plt.xlabel('eu[0]s')
#plt.show()
#%%
import seaborn
c11s, anisotropics, c44s, stds = [numpy.array(a)[-1500:] for a in zip(*qs)]#

for name, data1 in zip(['C11', 'A Ratio', 'C44', 'std dev'],
                      [c11s, anisotropics, c44s, stds]):
    seaborn.distplot(data1, kde = False, fit = scipy.stats.norm)
    plt.title('{0}, $\mu$ = {1:0.3f}, $\sigma$ = {2:0.3f}'.format(name, numpy.mean(data1), numpy.std(data1)), fontsize = 36)
    plt.tick_params(axis='x', which='major', labelsize=16)
    plt.show()
    #seaborn.distplot(d[-650:], kde = False, fit = scipy.stats.norm)
    #plt.title('{0} u = {1:.3e}, std = {2:.3e}'.format(name, numpy.mean(data1), numpy.std(data1)))
    #plt.show()

#%%

while 1:
    U, gradU = UgradU(current_q)

    current_q += 0.0001 * gradU
#%%
# Forward problem

# This snippet is helpful to test the last accepted sample
c11, anisotropic, c44, std = qs[accepts[-1]]

c12 = -(c44 * 2.0 / anisotropic - c11)

dp, pv, ddpdX, ddpdY, ddpdZ, dpvdX, dpvdY, dpvdZ = polybasis.build(N, X, Y, Z)

C = numpy.array([[c11, c12, c12, 0, 0, 0],
                 [c12, c11, c12, 0, 0, 0],
                 [c12, c12, c11, 0, 0, 0],
                 [0, 0, 0, c44, 0, 0],
                 [0, 0, 0, 0, c44, 0],
                 [0, 0, 0, 0, 0, c44]])

K, M = polybasis.buildKM(C, dp, pv, density)
eigs, evecs = scipy.linalg.eigh(K, M, eigvals = (6, 6 + len(data) - 1))

print "computed, accepted"
for e1, dat in zip(eigs, data):
    print "{0:0.3f} {1:0.3f}".format(e1, dat)
