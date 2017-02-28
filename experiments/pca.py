#%%

import os
import pystan
import pickle
import matplotlib.pyplot as plt

os.chdir('/home/bbales2/modal')

f = open('traces/A49-ortho-1v2/A49-ortho-1v2')
hmc = pickle.load(f)
f.close()

if not hasattr(hmc, 'stdMin'):
    hmc.stdMin = 0.0

#%%
import numpy
#%%
import seaborn

for name, data1 in zip(*hmc.format_samples()):
    plt.plot(data1[-2000:])
    plt.title('{0}'.format(name, numpy.mean(data1), numpy.std(data1)), fontsize = 24)
    plt.tick_params(axis='y', which='major', labelsize=16)
    plt.show()
#%%
for name, data1 in zip(*hmc.format_samples()):
    data1 = data1[-2000:]
    seaborn.distplot(data1, kde = False, fit = scipy.stats.norm)
    plt.title('{0}, $\mu$ = {1:0.3f}, $\sigma$ = {2:0.3f}'.format(name, numpy.mean(data1), numpy.std(data1)), fontsize = 36)
    plt.tick_params(axis='x', which='major', labelsize=16)
    plt.show()
#%%
data = hmc.format_samples()

print len(data[0])
names = data[0]
stdIdx = data[0].index('std')
data = numpy.array(data[1])
names.pop(stdIdx)
data = numpy.delete(data, stdIdx, axis = 0).T

n2idx = { 'c11' : (0, 0),
          'c12' : (0, 1),
          'c44' : (3, 3),
          'c55' : (4, 4),
          'c23' : (1, 2),
          'c13' : (0, 2),
          'c33' : (2, 2),
          'c22' : (1, 1),
          'c66' : (5, 5) }

def vec2matrix(vec):
    matrix = numpy.zeros((6, 6))
    for name, val in zip(names, vec):
        matrix[n2idx[name]] = val

    return matrix

snames = sorted(names)

data2 = numpy.zeros(data.shape)

for i, name in enumerate(snames):
    print i, names.index(name)
    data2[:, i] = data[:, names.index(name)]

data = data2.copy()

print data.mean(axis = 0)

#%%

import sklearn.decomposition

pca = sklearn.decomposition.PCA()

pca.fit(data)

for i in range(pca.components_.shape[0]):
    m = vec2matrix(pca.components_[i])

    plt.imshow(m, cmap = plt.cm.viridis, interpolation = 'NONE')
    plt.colorbar()
    plt.show()

#%%
C = numpy.corrcoef(data.transpose())
plt.imshow(C, interpolation = 'NONE', cmap = plt.cm.viridis)
plt.gcf().set_size_inches((8, 8))
#plt.savefig("/home/bbales2/cluster_expansion/images/low_correlation.png", dpi = 72)
plt.show()
#%%

model = """
data {
    int<lower=0> N;
    int<lower=0> M;
    matrix[N, M] x;
}

parameters {
    vector[3] cc;
    vector<lower=0.0>[3] sigc;

    vector[M] cf;
    vector<lower=0.0>[M] sigf;
}

model {
    sigc ~ cauchy(0.0, 5.0);
    sigf ~ cauchy(0.0, 5.0);

    cf[1] ~ normal(cc[1], sigc[1]);
    cf[4] ~ normal(cc[1], sigc[1]);
    cf[6] ~ normal(cc[1], sigc[1]);

    cf[2] ~ normal(cc[2], sigc[2]);
    cf[3] ~ normal(cc[2], sigc[2]);
    cf[5] ~ normal(cc[2], sigc[2]);

    cf[7] ~ normal(cc[3], sigc[3]);
    cf[8] ~ normal(cc[3], sigc[3]);
    cf[9] ~ normal(cc[3], sigc[3]);

    x ~ normal(cf, sigf);
}
"""

m1 = pystan.StanModel(model_code = model)
#%%

fit = m1.sampling(data = {
    'N' : 1000,
    'M' : data.shape[1],
    'x' : data[-1000:]
})