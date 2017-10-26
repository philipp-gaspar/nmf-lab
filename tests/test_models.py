import numpy as np

# Creating synthetic data

rows = 50
columns = 100

print 'Generating synthetic data...\n'
V = np.random.rand(rows, columns)


from nmf_lab.models.nmf import NMF_MU

nmf_mu = NMF_MU()

n_comp = 20
num_trials = 100
results = nmf_mu.run_repeat(V, n_comp, num_trials,
                              max_iter=500,
                              cost_function='kullback-leibler',
                              verbose=True)

print(results.keys())
