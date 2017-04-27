import sys

import numpy as np

# Setup analysis path
analysis_path = '/home/philipp.gaspar/Software/nmf-lab/'
sys.path.append(analysis_path)

# Creating synthetic data

rows = 100
columns = 50

print 'Generating synthetic data...\n'
Y = np.random.rand(rows, columns)


from models.nmf import NMF_HALS

nmf_hals = NMF_HALS()

n_comp = 5
nmf_hals.run(Y, n_comp,
             init=None,
             max_iter=500,
             cost_function='frobenius',
             verbose=True)
