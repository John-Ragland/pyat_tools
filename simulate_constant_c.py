import numpy as np
from Simulation.kraken import pyat_tools
import multiprocessing as mp
import pickle
from tqdm import tqdm

# create ssp iterable
C0s = (np.linspace(1450,1550, 50))

ssps = []

for C0 in C0s:
    depths = np.arange(0,1528)
    Cs = np.ones(len(depths))*C0
    ssps.append(np.vstack((depths,Cs)).T)

print(mp.cpu_count())
a_pool = mp.Pool(processes = mp.cpu_count())


r = list(tqdm(a_pool.imap(pyat_tools.simulate_single_ssp, ssps), total=len(ssps)))

fn = 'TDGFs_constantC.pkl'

with open(fn, 'wb') as f:
    pickle.dump(r, f)