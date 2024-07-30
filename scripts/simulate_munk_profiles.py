import pyat_tools
import numpy as np
from matplotlib import pyplot as plt
import multiprocessing as mp
import pickle
from tqdm import tqdm

def simulate_munk_profile(k):
    '''
    simulate_munk_profiler - simulate TDGF for given iterable k. theta and cmin
        are determine by k, which maps to a hardcoded meshgrid
        
    Returns
    -------
    TDGF : np.array
        numpy array of shape (len(t),) where t is hardcoded in function
    '''
    
    cmins = np.linspace(1440,1510, 20)
    sigs = np.linspace(0,0.02,20)
    
    cmins_mesh, sigs_mesh = np.meshgrid(cmins, sigs)
    
    cmins_mesh = np.ndarray.flatten(cmins_mesh)
    sigs_mesh = np.ndarray.flatten(sigs_mesh)
    
    z = np.linspace(0,1500,1000)
    zhat = 2*(z - 400)/400

    c = cmins_mesh[k]*(1 + sigs_mesh[k]*(zhat - 1 + np.exp(-zhat)))
    
    ssp = np.vstack((z,c)).T
    
    tdgf = pyat_tools.simulate_single_ssp(ssp, verbose=False)
    return [cmins_mesh[k], sigs_mesh[k], tdgf]

a_pool = mp.Pool(processes = mp.cpu_count())

r = list(tqdm(a_pool.imap(simulate_munk_profile, range(100)), total=100))

fn = 'munk_TDGFs.pkl'

with open(fn, 'wb') as f:
    pickle.dump(r, f)