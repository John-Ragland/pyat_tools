'''
pyat_tools - a set of tools built on top of pyat (https://github.com/hunterakins/pyat)
for simulation of Axial Seamount caldera environment
'''

import numpy as np
from os import system
import os
from matplotlib import pyplot as plt
from pyat.pyat.env import *
from pyat.pyat.readwrite import *

import xarray as xr
import pandas as pd
from numpy import matlib
from scipy.interpolate import CubicSpline

from scipy.special import hankel2
from scipy import signal
import scipy
from tqdm import tqdm

import sys

import multiprocessing as mp

def load_ssp_from_file():
    # import shallow

    fn = '/Users/jhrag/Code/ocean_acoustics/NI_paper_figures/sound_speed_profile/gridded_data/soundspeed.nc'

    ssp = xr.load_dataarray(fn)

    ssp_slice = ssp.loc[pd.Timestamp('01-01-2016'):pd.Timestamp('01-01-2017'), 11:186]
    nan_mask = np.isnan(np.mean(ssp_slice.values, axis=1))

    ssp_slice[nan_mask,:] = np.nan
    
    fn = '/Users/jhrag/Code/ocean_acoustics/NI_paper_figures/sound_speed_profile/gridded_data/deep_profiler/sound_speed.nc'

    ssp_deep = xr.load_dataarray(fn)
    ssp_deep = ssp_deep.loc[187:1600]
    # repeat SSP for all dates
    ssp_deep_rep = matlib.repmat(ssp_deep.values,ssp_slice.shape[0],1)
    # stitch two SSP together
    ssp_stitch_np = np.hstack((ssp_slice.values, ssp_deep_rep))
    ssp_stitch_np[nan_mask,:] = np.nan
    depth_stitch = np.hstack((ssp_slice.depth, ssp_deep.depth))
    # Save as xr.DataArray
    ssp_stitch = xr.DataArray(ssp_stitch_np, dims = ['time', 'depth'], coords={'time':ssp_slice.time, 'depth':depth_stitch}, name='SSP stitched')
    
    return ssp_stitch

def convert_SSP_arlpy(ssp, time_idx):
    '''
    convert_SSP_arlpy - converts SSP in form of
    xr.DataArray to numpy array that works with ARLPY
    
    Parameters
    ----------
    ssp : xr.DataArray
        Gridded sound speed profile data with dimensions [time, depth]
    time_idx : int
        index of time dimension desired for ssp data
        
    Returns
    -------
    ssp_bellhop : numpy array
        numpy array with depth and SSP compatible with ARLPY
    '''
    
    if len(ssp.shape) == 2:
        ssp_slice = ssp[time_idx,:]
    if len(ssp.shape) == 1:
        ssp_slice = ssp
    
    # Stack SSPs in format that arlpy likes
    ssp_bellhop = np.vstack((ssp_slice.depth, ssp_slice.values)).T

    # remove all nan points
    nan_mask = ~np.isnan(ssp_bellhop[:,1])
    ssp_bellhop = ssp_bellhop[nan_mask,:]
    # add ocean surface points
    ssp_bellhop = np.vstack((np.array([0,ssp_bellhop[0,1]]),ssp_bellhop))

    
    return ssp_bellhop

def simulate_kraken(ssp_arlpy, bottom_depth, sd, rd, ranges, depths, freq, title, fn, verbose):
    '''
    simulate_kraken() - calculates the modes for the hardcoded environment
        and given SSP, bottom_depth, sd, rd, and freq. Code is tweaked from
        pyat/pyat/test/krak_test/test.py.
    an environment file is written for the simulation
    other variable names that are undefined are likely defined in Kraken 
        user manual
    
    Parameters
    ----------
    ssp_arlpy : numpy.array
        sound speed profile is the arlpy format. (2xm) numpy array
        containing sound speeds and depths
    bottom_depth : float
        depth of the flat ocean bottom
    sd : float
        depth of source in meters
    rd : list
        list of reciever depths
    ranges : numpy.array
        array of ranges (km) for simulation ex. np.arange(0,10,0.01)
    depths : numpy.array
        array of depths (m) for simulation ex. np.arange(0,100,0.5)
    freq : float
        frequency of the simulation
    title : str
        title of the experiment
    fn : str
        filename for .mod, .env, .prt files. should NOT contain extension
    verbose : bool
        specifies whether to print statements or not
    Returns
    -------
    modes : pyat.pyat.env.Modes
    '''
    # delete previous .mod if it exists
    try:
        os.remove(f'{fn}.mod')
        if verbose: print('removed .mod')
    except FileNotFoundError:
        if verbose: print('no .mod file')
    try:
        os.remove(f'{fn}.env')
        if verbose: print('removed .env')
    except FileNotFoundError:
        if verbose: print('no .env file')
    try:
        os.remove(f'{fn}.prt')
        if verbose: print('removed .prt')
    except FileNotFoundError:
        if verbose: print('no .prt file')
        
    Z = depths
    X = ranges

    pw		=	1
    aw		=	0
    cb		=	1600
    pb		=	1.8
    ab		=	0.2

    s = Source(sd)
    r = Dom(X, Z)
    pos = Pos(s,r)
    pos.s.depth	= [sd]
    pos.r.depth	 = Z
    pos.r.range		=	X
    pos.Nsd = 1
    pos.Nrd = len(rd)
    depth = [0, bottom_depth] 
    
    # Layer 1
    # Resample provided ssp to be uniform (using cubic spline)
    cs = CubicSpline(ssp_arlpy[:,0], ssp_arlpy[:,1])
    z1 = np.linspace(0,bottom_depth, bottom_depth) # 1 m resolution
    alphaR = cs(z1)

    betaR	=	0.0*np.ones(z1.shape)
    rho		=	pw*np.ones(z1.shape)
    alphaI	=	aw*np.ones(z1.shape)
    betaI	=	0.0*np.ones(z1.shape)

    ssp1 = SSPraw(z1, alphaR, betaR, rho, alphaI, betaI)

    #	Sound-speed layer specifications
    raw = [ssp1]
    NMedia		=	1
    Opt			=	'CVW'	
    N			=	[z1.size]
    sigma		=	[.5,.5]	 # roughness at each layer. only effects attenuation (imag part)
    ssp = SSP(raw, depth, NMedia, Opt, N, sigma)


    hs = HS(alphaR=cb, betaR=0, rho = pb, alphaI=ab, betaI=0)
    Opt = 'A'
    bottom = BotBndry(Opt, hs)
    top = TopBndry('CVW')
    bdy = Bndry(top, bottom)


    class Empty:
        def __init__(self):
            return

    cInt = Empty()
    cInt.High = cb
    cInt.Low = 0 # compute automatically
    RMax = max(X)

    write_env(f'{fn}.env', 'KRAKEN', title, freq, ssp, bdy, pos, [], cInt, RMax)
    system(f"krakenc.exe {fn}")
    fname = f'{fn}.mod'
    options = {'fname':fname, 'freq':0}
    modes = read_modes(**options)
    return modes

def pyat_test_case(freq):
    sd	=	20
    rd = [94.125, 99.755, 105.38, 111.00, 116.62, 122.25, 127.88, 139.12, 144.74, 150.38, 155.99, 161.62, 167.26, 172.88, 178.49, 184.12, 189.76, 195.38, 200.99, 206.62, 212.25]
    rr	=	2.5

    Z = np.arange(0, 100, .5)
    X = np.arange(0, 10, .01)

    cw		=	1500
    pw		=	1
    aw		=	0
    cb		=	1600
    pb		=	1.8
    ab		=	0.2

    s = Source(sd)
    r = Dom(X, Z)
    pos = Pos(s,r)
    pos.s.depth	= [sd]
    pos.r.depth	 = Z
    pos.r.range		=	X
    pos.Nsd = 1
    pos.Nrd = len(rd)

    bottom_depth = 100
    depth = [0, bottom_depth] 
    # Layer 1
    z1		=	depth[0:2]	
    z1 = np.linspace(depth[0], bottom_depth, 1000)
    alphaR	=	cw*np.ones(z1.shape)
    betaR	=	0.0*np.ones(z1.shape)
    rho		=	pw*np.ones(z1.shape)
    alphaI	=	aw*np.ones(z1.shape)
    betaI	=	0.0*np.ones(z1.shape)

    ssp1 = SSPraw(z1, alphaR, betaR, rho, alphaI, betaI)

    #	Sound-speed layer specifications
    raw = [ssp1]
    NMedia		=	1
    Opt			=	'CVW'	
    N			=	[z1.size]
    sigma		=	[.5,.5]	 # roughness at each layer. only effects attenuation (imag part)
    ssp = SSP(raw, depth, NMedia, Opt, N, sigma)


    hs = HS(alphaR=cb, betaR=0, rho = pb, alphaI=ab, betaI=0)
    Opt = 'A~'
    bottom = BotBndry(Opt, hs)
    top = TopBndry('CVW')
    bdy = Bndry(top, bottom)


    class Empty:
        def __init__(self):
            return

    cInt = Empty()
    cInt.High = cb
    cInt.Low = 0 # compute automatically
    RMax = max(X)

    write_env('py_env.env', 'KRAKEN', 'Pekeris profile', freq, ssp, bdy, pos, [], cInt, RMax)

    system("krakenc.exe py_env")
    fname = 'py_env.mod'
    options = {'fname':fname, 'freq':0}
    modes = read_modes(**options)
    return modes

def write_env_file_pyat(ssp_arlpy, bottom_depth, sd, rd, ranges, depths, freq, title, fn, verbose):
    '''
   write_env_file() - writes environment file for given
        and given SSP, bottom_depth, sd, rd, and freq. Code is tweaked from
        pyat/pyat/test/krak_test/test.py.
    an environment file is written for the simulation
    other variable names that are undefined are likely defined in Kraken 
        user manual
    
    Parameters
    ----------
    ssp_arlpy : numpy.array
        sound speed profile is the arlpy format. (2xm) numpy array
        containing sound speeds and depths
    bottom_depth : float
        depth of the flat ocean bottom
    sd : float
        depth of source in meters
    rd : list
        list of reciever depths
    ranges : numpy.array
        array of ranges (km) for simulation ex. np.arange(0,10,0.01)
    depths : numpy.array
        array of depths (m) for simulation ex. np.arange(0,100,0.5)
    freq : float
        frequency of the simulation
    title : str
        title of the experiment
    fn : str
        filename for .mod, .env, .prt files. should NOT contain extension
    verbose : bool
        specifies whether to print statements or not
    Returns
    -------
    modes : pyat.pyat.env.Modes
    '''
    # delete previous .mod if it exists
    try:
        os.remove(f'{fn}.mod')
        if verbose: print('removed .mod')
    except FileNotFoundError:
        if verbose: print('no .mod file')
    try:
        os.remove(f'{fn}.env')
        if verbose: print('removed .env')
    except FileNotFoundError:
        if verbose: print('no .env file')
    try:
        os.remove(f'{fn}.prt')
        if verbose: print('removed .prt')
    except FileNotFoundError:
        if verbose: print('no .prt file')
        
    Z = depths
    X = ranges

    pw		=	1
    aw		=	0
    cb		=	1600
    pb		=	1.8
    ab		=	0.2

    s = Source(sd)
    r = Dom(X, Z)
    pos = Pos(s,r)
    pos.s.depth	= [sd]
    pos.r.depth	 = Z
    pos.r.range		=	X
    pos.Nsd = 1
    pos.Nrd = len(rd)
    depth = [0, bottom_depth] 
    
    # Layer 1
    # Resample provided ssp to be uniform (using cubic spline)
    cs = CubicSpline(ssp_arlpy[:,0], ssp_arlpy[:,1])
    z1 = np.linspace(0,bottom_depth, bottom_depth) # 1 m resolution
    alphaR = cs(z1)

    betaR	=	0.0*np.ones(z1.shape)
    rho		=	pw*np.ones(z1.shape)
    alphaI	=	aw*np.ones(z1.shape)
    betaI	=	0.0*np.ones(z1.shape)

    ssp1 = SSPraw(z1, alphaR, betaR, rho, alphaI, betaI)

    #	Sound-speed layer specifications
    raw = [ssp1]
    NMedia		=	1
    Opt			=	'CVW'	
    N			=	[z1.size]
    sigma		=	[.5,.5]	 # roughness at each layer. only effects attenuation (imag part)
    ssp = SSP(raw, depth, NMedia, Opt, N, sigma)


    hs = HS(alphaR=cb, betaR=0, rho = pb, alphaI=ab, betaI=0)
    Opt = 'A'
    bottom = BotBndry(Opt, hs)
    top = TopBndry('CVW')
    bdy = Bndry(top, bottom)


    class Empty:
        def __init__(self):
            return

    cInt = Empty()
    cInt.High = cb
    cInt.Low = 0 # compute automatically
    RMax = max(X)

    write_env(f'{fn}.env', 'KRAKEN', title, freq, ssp, bdy, pos, [], cInt, RMax)
    return

def change_env_freq(env_file, freq, env_file_new = None):
    '''
    change frequency of an environment file without having to rewrite whole file
    
    Parameters
    ----------
    env_file : str
        path to environment file. Should contain file extension
    freq : float
        freq to write to file
    env_file_new : str
        path to new environment file. if None, then the original path is overwritten
        
    Returns
    -------
    None
    
    - file located at env_file is rewritten
    '''
    if env_file_new == None:
        a_file = open(env_file)
        line_ls = a_file.readlines()
        line_ls[1] = f'{freq}\n'

        a_file = open(env_file, "w")
        a_file.writelines(line_ls)
        a_file.close()
    else:
        os.system(f'cp {env_file} {env_file_new}')
        
        a_file = open(env_file_new)
        line_ls = a_file.readlines()
        line_ls[1] = f'{freq}\n'

        a_file = open(env_file_new, "w")
        a_file.writelines(line_ls)
        a_file.close()
    
def write_flp_file(source_depths, ranges, r_depths, fn, verbose=True):
    '''
    write_flp_file - write *.flp file for calculating Greene's function pressures
    
    Parameters
    ----------
    source_depths : array like
        depth of source in meters
    ranges : array like
        range of reciever in meters (i think, could be km)
    receiver_depth : float
        depth of recievers in meters
    fn : str
        file name of flp file. should NOT include extension.
    verbse : bool
        specifies whether to print statements or not
    '''
    
    try:
        os.remove(f'{fn}.flp')
        if verbose: print('removed .flp')
    except FileNotFoundError:
        if verbose: print('no .flp file')
    # remove previous
    s = Source(source_depths)
    r = Dom(ranges, r_depths)
    
    pos = Pos(s, r)
    
    write_fieldflp(fn, 'R', pos)
    return

def simulate_FDGF_SingleRange(fn, freqs, fband, krakenc=False, verbose=True):
    '''
    simulate_FDGF_singleRange - runs simulation for frequency domain Greene's function. the FDGF is sampled
        at freqs (array like). Before running this, a *.env and *.flp must be created with the
        name fn (located in python path). Currently hard coded to only work with single
        source_depth, reciever_depth, and range. simulate_FDGF - a build of this function to 
        handle multiple reciever locations, but single source location
        
    Parameters
    ----------
    fn : str
        name of env and flp file (must be the same)
    freqs : np.array()
        calculated with get_freq_time_vectors (freq_half)
        array of frequency to simulate over. should contain (N/2 + 1) points
        and span f = [0, Fs/2], where N is number of points in time domain.
        result is manually flipped to account for negative frequencies.
    fband : list
        list of length 2 given the low and high bound of the valid frequency band in Hz
        
    Returns
    -------
    pressures_flipped : np.array
        numpy array of length N containing the Frequency Domain Greene's Function (FDGF)
        array is forced to result in real valued p(t) where p(t) = ifft(P(f)).
        frequency bins for this array are given by freq_full from get_freq_time_vectors()
    '''
    freqs = list(freqs)
    pressures = []
    for freq in tqdm(freqs):
        # Check if frequency is in frequency band
        if (freq < fband[0]) | (freq > fband[1]):
            pressures.append(0)
            continue
        
        # delete mod and shd files
        try:
            os.remove(f'{fn}.mod')
        except FileNotFoundError:
            pass
        try:
            os.remove(f'{fn}.shd')
        except FileNotFoundError:
            pass
        
        # Skip DC component
        if freq == 0:
            pressures.append(0)
            continue
        change_env_freq(f'{fn}.env', freq)
        if krakenc:
            _ = os.system(f"krakenc.exe {fn}")
        else:
            _ = os.system(f"kraken.exe {fn}")
        
        try:
            # if there are no modes, then f bin
            fname = f'{fn}.mod'
            options = {'fname':fname, 'freq':freq}
            modes = read_modes(**options)
        except:
            pressures.append(0)
            continue

        if modes.M == 0:
            pressures.append(0)
            continue
        
        # Calculate pressure
        _ = os.system(f"field.exe {fn}")
        [x,x,x,x,Pos1,pressure]= read_shd(f'{fn}.shd')
        pressure = pressure[0,0,0,0]
        pressures.append(pressure)
        
    pressures = np.array(pressures)
    
    # Make sure that pressures_f is valid fourier transform
    # (aka force X[0] and X[N/2] to be real (for even N))
    pressures[0] = np.real(pressures[0])
    pressures[-1] = np.real(pressures[-1])
    
    # Create flipped frequency
    pressures_flipped = np.concatenate((
        pressures,
        np.flipud(np.conjugate(pressures[1:-1]))
    ))
    
    
    # Add butter worth filter to pressures
    #x = np.zeros(pressures_flipped.shape)
    #x[0] = 1
    #Wn = np.array([1, 90])/(freqs[-1])
    #b, a = signal.butter(4, Wn, btype='bandpass')
    #x_filt = signal.lfilter(b,a,x)
    #X_filt = np.abs(scipy.fft.fft(x_filt))
    #pressures_flipped_fltrd = np.abs(pressures_flipped)*X_filt*np.exp(1j*np.angle(pressures_flipped))
    
    return pressures_flipped

def simulate_FDGF_manual(fn, freqs, fband, simulated_modes=None, krakenc=False):
    '''
    simulate_FDGF_manual - runs simulation for frequency domain Greene's function. the FDGF is sampled
        at freqs (array like). Before running this, a *.env must be created with the
        name fn (located in python path). Currently hard coded to only work with single
        source_depth, reciever_depth, and range.
        
        uses equations from [1] instead of field.exe (as a sanity check)
        
        [1] L. E. Kinsler, A. R. Frey, A. B. Coppens, and J. V. Sanders,
        Fundamentals of Acoustics. John Wiley & Sons, 2000.

        
    Parameters
    ----------
    fn : str
        name of env and flp file (must be the same)
    freqs : np.array()
        calculated with get_freq_time_vectors (freq_half)
        array of frequency to simulate over. should contain (N/2 + 1) points
        and span f = [0, Fs/2], where N is number of points in time domain.
        result is manually flipped to account for negative frequencies.
    fband : list
        list of length 2 given the low and high bound of the valid frequency band in Hz
    simulated_modes : list
        modes to use to calculate the TDGF, if none, all are used. The first mode is
        referenced by 0
    krakenc : bool
        specifies whether to use kraken (False) or krakenc (True)
        
    Returns
    -------
    pressures_flipped : np.array
        numpy array of length N containing the Frequency Domain Greene's Function (FDGF)
        array is forced to result in real valued p(t) where p(t) = ifft(P(f)).
        frequency bins for this array are given by freq_full from get_freq_time_vectors()
    '''
    freqs = list(freqs)
    pressures = []
    kns = []
    
    r = 3200 #meters

    for count, freq in enumerate(tqdm(freqs)):
        # Check if frequency is in frequency band
        if (freq <= fband[0]) | (freq >= fband[1]):
            pressures.append(0)
            continue
        
        # delete mod files
        try:
            os.remove(f'{fn}.mod')
        except FileNotFoundError:
            pass
  
        # Skip DC component
        if freq == 0:
            pressures.append(0)
            continue
        
        change_env_freq(f'{fn}.env', freq)
        if krakenc:
            _ = os.system(f"krakenc.exe {fn}")
        else:
            _ = os.system(f"kraken.exe {fn}")
        
        # Read mode file
        fname = f'{fn}.mod'
        options = {'fname':fname, 'freq':freq}
        try:
            modes = read_modes(**options)
        except:# FileNotFoundError:
            pressures.append(0)
            continue
            
        if modes.M == 0:
            pressures.append(0)
            continue
        
        # Calculate pressure
        # depth hardcoded to 1498.5 m (-2 index) for source and reciever
        if (simulated_modes == None):
            arg1 = modes.phi[-2,:]*modes.phi[-2,:]
            arg2 = hankel2(0, modes.k*r)

            pressure = -1j*np.pi*np.sum(arg1*arg2)
            pressures.append(pressure)
        
        elif (np.max(simulated_modes)+1 > modes.M):
            arg1 = modes.phi[-2,:]*modes.phi[-2,:]
            arg2 = hankel2(0, modes.k*r)

            pressure = -1j*np.pi*np.sum(arg1*arg2)
            pressures.append(pressure)
            
            
        else:
            arg1 = modes.phi[-2,simulated_modes]*modes.phi[-2,simulated_modes]
            arg2 = hankel2(0, modes.k[simulated_modes]*r)

            pressure = -1j*np.pi*np.sum(arg1*arg2)
            pressures.append(pressure)
    
    # Make sure that pressures_f is valid fourier transform
    # (aka force X[0] and X[N/2] to be real (for even N))
    pressures[0] = np.real(pressures[0])
    pressures[-1] = np.real(pressures[-1])
    
    # Create flipped frequency
    pressures_flipped = np.concatenate((
        pressures,
        np.flipud(np.conjugate(pressures[1:-1]))
    ))
    
    # Add butter worth filter to pressures
    #x = np.zeros(pressures_flipped.shape)
    #x[0] = 1
    #Wn = np.array([1, 90])/(freqs[-1])
    #b, a = signal.butter(4, Wn, btype='bandpass')
    #x_filt = signal.lfilter(b,a,x)
    #X_filt = np.abs(scipy.fft.fft(x_filt))
    #pressures_flipped_fltrd = np.abs(pressures_flipped)*X_filt*np.exp(1j*np.angle(pressures_flipped))
    return pressures_flipped

def build_env_file(ssp_arlpy, env_top, env_bottom, fn):
    '''
    build_env_file - create an environment file that has manually defined properties and a
        a given sound speed profile (in arlpy format). Seperate .txt files for the top and
        bottom of the environment file must be pre-made
        
    Parameters
    ----------
    ssp_arlpy : np.array
       sound speed profiler of water column
    env_top : str
        path (relative or absolute) to text file (including file extension)
        that is the top half of the environment file
    env_bottom : str
        path (relative or absolute) to text file (including file extension)
        that is the bottom half of the enironment file
    fn : str
        file path for environment file to be written. Should include file extension
    '''
    # Create new environmental file
    if os.path.exists(fn):
        os.remove(fn)
    new_env_f =  open(fn, 'w')
    
    # open, read, and write top text file
    with open(env_top, 'r') as f:
        top = f.readlines()
    new_env_f.writelines(top)
    
    # Write SSP information
    # SSP
    
    # currently assume constant betaR, rho, alphaI, betaI
    betaR = np.ones(len(ssp_arlpy))*0
    rho = np.ones(len(ssp_arlpy))*1.03551
    alphaI = np.ones(len(ssp_arlpy))*(0)
    betaI = np.ones(len(ssp_arlpy))*0
    
    new_env_f.write('\n')
    for ii in range(len(ssp_arlpy)):
        new_env_f.write('\t {:6.2f} '.format(ssp_arlpy[ii][0]) + \
            '{:6.2f} '.format(ssp_arlpy[ii][1]) + \
            '{:6.2f} '.format(betaR[ ii ]) + \
            '{:6.6g}'.format(rho[ ii ] ) +  \
            ' {:10.6f} '.format(alphaI[ ii ]) + \
            '{:6.2f} '.format(betaI[ ii ]) + \
            '/ \t ! z c cs rho \r\n')
            
            
    # open, read, and write bottom text file
    with open(env_bottom, 'r') as f:
        bottom = f.readlines()

    new_env_f.writelines(bottom)
    return

def get_freq_time_vectors(Fs, To, verbose=True):
    '''
    get_freq_time_vectors - calculate vector of frequencies and times given
        sampling frequency and length of desired time segment
        
    Parameters
    ----------
    Fs : float
        sampling frequency in Hz
    To : float
        length of time segment in s
    verbose : bool
        whether or not to print the results
    
    Returns
    -------
    t : np.array
        vector of time bins (N,)
    freq_half : np.array
        vector of frequency bins to simulate over ((N/2 + 1),)
    freq_full : np.array
        vector of all frequncy bins (N,)
    '''
    
    N = Fs*To
    if N % 2 != 0:
        raise Exception('N must be even (solution is hard coded for even N) change Fs or To')

    t = 1/Fs*np.arange(N)
    freq_half = np.arange((N)/2 + 1)*Fs/(N)
    freq_full = np.arange(N)*Fs/N
    if verbose:
        print(f'N:\t{N}\nTo:\t{To} [s]\nΔf\t{Fs/(N*2)}')
        
    return t, freq_half, freq_full

def simulate_single_ssp(ssp, verbose=False):
    '''
    simulate_single_SSP builds upon simulate_TDGF() and calculates
        the 
    
    Parameters
    ----------
    ssp : numpy array
        sound speed profile in numpy format required by arlpy
        
    Returns
    -------
    TDGF : numpy array
    '''
    
    process_name = mp.current_process().name
    
    # Write Environment File
    fn = f'multiprocessing/caldera_{process_name}'
    
    build_env_file(ssp,'env_files/caldera_top_roughrock.txt','env_files/caldera_bottom_roughrock.txt', f'{fn}.env')
    
    # Build Frequency vector to accurate sample DTFT (so that x(t) can be reconstructed)
    Fs = 200
    To = 20
    t, freq_half, freq_full = get_freq_time_vectors(Fs, To, verbose=verbose)

    # Write field flp file
    s_depths=[1527] #meters
    ranges =  [3.186] #km
    r_depths = [1518] # meters

    write_flp_file(s_depths, ranges, r_depths, fn, verbose=verbose)
    
    # Simulate with Kraken
    pressures = simulate_FDGF(fn, freq_half, [1, 90], krakenc=False)

    pt = np.real(scipy.fft.ifft(pressures))
    
    return pt

def create_NCCF(pt):
    '''
    create_NCCF - takes TDGF array and turns it into flipped NCCF
        of shape (11999,) so that NCCF code works with simulated
        results   

    Parameters
    ----------
    pt : numpy.array
        array of arbitray shape containing less then 6000 points
    
    Returns
    -------
    NCCF : numpy array
        np array of shape (11999,) that has flipped simulated version
    '''
    
    t,_,_ = get_freq_time_vectors(200, 30, verbose=False)
    pt_pad = np.hstack((pt))#, np.zeros(3000)))

    pt_pad = np.expand_dims(np.hstack((np.flipud(pt_pad[1:]), pt_pad)),0)

    NCCF_sim = xr.DataArray(
        pt_pad,
        dims=['dates','delay'],
        coords = {'delay':np.hstack((-np.flipud(t[1:]), t))}
    )
    return NCCF_sim

def simulate_green_1cpu(env_fn, freq, file_dir, fband):
    '''
    simulate_green_1cpu - simulate the greens function for a single frequency
        given an environment file located at env_fn. New environment file will be
        written to file_dir and frequency value will be overwritten with freq
    This is designed to be called by multiprocessing
    
    Handling of frequencies where no modes propagate, or DC are approached by simply returning None
    
    Parameters
    ----------
    env_fn : str
        path to environment file
    freq : float
        frequency in Hertz to evaluate green's function
    file_dir : str
        location where new .env .flp. .mod and .shd files will be written
    freq_range : list of length 2
        lower and upper frequencies
    '''

    # Check if frequency is in frequency band
    if (freq < fband[0]) | (freq > fband[1]):
        pressures = None
        return pressures

    # Check if DC
    if freq == 0:
        pressures = None
        return pressures

    process_name = mp.current_process().name
    
    fn = f'{file_dir}{env_fn}{process_name}'

    # delete mod and shd files with same name
    try:
        os.remove(f'{file_dir}/{fn}.mod')
    except FileNotFoundError:
        pass
    try:
        os.remove(f'{file_dir}{fn}.shd')
    except FileNotFoundError:
        pass
    
    # write new env file
    change_env_freq(f'{env_fn}.env', freq, env_file_new = f'{fn}.env')
    
    # Run Kraken
    os.system(f'kraken.exe {fn} >/dev/null 2>&1')
    
    # read mod file
    try:
        fname = f'{fn}.mod'
        options = {'fname':fname, 'freq':freq}
        modes = read_modes(**options)
    # if there are no propagating modes, return None
    except:
        pressures = None
        return pressures

    # Copy flp file to multiprocessing directory
    os.system(f'cp {env_fn}.flp {fn}.flp')
    
    # Run field.exe
    os.system(f'field.exe {fn} >/dev/null 2>&1')
    
    try:
        [_,_,_,_,Pos1,pressure]= read_shd(f'{fn}.shd')
    #except FileNotFoundError:
    except: # this is terrible and should eventually be removed haha
        return None
    
    return pressure

def simulate_FDGF(fn, freqs, fband, mp_file_dir, data_lens, multiprocessing=True, verbose=False):
    '''
    simulate_FDGF - runs simulation for frequency domain Greene's function. the FDGF is sampled
        at freqs (array like). Before running this, a *.env and *.flp must be created with the
        name fn (located in python path).
        
    for calling this function from an external script, you need to put everything relating to
        this function and it's outputs inside if __name__ == '__main__': (who knows why)
    some notes:
        there are several errors non-deterministic errors in the multiprocessing where it
        fails for unpredictable frequencies... This is a big error that I need to figure out
    
    Parameters
    ----------
    fn : str
        filename of environment and flp files. should NOT contain ext
    freqs : np.array()
        calculated with get_freq_time_vectors (freq_half)
        array of frequency to simulate over. should contain (N/2 + 1) points
        and span f = [0, Fs/2], where N is number of points in time domain.
        result is manually flipped to account for negative frequencies.
    fband : list
        list of length 2 given the low and high bound of the valid frequency band in Hz
    mp_file_dir : str
        multiprocessing file directory - multiprocessing feature requires a directory where
        it can write multiple environmental files. This string should be appended by /
    data_lens : dict
        length of all the flp variables should have structure:
        {'s_depths': len of source depths,,
         'ranges': len of source depths,
         'r_depths': len of reciever depths
         }
    multiprocessing : bool
        mostly for debugging, if false a simple for loop is used.
    verbose=False
    
    Returns
    -------
    pressures_flipped : np.array
        array of shape[2*frequencies, source_depths, reciever range, reciever depth]
        
        The array is flipped to be conjugate (real fourier transform)
    '''
    
    if verbose: print('simulating environment...')
    # create starmap input
    inputs = []
    for k in range(len(freqs)):
        inputs.append((fn, freqs[k], mp_file_dir, fband))
    n_cpu = mp.cpu_count()
    
    if multiprocessing:
        #if __name__ == '__main__':
        with mp.Pool(processes = n_cpu) as pool:
            pressures_ls = pool.starmap(simulate_green_1cpu, inputs)
    else:
        pressures_ls = []
        for input_single in tqdm(inputs):
            #print(input_single[0], input_single[1],input_single[2],input_single[3])
            pressures_ls.append(simulate_green_1cpu(input_single[0], input_single[1],input_single[2],input_single[3]))
    print('constructing array...')
    # Construct a big 'ole array
    #print(len(freqs), data_lens['s_depths'], data_lens['ranges'], data_lens['r_depths'])
    pressures = np.zeros((len(freqs), data_lens['s_depths'], data_lens['r_depths'], data_lens['ranges'],)) + 1j*np.zeros((len(freqs), data_lens['s_depths'], data_lens['r_depths'], data_lens['ranges'],))
    
    for k in range(len(freqs)):
        if pressures_ls[k] is None:
            pressures[k,:,:,:] = np.zeros((data_lens['s_depths'], data_lens['r_depths'], data_lens['ranges'],))
        else:
            pressures[k,:,:,:] = pressures_ls[k][0,:,:,:]

    # set all nan to zero
    pressures[np.isnan(pressures)] = 0
    
    # Make sure that pressures_f is valid fourier transform
    # (aka force X[0] and X[N/2] to be real (for even N))
    pressures[0,:,:,:] = np.real(pressures[0,:,:,:])
    pressures[-1,:,:,:] = np.real(pressures[-1,:,:,:])
    
    # Create flipped frequency
    pressures_flipped = np.concatenate((
        pressures,
        np.conjugate(np.flip(pressures[1:-1,:,:,:], axis=0))
    ), axis=0)
    
    return pressures_flipped

def dummy(k):
    print(mp.current_process().name)
    return k