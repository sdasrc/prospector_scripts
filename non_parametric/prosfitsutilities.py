import os, sys
from tqdm.notebook import tqdm_notebook
import fitsio
from fitsio import FITS,FITSHDR
from astropy.io import fits
import numpy as np

def loadfits2dict(fitsfile):
    fitshdu = fits.open(fitsfile)
    fitsarr = fitsio.read(fitsfile)
    fitscols = fitshdu[1].columns
    fitskeys = {fitscols[ii].name:ii for ii in range(len(fitscols)) }
    fitsdict = {}
    for ths in tqdm_notebook(fitsarr): fitsdict[ths[1]] = ths
    for aa in list(fitsdict): 
        if aa <= 0: del fitsdict[aa]
    return fitsdict, fitskeys


def getvalwithsig(fdict,fdictkeys,fparam,common_keys,keydef=['_best','_16ptile','_50ptile','_84ptile']):
    '''
    getvalwithsig(fdict,fdictkeys,fparam,common_keys)
    '''
    fparamarr = np.array([fdict[xx][fdictkeys['{0}{1}'.format(fparam,keydef[0])]] for xx in common_keys])
    fparamsigplus = np.array([fdict[xx][fdictkeys['{0}{1}'.format(fparam,keydef[3])]] 
                              - fdict[xx][fdictkeys['{0}{1}'.format(fparam,keydef[2])]] 
                        for xx in common_keys])
    fparamsigminus =  np.array([fdict[xx][fdictkeys['{0}{1}'.format(fparam,keydef[2])]] 
                                - fdict[xx][fdictkeys['{0}{1}'.format(fparam,keydef[1])]]
                        for xx in common_keys])
    fparamsig = np.array(list(zip(fparamsigminus, fparamsigplus))).T  
    return fparamarr, fparamsig, fparamsigplus, fparamsigminus

def getbagpmass(bagpdict, bagpkeys, common_keys):
    # bagppipes
    bagpmass = np.array([10**(bagpdict[xx][bagpkeys['stellar_mass_50']]) for xx in common_keys])
    bagpmasssigplus = np.array([ 
        10**(bagpdict[xx][bagpkeys['stellar_mass_84']]) - 10**(bagpdict[xx][bagpkeys['stellar_mass_50']])
                  for xx in common_keys])
    bagpmasssigminus = np.array([ 
        10**(bagpdict[xx][bagpkeys['stellar_mass_50']]) - 10**(bagpdict[xx][bagpkeys['stellar_mass_16']])
                  for xx in common_keys])

    bagpmasssig = np.array(list(zip(bagpmasssigminus,bagpmasssigplus))).T
    
    return bagpmass, bagpmasssig, bagpmasssigplus, bagpmasssigminus

def get_common_keys(*args,**kwargs):
    if(len(args)<2): return []
    common_keys = np.intersect1d(list(args[0]),list(args[1]))
    N = len(args)-2
    for ii in range(N):
        common_keys = np.intersect1d(common_keys,list(args[ii+2]))
    return common_keys