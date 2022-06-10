def loadfits(fitname,picklename):
    import os, pickle, sys
    import numpy as np
    from tqdm import tqdm
    import fitsio
    from fitsio import FITS,FITSHDR

    from astropy.io import fits
 

    if os.path.isfile(picklename):
        with open(picklename, 'rb') as handle:
            tdict = pickle.load(handle)

    else :
        thdu = fits.open(fitname)
        tarr = fitsio.read(fitname)
        tcols = thdu[1].columns
        tkeys = {tcols[ii].name:ii for ii in range(len(tcols))}
        
        tdict = {}
        for ths in tqdm(tarr): tdict[ths[tkeys['id']]] = ths

        with open(picklename, 'wb') as handle:
            pickle.dump(tdict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return tdict

cigale_dict = loadfits('/beegfs/lofar/deepfields/sedfits/cigale_Fritz/lofar_team_test_cigale.fits','cigalefritz.pickle') 
