catdict = {}

catthdu = fits.open('EN1_sedfit_v1.0.fits')
catarr = list(catthdu[1].data)
catcols = catthdu[1].columns
catkeys = {catcols[ii].name:ii for ii in range(len(catcols))}

for ths in tqdm_notebook(catarr): catdict[ths[catkeys['radioID']]] = ths

# -----------------------------------------------------------------

crossdict = {}

crosshdu = fits.open('final_cross_match_catalogue-v1.0.fits')
crossarr = crosshdu[1].data
crosscols = crosshdu[1].columns
crosskeys = {crosscols[ii].name:ii for ii in range(len(crosscols))}

catnames = np.array([catdict[xx][catkeys['Source_Name']] for xx in catdict.keys()])
catobs = list(catdict.keys())

crossdict = {}
for xx in tqdm_notebook(range(len(crossarr))):
    tkey = np.where(catnames == crossarr[xx][crosskeys['Source_Name']])[0][0]
    crossdict[catobs[tkey]] = crossarr[xx]

# -----------------------------------------------------------------

catdict = {}

catthdu = fits.open('EN1_sedfit_v1.0.fits')
catarr = catthdu[1].data
catcols = catthdu[1].columns
catkeys = {catcols[ii].name:ii for ii in range(len(catcols))}
for ths in catarr: catdict[ths[catkeys['radioID']]] = ths

catnames = np.array([catdict[xx][catkeys['Source_Name']] for xx in catdict.keys()])
catrid = np.array([catdict[xx][catkeys['radioID']] for xx in catdict.keys()])

tfitsname = 'ELAIS_20200817.fits'
thdu = fits.open(tfitsname)
tarr = thdu[1].data
tcols = thdu[1].columns
tkeys = {tcols[ii].name:ii for ii in range(len(tcols))}

tdict = {}
namekey = 'galaxy_id'
for tt in tqdm_notebook(tarr):
    if len(np.where( tt[tkeys[namekey]] == catnames )[0]):
        tdict[catrid[np.where( tt[tkeys[namekey]] == catnames )[0][0]]] = tt