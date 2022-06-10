import os, sys
import fitsio
from fitsio import FITS,FITSHDR
from astropy.io import fits
import numpy as np
from tqdm import tqdm

filetags = '_output.txt'

print('v3')
print(sys.argv)

if len(sys.argv) != 3: 
    print ('run as python make_fits_catalog_v2.py <dir> <outputfits>')
    exit()
else:
    WORK_DIR = sys.argv[1]
    if WORK_DIR[-1] != '/': WORK_DIR = WORK_DIR+'/'
    fname = sys.argv[2]
    if fname[-5:].lower() == '.fits': fname = fname[:-5]
    fname = WORK_DIR+fname+'.fits'

# WORK_DIR = os.getcwd()+'/'
# fname = WORK_DIR+'trial'+'.fits'

# Import input catalog
catdict = {}
catthdu = fits.open('/beegfs/car/sdas/prospector_sedfits/fits/ELAIS_N1/EN1_sedfit_v1.0.fits')
catarr = list(catthdu[1].data)
catcols = catthdu[1].columns
catkeys = {catcols[ii].name:ii for ii in range(len(catcols))}
for ths in catarr: catdict[ths[catkeys['radioID']]] = ths
    
chi2_99_table = np.array([99.00,99.00,99.00,99.00,99.00,99.00,99.00,10.01,11.45,13.01,14.53,16.02,
             17.49,18.94,20.39,21.84,23.28,24.73,26.18,27.64,29.1,30.58,32.06,33.56,35.07,
             36.58,38.11,39.66,41.21,42.78,44.37])
    
# Deal with the actual problems now

outfits = FITS(fname,'rw')

subfolders = [ f.path for f in os.scandir(WORK_DIR) if f.is_dir() ]
objdirs = [ f.split('/')[-1] for f in subfolders if f.split('/')[-1].isnumeric() ]

nrows = 0
column_names  = ['index']
for this_dir in objdirs:
    outtxt = WORK_DIR+this_dir+'/'+this_dir+filetags
    if os.path.isfile(outtxt):
        nrows+=1
with open(WORK_DIR+this_dir+'/'+this_dir+filetags,'r') as f: outlines = f.readlines()
for this_line in outlines: column_names.append(this_line.split(' ')[0].replace('ptile',''))

columns = [('index','i8'),('radio_id','i8'),('sour_id','i8'),('sour_name','a30'),('z_best','f8'),
           ('lum_dist','f8'),('age_univ','f8'),('filters_obs','i8'),('flag_good','i8'),
           ('chi2_best','f8'),('chi2_99_flag','i8')]

for this_col in column_names[9:-1]: columns.append((this_col.replace('ptile',''),'f8'))
 
data = np.zeros(nrows, dtype=columns)


indx = 0
for this_dir in tqdm(objdirs):
    outtxt = WORK_DIR+this_dir+'/'+this_dir+filetags
    if os.path.isfile(outtxt):
        with open(outtxt,'r') as f: outlines = f.readlines()
        for this_line in outlines: exec(this_line.replace('ptile',''))
                
        data[indx]['index'] = indx
        data[indx]['radio_id'] = radio_id
        data[indx]['sour_id'] = sour_id
        data[indx]['sour_name'] = sour_name
        data[indx]['z_best'] = z_best
        data[indx]['lum_dist'] = lum_dist
        data[indx]['age_univ'] = age_univ
        data[indx]['filters_obs'] = good_fluxes
        data[indx]['flag_good'] = int(catdict[radio_id][catkeys['FLAG_GOOD']])
        data[indx]['chi2_best'] = chi2_best

        # From Smith +12
        ndof = int(-2.820+(0.661*good_fluxes)+(7.91*10**-3*good_fluxes**2 ))
        chi2cutoff = int( chi2_best > chi2_99_table[ndof] )
        data[indx]['chi2_99_flag'] = chi2cutoff

        for this_var in column_names[9:-1]: exec('data[{0}]["{1}"] = {1}'.format(indx,this_var))
        indx += 1
    else:
        continue

outfits.write(data)
outfits.close()

print('{0} out of {1} successfully parsed.'.format(indx+1,nrows))