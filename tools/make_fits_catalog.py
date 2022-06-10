import os, sys
import fitsio
from fitsio import FITS,FITSHDR
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

if len(sys.argv) < 2: 
    fname = 'EN1_thisrun.fits'
else:
    fname = sys.argv[1]
    if fname[-5:].lower() == '.fits': fname = fname[:-5]
    fname = fname+'.fits'
    
outfits = FITS(fname,'rw')

subfolders = [ f.path for f in os.scandir(os.getcwd()) if f.is_dir() ]
objdirs = [ f.split('/')[-1] for f in subfolders if f.split('/')[-1].isnumeric() ]

column_names  = ['index']
for this_dir in objdirs:
    outtxt = this_dir+'/'+this_dir+'_output.txt'
    if os.path.isfile(outtxt):
        break
with open(this_dir+'/'+this_dir+'_output.txt','r') as f: outlines = f.readlines()
for this_line in outlines: column_names.append(this_line.split(' ')[0])

nrows = len(objdirs)
columns = [('index','i8'),('radio_id','i8'),('sour_id','i8'),('sour_name','a30')]

for this_col in column_names[4:]: columns.append((this_col,'f8'))
 
data = np.zeros(nrows, dtype=columns)

paul_fits = '/home/overlord/Local/phd_project/prospector_sedfits/magphys/catalog_paul/ELAIS_20200817.fits'
if not os.path.isfile(paul_fits): paul_fits = '/beegfs/car/sdas/prospector_sedfits/magphys/catalog_paul/ELAIS_20200817.fits'

paul_data = fitsio.read(paul_fits)
paul_objs = np.array([ob[0] for ob in paul_data])

paul_col_dict = {'ssfr100_PHASKELL' : 36, 
                 'ssfr100_PHASKELL_sigma' : 37, 
                 'mass_PHASKELL' : 44, 
                 'mass_PHASKELL_sigma' : 45,
                 'sfr100_PHASKELL' : 124, 
                 'sfr100_PHASKELL_sigma' : 125, 
                 'ldust_PHASKELL' : 52, 
                 'ldust_PHASKELL_sigma' : 53,
                 'mdust_PHASKELL' : 116,
                 'mdust_PHASKELL_sigma' : 117}

compare_plot_labels = ['stellar mass', 'SFR (last 100 Myrs)', 'sSFR (last 100 Myrs)', 'dust luminosity', 'dust mass']
prospect_for_plot = [[], [], [], [], []]
prospect_sig_for_plot = [[], [], [], [], []]
phaskell_for_plot = [[], [], [], [], []]
phaskell_sig_for_plot = [[], [], [], [], []]

indx = 0
for this_dir in tqdm(objdirs):
    outtxt = this_dir+'/'+this_dir+'_output.txt'
    if os.path.isfile(outtxt):
        with open(outtxt,'r') as f: outlines = f.readlines()
        for this_line in outlines: 
            if 'PHASKELL' not in this_line: exec(this_line)
                
        data[indx]['index'] = indx
        for this_var in column_names[1:]: 
            if 'PHASKELL' not in this_var: cmnd = 'data[{0}]["{1}"] = {1}'.format(indx,this_var)
            else:
                indarr = np.where(sour_name == np.array(paul_objs))
                if np.shape(indarr)[1]:
                    inds = indarr[0][0]
                    cmnd = 'data[{0}]["{1}"] = {2}'.format(indx,this_var, paul_data[inds][paul_col_dict[this_var]] )
                    exec('{0} = {1}'.format(this_var,paul_data[inds][paul_col_dict[this_var]]))
                    exec('{0} = {1}'.format(this_var+'_sigma',paul_data[inds][paul_col_dict[this_var+'_sigma']]))
                else:              
                    cmnd = 'data[{0}]["{1}"] = {2}'.format(indx,this_var, 0 )
                    exec('{0} = {1}'.format(this_var,0))
                    exec('{0} = {1}'.format(this_var+'_sigma',0))
            exec(cmnd)
        indx += 1
        prospect_for_plot[0].append(mass_best)
        prospect_for_plot[1].append(sfr100_best)
        prospect_for_plot[2].append(ssfr100_best)
        prospect_for_plot[3].append(ldust_best)

        prospect_sig_for_plot[0].append(mass_sigma)
        prospect_sig_for_plot[1].append(sfr100_sigma)
        prospect_sig_for_plot[2].append(ssfr100_sigma)
        prospect_sig_for_plot[3].append(ldust_sigma)

        phaskell_for_plot[0].append(mass_PHASKELL)
        phaskell_for_plot[1].append(sfr100_PHASKELL)
        phaskell_for_plot[2].append(ssfr100_PHASKELL)
        phaskell_for_plot[3].append(ldust_PHASKELL)

        phaskell_sig_for_plot[0].append(mass_PHASKELL_sigma)
        phaskell_sig_for_plot[1].append(sfr100_PHASKELL_sigma)
        phaskell_sig_for_plot[2].append(ssfr100_PHASKELL_sigma)
        phaskell_sig_for_plot[3].append(ldust_PHASKELL_sigma)


    else:
        continue

outfits.write(data)
outfits.close()

plt.rcParams['figure.figsize'] = (12,12)
fig, ax = plt.subplots(2,2)
for ii in range(4):
    
    this_min = np.min([np.min(phaskell_for_plot[ii]),np.min(prospect_for_plot[ii])])
    this_max = np.max([np.max(phaskell_for_plot[ii]),np.max(prospect_for_plot[ii])])
    
    xx = np.linspace(this_min,this_max,100)
    if ii == 0: ax[0,0].plot(xx,1.6*xx,'-.k')
    ax[ii%2,ii//2].plot(xx,xx,'--')
        
    ax[ii%2,ii//2].errorbar(phaskell_for_plot[ii],prospect_for_plot[ii], xerr=phaskell_sig_for_plot[ii], yerr=phaskell_sig_for_plot[ii],ls='none')
    
    ax[ii%2,ii//2].set_xlabel(compare_plot_labels[ii]+' (P Haskell)')
    ax[ii%2,ii//2].set_ylabel(compare_plot_labels[ii]+' (Prospector)')

fig.suptitle('Phaskell vs Prospector comparison')

plt.tight_layout()

plt.savefig('phaskell_prospector_comparison.jpg')
