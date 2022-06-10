import os, sys
import fitsio
from fitsio import FITS,FITSHDR
import numpy as np
from tqdm import tqdm

filetags = '_output.txt'

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
    
outfits = FITS(fname,'rw')

subfolders = [ f.path for f in os.scandir(WORK_DIR) if f.is_dir() ]
objdirs = [ f.split('/')[-1] for f in subfolders if f.split('/')[-1].isnumeric() ]

column_names  = ['index']
for this_dir in objdirs:
    outtxt = WORK_DIR+this_dir+'/'+this_dir+filetags
    if os.path.isfile(outtxt):
        print(this_dir)
        break
with open(WORK_DIR+this_dir+'/'+this_dir+filetags,'r') as f: outlines = f.readlines()
for this_line in outlines: column_names.append(this_line.split(' ')[0])

nrows = len(objdirs)
columns = [('index','i8'),('radio_id','i8'),('sour_id','i8'),('sour_name','a30')]

for this_col in column_names[4:]: columns.append((this_col,'f8'))
 
data = np.zeros(nrows, dtype=columns)


indx = 0
for this_dir in tqdm(objdirs):
    outtxt = WORK_DIR+this_dir+'/'+this_dir+filetags
    if os.path.isfile(outtxt):
        with open(outtxt,'r') as f: outlines = f.readlines()
        for this_line in outlines: exec(this_line)
                
        data[indx]['index'] = indx
        for this_var in column_names[1:]: exec('data[{0}]["{1}"] = {1}'.format(indx,this_var))
        indx += 1
    else:
        continue

outfits.write(data)
outfits.close()

print('{0} out of {1} successfully parsed.'.format(indx+1,nrows))