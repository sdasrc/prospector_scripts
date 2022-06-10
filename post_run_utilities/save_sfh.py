import os, csv, h5py
import numpy as np
from prospect.io import read_results
import corner
import matplotlib.pyplot as plt
dirs = np.array([xx for xx in os.listdir('.') if 'dir' in xx])

from scipy.integrate import simpson


plt.rcParams['figure.figsize'] = (20,10)
plt.rcParams['font.size'] = 20

free_parms = np.array([2,4,6,9,11,13,14,15])
cnt = 1
for this_dir in dirs:
    print(cnt,this_dir)
    objid = this_dir.split('_')[2]
    outfile = "{0}/en1_{1}_emcee_128walkers.h5".format(this_dir,objid)
    csvfile = "{0}/{1}_128wlk_4096it__f1_model_params.csv".format(this_dir,objid)
    plotname = "{0}/{1}_128wlk_4096it__fig_2_sfr.png".format(this_dir,objid)
    res, obs, _ = read_results.results_from(outfile, dangerous=False)
    
    with open(csvfile, newline='') as f:
        csvread = csv.reader(f, delimiter='\t')
        data = list(csvread)
    
    theta = []
    for ii in free_parms:
        theta.append(float(data[3][ii]))

    plt.figure()
    tages, taus, ages, frac_ages, outages = [], [], [], [], []
    maccept, miccept = np.max(res['acceptance']),np.min(res['acceptance'])
    for ind in range(np.shape(res['chain'])[0]):
        this_chain = res['chain'][ind]
        accept = res['acceptance'][ind]
        tage, tau, = this_chain[-1][2], this_chain[-1][3]
        tages.append(tage)
        taus.append(tau)
        tl = np.linspace(0.3,tage,1000) # All ages in Gyrs
        sfr = ( (tage - tl)/tau ) * np.exp( -( (tage - tl)/tau ) )
        area = simpson(sfr,x=tl)   # area under the curve gives stellar mass
        ages.append(area*1.e9)   # Since time is in gyrs, converting it to years
        frac_ages.append((area*1.e9)/this_chain[-1][0])
        outages.append(this_chain[-1][0])
        if accept > np.mean([miccept,maccept]):
            plt.plot(tl, sfr, color='blue')
        else:
            plt.plot(tl, sfr, color='orange', ls='--')

    plt.xlabel(r'lookback time $(t_l)$ (Gyr)')
    plt.ylabel(r'$\psi(t_l)$')

    plt.title('Obj {0} {1}walkers {2}iter - SFR vs Lookback time'.format(res['obs']['sour_id'],\
            np.shape(res['chain'])[0],np.shape(res['chain'])[1] ) +'\n' + \
          't_age ({0:.3f}, {1:.3f}), tau ({2:.3f}, {3:.3f})'.format(np.min(tages), \
            np.max(tages),np.min(taus),np.min(taus)))
    plt.tight_layout()
    plt.savefig('{3:.3f}_{0}_{1}w_{2}it_sfr.jpg'.format(res['obs']['sour_id'],\
            np.shape(res['chain'])[0],np.shape(res['chain'])[1],np.median(frac_ages)))

    plt.figure()
    plt.plot(ages, '.b',label=r'$M_{area~under~sfr~curve}$'+'; (min={0:.2e}, max={1:.2e})'.format(np.min(ages),np.max(ages)))
    plt.plot(outages, '.r', label=r'$M_{prospect~output}$'+'; (min={0:.2e}, max={1:.2e})'.format(np.min(outages),np.max(outages)))
    plt.legend(loc='best')
    plt.xlabel('walkers')
    plt.ylabel(r'Stellar mass ($M_\odot/yr$)')
    plt.title('Obj {0} {1}walkers {2}iter - stellar mass outputs'.format(res['obs']['sour_id'],\
                np.shape(res['chain'])[0],np.shape(res['chain'])[1] ))
    plt.savefig('{3:.3f}_{0}_{1}w_{2}it_smass_out.jpg'.format(res['obs']['sour_id'],\
                np.shape(res['chain'])[0],np.shape(res['chain'])[1],np.median(frac_ages)))

    plt.figure()
    print(np.median(frac_ages))
    plt.plot(frac_ages, '.k', label='median = {0:.3f}'.format(np.median(frac_ages)))
    plt.xlabel('walkers')
    plt.ylabel(r'$\frac{M_{area~under~sfr~curve}}{M_{prospect~output}}$')
    plt.axhline(1, ls='--')
    plt.title('Obj {0} {1}walkers {2}iter - stellar mass output ratio'.format(res['obs']['sour_id'],\
                np.shape(res['chain'])[0],np.shape(res['chain'])[1] ))
    plt.legend(loc='best')
    plt.savefig('{3:.3f}_{0}_{1}w_{2}it_smass_ratio.jpg'.format(res['obs']['sour_id'],\
                np.shape(res['chain'])[0],np.shape(res['chain'])[1],np.median(frac_ages)))

print('Done')
