import os, csv, h5py
import numpy as np
from prospect.io import read_results
import corner
import matplotlib.pyplot as plt
dirs = np.array([xx for xx in os.listdir('.') if 'dir' in xx])

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
    result, obs, _ = read_results.results_from(outfile, dangerous=False)
    
    with open(csvfile, newline='') as f:
        csvread = csv.reader(f, delimiter='\t')
        data = list(csvread)
    
    theta = []
    for ii in free_parms:
        theta.append(float(data[3][ii]))

    plt.figure()
    fig, ax = plt.subplots(nrows=1, ncols=2)


    tage, tau, tl = theta[2], theta[3], np.linspace(0,13,1000) # All ages in Gyrs
    print('tage = {0}, tau = {1}'.format(tage, tau))
    sfr = ( (tage - tl)/tau ) * np.exp( -( (tage - tl)/tau ) )

    ax[0].plot(tl, sfr, label='tage = {0:.3f}, tau = {1:.3f}'.format(tage, tau))
    ax[0].axhline(0,ls='--',color='orange')
    ax[0].axvline(0.3,ls='--',color='cyan')
    ax[0].axvline(10,ls='--',color='cyan')
    ax[0].set_xlabel(r'lookback time $(t_l)$ (Gyr)')
    ax[0].set_ylabel(r'A $\psi(t_l)$')
    ax[0].legend()


    ax[1].loglog(tl, sfr, label='tage = {0:.3f}, tau = {1:.3f}'.format(tage, tau))
    ax[1].axhline(0,ls='--',color='orange')
    ax[1].axvline(0.3,ls='--',color='cyan')
    ax[1].axvline(10,ls='--',color='cyan')
    ax[1].set_xlabel(r'lookback time $(t_l)$ (Gyr)')
    ax[1].set_ylabel(r'A $\psi(t_l)$')
    ax[1].legend()

    fig.suptitle('Obj {0} 128 walkers 4096iter - SFR vs Lookback time'.format(objid))
    fig.tight_layout()
    fig.savefig(plotname)
print('Done')
