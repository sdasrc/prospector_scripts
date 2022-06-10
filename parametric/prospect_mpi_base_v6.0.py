import os, time, sys, psutil, shutil
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Good looking plt params
from matplotlib.font_manager import FontProperties
from matplotlib import gridspec
plt.rcParams['figure.figsize'] = (15,10)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 20

plt.rcParams['xtick.major.pad'] = 7.0
plt.rcParams['xtick.major.size'] = 7.5
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['xtick.minor.pad'] = 7.0
plt.rcParams['xtick.minor.size'] = 3.5
plt.rcParams['xtick.minor.width'] = 1.0

plt.rcParams['ytick.major.pad'] = 7.0
plt.rcParams['ytick.major.size'] = 7.5
plt.rcParams['ytick.major.width'] = 1.5
plt.rcParams['ytick.minor.pad'] = 7.0
plt.rcParams['ytick.minor.size'] = 3.5
plt.rcParams['ytick.minor.width'] = 1.0

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['image.cmap'] = 'Spectral'


# Import prospector and allied libraries
import prospect
import h5py, fsps, sedpy
import dynesty
import corner

# Necessary prospector functions
from prospect import prospect_args
from prospect.fitting import fit_model
from prospect.io import write_results as writer
from prospect.fitting import lnprobfn
from functools import partial
from prospect.plotting.corner import quantile
from prospect.models.transforms import zfrac_to_masses, zfrac_to_sfr

# Implement a low level mpi
from mpi4py import MPI

# Astropy to calculate cosmological params
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u

import traceback
import pickle

from science_helpers import *

script_version = '6.0'
# Whats new
# Dynesty with 2 component dust model

# --------------------------------------------------------------------------- #
#              I  M  P  O  R  T        H  E  L  P  E  R  S                    #
# ----------------------------------------------------------------------------#
from prospector_helpers import *
from science_helpers import *
from input_helpers import *
from result_helpers import *
from prosfitsutilities import *

# --------------------------------------------------------------------------- #
#              P  R  O  S  P  E  C  T  O  R        C O D E                    #
# ----------------------------------------------------------------------------#

def run_prospector(this_obj, **extras):

    obs = build_obs(obs_data,filters,objid=this_obj)
    print('this obj id : ',this_obj,', source id : ',obs['radio_id'])

    # Separate folders to spit out figures and stats of each object
    obj_dir = WORK_DIR+obs['radio_id']

    DELETEOLD = True

    if os.path.isdir(obj_dir) and DELETEOLD: shutil.rmtree(obj_dir, ignore_errors=True)

    os.makedirs(obj_dir)


    src_tag = '{0}/{1}'.format(obj_dir,obs['radio_id'])
    plt_tag = '{0} (Radio ID = {1}, z = {2:.3f})'.format(obs['sour_name'],obs['radio_id'],obs['z_best'])

    fitdata = {}


    # Build the model
    model, model_params = build_model(obs, fixed_metallicity=0.0, add_burst=ADD_BURST, add_duste=ADD_DUSTE,
                    add_neb=ADD_NEB, add_agn=ADD_AGN)

    NDIM = model.ndim

    model_file = '{0}_{1}_{2}_model_desc.txt'.format(src_tag,METHOD,SAMPLER)
    with open(model_file,'w') as f: f.write(model.description)

    sps = build_sps(zcontinuous=ZCONTINUOUS)  

    # ----------------------------------------------------
    #         Dynesty begins here
    # ----------------------------------------------------

    run_params = {}
    # Set this to False if you don't want to do another optimization
    # before dynesty sampling (but note that the "optimization" entry 
    # in the output dictionary will be (None, 0.) in this case)
    # If set to true then another round of optmization will be performed 
    # before sampling begins and the "optmization" entry of the output
    # will be populated.

    run_params["dynesty"] = True
    run_params["optmization"] = False
    run_params["emcee"] = False
    run_params["nested_method"] = NESTED_METHOD
    run_params["nlive_init"] = NLIVE_INIT
    run_params["nlive_batch"] = NLIVE_BATCH
    run_params["nested_dlogz_init"] = NESTED_DLOGZ_INIT
    run_params["nested_posterior_thresh"] = NESTED_POSTERIOR_THRESH
    run_params["verbose"] = False
    run_params["print_progress"] = False
    run_params["nested_maxcall"] = int(1e7)

    # run_params["nested_sample"] = 'rwalk'
    # run_params["nested_bound"] = 'multi'
    # run_params["nested_walks"] = '50'
    # run_params["nested_nlive_init"] = 200
    # run_params["nested_nlive_batch"] = 200
    # run_params["nested_dlogz_init"] = 0.01
    # run_params["nested_weight_kwargs"] = {'pfrac': 1.0},
    # run_params["verbose"] = False
    # run_params["debug"] = False

    output = fit_model(obs, model, sps, lnprobfn=lnprobfn, **run_params)
    print('done dynesty in {0}s'.format(output["sampling"][1]))

    # -------------------------------------------------------------- #
    # data i/o h5 files                                              #
    # -------------------------------------------------------------- #

    from prospect.io import write_results as writer
    hfile = "{0}_{1}_{2}_output.h5".format(src_tag,METHOD,SAMPLER)
    writer.write_hdf5(hfile, run_params, model, obs, output["sampling"][0], output["optimization"][0], tsample=output["sampling"][1], toptimize=output["optimization"][1])


    # import prospect.io.read_results as reader
    # result, obs_fromfile, _ = reader.results_from(hfile, dangerous=False)

    result = output["sampling"][0]

    # -------------------------------------------------------------- #
    # diagnostic trace and corner plots                              #
    # -------------------------------------------------------------- #

    diagnostics = DIAGNOSTICS

    if diagnostics:
        tracefig = reader.traceplot(result, figsize=(16,8))
        plt.rcParams['font.size'] = 10
        plt.tight_layout()
        plotname = '{0}_dynesty_trace.pdf'.format(src_tag)
        plt.savefig(plotname)

        plt.rcParams['figure.figsize'] = (20,20)
        plt.rcParams['font.size'] = 12
        thin = 1
        cornerfig = reader.subcorner(result, start=0, thin=thin,
                                   fig=plt.subplots(NDIM,NDIM)[0])
        plotname = '{0}_dynesty_corner.pdf'.format(src_tag)
        plt.savefig(plotname)

    
    # ----------------------------------------------------
    #         Dynesty ended here
    # ----------------------------------------------------  


    # ----------------------------------------------------
    #         Calculate uncertainties
    #        taken from prospector demo rst file
    # ----------------------------------------------------  


    # use the prospector utilities to calculate uncertainties in a go
    max_lnp = np.argmax(result['logl'])    
    theta_max = result['samples'][max_lnp, :].copy()
    flatchain = result['samples']

    # 16th, 50th, and 84th percentiles of the posterior
    weights = np.exp(result['logwt'] - result['logz'][-1])
    post_pcts = quantile(flatchain.T, q=[0.16, 0.50, 0.84], weights=weights)

    cosmo = FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc, Tcmb0=2.725 * u.K, Om0=0.3)
    lumdist  = cosmo.luminosity_distance(obs['z_best']).value

    # ----------------------------------------------------
    #         Initiate params for plotting
    # ---------------------------------------------------- 

    theta_labels = model.free_params
    theta_keys = {}
    for xx in range(len(theta_labels)) : theta_keys[theta_labels[xx]]=xx       

    # ----------------------------------------------------
    #         We'll be dealing with rest frame here
    #         Let's set up accordingly
    # ----------------------------------------------------
    obs_phot, obs_err, phot_wave = obs['maggies'], obs['maggies_unc'], obs['phot_wave']
    spec_wave = sps.wavelengths # rest frame

    wframe = 'rest'

    phot_wave = phot_wave/(1+obs['z_best']) # plotting curves in rest frame wavelength
    # spec_wave = (wave_factor)*spec_wave # plotting curves in rest frame wavelength

    # ---------------------------------------------------------------------
    # Store all thetas in a dictionary
    # ---------------------------------------------------------------------
    theta_dict = {}
    for nn in range(NDIM):
        theta_dict[theta_labels[nn]] = {'best' : theta_max[nn]}

    for tt in range(len(theta_labels)):
        p16,p50,p84 = post_pcts[tt]
        theta_dict[theta_labels[tt]]['50ptile'] = p50
        theta_dict[theta_labels[tt]]['16ptile'] = p16
        theta_dict[theta_labels[tt]]['84ptile'] = p84


    # -------------------------------------------------------------- #
    # Define arrays for physical parameters
    # -------------------------------------------------------------- #   

    # Binning for SED
    specbins = 200
    log_spec = np.log(spec_wave)  # Logarithmic binning for SED plots
    wave_bin_log_width = (log_spec[0]-log_spec[-1])/specbins
    wave_bin_log = np.linspace(log_spec[0], log_spec[-1],specbins)
    wave_bin_log_mid = wave_bin_log - (wave_bin_log_width/2.)
    np.append(wave_bin_log_mid,[ wave_bin_log[-1] + (wave_bin_log_width/2.)])

    wave_bin, wave_bin_mid = np.exp(wave_bin_log), np.exp(wave_bin_log_mid)

    digitize_spec_bins = np.digitize(spec_wave,wave_bin)
    # Empty arrays to put stuffs into
    binned_sed = [[] for ii in wave_bin_mid]
    binned_weight = [[] for ii in wave_bin_mid]

    # Get empty arrays for stellar masses and dust
    ldust_arr, mfrac_arr, mcurr_arr = [], [], []    
    all_sfr_arr, sfrqtiles, mstar100_arr, sfr100_arr, ssfr100_arr = [],[],[],[],[]


    # Define arrays for the sfh plots
    t_end, bin_size = 13.8, SFR_BIN_SIZE
    # Divide the time axis into multiple bins - then average over all the sfr points in there
    # Create an array containing mid point of each time bins           
    tbins = np.arange(SFR_BIN_SIZE,t_end,SFR_BIN_SIZE)
    binmids = tbins - SFR_BIN_SIZE/2. 
    binmids = np.append(binmids,[tbins[-1] + SFR_BIN_SIZE/2.] )
    binsfrs = [[] for ii in binmids]
    binsfrweights = [[] for ii in binmids]
    mstar_100myr_arr, sfr_100myr_arr, ssfr_100myr_arr = [], [], []
    shouldiplot = np.random.choice(range(len(flatchain)), 10, p=weights)
    # Will create an array of random states whose sfhs will be plotted
    tltoplot,sfhtoplot = [],[]
        
    # -------------------------------------------------------------- #
    # Iterate over chains and calculate physical params
    # -------------------------------------------------------------- # 

    for tt in range(len(flatchain)):
        this_state = flatchain[tt]
        this_weight = weights[tt]
        this_spec, this_phot, this_mfrac = model.predict(this_state,obs=obs,sps=sps)
        for ii in range(len(spec_wave)):
            if digitize_spec_bins[ii] < len(binned_sed):
                binned_sed[digitize_spec_bins[ii]].append([this_spec[ii]])
                binned_weight[digitize_spec_bins[ii]].append(this_weight)
                
        # Luminosity and mass frac parts                
        this_mcurr = this_state[theta_keys['mass']]*this_mfrac
        ldust = get_ldust(spec_wave,magp_phot_Llambda(spec_wave, pros_to_magp_phot(spec_wave, this_spec, obs['z_best'])))

        ldust_arr.append(ldust)
        mfrac_arr.append(this_mfrac)
        mcurr_arr.append(this_mcurr)

        # SFR Parts
        tl, sfr, sfr_uplim, st_mass_100myr,sfr_100myr,ssfr_100myr = get_sfr(this_state,theta_keys,isburst=ADD_BURST)
        # Divide that by the time to get star formation rate averaged over 100 myrs
        mstar100_arr.append(st_mass_100myr)
        sfr100_arr.append(sfr_100myr)
        ssfr100_arr.append(ssfr_100myr)

        if tt in shouldiplot:
            tltoplot.append(np.array(tl))
            sfhtoplot.append(sfr)

        binnify = np.digitize(tl,tbins)
        for ii in range(len(tl)):
            binsfrs[binnify[ii]].append([sfr[ii]])
            binsfrweights[binnify[ii]].append(this_weight)

    # -------------------------------------------------------------- #
    # Calculate all quantiles
    # -------------------------------------------------------------- # 

    sed_50, sed_16, sed_84 = [],[],[]
    for ii in range(len(wave_bin_mid)):
        if len(binned_sed[ii])>0:
            sed16,sed50,sed84 = weighted_quantile(np.ndarray.flatten(np.array(binned_sed[ii])),[0.16,0.5,0.84],sample_weight=binned_weight[ii])
        else: sed16,sed50,sed84 = np.NaN, np.NaN, np.NaN
        sed_16.append(sed16)
        sed_50.append(sed50)
        sed_84.append(sed84)

    mstar100_16ptile, mstar100_50ptile, mstar100_84ptile = weighted_quantile(mstar100_arr,[0.16,0.5,0.84],sample_weight=weights)
    sfr100_16ptile, sfr100_50ptile, sfr100_84ptile = weighted_quantile(sfr100_arr,[0.16,0.5,0.84],sample_weight=weights)
    ssfr100_16ptile, ssfr100_50ptile, ssfr100_84ptile = weighted_quantile(ssfr100_arr,[0.16,0.5,0.84],sample_weight=weights)

    # -------------------------------------------------------------- #
    # Best fits
    # -------------------------------------------------------------- # 

    best_spec, best_phot, best_mfrac = model.predict(theta_max,obs=obs,sps=sps)
    best_mcurr = theta_max[theta_keys['mass']]*best_mfrac

    # -------------------------------------------------------------- #
    # Get all data
    # -------------------------------------------------------------- # 

    ## Get all the data in
    # Observed Phot
    obs_phot_Lnu_Lsol = pros_to_magp_phot(phot_wave, obs_phot, obs['z_best'])
    obs_phot_Llambda_Lsol = magp_phot_Llambda(phot_wave, obs_phot_Lnu_Lsol)


    obs_phot_Lnu_Lsol_err = pros_to_magp_phot(phot_wave, obs_err, obs['z_best'])
    obs_phot_Llambda_Lsol_err = magp_phot_Llambda(phot_wave, obs_phot_Lnu_Lsol_err)

    # Prospector phot
    pros_phot_Lnu_Lsol = pros_to_magp_phot(phot_wave, best_phot, obs['z_best'])
    pros_phot_Llambda_Lsol = magp_phot_Llambda(phot_wave, pros_phot_Lnu_Lsol)

    # Magphys phot
    magp_phot,paulchi2 = get_magp_phot(obs['sour_name']) # get
    magp_phot_Llambda_Lsol = magp_phot_Llambda(phot_wave, magp_phot)

    # pros spec
    pros_spec_Lnu_Lsol_best = pros_to_magp_phot(spec_wave, best_spec, obs['z_best'])
    pros_spec_Llambda_Lsol_best = magp_phot_Llambda(spec_wave, pros_spec_Lnu_Lsol_best)

    pros_spec_Lnu_Lsol_16 = pros_to_magp_phot(wave_bin_mid, sed_16, obs['z_best'])
    pros_spec_Llambda_Lsol_16 = magp_phot_Llambda(wave_bin_mid, pros_spec_Lnu_Lsol_16)

    pros_spec_Lnu_Lsol_50 = pros_to_magp_phot(wave_bin_mid, sed_50, obs['z_best'])
    pros_spec_Llambda_Lsol_50 = magp_phot_Llambda(wave_bin_mid, pros_spec_Lnu_Lsol_50)

    pros_spec_Lnu_Lsol_84 = pros_to_magp_phot(wave_bin_mid, sed_84, obs['z_best'])
    pros_spec_Llambda_Lsol_84 = magp_phot_Llambda(wave_bin_mid, pros_spec_Lnu_Lsol_84)

    # magp spec
    magp_spec_Llambda_Lsol, magp_spec_wave = get_magphys_sed(obs['sour_name']) # get    

    # all units in lambda L_lambda/ L_sol

    # -------------------------------------------------------------- #
    # Plot SED + overplot with Magphys
    # -------------------------------------------------------------- # 

    # all units in lambda L_lambda/ L_sol

    plt.rcParams['figure.figsize'] = (15,12)
    plt.rcParams['font.size'] = 12

    fitdata = {}

    plotname = '{0}_sed'.format(src_tag)
    plottitle = plt_tag+ ' - ' + 'SED'
    fitdata[0] = {'fit_phot':best_phot, 'fit_spec':best_spec, 'datalabel':'PROSPECTOR fit'}
    nfitdata = 1

    fig = plt.figure()

    # ACTUAL PLOTTING NOW
    # chi error for residual plot
    fig.add_axes((.1,.3,.8,.6))


    plt.fill_between(wave_bin_mid, wave_bin_mid*pros_spec_Llambda_Lsol_16, wave_bin_mid*pros_spec_Llambda_Lsol_84, 
                     color='lightblue', alpha=0.5, zorder=1)

    #  spectra
    plt.loglog(magp_spec_wave/(1+obs['z_best']), magp_spec_wave*magp_spec_Llambda_Lsol,
               'deeppink',lw=1, label='magphys spectrometry', alpha=0.7 )
    plt.loglog(spec_wave, spec_wave*pros_spec_Llambda_Lsol_best, 
               lw=1, color='blue', label='best fit spectrometry', zorder=3, alpha=0.5)
    plt.loglog(wave_bin_mid, wave_bin_mid*pros_spec_Llambda_Lsol_50,
               label='50th percentile SED Fit',lw=1, ls='--',color='blue', zorder=3)



    #  photometry
    plt.loglog(phot_wave, phot_wave*magp_phot_Llambda_Lsol, 
            ms=10, marker='s',linewidth=30, color='deeppink', markeredgecolor='deeppink', markerfacecolor='none',  
           ls = '', label='magphys photometry', zorder=3)
    plt.loglog(phot_wave, phot_wave*pros_phot_Llambda_Lsol, 
               ms=10, marker='s',linewidth=30, color='blue', markeredgecolor='darkblue', 
                 markerfacecolor='none',  ls = '', label='best fit photometry', zorder=3)

    # Plot observed phot
    plt.errorbar(phot_wave, phot_wave*obs_phot_Llambda_Lsol, yerr = phot_wave*obs_phot_Llambda_Lsol_err, 
            mew=2, marker='.', color='black',  ls = '', ecolor='gray', label='Observed photometry', zorder=3)

    plt.xlim(5.e2,2.e7)

    txmin, txmax = np.where(spec_wave > 2.e3)[0][0], np.where(spec_wave < 1.e7)[0][-1]
    tymin = spec_wave[txmin:txmax]*pros_spec_Llambda_Lsol_best[txmin:txmax]
    print('{0:.3e}'.format(0.1*np.nanmin(tymin)))
    print('{0:.3e}'.format(10*np.nanmax(tymin)))

    yup = np.nanmax( [ np.nanmax(magp_spec_wave*magp_spec_Llambda_Lsol), np.nanmax(wave_bin_mid*pros_spec_Llambda_Lsol_84), 
         np.nanmax(phot_wave*magp_phot_Llambda_Lsol), np.nanmax(phot_wave*pros_phot_Llambda_Lsol), 
                      np.nanmax(phot_wave*(obs_phot_Llambda_Lsol+obs_phot_Llambda_Lsol_err)) ])

    plt.ylim(0.1*np.nanmin(tymin), 2*yup)

    plt.xscale('log')
    plt.yscale('log')
    plt.title(plottitle)
    plt.ylabel(r'$\lambda L_{\lambda}/L_{\odot}$')
    plt.legend(loc='best')

    # RESIDUALS
    # get residuals
    mchi_arr, mchi_sq, mchi_nan_mask = get_chi2(obs_phot_Llambda_Lsol, magp_phot_Llambda_Lsol, obs_phot_Llambda_Lsol_err)
    pchi_arr, pchi_sq, pchi_nan_mask = get_chi2(obs_phot_Llambda_Lsol, pros_phot_Llambda_Lsol, obs_phot_Llambda_Lsol_err)

    # plot residuals
    fig.add_axes((.1,.1,.8,.2))
    plt.errorbar(phot_wave[mchi_nan_mask], mchi_arr[mchi_nan_mask], linewidth=8,
                 mew=1, marker='.', color='magenta', ls = '', label=r"magphys $\chi^2 =$ "+str(int(mchi_sq)))
    plt.errorbar(phot_wave[pchi_nan_mask], pchi_arr[pchi_nan_mask], linewidth=8,
                 mew=1, marker='.', color='darkblue', ls = '', label=r"prospector $\chi^2 =$ "+str(int(pchi_sq)))

    plt.xlabel('rest frame wavelength [A]')

    chi_max_p = np.max((np.abs(np.nanmin(pchi_arr)),np.abs(np.nanmax(pchi_arr))))
    chi_max_m = np.max((np.abs(np.nanmin(mchi_arr)),np.abs(np.nanmax(mchi_arr))))
    chi_max = np.max([chi_max_m,chi_max_p])

    plt.xlim(5.e2,2.e7)
    plt.ylim(-1.2*chi_max,1.2*chi_max)
    plt.axhline(0,ls='--')
    plt.xscale('log')
    plt.ylabel(r"$\chi$")

    print((-1.2*chi_max,1.2*chi_max))

    for ff in obs['filters']:
        waves, trans = ff.wavelength, ff.transmission
        trans = trans/(trans.max()*3631)
        plt.plot(waves,trans,alpha=0.3,color='gray')


    plt.legend(loc='best')
    plt.savefig(plotname+'_overplot_magp.pdf')

    # -------------------------------------------------------------- #
    # Plot SED 
    # -------------------------------------------------------------- # 

    # all units in lambda L_lambda/ L_sol

    plt.rcParams['figure.figsize'] = (15,12)
    plt.rcParams['font.size'] = 12

    fitdata = {}

    plotname = '{0}_sed'.format(src_tag)
    plottitle = plt_tag+ ' - ' + 'SED'
    fitdata[0] = {'fit_phot':best_phot, 'fit_spec':best_spec, 'datalabel':'PROSPECTOR fit'}
    nfitdata = 1

    fig = plt.figure()

    # ACTUAL PLOTTING NOW
    # chi error for residual plot
    fig.add_axes((.1,.3,.8,.6))


    plt.fill_between(wave_bin_mid, wave_bin_mid*pros_spec_Llambda_Lsol_16, wave_bin_mid*pros_spec_Llambda_Lsol_84, 
                     color='lightblue', alpha=0.5, zorder=1)

    #  spectra
    plt.loglog(spec_wave, spec_wave*pros_spec_Llambda_Lsol_best, 
               lw=1, color='blue', label='best fit spectrometry', zorder=3, alpha=0.5)
    plt.loglog(wave_bin_mid, wave_bin_mid*pros_spec_Llambda_Lsol_50,
               label='50th percentile SED Fit',lw=1, ls='--',color='blue', zorder=3)



    #  photometry
    plt.loglog(phot_wave, phot_wave*pros_phot_Llambda_Lsol, 
               ms=10, marker='s',linewidth=30, color='blue', markeredgecolor='darkblue', 
                 markerfacecolor='none',  ls = '', label='best fit photometry', zorder=3)

    # Plot observed phot
    plt.errorbar(phot_wave, phot_wave*obs_phot_Llambda_Lsol, yerr = phot_wave*obs_phot_Llambda_Lsol_err, 
            mew=2, marker='.', color='black',  ls = '', ecolor='gray', label='Observed photometry', zorder=3)

    plt.xlim(5.e2,2.e7)
    plt.ylim(0.1*np.nanmin(tymin), 2*yup)

    plt.xscale('log')
    plt.yscale('log')
    plt.title(plottitle)
    plt.ylabel(r'$\lambda L_{\lambda}/L_{\odot}$')
    plt.legend(loc='best')

    # RESIDUALS
    # plot residuals
    fig.add_axes((.1,.1,.8,.2))
    plt.errorbar(phot_wave[pchi_nan_mask], pchi_arr[pchi_nan_mask], linewidth=8,
                 mew=1, marker='.', color='darkblue', ls = '', label=r"prospector $\chi^2 =$ "+str(int(pchi_sq)))

    plt.xlabel('rest frame wavelength [A]')

    plt.xlim(5.e2,2.e7)
    plt.ylim(-1.2*chi_max,1.2*chi_max)
    plt.axhline(0,ls='--')
    plt.xscale('log')
    plt.ylabel(r"$\chi$")

    print((-1.2*chi_max,1.2*chi_max))

    for ff in obs['filters']:
        waves, trans = ff.wavelength, ff.transmission
        trans = trans/(trans.max()*3631)
        plt.plot(waves,trans,alpha=0.3,color='gray')


    plt.legend(loc='best')
    plt.savefig(plotname+'.pdf')

    # -------------------------------------------------------------- #
    # Save the SED and Photometries to a file
    # -------------------------------------------------------------- #
    fitfile = "{0}/EN1_{1}_photfit.dat".format(obj_dir,obs['radio_id'])
    with open(fitfile,'w') as f:
        print('# wavelength [A]    obs_phot    obs_phot_err    best_fit_phot [Lsol/A]', file=f)
        np.savetxt(f, np.transpose([phot_wave, obs_phot_Llambda_Lsol, obs_phot_Llambda_Lsol_err, pros_phot_Llambda_Lsol]), fmt='%1.5e')

    specfile = "{0}/EN1_{1}_sedfit.dat".format(obj_dir,obs['radio_id'])
    with open(specfile,'w') as f:
        print('# wavelength [A]    best_fit_spec [Lsol/A]', file=f)
        np.savetxt(f, np.transpose([spec_wave, pros_spec_Llambda_Lsol_best]), fmt='%1.5e')

    sedfile = "{0}/EN1_{1}_binnedsed.dat".format(obj_dir,obs['radio_id'])
    with open(sedfile,'w') as f:
        print('# wavelength [A]    spec_16ptile    spec_50ptile    spec_84ptile [Lsol/A]', file=f)
        np.savetxt(f, np.transpose([wave_bin_mid, pros_spec_Llambda_Lsol_16, pros_spec_Llambda_Lsol_50, pros_spec_Llambda_Lsol_84]), fmt='%1.5e')

    # -------------------------------------------------------------- #
    # Get dust, mfrac stats
    # -------------------------------------------------------------- #
    p16_ldust, p50_ldust, p84_ldust = weighted_quantile(ldust_arr,[0.16,0.5,0.84],sample_weight=weights)
    ldust_best = get_ldust(spec_wave,magp_phot_Llambda(spec_wave, pros_to_magp_phot(spec_wave, best_spec, obs['z_best'])))

    p16_mfrac, p50_mfrac, p84_mfrac = weighted_quantile(mfrac_arr,[0.16,0.5,0.84],sample_weight=weights)
    p16_mcurr, p50_mcurr, p84_mcurr = weighted_quantile(mcurr_arr,[0.16,0.5,0.84],sample_weight=weights)

    # ---------------------------------------------------------------
    # Star formation rates
    # ---------------------------------------------------------------

    sfhfile = "{0}/EN1_{1}_sfh.dat".format(obj_dir,obs['radio_id'])
    
    plt.rcParams['figure.figsize'] = (12,8)
    plt.rcParams['font.size'] = 12
    plt.figure()
        
    t_end, bin_size = 13.8, SFR_BIN_SIZE

    for tt in range(len(sfhtoplot)):
        tl,tsfr = tltoplot[tt], sfhtoplot[tt]
        plt.plot(tl, tsfr, color='black',alpha=0.4,zorder=1)

    # Get the sfr for theta max
    tl_best, sfr_best, sfr_uplim_best, st_mass_100myr_best,sfr_100myr_best,ssfr_100myr_best = get_sfr(theta_max,theta_keys,isburst=ADD_BURST)
    plt.plot(tl_best, sfr_best, color='red', lw=1, zorder=1, label='highest logprob')


    sfh_50, sfh_16, sfh_84 = [],[],[]
    for ii in range(len(binmids)):
        if len(binsfrs[ii])>0:
            sf16, sf50, sf84 = weighted_quantile(np.ndarray.flatten(np.array(binsfrs[ii])),[0.16,0.5,0.84],sample_weight=binsfrweights[ii])
        else: sf16, sf50, sf84 = np.NaN, np.NaN, np.NaN
        sfh_50.append(sf50)
        sfh_16.append(sf16)
        sfh_84.append(sf84)

    plt.plot(binmids,sfh_50,label='50 ptile',lw=1,  color='blue',zorder=3)
    plt.fill_between(binmids, sfh_16, sfh_84, color='lightblue', alpha=0.5,zorder=2)

    plt.xlabel(r'lookback time $(t_l)$ (Gyr)')
    plt.ylabel(r'$SFR~(M_\odot/yr)$')
    plt.legend(loc='best')
    plt.yscale('log')
    plt.ylim([1.e-2,1.e5])
    plt.tight_layout()


    plt.title(plottitle)
    plotname = '{0}_sfr'.format(src_tag)
    plt.legend()
    # Add legend
    # Move sfr calc to main loop
    # Get errors on that

    plt.savefig(plotname+'.pdf')
    
    with open(sfhfile,'w') as f: # Save SFH to a file
        print('# agebin mid [yr]    sfh_16ptile    sfh_50ptile    sfh_84ptile [Msol/yr]', file=f)
        np.savetxt(f, np.transpose([binmids, sfh_16, sfh_50, sfh_84]), fmt='%1.5e')

    # -------------------------------------------------------------- #
    
    # -------------------------------------------------------------- #

    spitoutfile = obj_dir+'/'+str(obs['radio_id'])+'_output.txt'

    with open(spitoutfile,'w') as f:
        # Galaxy identification
        print('radio_id = ', obs['radio_id'],file=f)
        print('sour_id = ', obs['sour_id'],file=f)
        print('sour_name = "{}"'.format(obs['sour_name']) ,file=f)

        # Redshift information
        print('z_best = ', obs['z_best'],file=f)
        print('lum_dist = ', obs['lum_dist'],file=f)
        print('age_univ = ', obs['age_univ'],file=f)

        # Prospector run params
        print('chi2_best = ', pchi_sq ,file=f)
        print('chi2_magphys = ', mchi_sq ,file=f)
        

        # Result free params
        # - mstar
        for tt in range(NDIM):
            print(theta_labels[tt]+'_best = ', theta_dict[theta_labels[tt]]['best'] ,file=f)
            print(theta_labels[tt]+'_16ptile = ', theta_dict[theta_labels[tt]]['16ptile'] ,file=f)
            print(theta_labels[tt]+'_50ptile = ', theta_dict[theta_labels[tt]]['50ptile'] ,file=f)
            print(theta_labels[tt]+'_84ptile = ', theta_dict[theta_labels[tt]]['84ptile'] ,file=f)
                
        ### Physical parameters

        # - SFR       - star formation rate in Msun/yr averaged over last 0.1 Gyr
        print('st_mass_100myr_best = ', st_mass_100myr_best ,file=f)
        print('mstar100_16ptile = ', mstar100_16ptile ,file=f)
        print('mstar100_50ptile = ', mstar100_50ptile ,file=f)
        print('mstar100_84ptile = ', mstar100_84ptile ,file=f)


        print('mfrac_best = ', best_mfrac ,file=f)
        print('mfrac_16ptile = ', p16_mfrac ,file=f)
        print('mfrac_50ptile = ', p50_mfrac ,file=f)
        print('mfrac_84ptile = ', p84_mfrac ,file=f)

        print('mcurr_best = ', best_mcurr ,file=f)
        print('mcurr_16ptile = ', p16_mcurr ,file=f)
        print('mcurr_50ptile = ', p50_mcurr ,file=f)
        print('mcurr_84ptile = ', p84_mcurr ,file=f)

        print('sfr100_best = ', sfr_100myr_best ,file=f)
        print('sfr100_16ptile = ', sfr100_16ptile ,file=f)
        print('sfr100_50ptile = ', sfr100_50ptile ,file=f)
        print('sfr100_84ptile = ', sfr100_84ptile ,file=f)

        print('ssfr100_best = ', ssfr_100myr_best ,file=f)
        print('ssfr100_16ptile = ', ssfr100_16ptile ,file=f)
        print('ssfr100_50ptile = ', ssfr100_50ptile ,file=f)
        print('ssfr100_84ptile = ', ssfr100_84ptile ,file=f)
        
        print('ldust_best = ', ldust_best ,file=f)
        print('ldust_16ptile = ', p16_ldust ,file=f)
        print('ldust_50ptile = ', p50_ldust ,file=f)
        print('ldust_84ptile = ', p84_ldust ,file=f)        

        for mm in model.fixed_params:
            print('{0} = {1}'.format(mm,model.params[mm][0]), file=f)


        print('good_fluxes = ', obs['good_fluxes'] ,file=f)

    return obs['radio_id'], pchi_sq






if __name__ == '__main__':
    script_start = time.time()


    if len(sys.argv) < 3:
        print('{0} outputs received, inadequate. Please supply valid working directory and parameter file inputs. Will exit.'.format(len(sys.argv)))
        exit()
    
    WORK_DIR = sys.argv[1] 
    inp_params = sys.argv[2]

    if WORK_DIR[-1] != '/': WORK_DIR = WORK_DIR+'/'
    if not os.path.isfile(WORK_DIR+inp_params):
        print('Please supply a parameter file. Will exit.')
        exit()
    with open(WORK_DIR+inp_params,'r') as f: inplines = f.readlines()
    for this_line in inplines: exec(this_line) 

    TOTAL_OBJS = NODES * NPROCS * NRUNS

    NOBJS = NPROCS * NRUNS

    begn, no_of_rows = OBJID, OBJID+NOBJS

    # Will work irrespective of where its run
    for itdir in FITS_DIR_LIST:
        if os.path.isdir(itdir):
            FITS_DIR = itdir
            break


    datafile = FITS_DIR+DATA_FITS_FILE

    if SPECOBJS:
        if len(OBJLISTFILE):
            OBJLISTFILE = WORK_DIR+OBJLISTFILE
            OBJLIST =  list(np.loadtxt(OBJLISTFILE,dtype=int))
        else: OBJLIST = OBJLISTARR
    else: OBJLIST = []

    obs_data, filter_names, cnt2 = data_import(fits_fname=datafile,specobjs=SPECOBJS,objlist=OBJLIST)

    # Will work irrespective of where its run
    for itdir in FILT_DIR_LIST:
        if os.path.isdir(itdir):
            FILT_DIR = itdir
            break

    filters = get_filters(FILT_DIR, filter_names)

    # Implement a rudimentary mpi
    # Get the rank of the processor - and run that object in SERIAL

    # Implement date string later
    # dtstr = datetime.now().strftime('_%y%m%d_%H%M%S')
    dtstr = ''

    succesfile = WORK_DIR+dtstr+'_success.txt'
    timefile = WORK_DIR+dtstr+'_script_runtime.txt'
    
    if not os.path.isfile(succesfile):
        with open(succesfile,'w') as f:
            print('# obj    proc    time     chi2\n# ----------\n', file=f) 

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank() # get your process ID


    for this_run in range(NRUNS):
        this_start = time.time()
        this_obj = OBJID + this_run*NPROCS + rank

        diditrun = True
        excepfile = WORK_DIR+dtstr+'_exception_'+str(this_obj)+'.txt'

        try:
            which_obj, best_chi_sq = run_prospector(this_obj)
        except Exception as e:
            with open(excepfile, 'w') as f:
                print(traceback.format_exc(),file=f)
                diditrun = False
        
        this_end = time.time()
        if diditrun:
            with open(succesfile,'a') as f:
                # Obj processor time chi2
                print('{0}    {1}    {2:.3f}    {3}'.format(which_obj, rank, (this_end - this_start)/60, best_chi_sq), file=f)


    script_end = time.time()
    with open(timefile,'a') as f:
        print('\n# Script took {0:.3f} minutes.'.format((script_end - script_start)/60), file=f) 
