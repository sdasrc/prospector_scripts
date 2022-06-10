import os, time, sys, psutil, tqdm
import numpy as np
import matplotlib.pyplot as plt

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

# Statistics
from scipy.stats import norm
import matplotlib.mlab as mlab
from scipy.integrate import simpson

# Import prospector and allied libraries
import prospect
import h5py, fsps, sedpy
import dynesty, dynesty
import corner

# Necessary prospector functions
from prospect import prospect_args
from prospect.fitting import fit_model
from prospect.io import write_results as writer
from prospect.fitting import lnprobfn
from functools import partial

# Implement a low level mpi
from mpi4py import MPI

script_version = '4.0.1'
# Whats new
# Dynesty with 2 component dust model

# --------------------------------------------------------------------------- #
#              P  R  O  S  P  E  C  T  O  R        P  A  R  T  S              #
# ----------------------------------------------------------------------------#

def data_import(fits_fname='EN1_sedfit_v1.0.fits',rows=[0,0],**kwargs):
    from astropy.io import fits
    import numpy as np
    """
    Function to import fluxes and source data from ELAIS N1 Source Catalog
    Will read fits files and convert fluxes into maggies. Return dict.
    usage: data_import(filename, rows=[first obj (starts at 0), no of objs],
        additional args)
    rows = [0,0] will export the entire file
    """
    
    # check if the fits_data dict is already defined, if not initialize
    if not ('fits_data' in locals() or 'fits_data' in globals()):
        fits_data = {}

    beg_row,no_of_rows = rows
    with fits.open(fits_fname) as en1_hdul:
        # if both rows 0, find no of files
        if beg_row == no_of_rows == 0:
            no_of_rows = en1_hdul[1].header['NAXIS2']
        # if both begin is greater than end, flip
        else:
            no_of_rows = (beg_row + no_of_rows) if (beg_row + no_of_rows) <= en1_hdul[1].header['NAXIS2'] else en1_hdul[1].header['NAXIS2']            
        en1_data = en1_hdul[1].data[beg_row:no_of_rows]

    filters = en1_data.columns.names
    flux_cols, each_flux_data_len, ncols = 14, 3, len(filters)
    filterlist = [filters[ii][:-2] for ii in range(flux_cols,ncols,each_flux_data_len)]

    for ii in range(beg_row-beg_row,no_of_rows-beg_row):
        this_obj = en1_data.item(ii)
        fits_data[ii] = ( {'radio_id' : str(this_obj[13]),
            'sour_id' : str(this_obj[1]), 
            'sour_name': this_obj[0].decode('UTF-8'), 
            'flag': this_obj[2], 'z_best' : this_obj[3], 
            'flux_maggies' : np.array([this_obj[ii]/3631.  for ii in range(flux_cols,ncols,each_flux_data_len)]), 
            'flux_err_maggies' : np.array([this_obj[ii]/3631.  for ii in range(flux_cols+1,ncols,each_flux_data_len)]) })

        if np.isnan(fits_data[ii]['z_best']): fits_data[ii]['z_best'] = 0.0

    return fits_data, no_of_rows - beg_row, filterlist

# --------------------------------------------------------
#      Query filters from sedpy
# --------------------------------------------------------

def get_filters(FILT_DIR = '', filts_from_file = [], **kwargs):
    """
    Need to make it better instead of just hammering through it
    """
    filt_names = ['phaskell_gpc1_g', 'phaskell_gpc1_r', 'phaskell_gpc1_i', 'phaskell_gpc1_y', 'phaskell_gpc1_z', 'phaskell_suprime_g', 'phaskell_suprime_r', 'phaskell_suprime_i', 'phaskell_suprime_z', 'phaskell_suprime_y', 'phaskell_suprime_n921', 'phaskell_ukidss_j', 'phaskell_ukidss_k', 'phaskell_irac_i1', 'phaskell_irac_i2', 'phaskell_irac_i3', 'phaskell_irac_i4', 'phaskell_irac_i1', 'phaskell_irac_i2', 'phaskell_mips_24', 'phaskell_pacs_green_100', 'phaskell_pacs_red_160', 'phaskell_spire_250', 'phaskell_spire_350', 'phaskell_spire_500']
    return sedpy.observate.load_filters(filt_names, directory=FILT_DIR)




# --------------------------------------------------------
#      Build up the model
# --------------------------------------------------------

def build_model(object_redshift=0.0, fixed_metallicity=None, add_duste=False,
                add_neb=False, add_agn=False, luminosity_distance=0.0, **extras):

    from prospect.models.templates import TemplateLibrary
    from prospect.models import priors, sedmodel

    # --- Get a basic delay-tau SFH parameter set. ---
    # This has 5 free parameters:
    #   "mass", "logzsol", "dust2", "tage", "tau"
    # And two fixed parameters
    #   "zred"=0.1, "sfh"=4
    # See the python-FSPS documentation for details about most of these
    # parameters.  Also, look at `TemplateLibrary.describe("parametric_sfh")` to
    # view the parameters, their initial values, and the priors in detail.
    model_params = TemplateLibrary["parametric_sfh"]

    # Add lumdist parameter.  If this is not added then the distance is
    # controlled by the "zred" parameter and a WMAP9 cosmology.
    if luminosity_distance > 0:
        model_params["lumdist"] = {"N": 1, "isfree": False,
                                   "init": luminosity_distance, "units":"Mpc"}

    # Adjust model initial values (only important for optimization or dynesty)
    model_params["dust2"]["init"] = 0.1
    model_params["logzsol"]["init"] = -0.3
    model_params["tage"]["init"] = 13.
    model_params["mass"]["init"] = 1e8

    # If we are going to be using dynesty, it is useful to provide an
    # initial scale for the cloud of walkers (the default is 0.1)
    # For dynesty these can be skipped
    model_params["mass"]["init_disp"] = 1e7
    model_params["tau"]["init_disp"] = 3.0
    model_params["tage"]["init_disp"] = 5.0
    model_params["tage"]["disp_floor"] = 2.0
    model_params["dust2"]["disp_floor"] = 0.1

    # adjust priors
    model_params["dust2"]["prior"] = priors.TopHat(mini=0.0, maxi=2.0)
    # model_params["tage"]["prior"] = priors.LogUniform(mini=2.0, maxi=13.0)
    # model_params["tau"]["prior"] = priors.LogUniform(mini=1e-1, maxi=13.0)
    model_params["mass"]["prior"] = priors.LogUniform(mini=1.e5, maxi=1.e13)

    # Change the model parameter specifications based on some keyword arguments
    if fixed_metallicity is not None:
        # make it a fixed parameter
        model_params["logzsol"]["isfree"] = False
        #And use value supplied by fixed_metallicity keyword
        model_params["logzsol"]['init'] = fixed_metallicity

    if object_redshift != 0.0:
        # make sure zred is fixed
        model_params["zred"]['isfree'] = False
        # And set the value to the object_redshift keyword
        model_params["zred"]['init'] = object_redshift

    else:
        # make sure zred is fixed
        model_params["zred"]['isfree'] = True
        # And set the value to the object_redshift keyword
        model_params["zred"]['init'] = 0.5
        model_params["zred"]["prior"] = priors.TopHat(mini=0.0, maxi=10.0)

    if add_duste:
        # Add dust emission (with fixed dust SED parameters)
        model_params.update(TemplateLibrary["dust_emission"])
        model_params['duste_umin']['isfree'] = True
        model_params['duste_gamma']['isfree'] = True
        model_params['dust_type']['init'] = 4
        # Adding the two component dust model from prospector alpha
        alpha_model = TemplateLibrary['alpha']
        model_params['dust2'] = alpha_model['dust2']
        model_params['duste_qpah'] = alpha_model['duste_qpah']
        model_params['dust1'] = alpha_model['dust1']
        model_params['dust_ratio'] = alpha_model['dust_ratio']
        model_params['dust_index'] = alpha_model['dust_index']

    if add_agn:
        # Add dust emission (with fixed dust SED parameters)
        model_params.update(TemplateLibrary["agn"])
        model_params['fagn']['isfree'] = True    # make agn luminosity free, default priors
        model_params['agn_tau']['isfree'] = True
        model_params['add_agn_dust']['init'] = True

    if add_neb:
        # Add nebular emission (with fixed parameters)
        model_params.update(TemplateLibrary["nebular"])

    # Now instantiate the model using this new dictionary of parameter specifications
    model = sedmodel.SedModel(model_params)

    return model, model_params

# --------------------------------------------------------
#      Query observed data
# --------------------------------------------------------

def build_obs(obs_data, filt_list, objid=0, **kwargs):

    from prospect.utils.obsutils import fix_obs

    fluxes = obs_data[objid]['flux_maggies'].tolist()
    fluxes = fluxes[1:]
    fluxes_unc = obs_data[objid]['flux_err_maggies'].tolist()
    fluxes_unc = fluxes_unc[1:]

    obs = {}

    obs = {'radio_id': obs_data[objid]['radio_id'],
      'sour_id': obs_data[objid]['sour_id'],
      'sour_name': obs_data[objid]['sour_name'],
      'flag': obs_data[objid]['flag'],
      'z_best': obs_data[objid]['z_best'] }
    obs['maggies'] = np.array(fluxes)
    obs['maggies_unc'] = np.array(fluxes_unc)

    obs['filters'] = filt_list

    obs['phot_mask'] = np.array([True for ff in obs['filters'] ])
    obs['phot_wave'] = np.array([ff.wave_effective for ff in obs['filters']])

    obs['wavelength'] = None
    obs['spectrum'] = None
    obs['unc'] = None
    obs['mask'] = None
    obs['objid'] = objid

    # This ensures all required keys are present and adds some extra useful info
    obs = fix_obs(obs)

    return obs

# --------------
# SPS Object
# --------------


def build_sps(zcontinuous=1, compute_vega_mags=False, **extras):
    from prospect.sources import CSPSpecBasis
    sps = CSPSpecBasis(zcontinuous=zcontinuous, compute_vega_mags=compute_vega_mags)
    return sps

# -----------------
# Noise Model
# ------------------

def build_noise(**extras):
    return None, None


# --------------------------------------------------------------------------- #
#                          S C I E N C E       P A R T S                      #
# ----------------------------------------------------------------------------#

def calc_dmpc(z,H0=73, WM=0.27, WV=0.73, **kwargs):
    # From https://www.astro.ucla.edu/%7Ewright/CosmoCalc.html
    # James Schombert has written this Python version of the calculator. 
    # I ported it to python 3 hehe :)
    # initialize constants

    WR = 0.        # Omega(radiation)
    WK = 0.        # Omega curvaturve = 1-Omega(total)
    c = 299792.458 # velocity of light in km/sec
    Tyr = 977.8    # coefficent for converting 1/H into Gyr
    DTT = 0.5      # time from z to now in units of 1/H0
    DTT_Gyr = 0.0  # value of DTT in Gyr
    age = 0.5      # age of Universe in units of 1/H0
    age_Gyr = 0.0  # value of age in Gyr
    zage = 0.1     # age of Universe at redshift z in units of 1/H0
    zage_Gyr = 0.0 # value of zage in Gyr
    DCMR = 0.0     # comoving radial distance in units of c/H0
    DCMR_Mpc = 0.0 
    DCMR_Gyr = 0.0
    DA = 0.0       # angular size distance
    DA_Mpc = 0.0
    DA_Gyr = 0.0
    kpc_DA = 0.0
    DL = 0.0       # luminosity distance
    DL_Mpc = 0.0
    DL_Gyr = 0.0   # DL in units of billions of light years
    V_Gpc = 0.0
    a = 1.0        # 1/(1+z), the scale factor of the Universe
    az = 0.5       # 1/(1+z(object))

    h = H0/100.
    WR = 4.165E-5/(h*h)   # includes 3 massless neutrino species, T0 = 2.72528
    WK = 1-WM-WR-WV
    az = 1.0/(1+1.0*z)
    age = 0.
    n=1000         # number of points in integrals
    
    for i in range(n):
        a = az*(i+0.5)/n
        adot = np.sqrt(WK+(WM/a)+(WR/(a*a))+(WV*a*a))
        age = age + 1./adot

    zage = az*age/n
    zage_Gyr = (Tyr/H0)*zage
    DTT = 0.0
    DCMR = 0.0

    # do integral over a=1/(1+z) from az to 1 in n steps, midpoint rule
    for i in range(n):
        a = az+(1-az)*(i+0.5)/n
        adot = np.sqrt(WK+(WM/a)+(WR/(a*a))+(WV*a*a))
        DTT = DTT + 1./adot
        DCMR = DCMR + 1./(a*adot)

    DTT = (1.-az)*DTT/n
    DCMR = (1.-az)*DCMR/n
    age = DTT+zage
    age_Gyr = age*(Tyr/H0)
    DTT_Gyr = (Tyr/H0)*DTT
    DCMR_Gyr = (Tyr/H0)*DCMR
    DCMR_Mpc = (c/H0)*DCMR

    # tangential comoving distance

    ratio = 1.00
    x = np.sqrt(abs(WK))*DCMR
    if x > 0.1:
        if WK > 0:
            ratio =  0.5*(np.exp(x)-np.exp(-x))/x 
        else:
            ratio = np.sin(x)/x
    else:
        y = x*x
        if WK < 0: y = -y
        ratio = 1. + y/6. + y*y/120.
    DCMT = ratio*DCMR
    DA = az*DCMT
    DA_Mpc = (c/H0)*DA
    kpc_DA = DA_Mpc/206.264806
    DA_Gyr = (Tyr/H0)*DA
    DL = DA/(az*az)
    DL_Mpc = (c/H0)*DL
    DL_Gyr = (Tyr/H0)*DL

    # comoving volume computation

    ratio = 1.00
    x = np.sqrt(abs(WK))*DCMR
    if x > 0.1:
        if WK > 0:
            ratio = (0.125*(np.exp(2.*x)-np.exp(-2.*x))-x/2.)/(x*x*x/3.)
        else:
            ratio = (x/2. - np.sin(2.*x)/4.)/(x*x*x/3.)
    else:
        y = x*x
        if WK < 0: y = -y
        ratio = 1. + y/5. + (2./105.)*y*y
    VCM = ratio*DCMR*DCMR*DCMR/3.
    V_Gpc = 4.*np.pi*((0.001*c/H0)**3)*VCM

    return DCMR_Mpc, DL_Mpc


# Calculate chi^2
def get_chi2(x,y,xerr):
    ''' Calculate the chi^2 between observed (x) and 
    simulated (y) data given error in obs (xerr).
    Returns array of chi, chi^2, and mask array, where
    mask denotes the array indices where chi_arr is not null.
    '''
    chi_arr, chi_nan_mask = [],[]

    # chi error for residual plot
    chi_arr = (x - y)/xerr
    
    chi_nan_mask = np.array([ii for ii in range(len(chi_arr)) if not np.isnan(chi_arr[ii])])

    chi_sq_arr = np.array([ii**2 for ii in chi_arr])
    chi_sq = np.nansum(chi_sq_arr)

    return chi_arr, chi_sq, chi_nan_mask



# Get dust luminosity
def get_ldust(spec_wave,flux_maggies,zred):
    # Calculate the redshift comoving distance
    d_cmr, d_lum  = calc_dmpc(zred)
    F0 = 3631.e-23
    flux_ergs = 3631.e-23 * flux_maggies # maggies to ergs cm-2 Hz-1 s-1
    
    # The idea is to get the area under flux density vs frequency curve
    # Convert the wavelength array (in Angstroms) to frequency
    spec_freq = np.array([3.e8/(ii*1.e-10) for ii in spec_wave])
    
    # Dust part of the spectrum is b/w 8 to 1000um.
    dust_start, dust_end = np.where(spec_wave > 8.e4)[0][0], np.where(spec_wave < 1.e7)[0][-1]
    dust_freq, dust_spec = spec_freq[dust_start:dust_end], flux_ergs[dust_start:dust_end]
    
    # Area under the curve gives total dust flux
    total_flux_ergs = simpson(dust_spec,x=dust_freq) # ergs cm-2 s-1
    # Convert flux to luminosity
    total_lum_ergs = 4*np.pi*d_lum*d_lum*9.523e48 *total_flux_ergs # ergs s-1
    Lsol = 3.826e33 # ergs s-1
    total_lum_lsol = total_lum_ergs/Lsol 
    
    return -1*total_lum_lsol, d_cmr, d_lum # -1 since the integral limits 
    # are reversed when converting from wavelength to freq

# get sfr curves and assorted physical parameters for one set of theta
def get_sfr(this_theta):
    tage, tau = this_theta[2],this_theta[3] # Gyrs
    tl = np.linspace(0.001,tage, 500) # Gyrs
    # The sfr equation is normalized by the stellar mass
    sfr = ( (tage - tl)/tau ) * np.exp( -( (tage - tl)/tau ) )
    # So get the normalization factor for every state
    area = simpson(sfr,x=tl*1.e9) # convert to yrs
    sfr= (sfr/area)*this_theta[0] # True sfr, plot this
    
    # Calculate stellar mass in the last 100 myrs    
    tbelow100 = np.where(tl <= 0.1)
    # area under the curve to calculate stellar mass in the last 100 myrs
    st_mass_100myr = simpson( sfr[:len(tbelow100[0])], x=tl[:len(tbelow100[0])]*1.e9)
    sfr_100myr= st_mass_100myr/100.e6
    ssfr_100myr = sfr_100myr/this_theta[0] #specific sfr = sfr100myr/stellar mass total
    
    return tl, sfr, st_mass_100myr, sfr_100myr, ssfr_100myr


import prospect.io.read_results as reader


WORK_DIR = os.getcwd() 
inp_params = 'params_0.py'

if WORK_DIR[-1] != '/': WORK_DIR = WORK_DIR+'/'
if not os.path.isfile(WORK_DIR+inp_params):
    print('Please supply a parameter file. Will exit.')
    exit()
with open(WORK_DIR+inp_params,'r') as f: inplines = f.readlines()
for this_line in inplines: exec(this_line) 

TOTAL_OBJS = NODES * NPROCS * NRUNS

NOBJS = NPROCS * NRUNS

subfolders = [ f.path for f in os.scandir(os.getcwd()) if f.is_dir() ]
objdirs = [ f.split('/')[-1] for f in subfolders if f.split('/')[-1].isnumeric() ]


# Will work irrespective of where its run
for itdir in FITS_DIR_LIST:
    if os.path.isdir(itdir):
        FITS_DIR = itdir
        break


datafile = FITS_DIR+DATA_FITS_FILE
obs_data, cnt, filter_names = data_import(fits_fname=datafile,rows=[0,0])

# Will work irrespective of where its run
for itdir in FILT_DIR_LIST:
    if os.path.isdir(itdir):
        FILT_DIR = itdir
        break

filters = get_filters(FILT_DIR, filter_names)
sps = build_sps(zcontinuous=1) 

cnt = 0

comm = MPI.COMM_WORLD
rank = comm.Get_rank() # get your process ID

breakdirs = np.array_split(objdirs,8)

for this_dir in breakdirs[rank]:

    # print('{0}/{1}'.format(cnt,len(objdirs)))
    this_obj = int(this_dir)
    cnt+=1

    outtxt = this_dir+'/'+this_dir+'_output.txt'
    if os.path.isfile(outtxt):
        with open(outtxt,'r') as f: outlines = f.readlines()
        for this_line in outlines: exec(this_line)

    else:
        continue

    # print(this_dir)

    obj_dir = WORK_DIR+this_dir

    src_tag = '{0}/{1}_dynesty_{2}init_{3}batch_{4}_'.format(obj_dir,this_dir,
                    NLIVE_INIT, NLIVE_BATCH, NESTED_METHOD)
    plt_tag = 'Source {0} dynesty {1}init {2}batch {3}'.format(this_dir,
                    NLIVE_INIT, NLIVE_BATCH, NESTED_METHOD)


    nplot = 100
    fitdata = {}

    hfile = "{0}/en1_{1}_dynesty_{2}init_{3}batch_{4}.h5".format(obj_dir,this_dir, 
                    NLIVE_INIT, NLIVE_BATCH, NESTED_METHOD)


    result, this_dirfromfile, _ = reader.results_from(hfile, dangerous=False)

    obs = build_obs(obs_data,filters,objid=this_obj)


    model, model_params = build_model(object_redshift=obs['z_best'],fixed_metallicity=0.0,\
                add_duste=True,add_agn=True)
 

    NDIM = model.ndim

    # -------------------------------------------------------------- #
    # diagnostic trace and corner plots                              #
    # -------------------------------------------------------------- #

    # tracefig = reader.traceplot(result, figsize=(20,10))
    # plotname = '{0}_fig_{1}_dynesty_randwalker_trace.jpg'.format(src_tag,nplot)
    # plt.savefig(plotname)

    # plt.rcParams['figure.figsize'] = (32,32)
    # plt.rcParams['font.size'] = 27
    # thin = 1
    # cornerfig = reader.subcorner(result, start=0, thin=thin,
    #                                fig=plt.subplots(NDIM,NDIM,figsize=(27,27))[0])
    # nplot+=1
    # plotname = '{0}_fig_{1}_dynesty_randwalker_trace.jpg'.format(src_tag,nplot)
    # plt.savefig(plotname)

    
    # # ----------------------------------------------------
    # #         Dynesty ended here
    # # ----------------------------------------------------        

    # this_dirbkup = obs
    # theta = model.theta.copy()
    # dynesty_theta = theta

    # output_sampling = output['sampling'][0]

    this_dirphot, this_direrr, phot_wave = obs['maggies'], obs['maggies_unc'], obs['phot_wave']
    # spec_wave = sps.wavelengths


    # dynesty_spec, dynesty_phot, dynesty_mfrac = model.mean_model(theta, obs=obs,sps=sps)
    # plotname = '{0}_fig_{1}_dynesty_sed'.format(src_tag,nplot)
    # plottitle = plt_tag+ ' - ' + 'MCMC sampled SED'
    # fitdata[0] = {'fit_phot':dynesty_phot, 'fit_spec':dynesty_spec, 'datalabel':'dynesty SED'}
    
    # dynesty_chi_arr, dynesty_chi_sq, dynesty_chi_nan_mask = get_chi2(this_dirphot,dynesty_phot,this_direrr)
    

    # ----------------------------------------------------
    #         Game of log probs
    # ----------------------------------------------------  

    # fig = plt.figure()
    # nplot+=1
    # plt.rcParams['figure.figsize'] = (10,6)
    # plt.rcParams['font.size'] = 6

    # plt.rcParams['figure.figsize'] = (15,10)
    # plt.rcParams['font.size'] = 10
    # plt.plot(result['lnlikelihood'],'.k')
    # plt.title('lnprobability > 0.9 percent maximum')
    # plt.xlabel('iteration')
    # plt.ylabel('lnP')
    # plt.ylim([0.8*np.max(result['lnlikelihood']), np.max(result['lnlikelihood'])])
    # plt.tight_layout()

    # plotname = "{0}_fig_{1}_lnP_clipped_0.8maxlnP.jpg".format(src_tag, nplot)
    # plt.savefig(plotname)
    

    maxlnP = np.where(result['lnlikelihood']>0.90*np.max(result['lnlikelihood']))


    # -------------------------------------------------------------- #
    # validate the best fit output by prospector
    # -------------------------------------------------------------- # 
    max_probs = np.argmax(result['lnprobability'])
    theta_best = result['chain'][max_probs]

    # randstates = result['chain'][np.random.randint(low=0,high=np.shape(result['chain'])[0],size=np.shape(result['chain'])[0]//10)]
    randstates = result['chain']

    theta_best = result['chain'][max_probs]
    best_spec, best_phot, best_mfrac = model.predict(theta_best,obs=obs,sps=sps)

    # fitdata = {}

    # plotname = '{0}_fig_{1}_compare_thetas_sed'.format(src_tag,nplot)
    # plottitle = plt_tag+ ' - ' + 'SED'
    # fitdata[0] = {'fit_phot':best_phot, 'fit_spec':best_spec, 'datalabel':'PROSPECTOR fit'}
    # best_chi_arr, best_chi_sq, nplot = \
    #           plot_seds_shade(obs, model, randstates, sps, figname=plotname, plottitle=plottitle, wframe='rest', nplot=nplot, nfitdata = 1, specbins=200, fitdata=fitdata)

    best_chi_arr, best_chi_sq, best_chi_nan_mask = get_chi2(this_dirphot,best_phot,this_direrr)


    # -------------------------------------------------------------- #
    # plot dust spec
    # -------------------------------------------------------------- #

    ldust_arr, mfrac_arr = [], []
    obs_phot, obs_err, phot_wave = obs['maggies'], obs['maggies_unc'], obs['phot_wave']
    spec_wave = sps.wavelengths
    for this_state in randstates:
        this_spec, this_phot, this_mfrac = model.predict(this_state,obs=obs,sps=sps)
        ldust,d_cmr, d_lum = get_ldust(spec_wave,this_spec,obs['z_best'])
        ldust_arr.append(ldust)
        mfrac_arr.append(this_mfrac)

    p50_ldust, p16_ldust, p84_ldust = np.percentile(ldust_arr,50), np.percentile(ldust_arr,16), np.percentile(ldust_arr,84)
    ldust_best,d_cmr, d_lum = get_ldust(spec_wave,best_spec,obs['z_best'])
    # ldust_dynesty,d_cmr, d_lum = get_ldust(spec_wave,dynesty_spec,obs['z_best'])

    p50_mfrac, p16_mfrac, p84_mfrac = np.percentile(mfrac_arr,50), np.percentile(mfrac_arr,16), np.percentile(mfrac_arr,84)
    


    # ---------------------------------------------------------------------
    # Store all thetas in a dictionary
    # ---------------------------------------------------------------------
    theta_dict = {}
    for nn in range(NDIM):
        theta_dict[result['theta_labels'][nn]] = {'best' : theta_best[nn]}


    # ---------------------------------------------------------------------
    # Plot the walker state histograms
    # ---------------------------------------------------------------------

    nplot+=1
    plotname = "{0}_fig_{1}_hist_alliter_ptiles.jpg".format(src_tag, nplot)
    allstates = result['chain'][maxlnP]


    NCOLS = 4
    NROWS = int(np.ceil(NDIM/NCOLS))

    fig, ax = plt.subplots(nrows=NROWS, ncols=NCOLS)
    
    plt.rcParams['figure.figsize'] = (NCOLS*5,NROWS*5)
    plt.rcParams['font.size'] = 10

    for tt in range(len(result['theta_labels'])):
        this_theta = allstates[:,tt]
            
        (mu, sigma) = norm.fit(this_theta)
        p16,p50,p84 = np.percentile(this_theta,16), np.percentile(this_theta,50), np.percentile(this_theta,84)

        # the histogram of the data
        n, bins, patches = ax[tt%NROWS,tt//NROWS].hist(this_theta, 20, density=True, edgecolor='k', facecolor='green', alpha=0.5)

        # add a 'best fit' line
        y = norm.pdf( bins, mu, sigma)
        l = ax[tt%NROWS,tt//NROWS].plot(bins, y, 'r--', linewidth=2)
        ax[tt%NROWS,tt//NROWS].axvline(p16, ls=':', color='orange', linewidth=2, label='16 ptile = {0:.3e}'.format(p16))
        ax[tt%NROWS,tt//NROWS].axvline(p50, ls=':', color='red', linewidth=2, label='50 ptile = {0:.3e}'.format(p50))
        ax[tt%NROWS,tt//NROWS].axvline(p84, ls=':', color='purple', linewidth=2, label='84 ptile = {0:.3e}'.format(p84))
        # ax[tt%NROWS,tt//NROWS].axvline(dynesty_theta[tt], ls=':', linewidth=2, color='blue', label='prospect = {0:.3e}'.format(dynesty_theta[tt]))
        ax[tt%NROWS,tt//NROWS].axvline(theta_best[tt], ls=':', color='black', linewidth=2, label='best = {0:.3e}'.format(theta_best[tt]))
        ax[tt%NROWS,tt//NROWS].legend()
        ax[tt%NROWS,tt//NROWS].set_xlabel(result['theta_labels'][tt])
        ax[tt%NROWS,tt//NROWS].set_ylabel('Count')

        theta_dict[result['theta_labels'][tt]]['50ptile'] = p50
        theta_dict[result['theta_labels'][tt]]['16ptile'] = p16
        theta_dict[result['theta_labels'][tt]]['84ptile'] = p84
        
        fig.tight_layout()

        fig.suptitle(plt_tag+ ' - theta histogram (all iter)')

    plt.savefig(plotname)


    # ---------------------------------------------------------------
    # Star formation rates
    # ---------------------------------------------------------------

    # fig = plt.figure()

    # plt.rcParams['figure.figsize'] = (15, 10)

    # plt.rcParams['font.size']=15  
    # Array of random states
    t_end, bin_size = 13.8, SFR_BIN_SIZE

    # Divide the time axis into multiple bins - then average over all the sfr points in there
    # Create an array containing mid point of each time bins           
    tbins = np.arange(SFR_BIN_SIZE,t_end,SFR_BIN_SIZE)
    binmids = tbins - SFR_BIN_SIZE/2. 
    binmids = np.append(binmids,[tbins[-1] + SFR_BIN_SIZE/2.] )
    binsfrs = [[] for ii in binmids]

    mstar_100myr_arr, sfr_100myr_arr, ssfr_100myr_arr = [], [], []

    for this_state in randstates:
        tl, sfr, st_mass_100myr,sfr_100myr,ssfr_100myr = get_sfr(this_state)

        # Divide that by the time to get star formation rate averaged over 100 myrs
        mstar_100myr_arr.append(st_mass_100myr)
        sfr_100myr_arr.append(sfr_100myr)
        ssfr_100myr_arr.append(ssfr_100myr)

        # plt.plot(tl, sfr, color='dodgerblue',alpha=0.3,zorder=1)

        binnify = np.digitize(tl,tbins)
        for ii in range(len(tl)):
            binsfrs[binnify[ii]].append([sfr[ii]])

    # Get the sfr for theta best and theta dynesty
    # tl_dynesty, sfr_dynesty, st_mass_100myr_dynesty,sfr_100myr_dynesty,ssfr_100myr_dynesty = get_sfr(dynesty_theta)
    tl_best, sfr_best, st_mass_100myr_best,sfr_100myr_best,ssfr_100myr_best = get_sfr(theta_best)


    # plt.plot(tl_dynesty, sfr_dynesty, color='black', lw=2, ls=':', zorder=1, label='dynesty out')
    # plt.plot(tl_best, sfr_best, color='black', lw=2, ls='-.', zorder=1, label='highest logprob')

    sfh_median, sfh_16, sfh_84 = [],[],[]

    for ii in range(len(binmids)):
        sfh_median.append(np.percentile(binsfrs[ii],50) if len(binsfrs[ii])>0 else np.NaN)
        sfh_16.append(np.percentile(binsfrs[ii],16) if len(binsfrs[ii])>0 else np.NaN)
        sfh_84.append(np.percentile(binsfrs[ii],84) if len(binsfrs[ii])>0 else np.NaN)

    # plt.plot(binmids,sfh_median,label='median',lw=1,  color='firebrick',zorder=3)
    # plt.plot(binmids,sfh_16,label='16th percentile',lw=1,  color='orangered',zorder=3)
    # plt.plot(binmids,sfh_84,label='84th percentile',lw=1, color='darkorchid',zorder=3)

    # plt.fill_between(binmids, sfh_16, sfh_84, color='yellow', alpha=0.4,zorder=2)

    # plt.xlabel(r'lookback time $(t_l)$ (Gyr)')
    # plt.ylabel(r'$SFR~(M_\odot/yr)$')
    # plt.legend(loc='upper right')
    # plt.title('{0} - SFR from randomly chosen {1} states {2} Gyr bins'.format(plt_tag,SFR_PICK_NRAND,bin_size))
    # plt.tight_layout()
    # nplot+=1
    # plt.savefig('{0}_fig_{1}_sfh_{2}states_{3}bins.jpg'.format(src_tag, nplot,SFR_PICK_NRAND,bin_size))


    # Get the stellar mass histogram
    p16_mstar_100myr,p50_mstar_100myr,p84_mstar_100myr = np.percentile(mstar_100myr_arr,16), np.percentile(mstar_100myr_arr,50), np.percentile(mstar_100myr_arr,84)
    p16_sfr_100myr,p50_sfr_100myr,p84_sfr_100myr = np.percentile(sfr_100myr_arr,16), np.percentile(sfr_100myr_arr,50), np.percentile(sfr_100myr_arr,84)
    p16_ssfr_100myr,p50_ssfr_100myr,p84_ssfr_100myr = np.percentile(ssfr_100myr_arr,16), np.percentile(ssfr_100myr_arr,50), np.percentile(ssfr_100myr_arr,84)



    # ------------------------------------------------
    plt.close('all')

    spitoutfile = obj_dir+'/'+str(this_dir)+'_output2.txt'

    with open(spitoutfile,'w') as f:
        # Galaxy identification
        print('radio_id = ', this_dir,file=f)
        print('sour_id = ', obs['sour_id'],file=f)
        print('sour_name = "{}"'.format(obs['sour_name']) ,file=f)

        # Redshift information
        print('z_best = ', obs['z_best'],file=f)
        print('d_cmr = ', d_cmr,file=f)
        print('d_lum = ', d_lum,file=f)

        # Prospector run params
        print('chi2_best = ', best_chi_sq ,file=f)
        

        # Result free params
        # - mstar
        for tt in range(NDIM):
            print(result['theta_labels'][tt]+'_best = ', theta_dict[result['theta_labels'][tt]]['best'] ,file=f)
            # print(result['theta_labels'][tt]+'_dynesty = ', theta_dict[result['theta_labels'][tt]]['dynesty'] ,file=f)
            print(result['theta_labels'][tt]+'_50ptile = ', theta_dict[result['theta_labels'][tt]]['50ptile'] ,file=f)
            print(result['theta_labels'][tt]+'_16ptile = ', theta_dict[result['theta_labels'][tt]]['16ptile'] ,file=f)
            print(result['theta_labels'][tt]+'_84ptile = ', theta_dict[result['theta_labels'][tt]]['84ptile'] ,file=f)
            
                
        ### Physical parameters

        # - SFR       - star formation rate in Msun/yr averaged over last 0.1 Gyr
        print('mstar100_best = ', st_mass_100myr_best ,file=f)
        # print('mstar100_dynesty = ', st_mass_100myr_dynesty ,file=f)
        print('mstar100_50ptile = ', p50_mstar_100myr ,file=f)
        print('mstar100_16ptile = ', p16_mstar_100myr ,file=f)
        print('mstar100_84ptile = ', p84_mstar_100myr ,file=f)


        print('mfrac_best = ', best_mfrac ,file=f)
        # print('mfrac_dynesty = ', dynesty_mfrac ,file=f)
        print('mfrac_50ptile = ', p50_mfrac ,file=f)
        print('mfrac_16ptile = ', p16_mfrac ,file=f)
        print('mfrac_18ptile = ', p84_mfrac ,file=f)

        print('sfr100_best = ', sfr_100myr_best ,file=f)
        # print('sfr100_dynesty = ', sfr_100myr_dynesty ,file=f)
        print('sfr100_50ptile = ', p50_sfr_100myr ,file=f)
        print('sfr100_16ptile = ', p16_sfr_100myr ,file=f)
        print('sfr100_84ptile = ', p84_sfr_100myr ,file=f)

        print('ssfr100_best = ', ssfr_100myr_best ,file=f)
        # print('ssfr100_dynesty = ', ssfr_100myr_dynesty ,file=f)
        print('ssfr100_50ptile = ', p50_ssfr_100myr ,file=f)
        print('ssfr100_16ptile = ', p16_ssfr_100myr ,file=f)
        print('ssfr100_84ptile = ', p84_ssfr_100myr ,file=f)
        
        print('ldust_best = ', ldust_best ,file=f)
        # print('ldust_dynesty = ', ldust_dynesty ,file=f)
        print('ldust_50ptile = ', p50_ldust ,file=f)
        print('ldust_16ptile = ', p16_ldust ,file=f)
        print('ldust_84ptile = ', p84_ldust ,file=f)






