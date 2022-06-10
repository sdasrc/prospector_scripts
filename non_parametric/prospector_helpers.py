# --------------------------------------------------------------------------- #
#              P  R  O  S  P  E  C  T  O  R        P  A  R  T  S              #
# ----------------------------------------------------------------------------#


# --------------------------------------------------------
#      Build up the model
# --------------------------------------------------------

def build_model(obs, fixed_metallicity=None, add_duste=False, add_neb=False, 
    add_agn=False, change_dirich_prior=False, **kwargs):

    import numpy as np
    from prospect.models.templates import TemplateLibrary
    from prospect.models import priors, sedmodel
    from prospect.models.templates import adjust_dirichlet_agebins
    from prospect.models.priors import Beta

    # --- Get a basic delay-tau SFH parameter set. ---
    # This has 5 free parameters:
    #   "mass", "logzsol", "dust2", "tage", "tau"
    # And two fixed parameters
    #   "zred"=0.1, "sfh"=4
    # See the python-FSPS documentation for details about most of these
    # parameters.  Also, look at `TemplateLibrary.describe("parametric_sfh")` to
    # view the parameters, their initial values, and the priors in detail.
    model_params = TemplateLibrary["alpha"]

    # add cosmological params if redshift is known from catalog
    if obs['z_best'] != 0.0:
        model_params["zred"]['isfree'] = False
        model_params["zred"]['init'] = obs['z_best']
        model_params["lumdist"] = {"N": 1, "isfree": False, "init": obs['lum_dist'], "units":"Mpc"}
    else:
        # Make redshift a free param
        model_params["zred"]['isfree'] = True
        model_params["zred"]['init'] = 0.0
        model_params["zred"]["prior"] = priors.TopHat(mini=0.0, maxi=11.0)

    # params we dont change for now :
    # Mass, sfh (=3), dust, dust type, imf

    # adjust priors

    # Change the agebins by the age of the universe at obj redshift
    # I aim to keep width of each time bin (in log10) greater than 0.2
    # So, the number of bins change with redshift
    # The first two bins stay the same though
    if obs['z_best'] > 7: ntbins = 3
    elif obs['z_best'] > 5: ntbins = 4
    elif obs['z_best'] > 3: ntbins = 5
    else: ntbins = 6   
    # Calculate the new bin endpoints
    tbinends = np.concatenate([[1,np.log10(30e6)],np.linspace(np.log10(100e6),np.log10(obs['age_univ']*1.e9),ntbins)])
    # Update the model params with the prospector function
    # It calculates the new Dirichlet prior on z_fraction from the 
    # given values of agebins
    model_params = adjust_dirichlet_agebins(model_params,tbinends)

    # Change the Dirichlet prior
    if change_dirich_prior:
        n_zfrac = model_params['z_fraction']['N']
        zalpha, zbeta = kwargs['alpha'],  kwargs['beta']
        model_params['z_fraction']['prior'] = Beta(mini=0.0,maxi=1.0,alpha=zalpha*np.ones(n_zfrac),beta=zbeta*np.ones(n_zfrac))
        model_params['z_fraction']['init'] =  kwargs['init']*np.ones(n_zfrac)

        # Cover a larger mass range than the default template
        model_params["total_mass"]["prior"] = priors.LogUniform(mini=1.e5, maxi=1.e13)
        model_params["total_mass"]["init"] = 1e8

    # Change the model parameter specifications based on some keyword arguments
    if fixed_metallicity is not None:
        # make it a fixed parameter
        model_params["logzsol"]["isfree"] = False
        #And use value supplied by fixed_metallicity keyword
        model_params["logzsol"]['init'] = fixed_metallicity

    if not add_duste:
        # then delete all dust emission parameters of the model
        del model_params['duste_umin']
        del model_params['duste_gamma']
        del model_params['dust_type']

    if not add_agn:
        # then delete all agn emission parameters of the model
        del model_params['fagn']
        del model_params['agn_tau']
        del model_params['add_agn_dust']

    if not add_neb:
        # then delete all nebular emission parameters of the model
        del model_params['add_neb_emission']
        del model_params['add_neb_continuum']
        del model_params['nebemlineinspec']
        del model_params['gas_logz']
        del model_params['gas_logu']

    # Now instantiate the model using this new dictionary of parameter specifications
    model = sedmodel.SedModel(model_params)

    return model, model_params

# --------------------------------------------------------
#      Query observed data
# --------------------------------------------------------

def build_obs(obs_data, filt_list, objid=0, **kwargs):
    import numpy as np
    from prospect.utils.obsutils import fix_obs
    # Astropy to calculate cosmological params
    from astropy.cosmology import FlatLambdaCDM
    import astropy.units as u

    # Will add the cosmological info to the observation dict as well
    #  Get all cosmological params out of the way first
    cosmo = FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc, Tcmb0=2.725 * u.K, Om0=0.3)
    lumdist  = cosmo.luminosity_distance(obs_data[objid]['z_best']).value     # in mpc
    tl_univ_z = cosmo.age(obs_data[objid]['z_best']).value # age of the universe at that redshift

    fluxes = obs_data[objid]['flux_maggies'].tolist()
    fluxes_unc = obs_data[objid]['flux_err_maggies'].tolist()

    obs = {}

    obs = {'radio_id': obs_data[objid]['radio_id'],
      'sour_id': obs_data[objid]['sour_id'],
      'sour_name': obs_data[objid]['sour_name'],
      'flag': obs_data[objid]['flag'],
      'z_best': obs_data[objid]['z_best'],
      'lum_dist':  lumdist,
      'age_univ': tl_univ_z
      }
    obs['maggies'] = np.array(fluxes)
    obs['maggies_unc'] = np.array(fluxes_unc)

    obs['filters'] = filt_list

    obs['phot_mask'] = np.array([True for ff in obs['filters'] ])
    obs['phot_wave'] = np.array([ff.wave_effective for ff in obs['filters']])

    obs['wavelength'] = None
    obs['spectrum'] = None
    obs['unc'] = None
    obs['mask'] = None
    obs['good_fluxes'] = obs_data[objid]['good_fluxes']
    obs['objid'] = objid

    # This ensures all required keys are present and adds some extra useful info
    obs = fix_obs(obs)

    return obs

# --------------
# SPS Object
# --------------


def build_sps(**kwargs):
    from prospect.sources import FastStepBasis
    sps = FastStepBasis()
    return sps

# -----------------
# Noise Model
# ------------------

def build_noise(**kwargs):
    return None, None
