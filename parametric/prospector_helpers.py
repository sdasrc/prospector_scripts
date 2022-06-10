# --------------------------------------------------------------------------- #
#              P  R  O  S  P  E  C  T  O  R        P  A  R  T  S              #
# ----------------------------------------------------------------------------#


# --------------------------------------------------------
#      Build up the model
# --------------------------------------------------------

def build_model(obs, fixed_metallicity=None, add_duste=False, add_neb=False, 
    add_agn=False, add_burst=False, **kwargs):

    import numpy as np
    from prospect.models.templates import TemplateLibrary
    from prospect.models import priors, sedmodel
    # Astropy to calculate cosmological params
    from astropy.cosmology import FlatLambdaCDM
    import astropy.units as u

    # --- Get a basic delay-tau SFH parameter set. ---
    # This has 5 free parameters:
    #   "mass", "logzsol", "dust2", "tage", "tau"
    # And two fixed parameters
    #   "zred"=0.1, "sfh"=4
    # See the python-FSPS documentation for details about most of these
    # parameters.  Also, look at `TemplateLibrary.describe("parametric_sfh")` to
    # view the parameters, their initial values, and the priors in detail.
    model_params = TemplateLibrary["parametric_sfh"]

    # addburst = 0 : no starburst
    # addburst = 1 : fage_burst free, fburst is not
    # addburst = 2 : both fage_burst and fburst free

    if add_burst: 
        model_params.update(TemplateLibrary["burst_sfh"])
        model_params['fage_burst']['isfree'] = True
        model_params["fage_burst"]["prior"] = priors.TopHat(mini=0.0, maxi=1.0)
        model_params['fburst']['isfree'] = True
    else:
        tsv = True # tsv:: thestupidvariable

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

    # Adjust model initial values (only important for optimization or emcee)
    model_params["dust2"]["init"] = 0.1
    model_params["logzsol"]["init"] = -0.3
    model_params["mass"]["init"] = 1e8

    # adjust priors
    model_params["dust2"]["prior"] = priors.TopHat(mini=0.0, maxi=2.0)
    model_params["tage"]["prior"] = priors.TopHat(mini=0.001, maxi=obs['age_univ'])
    model_params["tau"]["prior"] = priors.LogUniform(mini=0.1, maxi=50.0)
    model_params["mass"]["prior"] = priors.LogUniform(mini=1.e5, maxi=1.e13)
    model_params["tage"]["init"] = obs['age_univ']

    # Change the model parameter specifications based on some keyword arguments
    if fixed_metallicity is not None:
        # make it a fixed parameter
        model_params["logzsol"]["isfree"] = False
        #And use value supplied by fixed_metallicity keyword
        model_params["logzsol"]['init'] = fixed_metallicity

    # Will add the option to get phot z later maybe?
    # if object_redshift != 0.0:
    #     # make sure zred is fixed
    #     model_params["zred"]['isfree'] = False
    #     # And set the value to the object_redshift keyword
    #     model_params["zred"]['init'] = object_redshift

    # else:
    #     # make sure zred is fixed
    #     model_params["zred"]['isfree'] = True
    #     # And set the value to the object_redshift keyword
    #     model_params["zred"]['init'] = 0.5
    #     model_params["zred"]["prior"] = priors.TopHat(mini=0.0, maxi=10.0)

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


        # model_params["duste_gamma"]["prior"] = priors.LogUniform(mini=1.e-5, maxi=0.2)
        # model_params["duste_qpah"]["prior"] = priors.TopHat(mini=0.5, maxi=10)
        # model_params["duste_umin"]["prior"] = priors.TopHat(mini=0.1, maxi=50)

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


def build_sps(zcontinuous=1, compute_vega_mags=False, **extras):
    from prospect.sources import CSPSpecBasis
    sps = CSPSpecBasis(zcontinuous=zcontinuous, compute_vega_mags=compute_vega_mags)
    return sps

# -----------------
# Noise Model
# ------------------

def build_noise(**kwargs):
    return None, None
