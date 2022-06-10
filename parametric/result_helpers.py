# Calculate chi^2
def get_chi2(x,y,xerr):
    ''' Calculate the chi^2 between observed (x) and 
    simulated (y) data given error in obs (xerr).
    Returns array of chi, chi^2, and mask array, where
    mask denotes the array indices where chi_arr is not null.
    return chi_arr, chi_sq, chi_nan_mask
    '''
    import numpy as np
    chi_arr, chi_nan_mask = [],[]

    # chi error for residual plot
    chi_arr = (x - y)/xerr
    
    chi_nan_mask = np.array([ii for ii in range(len(chi_arr)) if not np.isnan(chi_arr[ii])])

    chi_sq_arr = np.array([ii**2 for ii in chi_arr])
    chi_sq = np.nansum(chi_sq_arr)

    return chi_arr, chi_sq, chi_nan_mask


# get sfr curves and assorted physical parameters for one set of theta
def get_sfr(this_theta,theta_keys,isburst=False):
    import numpy as np

    # Statistics
    from scipy.stats import norm
    import matplotlib.mlab as mlab
    from scipy.integrate import simpson

    mstar_tot = this_theta[theta_keys['mass']]
    tage, tau = this_theta[theta_keys['tage']],this_theta[theta_keys['tau']] # Gyrs
    fburst = 0.
    if isburst: fburst, tburst = this_theta[theta_keys['fburst']], this_theta[theta_keys['fage_burst']]*tage # Gyrs
    
    tl = np.linspace(0.001,tage, 500) # Gyrs
    # The sfr equation is normalized by the stellar mass
    sfr = ( (tage - tl)/tau ) * np.exp( -( (tage - tl)/tau ) )
    # So get the normalization factor for every state
    area = simpson(sfr,x=tl*1.e9) # convert to yrs
    sfr= (sfr/area)*mstar_tot*(1 - fburst) # True sfr, plot this

    sfr_uplim = np.max(sfr)
    
    # Add a delta function with fburst frac of stellar mass
    if isburst:
        sfh_burst = np.zeros(len(tl))
        sfh_burst[np.where(tl>tburst)[0][0]] = mstar_tot*fburst
        sfr = sfr+sfh_burst
    
    # Calculate stellar mass in the last 100 myrs    
    tbelow100 = np.where(tl <= 0.1)
    # area under the curve to calculate stellar mass in the last 100 myrs
    st_mass_100myr = simpson( sfr[:len(tbelow100[0])], x=tl[:len(tbelow100[0])]*1.e9)
    sfr_100myr= st_mass_100myr/100.e6
    ssfr_100myr = sfr_100myr/mstar_tot #specific sfr = sfr100myr/stellar mass total
    
    return tl, sfr, sfr_uplim, st_mass_100myr, sfr_100myr, ssfr_100myr

def weighted_quantile(values, quantiles, sample_weight=None, 
                      values_sorted=False, old_style=False):
    """ Very close to numpy.percentile, but supports weights.
    NOTE: quantiles should be in [0, 1]!
    :param values: numpy.array with data
    :param quantiles: array-like with many quantiles needed
    :param sample_weight: array-like of the same length as `array`
    :param values_sorted: bool, if True, then will avoid sorting of
        initial array
    :param old_style: if True, will correct output to be consistent
        with numpy.percentile.
    :return: numpy.array with computed quantiles.
    """
    import numpy as np
    values = np.array(values)
    quantiles = np.array(quantiles)
    if sample_weight is None:
        sample_weight = np.ones(len(values))
    sample_weight = np.array(sample_weight)
    assert np.all(quantiles >= 0) and np.all(quantiles <= 1), \
        'quantiles should be in [0, 1]'

    if not values_sorted:
        sorter = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]

    weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
    if old_style:
        # To be convenient with numpy.percentile
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    else:
        weighted_quantiles /= np.sum(sample_weight)
    return np.interp(quantiles, weighted_quantiles, values)


def get_ldust(wave_angs,Llambda_Lsol):
    import numpy as np
    from numpy import trapz
    dust_start, dust_end = np.where(wave_angs > 8.e4)[0][0], np.where(wave_angs < 1.e7)[0][-1]
    dust_wave = wave_angs[dust_start:dust_end]
    dust_spec = Llambda_Lsol[dust_start:dust_end]
    ldust = trapz(dust_spec,x=dust_wave)

    return ldust

# Get dust luminosity
# def get_ldust(spec_wave,flux_maggies,zred):

#     import numpy as np
#     # Astropy to calculate cosmological params
#     from astropy.cosmology import FlatLambdaCDM
#     import astropy.units as u

#     # Statistics
#     from scipy.stats import norm
#     import matplotlib.mlab as mlab
#     from scipy.integrate import simpson

#     # call the cosmo object from astropy
#     cosmo = FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc, Tcmb0=2.725 * u.K, Om0=0.3)

#     # Calculate the redshift comoving distance
#     d_lum  = cosmo.luminosity_distance(zred).value
#     F0 = 3631.e-23
#     flux_ergs = 3631.e-23 * flux_maggies # maggies to ergs cm-2 Hz-1 s-1
    
#     # The idea is to get the area under flux density vs frequency curve
#     # Convert the wavelength array (in Angstroms) to frequency
#     spec_freq = np.array([3.e8/(ii*1.e-10) for ii in spec_wave])
    
#     # Dust part of the spectrum is b/w 8 to 1000um.
#     dust_start, dust_end = np.where(spec_wave > 8.e4)[0][0], np.where(spec_wave < 1.e7)[0][-1]
#     dust_freq, dust_spec = spec_freq[dust_start:dust_end], flux_ergs[dust_start:dust_end]
    
#     # Area under the curve gives total dust flux
#     total_flux_ergs = simpson(dust_spec,x=dust_freq) # ergs cm-2 s-1
#     # Convert flux to luminosity
#     total_lum_ergs = 4*np.pi*d_lum*d_lum*9.523e48 *total_flux_ergs # ergs s-1
#     Lsol = 3.826e33 # ergs s-1
#     total_lum_lsol = total_lum_ergs/Lsol 
    
#     return -1*total_lum_lsol,  d_lum # -1 since the integral limits 
#     # are reversed when converting from wavelength to freq

def get_magphys_sed(sour_name):
    '''
    get_magphys_sed(sour_name,z_red):
    return magpattenuated, magpwave
    input - sour_name str, z_red float
    returns - unattenuated luminosity L_lambda/L_sol in 1/A
          and rest frame wavelength in A
    '''
    import numpy as np
    en1_sed_dir = "/beegfs/lofar/deepfields/magphys/EN1/EN1_SEDs/"
    sed_fname = "{0}{1}.sed".format(en1_sed_dir,sour_name)
    magpsed = np.loadtxt(sed_fname,comments='#')
    magpwave = 10**magpsed[:,0]   # obs wavelength in A
    magpattenuated = 10**magpsed[:,1] # L_lambda/L_sol in 1/A
    # magpwave = magpwave/(1.+z_red) # convert to rest frame
    # magpwave = magpwave
    return magpattenuated, magpwave

def get_magp_phot(sour_name):
    '''
    def get_magp_phot(sour_name):
    return magp_phot
    - given the source name, return np array of 26 photometries
      in Jy/Hz
    '''
    import numpy as np
    magp_phot_dir = '/beegfs/car/phaskell/080720_Vn1_photometry/'
    magp_phot_fitfile = '{0}{1}.fit'.format(magp_phot_dir,sour_name)
    with open(magp_phot_fitfile,'r') as ff:
        alllines = ff.readlines()

    magpfilts = np.array(alllines[12][2:-1].split())
    paulchi2 = np.array(alllines[9][2:-1].split())[2]
    magp_filt_order = ['u','g','r','i','z','y','g_hsc','r_hsc','i_hsc','z_hsc','y_hsc','nb921',
                       'j', 'k',  'ch1_swire', 'ch2_swire', 'ch3_swire', 'ch4_swire', 'ch1_servs', 'ch2_servs',
                       'mips_24', 'pacs_100', 'pacs_160', 'spire_250', 'spire_350', 'spire_500']


    magp_phot_unordered = np.array(list(map(float,alllines[13].split())))
    magp_phot = np.zeros(26)

    filt_names = ['megacam_u','phaskell_gpc1_g', 'phaskell_gpc1_r', 'phaskell_gpc1_i',  
    'phaskell_gpc1_z','phaskell_gpc1_y', 'phaskell_suprime_g', 'phaskell_suprime_r', 'phaskell_suprime_i', 'phaskell_suprime_z', 
    'phaskell_suprime_y', 'phaskell_suprime_n921', 'phaskell_ukidss_j', 'phaskell_ukidss_k', 'phaskell_irac_i1', 
    'phaskell_irac_i2', 'phaskell_irac_i3', 'phaskell_irac_i4', 'phaskell_irac_i1', 'phaskell_irac_i2', 'phaskell_mips_24', 
    'phaskell_pacs_green_100', 'phaskell_pacs_red_160', 'phaskell_spire_250', 'phaskell_spire_350', 'phaskell_spire_500']
    
    for xx in range(len(filt_names)):
        tindx = np.where(magpfilts == magp_filt_order[xx])[0][0]
        magp_phot[xx] = magp_phot_unordered[tindx]

    return magp_phot, paulchi2

def copy_magp_outputs(sour_name,dest_dir):   
    import shutil
    plotdir = '/beegfs/lofar/deepfields/magphys/EN1/EN1_SED_plots/'
    plotfile = '{0}{1}_sed.pdf'.format(plotdir,sour_name)
    seddir = '/beegfs/lofar/deepfields/magphys/EN1/EN1_SEDs/'
    sedfile = '{0}{1}.sed'.format(seddir,sour_name)
    photdir = '/beegfs/car/phaskell/080720_Vn1_photometry/'
    photfile = '{0}{1}.fit'.format(photdir,sour_name)
    shutil.copy2(plotfile, dest_dir)
    shutil.copy2(sedfile, dest_dir)
    shutil.copy2(photfile, dest_dir)
    return 0


def pros_to_magp_phot(rest_wave, phot_flux, zred):
    import numpy as np
    # Astropy to calculate cosmological params
    from astropy.cosmology import FlatLambdaCDM
    import astropy.units as u

    # call the cosmo object from astropy
    cosmo = FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc, Tcmb0=2.725 * u.K, Om0=0.3)
    # Calculate the redshift comoving distance
    lumdist  = cosmo.luminosity_distance(zred).value     # in mpc
    lumdist_cm = lumdist*3.086e24 # in cm

    # Convert to rest frame
    phot_flux_rest_maggies = phot_flux/(1+zred) # in maggies

    Lo = 3.826e33
    phot_Lnu_Lsol = ((4*np.pi*3631.e-23)/Lo)*phot_flux_rest_maggies*lumdist_cm**2

    return phot_Lnu_Lsol # in Lsol/Hz

def magp_phot_Llambda(rest_wave_A, Lnu_Lsol):
    import numpy as np
    c  = 299792458
    Llambda_Lsol = (1.e10 *Lnu_Lsol*c)/( rest_wave_A**2)
    return Llambda_Lsol