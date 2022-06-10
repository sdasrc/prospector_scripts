# --------------------------------------------------------------------------- #
#                          S C I E N C E       P A R T S                      #
# ----------------------------------------------------------------------------#

def maggie_to_jy(fmaggie):
    '''    1 maggie = 3631 jy    '''
    return fmaggie*3631.

def jy_to_maggie(fjy):
    '''    1 maggie = 3631 jy    '''
    return fjy/3631.

def jy_to_ergs(fjy):
    '''    1 jy = 10^-23 ergs cm-2 s-1 Hz-1    '''
    return fjy*1.e-23


def mpc_to_cm(lmpc):
    '''    1 jy = 10^-23 ergs cm-2 s-1 Hz-1    '''
    return lmpc*3.086e24

def fnu_to_flambda(fluxnu,wavelnths):
    '''
    fnu_to_flambda(fluxnu,wavelnths)
    Returns f_lambda (Jy/m) given f_nu in Jy/Hz 
    and corresponding wavelength in metres
    '''
    c = 299792458
    flambda = (c*fluxnu)/(wavelnths*wavelnths)
    return flambda

def flambda_to_fnu(fluxlambda,freqarr):
    '''
    flambda_to_fnu(fluxlambda,freqarr)
    Returns f_lambda (Jy/m) given f_nu in Jy/Hz 
    and corresponding wavelength in metres
    '''
    c = 299792458
    fnu = (c*fluxlambda)/(freqarr*freqarr)
    return fnu

def flux_to_lum(fluxarr,lumdistmpc):
    '''
    flux_to_lum(fluxarr,lumdistmpc)
    Returns lumarr (ergs/s) given fluxarr in ergs/cm2/s/Hz
    and corresponding luminosity distance in Mpc
    '''
    import numpy as np
    lumdistcm = mpc_to_cm(lumdistmpc)
    lumarr = fluxarr*4*np.pi*lumdistcm*lumdistcm
    return lumarr

def get_lambLum_Lsol(thiswave,thisobs,lumdist):
    '''
    Converts fnu (maggies/Jy) to lambda L_lambda/Lsol
    (same units as magphys)
    '''
    import numpy as np
    if not isinstance(thiswave, np.ndarray): thiswave = np.array(thiswave)
    if not isinstance(thisobs, np.ndarray): thisobs = np.array(thisobs)
    thiswave_in_metre = thiswave*1.e-10
    fnu_jy = maggie_to_jy(thisobs)
    flambda_jy = fnu_to_flambda(fnu_jy,thiswave_in_metre)
    flux_ergs = jy_to_ergs(flambda_jy)

    lum_ergs = flux_to_lum(flux_ergs,lumdist)
    Lo = 3.826e33

    lambLum_Lsol = (lum_ergs*thiswave_in_metre)/Lo
    return lambLum_Lsol