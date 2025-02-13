import astropy.units as u
from astropy.cosmology import FlatLambdaCDM
from astropy.cosmology import Planck18 as cosmo
from scipy.integrate import trapezoid
import numpy as np

def proper_distance(zed):
    '''Computes and returns the proper distance, assuming a Plank18 cosmology.'''
    return float(cosmo.luminosity_distance(zed)/(1+zed)/u.Mpc)

def inner_product(fnc_i, fnc_j, PSD, frequency):
    '''Compute the noise--weighted inner product of two functions for a given PSD.
    fnc_i, fnc_j: the two functions to be integrated over;
    PSD: the PSD data for the detector being considered (see waveforms.NSBH.get_PSD_data);
    frequency: frequency range over which to integrate.'''
    integrand = 2 * (np.conjugate(fnc_i)*fnc_j + fnc_i*np.conjugate(fnc_j))/PSD
    inner_prod = trapezoid(y=integrand, x=frequency)
    return inner_prod

def get_IFIM_errors(FIM):
    '''Computes and returns the inverse of a given Fisher information matrix.
    FIM: NxN Fisher information matrix.'''
    IFIM = np.linalg.inv(FIM)
    errors = np.sqrt(np.diag(IFIM))
    return IFIM, errors