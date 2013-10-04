'''
Created on Sep 9, 2013

@author: Steven
'''
import numpy as np
import scipy.integrate as intg
import pycamb
from scipy.interpolate import InterpolatedUnivariateSpline as spline
import matplotlib.pyplot as plt

def power_to_corr(lnP, lnk, R):
    """
    Calculates the correlation function given a power spectrum
    
    NOTE: no check is done to make sure k spans [0,Infinity] - make sure of this
          before you enter the arguments.
    
    INPUT
        lnP: vector of values for the log power spectrum
        lnk: vector of values (same length as lnP) giving the log wavenumbers 
             for the power (EQUALLY SPACED)
        r:   radi(us)(i) at which to calculate the correlation
    """

    k = np.exp(lnk)
    P = np.exp(lnP)

    if not np.iterable(R):
        R = [R]

    corr = np.zeros_like(R)

    for i, r in enumerate(R):
        integ = P * k ** 2 * np.sin(k * r) / r


        corr_cum = (0.5 / np.pi ** 2) * intg.cumtrapz(integ, dx=lnk[1] - lnk[0])

        #Try this: we cut off the integral when we can no longer fit 5 steps between zeros.
        max_k = np.pi / (5 * r * (np.exp(lnk[1] - lnk[0]) - 1))
        max_index = np.where(k < max_k)[-1][-1]
        #Take average of last 20 values before the max_index
        corr[i] = np.mean(corr_cum[max_index - 20:max_index])


    return corr


def non_linear_power(lnk_out=None, **camb_kwargs):
    """
    Calculates the non-linear power spectrum from camb + halofit and outputs
    it at the given lnk_out if given.
    
    INPUT
    lnk_out: [None] The values of ln(k) at which the power spectrum should be output
    normalization: [1] Normalization constant for power (P = P*norm^2)
    **camb_kwargs: any argument for CAMB
    
    OUTPUT
    lnk: The lnk values at which the power is evaluated
    lnp: The log of the nonlinear power from halofit. 
    """
    #Must set scalar_amp small for it to work...
    camb_kwargs['scalar_amp'] = 1E-9

    k, P = pycamb.matter_power(NonLinear=1, **camb_kwargs)
    if lnk_out is not None:
        power_func = spline(np.log(k), np.log(P), k=1)
        P = np.exp(power_func(lnk_out))
        k = np.exp(lnk_out)

    return np.log(k), np.log(P)

def virial_mass(r, mean_dens, delta_halo):
    """
    Returns the virial mass of a given halo radius
    """
    return 4 * np.pi * r ** 3 * mean_dens * delta_halo / 3

def virial_radius(m, mean_dens, delta_halo):
    """
    Returns the virial mass of a given halo radius
    """
    return ((3 * m) / (4 * np.pi * mean_dens * delta_halo)) ** (1. / 3.)

def overlapping_halo_prob(r, rv1, rv2):
    """
    The probability of non-overlapping ellipsoidal haloes (Tinker 2005 Appendix B)
    """
    x = r / (rv1 + rv2)
    y = (x - 0.8) / 0.29
    if y <= 0:
        return 0
    elif y >= 1:
        return 1
    else:
        return 3 * y ** 2 - 2 * y ** 3



