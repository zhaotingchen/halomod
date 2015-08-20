'''
Created on 08/12/2014

@author: Steven
'''
from scipy.interpolate import InterpolatedUnivariateSpline as spline
import numpy as np
from hmf._framework import Model

class CMRelation(Model):
    r"""
    Base-class for Concentration-Mass relations

    This class should not be called directly, rather use a subclass which is
    specific to a certain relation.
    """
    _pdocs = \
    """

    Parameters
    ----------
    nu  : array
        A vector of peak-heights, :math:`\delta_c^2/\sigma^2`.

    z   : float, optional, default 0
        The redshift

    growth : type :class:`hmf.growth_factor.GrowthFactor`
        A model to calculate the growth factor

    M   : array
        masses

    \*\*model_parameters :
        These parameters are model-specific. For any model, list the available
        parameters (and their defaults) using ``<model>._defaults``

    """
    __doc__ += _pdocs
    _defaults = {}

    use_cosmo = False
    def __init__(self, nu=None, z=0.0, growth=None, M=None, **model_parameters):
        # Save instance variables
        self.nu = nu
        self.z = z
        self.growth = growth
        self.M = M

        super(CMRelation, self).__init__(**model_parameters)

class Bullock01(CMRelation):
    _defaults = {"F":0.001, "K":3.4}

    @property
    def zc(self):
        g = self.growth.growth_factor_fn(inverse=True)
        zc = g(np.sqrt(self.nu))
        zc[zc < 0] = 0.0  # hack?
        return zc

    def cm(self, m):
        return self.params["K"] * (self.zc + 1.0) / (self.z + 1.0)

class Cooray(CMRelation):
    _defaults = {"a":9.0, "b":0.13, "c":1.0, "ms":None}
    def ms(self):
        d = self.nu[1:] - self.nu[:-1]
        try:
            # this to start below "saturation level" in sharp-k filters.
            pos = np.where(d < 0)[0][-1]
        except IndexError:
            pos = 0
        nu = self.nu[pos:]
        ms = self.M[pos:]
        s = spline(nu, ms)
        return s(1.0)

    def cm(self, m):
        ms = (self.params['ms'] or self.ms()) * m.unit
        return self.params['a'] / (1 + self.z) ** self.params['c'] * (ms / m) ** self.params['b']

class Duffy(Cooray):
    _defaults = {"a":6.71, "b":0.091, "c":0.44, "ms":2e12}
