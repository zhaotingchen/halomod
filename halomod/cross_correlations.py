"""
Define cross-correlated samples. Has classes for both pure HOD cross-correlations (i.e. number of cross-pairs) and
for HaloModel-derived quantities based on these cross-pairs.
"""

from .halo_model import TracerHaloModel
from hmf._framework import Component, Framework, get_model_
from hmf._cache import parameter, cached_quantity#,
#subframework#
from abc import ABCMeta, abstractmethod
import numpy as np
from scipy import integrate as intg
from halomod import tools


from .halo_model import DMHaloModel

class HODCross(Component):
    """
    Provides methods necessary to compute cross-correlation pairs for HOD models.
    """
    _defaults = {}
    __metaclass__ = ABCMeta

    def __init__(self, hods, **model_params):
        super(HODCross, self).__init__(**model_params)

        assert len(hods)==2
        self.hods = hods

    @abstractmethod
    def R_ss(self, m):
        """
        The cross-correlation of numbers of pairs within a halo.

        Notes
        -----
        Defined by

        .. math:: \langle T_1 T_2 \rangle  = \langle T_1 \rangle \langle T_2 \rangle + \sigma_1 \sigma_2 R_{ss},

        where :math:`T` is the total amount of tracer in the halo's profile (i.e. not counting the central component,
        if this exists).
        """
        pass

    @abstractmethod
    def R_cs(self, m):
        """
        The cross-correlation of central-satellite pairs within a halo (central from first hod, satellite from second)

        Notes
        -----
        Defined by

        .. math:: \langle T^c_1 T^s_2 \rangle  = \langle T^c_1 \rangle \langle T^s_2 \rangle + \sigma^c_1 \sigma^s_2 R_{cs},

        where :math:`T^s` is the total amount of tracer in the halo's profile (i.e. not counting the central component,
        if this exists).
        """
        pass


    @abstractmethod
    def R_sc(self, m):
        """
        The cross-correlation of satellite-central pairs within a halo (central from second hod, satellite from first)

        Notes
        -----
        Defined by

        .. math:: \langle T^s_1 T^c_2 \rangle  = \langle T^s_1 \rangle \langle T^c_2 \rangle + \sigma^s_1 \sigma^c_2 R_{sc},

        where :math:`T^s` is the total amount of tracer in the halo's profile (i.e. not counting the central component,
        if this exists).
        """
        pass

    @abstractmethod
    def self_pairs(self, m):
        "The expected number of cross-pairs at a separation of zero."
        pass

    def ss_cross_pairs(self, m):
        "The average value of cross-pairs in a halo of mass m, :math:`\langle T^s_1 T^s_2 \rangle - Q`."
        h1, h2 = self.hods

        return h1.satellite_occupation(m) * h2.satellite_occupation(m) + h1.sigma_satellite(m) * h2.sigma_satellite(m) * self.R_ss(m) - self.self_pairs(m)

    def cs_cross_pairs(self, m):
        "The average value of cross-pairs in a halo of mass m, :math:`\langle T^c_1 T^s_2 \rangle`."
        h1, h2 = self.hods

        return h1.central_occupation(m) * h2.satellite_occupation(m) + h1.sigma_central(m) * h2.sigma_satellite(
            m) * self.R_cs(m)

    def sc_cross_pairs(self, m):
        "The average value of cross-pairs in a halo of mass m, :math:`\langle T^s_1 T^c_2 \rangle`."
        h1, h2 = self.hods

        return h2.central_occupation(m) * h1.satellite_occupation(m) + h2.sigma_central(m) * h1.sigma_satellite(
            m) * self.R_sc(m)


class ConstantCorr(HODCross):
    _defaults = {"R_ss":0.0, "R_cs":0.0, "R_sc":0.0}

    @abstractmethod
    def R_ss(self, m):
        return self.params['R_ss']

    @abstractmethod
    def R_cs(self, m):
        return self.params['R_cs']

    @abstractmethod
    def R_sc(self, m):
        return self.params['R_sc']

    @abstractmethod
    def self_pairs(self, m):
        "The expected number of cross-pairs at a separation of zero."
        return 0


class CrossCorrelations(Framework):
    def __init__(self, cross_hod_model,
                 cross_hod_params={},
                 halo_model_1_params = {},
                 halo_model_2_params = {}):

        self.cross_hod_model = cross_hod_model
        self.cross_hod_params = cross_hod_params

        self._halo_model_1_params = halo_model_1_params
        self._halo_model_2_params = halo_model_2_params

    @parameter("model")
    def cross_hod_model(self, val):
        if not isinstance(val, str) and not np.issubclass_(val, HODCross):
            raise ValueError("cross_hod_model must be a subclass of cross_correlations.HODCross")
        elif isinstance(val, str):
            return get_model_(val, "")
        else:
            return val

    @parameter("param")
    def cross_hod_params(self, val):
        return val

    #@subframework
    @cached_quantity
    def halo_model_1(self):
        return TracerHaloModel(**self._halo_model_1_params)

    #@subframework
    @cached_quantity
    def halo_model_2(self):
        return TracerHaloModel(**self._halo_model_2_params)

    # ===========================================================================
    # Cross-correlations
    # ===========================================================================
    @cached_quantity
    def cross_hod(self):
        return self.cross_hod_model([self.halo_model_1.hod, self.halo_model_2.hod], **self.cross_hod_params)

    @cached_quantity
    def power_1h_cross(self):
        """
        Total 1-halo cross-power
        """
        hm1, hm2 = self.halo_model_1, self.halo_model_2
        mask = np.logical_and(
            np.logical_and(
                np.logical_not(np.isnan(self.cross_hod.ss_cross_pairs(hm1.m))),
                np.logical_not(np.isnan(self.cross_hod.sc_cross_pairs(hm1.m)))
            ),
            np.logical_not(np.isnan(self.cross_hod.cs_cross_pairs(hm1.m)))
        )

        m = hm1.m[mask]
        u1 = hm1.tracer_profile_ukm[:, mask]
        u2 = hm2.tracer_profile_ukm[:, mask]

        integ = hm1.dndm[mask] * (u1 * u2 * self.cross_hod.ss_cross_pairs(m) +
                                  u1*self.cross_hod.sc_cross_pairs(m) +
                                  u2*self.cross_hod.cs_cross_pairs(m))

        p = intg.simps(integ, m)

        return p / (hm1.mean_tracer_den * hm2.mean_tracer_den)

    @cached_quantity
    def corr_1h_cross(self):
        """The 1-halo term of the cross correlation"""
        return tools.power_to_corr_ogata(self.power_1h_cross,
                                         self.halo_model_1.k, self.halo_model_1.r)

    @cached_quantity
    def power_2h_cross(self):
        "The 2-halo term of the cross-power spectrum"
        hm1, hm2 = self.halo_model_1, self.halo_model_2

        u1 = hm1.tracer_profile_ukm[:, hm1._tm]
        u2 = hm2.tracer_profile_ukm[:, hm2._tm]

        bias = hm1.bias

        # Do this the simple way for now
        b1 = intg.simps(hm1.dndm[hm1._tm] * bias[hm1._tm] * hm1.total_occupation[hm1._tm] * u1,
                        hm1.m[hm1._tm])
        b2 = intg.simps(hm2.dndm[hm2._tm] * bias[hm2._tm] * hm2.total_occupation[hm2._tm] * u2,
                        hm2.m[hm2._tm])

        return b1 * b2 * hm1._power_halo_centres / (hm1.mean_tracer_den * hm2.mean_tracer_den)

    @cached_quantity
    def corr_2h_cross(self):
        """The 2-halo term of the cross-correlation"""

        return tools.power_to_corr_ogata(self.power_2h_cross, self.halo_model_1.k, self.halo_model_1.r)

    @cached_quantity
    def power_cross(self):
        """
        Total tracer auto power spectrum
        """
        return self.power_1h_cross + self.power_2h_cross

    @cached_quantity
    def corr_cross(self):
        """The tracer auto correlation function"""
        return self.corr_1h_cross + self.corr_2h_cross + 1


class OpticalHIContinuous(HODCross):
    """
    A cross-correlation model for discrete optical galaxies and a continuous HI distribution throughout the halo.

    In this model, we simply assume some constant (with mass) correlation co-efficient between the occupations.
    """

    _defaults = {"R_ss": 0.5, "R_cs": 0}

    def R_ss(self, m):
        return self.params['R_ss']

    def R_cs(self, m):
        return 0

    def R_sc(self, m):
        return 0

    def self_pairs(self, m):
        return 0


class OpticalHIDiscrete(HODCross):
    """
    A cross-correlation model for discrete optical galaxies and a discrete HI distribution (see derivation_of_1halo_term for details).
    """

    def R_ss(self, m):
        return self.self_pairs(m) / (self.hods[0].sigma_satellite(m) * self.hods[1].sigma_satellite(m))

    def R_cs(self, m):
        return 0

    def R_sc(self, m):
        return 0

    def self_pairs(self, m):
        return self.hods[0].satellite_occupation(m)


class OpticalOptical(HODCross):
    """
    A cross-correlation model for discrete optical galaxies and a discrete HI distribution (see derivation_of_1halo_term for details).
    """

    _defaults = {"R_ss": 0.0, "R_cs": 0}

    def R_ss(self, m):
        return self.params['R_ss']

    def R_cs(self, m):
        return 0

    def R_sc(self, m):
        return 0

    def self_pairs(self, m):
        return 0
