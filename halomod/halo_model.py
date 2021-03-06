# ===============================================================================
# Some Imports
# ===============================================================================
from scipy.interpolate import InterpolatedUnivariateSpline as spline
import scipy.integrate as intg
import numpy as np
from scipy.optimize import minimize

# import scipy.special as sp

USEFORT = False

from hmf import MassFunction
from hmf._cache import cached_quantity, parameter
# import hmf.tools as ht
from . import tools
from . import hod
from .concentration import CMRelation
from .halo_exclusion import Exclusion, NoExclusion

if USEFORT:
    from .fort.routines import hod_routines as fort
from copy import copy, deepcopy
from numpy import issubclass_
from hmf._framework import get_model, get_model_
from . import profiles
from . import bias
from hmf.filters import TopHat
import warnings


class DMHaloModel(MassFunction):
    '''
    Calculates several quantities using the halo model.

    Parameters
    ----------
    r : array_like, optional, default ``np.logspace(-2.0,1.5,100)``
        The scales at which the correlation function is calculated in Mpc/*h*

    **kwargs: anything that can be used in the MassFunction class

    '''
    rlog = True

    def __init__(self, rmin=0.1, rmax=50.0, rnum=20,
                 halo_profile_model='NFW', halo_profile_params={},
                 halo_concentration_model='Duffy08', halo_concentration_params={},
                 bias_model="Tinker10", bias_params={},
                 sd_bias_model="Tinker_SD05", sd_bias_params={},
                 exclusion_model="NgMatched", exclusion_params={},
                 hc_spectrum="nonlinear", Mmin=0, Mmax=18,
                 force_1halo_turnover=True,
                 **hmf_kwargs):

        super(DMHaloModel, self).__init__(Mmin=Mmin, Mmax=Mmax, **hmf_kwargs)

        # Initially save parameters to the class.
        self.halo_profile_model, self.halo_profile_params = halo_profile_model, halo_profile_params
        self.halo_concentration_model, self.halo_concentration_params = halo_concentration_model, halo_concentration_params
        self.bias_model, self.bias_params = bias_model, bias_params
        self.sd_bias_model, self.sd_bias_params = sd_bias_model, sd_bias_params
        self.exclusion_model, self.exclusion_params = exclusion_model, exclusion_params

        self.rmin = rmin
        self.rmax = rmax
        self.rnum = rnum
        self.hc_spectrum = hc_spectrum
        self.force_1halo_turnover = force_1halo_turnover



    # ===============================================================================
    # Parameters
    # ===============================================================================
    @parameter("param")
    def bias_params(self, val):
        return val

    @parameter("switch")
    def bias_model(self, val):
        if not isinstance(val, str) and not issubclass_(val, bias.Bias):
            raise ValueError("bias_model must be a subclass of bias.Bias")
        elif isinstance(val, str):
            return get_model_(val, "halomod.bias")
        else:
            return val

    @parameter("switch")
    def hc_spectrum(self, val):
        return val

    @parameter("param")
    def sd_bias_params(self, val):
        return val

    @parameter("switch")
    def nonlinear(self, val):
        """Logical indicating whether the power is to be nonlinear or not"""
        try:
            if val:
                return True
            else:
                return False
        except:
            raise ValueError("nonlinear must be a logical value")

    @parameter("param")
    def halo_profile_params(self, val):
        """Dictionary of parameters for the Profile model"""
        return val

    @parameter("model")
    def halo_profile_model(self, val):
        """The halo density halo_profile model"""
        if not isinstance(val, str) and not issubclass_(val, profiles.Profile):
            raise ValueError("halo_profile_model must be a subclass of profiles.Profile")
        elif isinstance(val, str):
            return get_model_(val, "halomod.profiles")
        else:
            return val

    @parameter("model")
    def halo_concentration_model(self, val):
        """A halo_concentration-mass relation"""
        if not isinstance(val, str) and not issubclass_(val, CMRelation):
            raise ValueError("halo_concentration_model must be a subclass of halo_concentration.CMRelation")
        elif isinstance(val, str):
            return get_model_(val, "halomod.concentration")
        else:
            return val

    @parameter("param")
    def halo_concentration_params(self, val):
        return val

    @parameter("switch")
    def rmin(self, val):
        return val

    @parameter("res")
    def rmax(self, val):
        return val

    @parameter("res")
    def rnum(self, val):
        return val

    @parameter("switch")
    def sd_bias_model(self, val):
        if not isinstance(val, str) and not issubclass_(val, bias.ScaleDepBias) and val is not None:
            raise ValueError("scale_dependenent_bias must be a subclass of bias.ScaleDepBias")
        elif isinstance(val, str):
            model = get_model_(val, "halomod.bias")
            if not issubclass_(model, bias.ScaleDepBias):
                raise ValueError("scale_dependenent_bias must be a subclass of bias.ScaleDepBias")
            return model
        return val

    @parameter("switch")
    def force_1halo_turnover(self,val):
        return bool(val)

    @parameter("param")
    def exclusion_params(self, val):
        """Dictionary of parameters for the Exclusion model"""
        return val

    @parameter("switch")
    def exclusion_model(self, val):
        """A string identifier for the type of halo exclusion used (or None)"""
        if val is None:
            val = "NoExclusion"

        if issubclass_(val, Exclusion):
            return val
        else:
            return get_model_(val, "halomod.halo_exclusion")

    # ===========================================================================
    # Basic Quantities
    # ===========================================================================
    @cached_quantity
    def r(self):
        if type(self.rmin) == list or type(self.rmin) == np.ndarray:
            r = np.array(self.rmin)
        else:
            if self.rlog:
                r = np.exp(np.linspace(np.log(self.rmin), np.log(self.rmax), self.rnum))
            else:
                r = np.linspace(self.rmin, self.rmax, self.rnum)

        return r

    @cached_quantity
    def m(self):
        return 10 ** np.arange(self.Mmin, self.Mmax, self.dlog10m)

    @cached_quantity
    def bias(self):
        """The halo bias as a function of halo mass."""

        return self.bias_model(nu=self.nu, delta_c=self.delta_c,
                               m=self.m, mstar=self.mass_nonlinear,
                               delta_halo=self.delta_halo, n=self.n, Om0=self.cosmo.Om0,
                               h=self.cosmo.h, sigma_8=self.sigma_8,
                               **self.bias_params).bias()

    @cached_quantity
    def halo_cm(self):
        """A class containing the elements necessary to calculate the halo halo_concentration-mass relation"""
        this_filter = copy(self.filter)
        this_filter.power = self._power0
        this_profile = self.halo_profile_model(None, self.mean_density0, self.delta_halo, self.z, **self.halo_profile_params)

        return self.halo_concentration_model(filter0=this_filter, mean_density0=self.mean_density0,
                                             growth=self.growth, delta_c=self.delta_c, profile=this_profile,
                                             cosmo=self.cosmo, delta_halo=self.delta_halo,
                                             **self.halo_concentration_params)

    @cached_quantity
    def halo_concentration(self):
        """
        The halo concentrations corresponding to :meth:`m`.
        """
        return self.halo_cm.cm(self.m, self.z)

    @cached_quantity
    def halo_profile(self):
        """A class containing the elements necessary to calculate halo halo_profile quantities"""
        return self.halo_profile_model(cm_relation=self.halo_cm,
                                       mean_dens=self.mean_density0,
                                       delta_halo=self.delta_halo, z=self.z,
                                       **self.halo_profile_params)

    @cached_quantity
    def sd_bias(self):
        """A class containing relevant methods to calculate scale-dependent bias corrections"""
        if self.sd_bias_model is None:
            return None
        else:
            return self.sd_bias_model(self._corr_mm_base, **self.sd_bias_params)


    # ===========================================================================
    # Derived DM Quantities
    # ===========================================================================
    #
    #
    # @cached_quantity
    # def bias_effective_matter(self):
    #     """
    #     The effective bias on linear scales for dark matter
    #     """
    #     integrand = self.m ** 2*self.dndm*self.bias
    #     return intg.trapz(integrand, dx=np.log(10)*self.dlog10m)/self.mean_density0



    # ===========================================================================
    # Halo/DM Statistics
    # ===========================================================================
    @cached_quantity
    def _power_halo_centres(self):
        """
        Power spectrum of halo centres, unbiased.

        This defines the halo-centre power spectrum, which is a part of the 2-halo
        term calculation. Formally, we make the assumption that the halo-centre
        power spectrum is linearly biased, and this function returns

        .. math :: P^{hh}_c (k) /(b_1(m_1)b_2(m_2))

        This should probably be expanded to its own component module.
        """
        if self.hc_spectrum == "linear":
            return self.power
        elif self.hc_spectrum == "nonlinear":
            return self.nonlinear_power
        elif self.hc_spectrum == "filtered-nl":
            f = TopHat(None, None)
            out = self.nonlinear_power*f.k_space(self.k*3.0)
            first_zero = np.where(out <= 0)[0][0]
            out[first_zero:] = 0
            return out
        elif self.hc_spectrum == "filtered-lin":
            f = TopHat(None, None)
            out = self.power*f.k_space(self.k*2.0)
            first_zero = np.where(out <= 0)[0][0]
            out[first_zero:] = 0
            return out

    @cached_quantity
    def corr_linear_mm(self):
        "The linear auto-correlation function of dark matter"
        return tools.power_to_corr_ogata(self.power, self.k, self.r)

    @cached_quantity
    def corr_halofit_mm(self):
        "The nonlinear (directly from halofit) auto-correlation function of dark matter"
        return tools.power_to_corr_ogata(self.nonlinear_power, self.k, self.r)

    @cached_quantity
    def _corr_mm_base(self):
        "The matter correlation function used throughout the calculations"
        if self.hc_spectrum == "linear":
            return self.corr_linear_mm
        elif self.hc_spectrum == "nonlinear" or self.hc_spectrum == "filtered-nl":
            return self.corr_halofit_mm

    def power_hh(self, mmin=None, mmax=None, mmin2=None, mmax2=None):
        """
        The halo-centre power spectrum of haloes in a given mass range.

        The power of a given pair of halo masses is assumed to be linearly biased,
        :math:`P_hh(k) = b(m_1)b(m_2)P_{lin}(k)`

        Parameters
        ----------
        mmin : real, default :attr:`.Mmin`
            The minimum halo mass of the range (for the first of the halo pairs).
            Note: masses here are log10 masses.

        mmax : real, default :attr:`.Mmax`
            The maximum halo mass of the range (for the first of the halo pairs).
            If a single halo mass is desired, set mmax==mmin.

        mmin2 : real, default `None`
            The minimum halo mass of the range (for the second of the halo pairs).
            By default, takes the same value as `mmin`.

        mmax : real, default `None`
            The maximum halo mass of the range (for the second of the halo pairs).
            By default, takes the same value as `mmin`.
        """
        if mmin is None:
            mmin = self.Mmin
        if mmax is None:
            mmax = self.Mmax
        if mmin2 is None:
            mmin2 = mmin
        if mmax2 is None:
            mmax2 = mmax

        if mmin == mmax or mmin2 == mmax2:
            spl = spline(np.log10(self.m), self.bias)
        if mmin == mmax:
            b1 = spl(mmin)
        else:
            mask = np.logical_and(self.m >= 10 ** mmin, self.m <= 10 ** mmax)
            b1 = intg.simps(self.bias[mask]*self.dndm[mask], self.m[mask])/intg.simps(self.dndm[mask], self.m[mask])

        if mmin2 == mmax2:
            b2 = spl(mmin2)
        else:
            mask = np.logical_and(self.m >= 10 ** mmin2, self.m <= 10 ** mmax2)
            b2 = intg.simps(self.bias[mask]*self.dndm[mask], self.m[mask])/intg.simps(self.dndm[mask], self.m[mask])

        return b1*b2*self._power_halo_centres

    # ===========================================================================
    # Halo Profile cached quantities
    # ===========================================================================
    @cached_quantity
    def halo_profile_ukm(self):
        "Mass-normalised fourier halo profile, with shape (len(k), len(m))"
        return self.halo_profile.u(self.k, self.m)

    @cached_quantity
    def halo_profile_rho(self):
        "Mass-normalised halo density profile, with shape (len(r), len(m))"
        return self.halo_profile.rho(self.r, self.m, norm="m")

    @cached_quantity
    def halo_profile_lam(self):
        "Mass-normalised halo profile self-convolution, with shape (len(r), len(m))"
        return self.halo_profile.lam(self.r, self.m)

    # ===========================================================================
    # 2-point DM statistics
    # ===========================================================================
    @cached_quantity
    def power_1h_auto_matter(self):
        """
        The halo model-derived nonlinear 1-halo dark matter auto-power spectrum
        """
        u = self.halo_profile_ukm
        integrand = self.dndm*self.m ** 3*u ** 2

        return intg.trapz(integrand, dx=np.log(10)*self.dlog10m)/self.mean_density0 ** 2

    @cached_quantity
    def corr_1h_auto_matter(self):
        """
        The halo model-derived nonlinear 1-halo dark matter auto-correlation function
        """
        if self.halo_profile.has_lam:
            lam = self.halo_profile_lam
            integrand = self.dndm*self.m ** 3*lam

            return intg.trapz(integrand, dx=np.log(10)*self.dlog10m)/self.mean_density0 ** 2 - 1
        else:
            return tools.power_to_corr_ogata(self.power_1h_auto_matter, self.k, self.r)

    @cached_quantity
    def power_2h_auto_matter(self):
        "The halo model-derived nonlinear 2-halo dark matter auto-power spectrum"
        # TODO: check what to do here.
        # Basically, HMcode assumes that the large-scale power is equivalent
        # to the linear power, with no biasing. I think this *has* to be true
        # since the matter power is for *all* mass. But other codes (eg. chomp)
        # do the normal integral which includes biasing...
        return self._power_halo_centres

        # if self.exclusion_model != NoExclusion:
        #     u = self.halo_profile_ukm
        #
        #     if self.sd_bias_model is not None:
        #         bias = np.outer(self.sd_bias.bias_scale(), self.bias)
        #     else:
        #         bias = self.bias
        #
        #
        #     inst = self.exclusion_model(m=self.m, density=self.dndlnm,
        #                                 I=self.dndlnm*u/self.rho_gtm[0], bias=bias, r=self.r,
        #                                 delta_halo=self.delta_halo,
        #                                 mean_density=self.mean_density0,
        #                                 **self.exclusion_params)
        #     mult = inst.integrate()
        #
        #     # hackery to ensure large scales are unbiased independent of low-mass limit
        #     mult /= mult[-1]
        #
        # else:
        #     inst = 0
        #     mult = 1
        #
        # if hasattr(inst, "density_mod"):
        #     self.__density_mod_mm = inst.density_mod
        #     # FIXME: this is a bit of a hack, to take account of the fact that m[0] is not exactly 0, but should
        #     # be in the analytic integral.
        #     self.__density_mod_mm *= self.mean_density0/self.rho_gtm[0]
        # else:
        #     self.__density_mod_mm = self.mean_density0
        #
        # return mult*self._power_halo_centres

    @cached_quantity
    def corr_2h_auto_matter(self):
        "The halo-model-derived nonlinear 2-halo dark matter auto-correlation function"
        if self.exclusion_model is NoExclusion:
            return tools.power_to_corr_ogata(self.power_2h_auto_matter, self.k, self.r)
        else:
            return tools.power_to_corr_ogata_matrix(self.power_2h_auto_matter, self.k, self.r)

        ## modify by the new density
        #return (self.__density_mod_mm/self.mean_density0) ** 2*(1 + corr) - 1

    @cached_quantity
    def corr_auto_matter(self):
        """The halo-model-derived nonlinear dark matter auto-correlation function"""
        return self.corr_1h_auto_matter + self.corr_2h_auto_matter + 1

    @cached_quantity
    def power_auto_matter(self):
        """The halo-model-derived nonlinear dark power auto-power spectrum"""
        return self.power_1h_auto_matter + self.power_2h_auto_matter


class TracerHaloModel(DMHaloModel):
    '''
    Calculates several quantities using the halo model.

    Parameters
    ----------
    r : array_like, optional, default ``np.logspace(-2.0,1.5,100)``
        The scales at which the correlation function is calculated in Mpc/*h*

    **kwargs: anything that can be used in the MassFunction class

    '''
    rlog = True

    def __init__(self,
                 hod_model="Zehavi05", hod_params={},
                 tracer_profile_model=None, tracer_profile_params=None,
                 tracer_concentration_model=None, tracer_concentration_params=None,
                 tracer_density=None,
                 force_1halo_turnover=True,
                 **halomodel_kwargs):

        super(TracerHaloModel, self).__init__(**halomodel_kwargs)

        # Initially save parameters to the class.
        self.hod_params = hod_params
        self.hod_model = hod_model
        self.tracer_profile_model, self.tracer_profile_params = tracer_profile_model, tracer_profile_params
        self.tracer_concentration_model, self.tracer_concentration_params = tracer_concentration_model, tracer_concentration_params

        self.force_1halo_turnover = force_1halo_turnover
        # A special argument, making it possible to define M_min by mean density
        self.tracer_density = tracer_density

        # Find mmin if we want to
        if tracer_density is not None:
            mmin = self._find_m_min(tracer_density)
            self.hod_params = {"M_min": mmin}

    def update(self, **kwargs):
        """
        Updates any parameter passed
        """
        if "tracer_density" in kwargs:
            self.tracer_density = kwargs.pop('tracer_density')
        elif "hod_params" in kwargs:
            if "M_min" in kwargs["hod_params"]:
                self.tracer_density = None

        super(TracerHaloModel, self).update(**kwargs)

        if self.tracer_density is not None:
            mmin = self._find_m_min(self.tracer_density)
            self.hod_params = {"M_min": mmin}

    # ===============================================================================
    # Parameters
    # ===============================================================================
    @parameter("param")
    def tracer_density(self, val):
        """Mean density of the tracer, ONLY if passed directly"""
        return val

    @parameter("param")
    def hod_params(self, val):
        """Dictionary of parameters for the HOD model"""
        return val

    @parameter("model")
    def hod_model(self, val):
        """:class:`~hod.HOD` class"""
        if not isinstance(val, str) and not issubclass_(val, hod.HOD):
            raise ValueError("hod_model must be a subclass of hod.HOD")
        elif isinstance(val, str):
            return get_model_(val, "halomod.hod")
        else:
            return val

    @parameter("param")
    def tracer_profile_params(self, val):
        """Dictionary of parameters for the Profile model"""
        return val

    @parameter("model")
    def tracer_profile_model(self, val):
        """The halo density halo_profile model"""
        if val is None:
            return val
        if not isinstance(val, str) and not issubclass_(val, profiles.Profile):
            raise ValueError("halo_profile_model must be a subclass of profiles.Profile")
        if isinstance(val, str):
            return get_model_(val, "halomod.profiles")
        else:
            return val

    @parameter("model")
    def tracer_concentration_model(self, val):
        """A halo_concentration-mass relation"""
        if val is None:
            return val
        if not isinstance(val, str) and not issubclass_(val, CMRelation):
            raise ValueError("halo_concentration_model must be a subclass of concentration.CMRelation")
        elif isinstance(val, str):
            return get_model_(val, "halomod.concentration")
        else:
            return val

    @parameter("param")
    def tracer_concentration_params(self, val):
        return val

    @parameter("switch")
    def force_1halo_turnover(self,val):
        return bool(val)

    # ===========================================================================
    # Basic Quantities
    # ===========================================================================
    # THE FOLLOWING IS LEFT IN AS A REMINDER NEVER TO DO IT
    # CHANGING THE MINIMUM MASS DYNAMICALLY DESTROYS MANY THINGS, LIKE THE ABILITY TO CROSS-CORRELATE TWO CLASSES.
    # @cached_quantity
    # def mmin(self):
    #     "This is the true minimum mass for this framework"
    #     return min(self.Mmin, self.hod.mmin)
    #
    # @cached_quantity
    # def m(self):
    #     return 10 ** np.arange(self.mmin, self.Mmax, self.dlog10m)

    @cached_quantity
    def _tm(self):
        """
        A tracer mask -- i.e. a mask on mass which restricts the range to those where the tracer exists
        for the given HOD.
        """
        if self.hod.mmin is None:
            return self.m >= self.m.min()

        if self.hod.mmin < self.Mmin:
            raise ValueError("The HOD is defined to lower masses than currently calculated. Please set Mmin lower.")

        return self.m >= 10 ** self.hod.mmin

    @cached_quantity
    def tracer_cm(self):
        """A class containing the elements necessary to calculate the halo_concentration-mass relation"""
        if self.tracer_concentration_model is None:
            return self.halo_concentration_model

        this_filter = copy(self.filter)
        this_filter.power = self._power0
        this_profile = self.profile_model(None, self.mean_density0, self.delta_halo, self.z, **self.tracer_profile_params)

        return self.tracer_concentration_model(filter0=this_filter, mean_density0=self.mean_density0,
                                        growth=self.growth, delta_c=self.delta_c, profile=this_profile,
                                        cosmo=self.cosmo, delta_halo=self.delta_halo,
                                        **self.tracer_concentration_params)

    @cached_quantity
    def tracer_concentration(self):
        """
        The concentrations corresponding to `.m`
        """
        return self.tracer_cm.cm(self.m, self.z)

    @cached_quantity
    def tracer_profile(self):
        """A class containing the elements necessary to calculate quantities of the tracer profile"""
        if self.tracer_profile_model is None:
            return self.halo_profile

        return self.profile_model(cm_relation=self.tracer_cm,
                                  mean_dens=self.mean_density0,
                                  delta_halo=self.delta_halo, z=self.z,
                                  **self.tracer_profile_params)

    @cached_quantity
    def hod(self):
        if self.tracer_profile_model is None:
            return self.hod_model(cm_relation=self.tracer_cm,
                              mean_dens=self.mean_density0,
                              delta_halo=self.delta_halo, z=self.z,
                              **self.hod_params, **self.halo_profile_params)

        return self.hod_model(cm_relation=self.tracer_cm,
                              mean_dens=self.mean_density0,
                              delta_halo=self.delta_halo, z=self.z,
                              **self.hod_params, **self.tracer_profile_params)

    # ===========================================================================
    # Basic HOD Quantities
    # ===========================================================================
    # @cached_quantity
    # def n_sat(self):
    #     """Average satellite occupancy of halo of mass m"""
    #     return self.hod.ns(self.m)
    #
    # @cached_quantity
    # def n_cen(self):
    #     """Average satellite occupancy of halo of mass m"""
    #     return self.hod.central_occupation(self.m)
    #
    # @cached_quantity
    # def n_tot(self):
    #     """Average satellite occupancy of halo of mass m"""
    #     return self.hod.ntot(self.m)

    @cached_quantity
    def total_occupation(self):
        "The mean occupation of the tracer as a function of halo mass"
        return self.hod.total_occupation(self.m)

    # ===========================================================================
    # Derived HOD Quantities
    # ===========================================================================
    @cached_quantity
    def mean_tracer_den(self):
        """
        The mean density of the tracer.

        This is always the *integrated* density. If `tracer_density` is supplied to the constructor,
        that value can be found as :meth:`.tracer_density`. It should be very close to this value.
        """
        integrand = self.dndm[self._tm] * self.total_occupation[self._tm]
        return intg.simps(integrand, self.m[self._tm])

    @cached_quantity
    def mean_tracer_den_unit(self):
        """
        The mean density of the tracer, in the units defined in the HOD.
        """
        return self.mean_tracer_den * self.hod.unit_conversion(self.cosmo, self.z)

    @cached_quantity
    def bias_effective(self):
        """
        The tracer occupation-weighted halo bias factor (Tinker 2005)
        """
        # Integrand is just the density of galaxies at mass m by bias
        integrand = self.m[self._tm] * self.dndm[self._tm] * self.total_occupation[self._tm] * self.bias[self._tm]
        b = intg.trapz(integrand, dx=np.log(self.m[1]/self.m[0]))
        return b/self.mean_tracer_den

    @cached_quantity
    def mass_effective(self):
        """
        Average host-halo mass (in log10 units)
        """
        # Integrand is just the density of galaxies at mass m by m
        integrand = self.m[self._tm] ** 2 * self.dndm[self._tm] * self.total_occupation[self._tm]

        m = intg.trapz(integrand, dx=np.log(self.m[1]/self.m[0]))
        return np.log10((m / self.mean_tracer_den))

    @cached_quantity
    def satellite_fraction(self):
        "The total fraction of tracers that are satellites (if this quantity exists for this kind of tracer)"
        if hasattr(self.hod, "satellite_occupation"):
            # Integrand is just the density of satellite galaxies at mass m
            integrand = self.m[self._tm] * self.dndm[self._tm] * self.hod.satellite_occupation(self.m[self._tm])
            s = intg.trapz(integrand, dx=np.log(self.m[1]/self.m[0]))
            return s/self.mean_tracer_den
        else:
            raise AttributeError("This HOD has no concept of a satellite")

    @cached_quantity
    def central_fraction(self):
        "The total fraction of tracers that are centrals (if this quantity exists for this kind of tracer)"
        return 1 - self.satellite_fraction

    @cached_quantity
    def tracer_density_m(self):
        """
        The total tracer density in halos of mass m
        """
        return self.dndm*self.total_occupation

    # ===========================================================================
    # Tracer Profile cached quantities
    # ===========================================================================
    @cached_quantity
    def tracer_profile_ukm(self):
        "The mass-normalised fourier density profile of the tracer, with shape (len(k), len(m))"
        return self.tracer_profile.u(self.k, self.m)

    @cached_quantity
    def tracer_profile_rho(self):
        "The mass-normalised density profile of the tracer, with shape (len(r), len(m))"
        return self.tracer_profile.rho(self.r, self.m, norm="m")

    @cached_quantity
    def tracer_profile_lam(self):
        "The mass-normalised profile self-convolution of the tracer, with shape (len(r), len(m))"
        return self.tracer_profile.lam(self.r, self.m)


    # ===========================================================================
    # 2-point tracer-tracer (HOD) statistics
    # ===========================================================================
    @cached_quantity
    def power_1h_ss_auto_tracer(self):
        """
        The satellite-satellite part of the 1-halo term of the tracer auto-power spectrum (if this exists)
        """
        if not hasattr(self.hod, "ss_pairs"):
            raise AttributeError("The HOD being used has no satellite occupation")

        u = self.tracer_profile_ukm[:, self._tm]
        integ = u ** 2*self.dndm[self._tm] * self.m[self._tm] * self.hod.ss_pairs(self.m[self._tm])

        if self.force_1halo_turnover:
            r = np.pi/self.k/10  # The 10 is a complete heuristic hack.
            mmin = 4*np.pi*r ** 3*self.mean_density0*self.delta_halo/3
            mask = np.outer(self.m[self._tm], np.ones_like(self.k)) < mmin
            integ[mask.T] = 0

        p = intg.trapz(integ, dx=self.dlog10m*np.log(10))

        return p/ self.mean_tracer_den ** 2

    @cached_quantity
    def corr_1h_ss_auto_tracer(self):
        """
        The satellite-satellite part of the 1-halo term of the tracer auto-correlation function (if this exists)
        """
        if not hasattr(self.hod, "ss_pairs"):
            raise AttributeError("The HOD being used has no satellite occupation")

        if self.tracer_profile.has_lam:
            lam = self.tracer_profile.lam(self.r, self.m[self._tm], norm="m")
            integ = self.m[self._tm] * self.dndm[self._tm] * self.hod.ss_pairs(self.m[self._tm]) * lam

            c = intg.trapz(integ, dx=self.dlog10m*np.log(10))

            return c/ self.mean_tracer_den ** 2 - 1
        else:
            return tools.power_to_corr_ogata(self.power_1h_ss_auto_tracer,
                                             self.k, self.r)

    @cached_quantity
    def power_1h_cs_auto_tracer(self):
        """The cen-sat part of the 1-halo term of the tracer auto-power spectrum (if this exists)"""
        if not hasattr(self.hod, "cs_pairs"):
            raise AttributeError("The HOD being used has no satellite occupation")

        u = self.tracer_profile_ukm[:, self._tm]
        integ = self.dndm[self._tm] * 2 * self.hod.cs_pairs(self.m[self._tm]) * u * self.m[self._tm]

        if self.force_1halo_turnover:
            r = np.pi/self.k/10  # The 10 is a complete heuristic hack.
            mmin = 4*np.pi*r ** 3*self.mean_density0*self.delta_halo/3
            mask = np.outer(self.m[self._tm], np.ones_like(self.k)) < mmin
            integ[mask.T] = 0

        c = intg.trapz(integ, dx=self.dlog10m*np.log(10))
        return c/ self.mean_tracer_den ** 2

    @cached_quantity
    def corr_1h_cs_auto_tracer(self):
        """The cen-sat part of the 1-halo term of the tracer auto-correlation function (if this exists)"""
        if not hasattr(self.hod, "cs_pairs"):
            raise AttributeError("The HOD being used has no satellite occupation")

        rho = self.tracer_profile_rho[:, self._tm]
        integ = self.dndm[self._tm] * 2 * self.hod.cs_pairs(self.m)[self._tm] * rho * self.m[self._tm]
        c = intg.trapz(integ, dx=self.dlog10m*np.log(10))

        return c/ self.mean_tracer_den ** 2 - 1

    @cached_quantity
    def power_1h_auto_tracer(self):
        """
        Total 1-halo galaxy power.
        """
        try:
            return self.power_1h_cs_auto_tracer + self.power_1h_ss_auto_tracer
        except AttributeError:
            u = self.tracer_profile_ukm[:, self._tm]
            integ = u ** 2 * self.dndm[self._tm] * self.m[self._tm] * self.total_occupation[self._tm] ** 2

            if self.force_1halo_turnover:
                r = np.pi / self.k / 10  # The 10 is a complete heuristic hack.
                mmin = 4 * np.pi * r ** 3 * self.mean_density0 * self.delta_halo / 3
                mask = np.outer(self.m[self._tm], np.ones_like(self.k)) < mmin
                integ[mask.T] = 0

            p = intg.trapz(integ, dx=self.dlog10m * np.log(10))

            return p / self.mean_tracer_den ** 2

    @cached_quantity
    def corr_1h_auto_tracer(self):
        """The 1-halo term of the galaxy correlations"""
        if self.tracer_profile.has_lam:
            if hasattr(self.hod, "ss_pairs"):
                rho = self.tracer_profile_rho[:, self._tm]
                lam = self.tracer_profile_lam[:, self._tm]
                integ = self.m[self._tm] * self.dndm[self._tm] * (self.hod.ss_pairs(self.m[self._tm]) * lam +
                                                                  2 * self.hod.cs_pairs(self.m[self._tm]) * rho)
                if self.hod._central:
                    integ *= self.n_cen[self._tm]

                c = intg.trapz(integ, dx=self.dlog10m * np.log(10))

                return c / self.mean_tracer_den ** 2 - 1
            else:
                lam = self.tracer_profile_lam[:, self._tm]
                integ = self.m[self._tm] * self.dndm[self._tm] * (self.total_occupation[self._tm] ** 2 * lam)
                c = intg.trapz(integ, dx=self.dlog10m * np.log(10))
                return c / self.mean_tracer_den ** 2 - 1
        else:
            try:
                return self.corr_1h_cs_auto_tracer + self.corr_1h_ss_auto_tracer + 1
            except AttributeError:
                return tools.power_to_corr_ogata(self.power_1h_auto_tracer,
                                                 self.k, self.r)

    @cached_quantity
    def _power_2h_auto_tracer(self):
        "The 2-halo term of the tracer auto-power spectrum"
        u = self.tracer_profile_ukm[:, self._tm]
        if self.sd_bias_model is not None:
            bias = np.outer(self.sd_bias.bias_scale(), self.bias)[:, self._tm]
        else:
            bias = self.bias[self._tm]
        inst = self.exclusion_model(m=self.m[self._tm], density=self.total_occupation[self._tm] * self.dndm[self._tm],
                                    I=self.total_occupation[self._tm] * self.dndm[self._tm] * u / self.mean_tracer_den,
                                    bias=bias, r=self.r, delta_halo=self.delta_halo,
                                    mean_density=self.mean_density0,
                                    **self.exclusion_params)

        if hasattr(inst, "density_mod"):
            self.__density_mod = inst.density_mod
        else:
            self.__density_mod = np.ones_like(self.r)*self.mean_tracer_den

        return inst.integrate()*self._power_halo_centres

    @cached_quantity
    def power_2h_auto_tracer(self):

        if self.r.min() > 0.3 / self.k.max():
            warnings.warn("Smallest r not small enough for highest k, setting to %s" % (0.1 / self.k.max()))
            self.update(rmin=0.3 / self.k.max())
        if self.r.max() < 15.0 / self.k.min():
            warnings.warn("Largest r not large enough for lowest k, setting to %s" % (10 / self.k.min()))
            self.update(rmax=15. / self.k.min())
        if np.log(self.r[1] / self.r[0]) > 0.1:
            warnings.warn("Probably not enough r steps. Check convergence at higher number.")

        return tools.corr_to_power_ogata(self.corr_2h_auto_tracer, self.r, self.k)

    @cached_quantity
    def corr_2h_auto_tracer(self):
        """The 2-halo term of the tracer auto-correlation"""

        if self.exclusion_model is NoExclusion and self.sd_bias_model is None:
            corr = tools.power_to_corr_ogata(self._power_2h_auto_tracer, self.k, self.r)
        else:
            corr = tools.power_to_corr_ogata_matrix(self._power_2h_auto_tracer, self.k, self.r)

        ## modify by the new density. This step is *extremely* sensitive to the
        ## exact value of __density_mod at large scales, where the ratio *should*
        ## be exactly 1.
        if self.r[-1] > 2*self.halo_profile._mvir_to_rvir(self.m[-1]):
            try:
                self.__density_mod *= self.mean_tracer_den / self.__density_mod[-1]
            except TypeError:
                pass

        return (self.__density_mod / self.mean_tracer_den) ** 2 * (1 + corr) - 1

    @cached_quantity
    def power_auto_tracer(self):
        """
        Total tracer auto power spectrum
        """
        return self.power_1h_auto_tracer + self.power_2h_auto_tracer

    @cached_quantity
    def corr_auto_tracer(self):
        """The tracer auto correlation function"""
        return self.corr_1h_auto_tracer + self.corr_2h_auto_tracer + 1

    # ===========================================================================
    # Cross-correlations
    # ===========================================================================
    @cached_quantity
    def power_1h_cross_tracer_matter(self):
        """
        Total 1-halo cross-power
        """
        ut = self.tracer_profile_ukm[:, self._tm]
        uh = self.halo_profile_ukm[:, self._tm]

        m = self.m[self._tm]

        integ = self.dndm[self._tm] * (uh * ut * m * self.total_occupation[self._tm] + uh * self.hod.satellite_occupation(m))
        p = intg.simps(integ, m)

        return p / (self.mean_tracer_den*self.mean_density)

    @cached_quantity
    def corr_1h_cross_tracer_matter(self):
        """The 1-halo term of the cross correlation"""
        return tools.power_to_corr_ogata(self.power_1h_cross_tracer_matter,
                                         self.k, self.r)

    @cached_quantity
    def power_2h_cross_tracer_matter(self):
        "The 2-halo term of the cross-power spectrum"
        ut = self.tracer_profile_ukm[:, self._tm]
        um = self.halo_profile_ukm

        # if self.sd_bias_model is not None:
        #     bias = np.outer(self.sd_bias.bias_scale(), self.bias)[:, self._tm]
        # else:
        bias = self.bias

        # Do this the simple way for now
        bt = intg.simps(self.dndm[self._tm] * bias[self._tm] * self.total_occupation[self._tm] * ut, self.m[self._tm])
        bm = intg.simps(self.dndm * bias * self.m * um, self.m)

        return bt * bm * self._power_halo_centres / (self.mean_tracer_den * self.mean_density)

        # inst = self.exclusion_model(m=self.m[self._tm],
        #                             density=self.total_occupation[self._tm] * self.dndm[self._tm],
        #                             I=self.total_occupation[self._tm] * self.dndm[
        #                                 self._tm] * u / self.mean_tracer_den,
        #                             bias=bias, r=self.r, delta_halo=self.delta_halo,
        #                             mean_density=self.mean_density0,
        #                             **self.exclusion_params)


        # if hasattr(inst, "density_mod"):
        #     self.__density_mod = inst.density_mod
        # else:
        #     self.__density_mod = np.ones_like(self.r) * self.mean_tracer_den

        # return inst.integrate() * self._power_halo_centres

    @cached_quantity
    def corr_2h_cross_tracer_matter(self):
        """The 2-halo term of the cross-correlation"""

        #if self.exclusion_model is NoExclusion and self.sd_bias_model is None:
        return tools.power_to_corr_ogata(self.power_2h_cross_tracer_matter, self.k, self.r)
        #else:
        #    corr = tools.power_to_corr_ogata_matrix(self.power_2h_auto_tracer, self.k, self.r)

        # ## modify by the new density. This step is *extremely* sensitive to the
        # ## exact value of __density_mod at large scales, where the ratio *should*
        # ## be exactly 1.
        # if self.r[-1] > 2 * self.halo_profile._mvir_to_rvir(self.m[-1]):
        #     try:
        #         self.__density_mod *= self.mean_tracer_den / self.__density_mod[-1]
        #     except TypeError:
        #         pass
        #
        # return (self.__density_mod / self.mean_tracer_den) ** 2 * (1 + corr) - 1

    @cached_quantity
    def power_cross_tracer_matter(self):
        """
        Total tracer auto power spectrum
        """
        return self.power_1h_cross_tracer_matter + self.power_2h_cross_tracer_matter

    @cached_quantity
    def corr_cross_tracer_matter(self):
        """The tracer auto correlation function"""
        return self.corr_1h_cross_tracer_matter + self.corr_2h_cross_tracer_matter + 1

    # ===========================================================================
    # Other utilities
    # ===========================================================================
    def _find_m_min(self, ng):
        """
        Calculate the minimum mass of a halo to contain a (central) galaxy
        based on a known mean galaxy density
        """

        self.power  # This just makes sure the power is gotten and copied
        c = deepcopy(self)
        c.update(hod_params={"M_min": self.Mmin}, dlog10m=0.01)

        integrand = c.m[c._tm] * c.dndm[c._tm] * c.total_occupation[c._tm]

        if self.hod.sharp_cut:
            integral = intg.cumtrapz(integrand[::-1], dx=np.log(c.m[1]/c.m[0]))

            if integral[-1] < ng:
                raise NGException("Maximum mean galaxy density exceeded. User input required density of %, " +
                                  "but maximum density (with HOD M_min == DM Mmin) is %s. Consider decreasing Mmin,"
                                  "or checking tracer_density."%(ng, integral[-1]))

            ind = np.where(integral > ng)[0][0]

            m = c.m[c._tm][::-1][1:][max(ind - 4, 0):min(ind + 4, len(c.m))]
            integral = integral[max(ind - 4, 0):min(ind + 4, len(c.m))]

            spline_int = spline(np.log(integral), np.log(m), k=3)
            mmin = spline_int(np.log(ng))/np.log(10)
        else:
            # Anything else requires us to do some optimization unfortunately.
            integral = intg.simps(integrand, dx=np.log(c.m[1]/c.m[0]))
            if integral < ng:
                raise NGException("Maximum mean galaxy density exceeded. User input required density of %, " +
                                  "but maximum density (with HOD M_min == DM Mmin) is %s. Consider decreasing Mmin,"
                                  "or checking tracer_density."%(ng, integral))

            def model(mmin):
                c.update(hod_params={"M_min": mmin})
                integrand = c.m[c._tm] * c.dndm[c._tm] * c.ean_occupation[c._tm]
                integral = intg.simps(integrand, dx=np.log(c.m[1]/c.m[0]))
                return abs(integral - ng)

            res = minimize(model, 12.0, tol=1e-3,
                           method="Nelder-Mead", options={"maxiter": 200})
            mmin = res.x[0]

        return mmin


    # =============================
    # For Compatibility
    # =============================
    @property
    def corr_gg_1h(self):
        warnings.warn("This method is deprecated in favour of corr_1h_auto_tracer")
        return self.corr_1h_auto_tracer

    @property
    def corr_gg_2h(self):
        warnings.warn("This method is deprecated in favour of corr_2h_auto_tracer")
        return self.corr_2h_auto_tracer

    @property
    def corr_gg(self):
        warnings.warn("This method is deprecated in favour of corr_auto_tracer")
        return self.corr_auto_tracer

    @property
    def power_gg_1h(self):
        warnings.warn("This method is deprecated in favour of power_1h_auto_tracer")
        return self.power_1h_auto_tracer

    @property
    def power_gg_2h(self):
        warnings.warn("This method is deprecated in favour of power_2h_auto_tracer")
        return self.power_2h_auto_tracer

    @property
    def power_gg(self):
        warnings.warn("This method is deprecated in favour of power_auto_tracer")
        return self.power_auto_tracer

    @property
    def corr_mm_1h(self):
        warnings.warn("This method is deprecated in favour of corr_1h_auto_matter")
        return self.corr_1h_auto_matter

    @property
    def corr_mm_2h(self):
        warnings.warn("This method is deprecated in favour of corr_2h_auto_matter")
        return self.corr_2h_auto_matter

    @property
    def corr_mm(self):
        warnings.warn("This method is deprecated in favour of corr_auto_matter")
        return self.corr_auto_matter

    @property
    def power_mm_1h(self):
        warnings.warn("This method is deprecated in favour of power_1h_auto_matter")
        return self.power_1h_auto_matter

    @property
    def power_mm_2h(self):
        warnings.warn("This method is deprecated in favour of power_2h_auto_matter")
        return self.power_2h_auto_matter

    @property
    def power_mm(self):
        warnings.warn("This method is deprecated in favour of power_auto_matter")
        return self.power_auto_matter


# For compatibility
HaloModel = TracerHaloModel


class NGException(Exception):
    pass
