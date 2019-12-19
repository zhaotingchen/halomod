"""
Module for defining HOD classes.

The HOD class exposes methods that deal directly with occupation statistics and don't interact with the broader halo
model. These include things like the average satellite/central occupation, total occupation, and "pair counts".

The HOD concept is here meant to be as general as possible. While traditionally the HOD has been thought of as a
number count occupation, the base class here is just as amenable to "occupations" that could be defined over the
real numbers -- i.e. continuous occupations. This could be achieved via each "discrete" galaxy being marked by some
real quantity (eg. each galaxy is on average a certain brightness, or contains a certain amount of gas), or it could
be achieved without assuming any kind of discrete tracer, and just assuming a matching of some real field to the
underlying halo mass. Thus  *all* kinds of occupations can be dealt with in these classes.

For the sake of consistency of implementation, all classes contain the notion that there may be a "satellite" component
of the occupation, and a "central" component. This is to increase fidelity in cases where it is known that a discrete
central object will necessarily be in the sample before any other object, because it is inherently "brighter" (for whatever
selection the sample uses). It is not necessary to assume some distinct central component, so for models in which
this does not make sense, it is safe to set the central component to zero.

The most subtle/important thing to note about these classes are the assumptions surrounding the satellite/central
decomposition. So here are the assumptions:

1. The average satellite occupancy is taken to be the average over *all* haloes, with and without centrals. This has
   subtle implications for how to mock up the galaxy population, because if one requires a central before placing a
   satellite, then the avg. number of satellites placed into *available* haloes is increased if the central occupation
   is less than 1.

2. We provide the option to enforce a "central condition", that is, the requirement that a central be found in a halo
   before any satellites are observed. To enforce this, set ``central=True`` in the constructor of any HOD. This has
   some ramifications:

3. If the central condition is enforced, then for all HOD classes (except see point 5), the mean satellite occupancy is
   modified. If the defined occupancy is Ns', then the returned occupancy is Ns = Nc*Ns'. This merely ensures that Ns=0
   when Nc=0. The user should note that this will change the interpretation of parameters in the Ns model, unless Nc is
   a simple step function.

4. The pair-wise counts involve a term <Nc*Ns>. When the central condition is enforced, this reduces trivially to <Ns>.
   However, if the central condition is not enforced we *assume* that the variates Nc and Ns are uncorrelated, and
   use <Nc*Ns> = <Nc><Ns>.

5. A HOD class that is defined with the central condition intrinsically satisfied, the class variable
   ``central_condition_inherent`` can be set to True in the class definition, which will avoid the extra modification.
   Do note that just because the class is specified such that the central condition can be satisfied (i.e. <Ns> is 0
   when <Nc> is zero), and thus the ``central_condition_inherent`` is True, does not mean that it is entirely enforced.
   The pairwise counts still depend on whether the user assumes that the central condition is enforced or not, which must
   be set at instantiation.

6. By default, the central condition is *not* enforced.
"""


import numpy as np
import scipy.special as sp
from hmf._framework import Component
from abc import ABCMeta, abstractmethod
import scipy.constants as const
import astropy.constants as astroconst


class HOD(Component):
    """
    Halo Occupation Distribution model base class.

    This class should not be called directly. The user
    should call a derived class.

    As with all :class:`hmf._framework.Model` classes,
    each class should specify its parameters in a _defaults dictionary at
    class-level.

    The exception to this is the M_min parameter, which is defined for every
    model (it may still be defined to modify the default). This parameter acts
    as the one that may be set via the mean density given all the other
    parameters. If the model has a sharp cutoff at low mass, corresponding to
    M_min, the extra parameter sharp_cut may be set to True, allowing for simpler
    setting of M_min via this route.

    See the derived classes in this module for examples of how to define derived
    classes of :class:`HOD`.
    """
    __metaclass__ = ABCMeta

    _defaults = {"M_min": 11.0}
    sharp_cut = False
    central_condition_inherent = False

    def __init__(self, cm_relation, mean_dens,
                 delta_halo=200.0, z=0.0, central=False, **model_parameters):

        self.delta_halo = delta_halo
        self.z = z
        self._cm_relation = cm_relation
        self.mean_dens = mean_dens

        self.has_lam = hasattr(self, "_l")
        self._central = central
        super(HOD, self).__init__(**model_parameters)


    @abstractmethod
    def nc(self, m):
        "Defines the average number of centrals at mass m. Useful for populating catalogues"
        pass

    @abstractmethod
    def ns(self, m):
        "Defines the average number of satellites at mass m. Useful for populating catalogues"
        pass

    @abstractmethod
    def _central_occupation(self, m):
        "The occupation function of the tracer"
        pass

    @abstractmethod
    def _satellite_occupation(self, m):
        "The occupation function of the tracer"
        pass

    @abstractmethod
    def ss_pairs(self,m):
        "The average amount of the tracer coupled with itself in haloes of mass m, <T_s T_s>"
        pass

    @abstractmethod
    def cs_pairs(self,m):
        "The average amount of the tracer coupled with itself in haloes of mass m, <T_s T_c>"
        pass

    @abstractmethod
    def sigma_satellite(self, m):
        "The standard deviation of the total tracer amount in haloes of mass m"
        pass

    @abstractmethod
    def sigma_central(self, m):
        "The standard deviation of the total tracer amount in haloes of mass m"
        pass

    def central_occupation(self, m):
        "The occupation function of the central component"
        return self._central_occupation(m)

    def satellite_occupation(self,m):
        "The occupation function of the satellite (or profile-dependent) component"
        if self._central and not self.central_condition_inherent:
            return self.nc(m) * self._satellite_occupation(m)
        else:
            return self._satellite_occupation(m)

    def total_occupation(self, m):
        "The total (average) occupation of the halo"
        return self.central_occupation(m) + self.satellite_occupation(m)

    def total_pair_function(self,m):
        "The total weight of the occupation paired with itself"
        return self.ss_pairs(m) + self.cs_pairs(m)

    def unit_conversion(self, cosmo, z):
        "A factor (potentially with astropy units) to convert the total occupation to a desired unit."
        return 1.0

    @property
    def mmin(self):
        "A function defining a reasonable minimum mass to set for this HOD to converge when integrated."
        return self.params['M_min']


class HODNoCentral(HOD):
    """
    Base class for all HODs which have no concept of a central/satellite split.
    """
    def __init__(self, **model_parameters):
        super(HODNoCentral, self).__init__(**model_parameters)
        self._central = False

    def nc(self, m):
        return 0

    def cs_pairs(self,m):
        return 0

    def _central_occupation(self, m):
        return 0

    def sigma_central(self, m):
        return 0


class HODBulk(HODNoCentral):
    "Base class for HODs that have no discrete tracers, just a bulk assignment of tracer to the halo"
    def ns(self,m):
        return 0

    def ss_pairs(self,m):
        return self.satellite_occupation(m)**2


class HODPoisson(HOD):
    """
    This class is a base class for all discrete HOD's for which the tracer has a poisson-distributed satellite
    count population, and for which the amount of the tracer is statistically independent of the number counts, but its
    average is directly proportional to it.

    This accounts for all Poisson-distributed number-count HOD's (which is all traditional HODs).
    """

    def nc(self, m):
        return self.central_occupation(m) /  self._tracer_per_central(m)

    def ns(self, m):
        return self.satellite_occupation(m) / self._tracer_per_satellite(m)

    def _tracer_per_central(self, m):
        return 1

    def _tracer_per_satellite(self,m):
        return self._tracer_per_central(m)

    def ss_pairs(self,m):
        return self.satellite_occupation(m)**2

    def cs_pairs(self,m):
        if self._central:
            return self.satellite_occupation(m) * self._tracer_per_central(m)
        else:
            return self.central_occupation(m) * self.satellite_occupation(m)

    def sigma_central(self, m):
        co = self.central_occupation(m)
        return np.sqrt(co*(1-co))

    def sigma_satellite(self, m):
        return np.sqrt(self.satellite_occupation(m))


class Zehavi05(HODPoisson):
    """
    Three-parameter model of Zehavi (2005)

    Parameters
    ----------
    M_min : float, default = 11.6222
        Minimum mass of halo that supports a central galaxy

    M_1 : float, default = 12.851
        Mass of a halo which on average contains 1 satellite

    alpha : float, default = 1.049
        Index of power law for satellite galaxies
    """
    _defaults = {"M_min":11.6222,
                 "M_1":12.851,
                 "alpha":1.049}
    sharp_cut = True

    def _central_occupation(self, M):
        """
        Number of central galaxies at mass M
        """
        n_c = np.zeros_like(M)
        n_c[M >= 10 ** self.params["M_min"]] = 1

        return n_c

    def _satellite_occupation(self, M):
        """
        Number of satellite galaxies at mass M
        """
        return (M / 10 ** self.params["M_1"]) ** self.params["alpha"]


class Zheng05(HODPoisson):
    """
    Five-parameter model of Zehavi (2005)

    Parameters
    ----------
    M_min : float, default = 11.6222
        Minimum mass of halo that supports a central galaxy

    M_1 : float, default = 12.851
        Mass of a halo which on average contains 1 satellite

    alpha : float, default = 1.049
        Index of power law for satellite galaxies

    sig_logm : float, default = 0.26
        Width of smoothed cutoff

    M_0 : float, default = 11.5047
        Minimum mass of halo containing satellites
    """
    _defaults = {"M_min":11.6222,
                 "M_1":12.851,
                 "alpha":1.049,
                 "M_0":11.5047,
                 "sig_logm":0.26
                 }

    def _central_occupation(self, M):
        """
        Number of central galaxies at mass M
        """
        nc = 0.5 * (1 + sp.erf((np.log10(M) - self.params["M_min"]) / self.params["sig_logm"]))
        return nc

    def _satellite_occupation(self, M):
        """
        Number of satellite galaxies at mass M
        """
        ns = np.zeros_like(M)
        ns[M > 10 ** self.params["M_0"]] = ((M[M > 10 ** self.params["M_0"]] - 10 ** self.params["M_0"]) / 10 ** self.params["M_1"]) ** self.params["alpha"]
        return ns

    @property
    def mmin(self):
        return self.params["M_min"] - 5 * self.params["sig_logm"]


class Contreras13(HODPoisson):
    """
    Nine-parameter model of Contreras (2013)

    Parameters
    ----------
    M_min : float, default = 11.6222
        Minimum mass of halo that supports a central galaxy

    M_1 : float, default = 12.851
        Mass of a halo which on average contains 1 satellite

    alpha : float, default = 1.049
        Index of power law for satellite galaxies

    sig_logm : float, default = 0.26
        Width of smoothed cutoff

    M_0 : float, default = 11.5047
        Minimum mass of halo containing satellites

    fca : float, default = 0.5
        fca

    fcb : float, default = 0
        fcb

    fs : float, default = 1
        fs

    delta : float, default  = 1
        delta

    x : float, default = 1
        x
    """
    _defaults = {"M_min":11.6222,
                 "M_1":12.851,
                 "alpha":1.049,
                 "M_0":11.5047,
                 "sig_logm":0.26,
                 "fca":0.5,
                 "fcb":0,
                 "fs":1,
                 "delta":1,
                 "x":1
                 }

    def _central_occupation(self, M):
        """
        Number of central galaxies at mass M
        """
        return self.params["fcb"] * (1 - self.params["fca"]) * np.exp(-np.log10(M / 10 ** self.params["M_min"]) ** 2 / (2 * (self.params["x"] * self.params["sig_logm"]) ** 2)) + self.params["fca"] * (1 + sp.erf(np.log10(M / 10 ** self.params["M_min"]) / self.params["x"] / self.params["sig_logm"]))

    def _satellite_occupation(self, M):
        """
        Number of satellite galaxies at mass M
        """
        return self.params["fs"] * (1 + sp.erf(np.log10(M / 10 ** self.params["M_1"]) / self.params["delta"])) * (M / 10 ** self.params["M_1"]) ** self.params["alpha"]


class Geach12(Contreras13):
    """
    8-parameter model of Geach et. al. (2012). This is identical to `Contreras13`,
    but with `x==1`.
    """
    pass


class Tinker05(Zehavi05):
    """
    3-parameter model of Tinker et. al. (2005).
    """
    _defaults = {"M_min":11.6222,
                 "M_1":12.851,
                 "M_cut":12.0}
    central_condition_inherent = True

    def _satellite_occupation(self, M):
        out = self.central_occupation(M)
        return out*np.exp(-10**self.params["M_cut"]/(M-10**self.params["M_min"]))*(M/10**self.params["M_1"])


class Zehavi05_WithMax(Zehavi05):
    """
    A version of the Zehavi05 model in which a maximum halo mass for occupancy also exists.
    """
    _defaults = {"alpha":0, # power-law slope
                 "M_1":11,  # mass at which mean occupation is A
                 "M_min":11, # Truncation mass
                 "M_max":18, # Truncation mass
                }

    def _central_occupation(self, M):
        """
        Number of central galaxies at mass M
        """
        n_c = np.zeros_like(M)
        n_c[np.logical_and(M >= 10 ** self.params["M_min"],M <= 10 ** self.params["M_max"])] = 1

        return n_c

    def _satellite_occupation(self, M):
        """
        Number of satellite galaxies at mass M
        """
        return (M / 10 ** self.params["M_1"]) ** self.params["alpha"]


class Zehavi05_Marked(Zehavi05_WithMax):
    """
    Exactly the Zehavi05 model, except with a possibility that the quantity is not number counts but some other
    quantity. NOTE: this should not give different results to Zehavi05 for any normalised statistic.
    """
    _defaults = {"M_min":11.6222,
                 "M_1":12.851,
                 "logA":0.0,
                 "alpha":1.049,
                 "M_max":18.0
                 }

    def sigma_central(self, m):
        co = super(Zehavi05_Marked, self)._central_occupation(m)
        return np.sqrt(self._tracer_per_central(m) * co * (1-co))

    def _tracer_per_central(self, m):
        return 10 ** self.params['logA']

    def _central_occupation(self, M):
        return super(Zehavi05_Marked, self)._central_occupation(M) * self._tracer_per_central(M)

    def _satellite_occupation(self, M):
        return super(Zehavi05_Marked, self)._satellite_occupation(M) * self._tracer_per_satellite(M)


class Zehavi05_tracer(Zehavi05_WithMax):
    """
    This class is based on the Zehavi05_WithMax class for a tracer with amplitude logA.
    The tracer can follow a different HOD than the underlying galaxy counts, which also follow a Zehavi05_WithMax HOD.
    """
    _defaults = {"M_min": 11.6222,
                 "M_1": 12.851,
                 "logA": 0.0,
                 "alpha": 1.049,
                 "M_max": 18.0,
                 "M_1_counts": 12.851,
                 "alpha_counts": 1.049,
                 }

    def unit_conversion(self, cosmo, z):
        "A factor (potentially with astropy units) to convert the total occupation to a desired unit."
        A12 = 2.869e-15
        nu21cm = 1.42e9
        Const = (3.0 * A12 * const.h * const.c ** 3.0) / (
                    32.0 * np.pi * (const.m_p + const.m_e) * const.Boltzmann * nu21cm ** 2);
        Mpcoverh_3 = ((astroconst.kpc.value * 1e3) / (cosmo.H0.value / 100.0)) ** 3
        hubble = cosmo.H0.value * cosmo.efunc(z) * 1.0e3 / (astroconst.kpc.value * 1e3)
        temp_conv = Const * ((1.0 + z) ** 2 / hubble)
        # convert to Mpc^3, solar mass
        temp_conv = temp_conv / Mpcoverh_3 * astroconst.M_sun.value
        return temp_conv

    def _central_occupation(self, M):
        """
        Number of central galaxies at mass M
        """
        n_c = np.zeros_like(M)
        n_c[np.logical_and(M >= 10 ** self.params["M_min"], M <= 10 ** self.params["M_max"])] = 1 * 10 ** self.params[
            'logA']

        return n_c

    def _satellite_occupation(self, M):
        """
        Number of satellite galaxies at mass M
        """
        return (M / 10 ** self.params["M_1"]) ** self.params["alpha"] * 10 ** self.params['logA']

    def sigma_central(self, m):
        co = super(Zehavi05_tracer, self)._central_occupation(m)
        return np.sqrt(self._tracer_per_central(m) * co * (1 - co))

    def _tracer_per_central(self, M):
        tpc = self._central_occupation(M) / self.nc(M)
        tpc[np.isnan(tpc)] = 0.0

        return tpc

    def _tracer_per_satellites(self, M):
        tps = np.zeros_like(M)
        index = self.ns(M) != 0.0
        tps[index] = self._satellite_occupation(M[index]) / self.ns(M[index])

        return tps

    def nc(self, M):
        n_c = np.zeros_like(M)
        n_c[np.logical_and(M >= 10 ** self.params["M_min"], M <= 10 ** self.params["M_max"])] = 1
        return n_c

    def ns(self, M):
        n_s = np.zeros_like(M)
        index = np.logical_and(M >= 10 ** self.params["M_min"], M <= 10 ** self.params["M_max"])
        n_s[index] = (M[index] / 10 ** self.params["M_1_counts"]) ** self.params["alpha_counts"]

        return n_s


class Zehavi05_centrals(Zehavi05):
    """
    A version of the Zehavi05 model in which a maximum halo mass for occupancy also exists.
    """
    _defaults = {"alpha": 0,  # power-law slope
                 "M_1": 11,  # mass at which mean occupation is A
                 "M_min": 11,  # Truncation mass
                 "M_max": 18,  # Truncation mass
                 "M_lim": 13
                 }

    def _central_occupation(self, M):
        """
        Number of central galaxies at mass M
        """
        n_c = np.zeros_like(M)
        n_c[np.logical_and(M >= 10 ** self.params["M_min"], M <= 10 ** self.params["M_max"])] = 1

        return n_c

    def _satellite_occupation(self, M):
        """
        Number of satellite galaxies at mass M
        """
        n_s = np.zeros_like(M)
        # index=np.logical_and(M >= 10 ** self.params["M_lim"],M <= 10 ** self.params["M_max"])
        # n_s[index]=(M[index] / 10 ** self.params["M_1"]) ** self.params["alpha"]

        return n_s


class Zehavi05_satellites(Zehavi05):
    """
    A version of the Zehavi05 model in which a maximum halo mass for occupancy also exists.
    """
    _defaults = {"alpha": 0,  # power-law slope
                 "M_1": 11,  # mass at which mean occupation is A
                 "M_min": 11,  # Truncation mass
                 "M_max": 18,  # Truncation mass
                 "M_lim": 13
                 }

    def _central_occupation(self, M):
        """
        Number of central galaxies at mass M
        """
        n_c = np.zeros_like(M)
        # n_c[np.logical_and(M >= 10 ** self.params["M_min"],M <= 10 ** self.params["M_max"])] = 1

        return n_c

    def _satellite_occupation(self, M):
        """
        Number of satellite galaxies at mass M
        """
        n_s = np.zeros_like(M)
        index = np.logical_and(M >= 10 ** self.params["M_lim"], M <= 10 ** self.params["M_max"])
        n_s[index] = (M[index] / 10 ** self.params["M_1"]) ** self.params["alpha"]

        return n_s


class Zehavi05_blue(Zehavi05):
    """
    A version of the Zehavi05 model in which a maximum halo mass for occupancy also exists.
    """
    _defaults = {"alpha": 0,  # power-law slope
                 "M_1": 11,  # mass at which mean occupation is A
                 "M_min": 11,  # Truncation mass
                 "M_max": 18,  # Truncation mass
                 }

    def _central_occupation(self, M):
        """
        Number of central galaxies at mass M
        """
        n_c = np.zeros_like(M)
        n_c[np.logical_and(M >= 10 ** self.params["M_min"], M <= 10 ** self.params["M_max"])] = 1

        return n_c

    def _satellite_occupation(self, M):
        """
        Number of satellite galaxies at mass M
        """
        n_s = np.zeros_like(M)
        index = np.logical_and(M >= 10 ** self.params["M_min"], M <= 10 ** self.params["M_max"])
        n_s[index] = (M[index] / 10 ** self.params["M_1"]) ** self.params["alpha"]

        return n_s


class ContinuousPowerLaw(HODBulk):
    """
    A continuous HOD which is tuned to match the Zehavi05 total occupation except for normalisation.
    """
    _defaults = {"alpha":0,  # power-law slope
                 "M_1":11,   # mass at which HI mass is A
                 "logA":9,   # gives HI mass at M_1
                 "M_min":11, # Truncation mass
                 "M_max":18,  # Truncation mass
                 "sigma_A":0 # The (constant) standard deviation of the tracer
                }
    sharp_cut = True

    def _satellite_occupation(self, m):
        alpha = self.params['alpha']
        M_1 = 10 ** self.params['M_1']
        A = 10 ** self.params['logA']
        M_min = 10 ** self.params['M_min']
        M_max = 10 ** self.params['M_max']

        out = np.where(np.logical_and(m >= M_min, m <= M_max), A * ((m / M_1) ** alpha + 1.), 0)
        return out

    def sigma_satellite(self, m):
        return np.ones_like(m) * self.params['sigma_A']


class Constant(HODBulk):
    "A toy model HOD in which every halo has the same amount of the tracer on average"
    _defaults = {"logA": 0, "M_min":11.0, "sigma_A": 0}

    def _satellite_occupation(self, m):
        return np.where(m > 10**self.params["M_min"], 10 ** self.params['logA'], 0)

    def sigma_satellite(self, m):
        return np.ones_like(m) * self.params['sigma_A']


class VN2018Continuous(HODBulk):
    """
    A continuous HOD following  Villaescusa-Navarro et al. 2018
    """
    _defaults = {"alpha": 0.24,  # power-law slope
                 "logA": 12.3324,  # gives HI mass amplitude
                 "M_min": 9, # Truncation Mass
                 "M_1": 12.3,  # Characteristic Mass
                 "sigma_A": 0,  # The (constant) standard deviation of the tracer
                 "beta": 0.35, # The slope within the exp term
                 "M_max": 18   # Truncation mass
                 }
    sharp_cut = False

    def _satellite_occupation(self, m):
        alpha = self.params['alpha']
        A = 10 ** self.params['logA']
        M_1 = 10 ** self.params['M_1']
        beta = self.params['beta']

        out = A * (m / M_1) ** alpha * np.exp(-(M_1 / m) ** beta)
        return out

    def sigma_satellite(self, m):
        return np.ones_like(m) * self.params['sigma_A']

    def unit_conversion(self, cosmo, z):
        "A factor (potentially with astropy units) to convert the total occupation to a desired unit."
        A12=2.869e-15
        nu21cm=1.42e9
        Const=( 3.0*A12*const.h*const.c**3.0 )/( 32.0*np.pi*(const.m_p+const.m_e)*const.Boltzmann * nu21cm**2);
        Mpcoverh_3=((astroconst.kpc.value*1e3)/(cosmo.H0.value/100.0) )**3
        hubble = cosmo.H0.value * cosmo.efunc(z)*1.0e3/(astroconst.kpc.value*1e3)
        temp_conv=Const * ((1.0+z)**2/hubble)
        # convert to Mpc^3, solar mass
        temp_conv=temp_conv/Mpcoverh_3 * astroconst.M_sun.value
        return temp_conv

class Padmanabhan(HODBulk):
    """
    A continuous HOD following Padmanabhan & Refregier (1607.01021)
    """
    _defaults = {"alpha": 0.09,  # gives HI mass amplitude
                 "f_Hc": 0.12,  # gives HI mass amplitude, fixed by Yp and Omegab
                 "beta": -0.58, #slop of mass
                 "M_min": 9, # Truncation Mass
                 "M_1": 11,  # Characteristic Mass
                 "sigma_A": 0,  # The (constant) standard deviation of the tracer
                 "M_max": 18,   # Truncation mass
                 "vc0": 36.31 #characteristic virial velocity, in km/s
                 }
    sharp_cut = False

    def _satellite_occupation(self, m):
        alpha = self.params['alpha']
        f_Hc = self.params['f_Hc']
        beta = self.params['beta']
        vc0 = self.params['vc0']
        M_1 = 10 ** self.params['M_1']

        out = alpha*f_Hc*m*(m/M_1)**beta*np.exp(-(vc0/self.virial_velocity(m))**3)
        return out

    def sigma_satellite(self, m):
        return np.ones_like(m) * self.params['sigma_A']

    def unit_conversion(self, cosmo, z):
        "A factor (potentially with astropy units) to convert the total occupation to a desired unit."
        A12=2.869e-15
        nu21cm=1.42e9
        Const=( 3.0*A12*const.h*const.c**3.0 )/( 32.0*np.pi*(const.m_p+const.m_e)*const.Boltzmann * nu21cm**2);
        Mpcoverh_3=((astroconst.kpc.value*1e3)/(cosmo.H0.value/100.0) )**3
        hubble = cosmo.H0.value * cosmo.efunc(z)*1.0e3/(astroconst.kpc.value*1e3)
        temp_conv=Const * ((1.0+z)**2/hubble)
        # convert to Mpc^3, solar mass
        temp_conv=temp_conv/Mpcoverh_3 * astroconst.M_sun.value
        return temp_conv

    def _mvir_to_rvir(self, m):
        """ Return the virial radius corresponding to m"""
        return (3 * m / (4 * np.pi * self.delta_halo * self.mean_dens)) ** (1. / 3.)

    def _rvir_to_mvir(self, r):
        """Return the virial mass corresponding to r"""
        return 4 * np.pi * r ** 3 * self.delta_halo * self.mean_dens / 3

    def _rs_from_m(self, m, c=None):
        """
        Return the scale radius for a halo of mass m

        Parameters
        ----------
        m : float
            mass of the halo

        c : float, default None
            halo_concentration of the halo (if None, use cm_relation to get it).
        """
        if c is None:
            c = self.cm_relation(m)
        rvir = self._mvir_to_rvir(m)
        return rvir / c

    def virial_velocity(self,m=None,r=None):
        """
        Return the virial velocity for a halo of virial mass `m`.

        Either `m` or `r` must be passed. If both are passed, `m`
        is preferentially used.

        Parameters
        ----------
        m : array_like, optional
            Masses of halos.

        r : array_like, optional
            Radii of halos.
        """
        if m is None and r is None:
            raise ValueError("Either m or r must be specified")
        if m is not None:
            r = self._mvir_to_rvir(m)
        else:
            m = self._rvir_to_mvir(r)
        return np.sqrt(6.673*1e-11*m/r) #convert to km/s?