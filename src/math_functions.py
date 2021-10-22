from typing import Union

import astropy.units as u
import numpy as np
import scipy.constants as const
from astropy.constants import M_sun
from numpy.core.defchararray import array
from scipy.special import digamma, gamma
from uncertainties.unumpy.core import uarray

# Constants in the equations:
E_p = const.m_p * const.c ** 2.0 * 1e7  # [erg]
c_1 = 6.26428e18  # [erg**-2 s**-1 G**-1]
c_3 = 1.86558e-23  # [erg G**-1 sr**-1]


def beam_size(FWHM: float) -> float:
    """Calculate beam size in sterad

    Args:
        FWHM (float): FWHm of the beam in [arcsec]

    Returns:
        float: the beam size in [sterad]
    """
    return np.pi / (4 * np.log(2)) * (1 / 3600 * np.pi / 180) ** 2 * FWHM ** 2


def b_field_revised(
    alpha: np.array,
    beam: float,
    I_nu: np.array,
    nu: float,
    K_0: int,
    alpha_min: float,
    alpha_max: float,
    pathlength: float,
    inclination: int = 0,
) -> np.array:
    """Calculate the B-Field from the equipartition estimate from Beck & Krause 2005

    Args:
        alpha (np.array): spectral index
        beam (float): beam size in [sterad]
        I_nu (np.array): the radio intensity in [Jy/beam] or the flux density in [Jy/kpc²]
        nu (float): the frequency of the radio intensity in [Hz]
        K_0 (int): estimated Proton/Electron ratio in CRs
        alpha_min (float): minimum (flattest) spectral index
        alpha_max (float): maximum (steepest) spectral index
        pathlength (float): estimated length through the galaxy along line of sight in [pc]
        inclination (int, optional): inclination of the galaxy. Defaults to 0.

    Returns:
        np.array: the magnetic field strength in [G]
    """
    # make sure to use negative spectral indices
    alpha_min = -abs(alpha_min)
    alpha_max = -abs(alpha_max)
    alpha = -abs(alpha)

    # convert I_nu from [Jy/beam] to [erg s**-1 cm**-2 Hz**-1 sr**-1]
    I_nu = I_nu / beam * 1.0e-23

    # Clip spectral indices
    alpha = np.clip(alpha, alpha_max, alpha_min)

    # Correct pathlength for inclination
    pathlength = (
        pathlength / np.cos(inclination * np.pi / 180) * const.parsec * 1.0e2
    )  # [cm]

    c_2 = (
        1.0
        / 4.0
        * c_3
        * (alpha - 5.0 / 3.0)
        / (alpha - 1.0)
        * gamma((-3.0 * alpha + 1.0) / 6.0)
        * gamma((-3.0 * alpha + 5.0) / 6.0)
    )
    c_4 = (2 / 3) ** (
        (1 - alpha) / 2
    )  # averaged cos over all i ### np.cos(inclination) ** ((gamma_e + 1) / 2)

    B_eq_no_exponent = (
        4.0
        * const.pi
        * (2.0 * alpha - 1.0)
        * (K_0 + 1.0)
        * I_nu
        * E_p ** (1.0 + 2.0 * alpha)
        * (nu / (2.0 * c_1)) ** (-alpha)
        / ((2.0 * alpha + 1.0) * c_2 * pathlength * c_4)
    )

    # if we calculate the spatially resolved magnetic field
    if hasattr(B_eq_no_exponent, "__len__"):
        # don't use the value if I_nu was negative and so was B_eq_no_exponent
        B_eq_no_exponent[B_eq_no_exponent <= 0] = None

    B_eq = B_eq_no_exponent ** (1.0 / (3.0 - alpha))
    return B_eq


def b_field_revised_error(
    alpha: np.array,
    alpha_error: np.array,
    beam: float,
    I_nu: np.array,
    I_nu_error: float,
    nu: float,
    K_0: int,
    K_0_error: float,
    alpha_min: float,
    alpha_max: float,
    pathlength: float,
    pathlength_error: float,
    inclination: int = 0,
) -> np.array:
    """Calculate the B-Field error from the equipartition estimate from Beck & Krause 2005

    Args:
        alpha (np.array): spectral index
        alpha_error (np.array): spectral index error
        beam (float): beam size in [sterad]
        I_nu (np.array): the radio intensity in [Jy/beam] or the flux density in [Jy/kpc²]
        I_nu_error (float): the radio intensity error in [Jy/beam] or the flux density error in [Jy/kpc²]
        nu (float): the frequency of the radio intensity in [Hz]
        K_0 (int): estimated Proton/Electron ratio in CRs
        K_0_error (float): estimated error on the Proton/Electron ratio in CRs
        alpha_min (float): minimum (flattest) spectral index
        alpha_max (float): maximum (steepest) spectral index
        pathlength (float): estimated length through the galaxy along line of sight in [pc]
        pathlength_error (float): estimated error on the length through the galaxy along line of sight in [pc]
        inclination (int, optional): inclination of the galaxy. Defaults to 0.

    Returns:
        np.array: the magnetic field strength error in [G]
    """
    # make sure to use negative spectral indices
    alpha_min = -abs(alpha_min)
    alpha_max = -abs(alpha_max)
    alpha = -abs(alpha)
    # Clip spectral index error at 1
    alpha_error = np.clip(abs(alpha_error), None, 1)

    B_eq = b_field_revised(
        alpha, beam, I_nu, nu, K_0, alpha_min, alpha_max, pathlength, inclination
    )

    # convert I_nu from [Jy/beam] to [erg s**-1 cm**-2 Hz**-1 sr**-1]
    I_nu = I_nu / beam * 1.0e-23
    I_nu_error = I_nu_error / beam * 1.0e-23

    # Python calculation errors occure sometimes because I_Nu**2 is too small
    if hasattr(I_nu, "__len__"):
        I_nu[I_nu ** 2 == 0] = np.nan
        I_nu_error[np.isnan(I_nu)] = np.nan

    # Clip spectral indices
    alpha = np.clip(alpha, alpha_max, alpha_min)

    # Correct pathlength for inclination
    pathlength = (
        pathlength / np.cos(inclination * np.pi / 180) * const.parsec * 1.0e2
    )  # [cm]

    B_eq_var = (B_eq / (3 - alpha)) ** 2 * (
        (I_nu_error / I_nu) ** 2
        + K_0_error ** 2 / (K_0 + 1) ** 2
        + pathlength_error ** 2 / pathlength ** 2
        + (
            +np.log(B_eq) / (3 - alpha)
            - np.log(nu / (2 * c_1 * E_p ** 2))
            + 4 / (4 * alpha ** 2 - 1)
            - (1 / 2) * np.log(2 / 3)
            + 1 / (alpha - 5 / 3)
            - 1 / (alpha - 1)
            - 1 / 2 * digamma((-3 * alpha + 1) / 6)
            - 1 / 2 * digamma((-3 * alpha + 5) / 6)
        )
        ** 2
        * alpha_error ** 2
    )
    if hasattr(B_eq_var, "__len__"):
        B_eq_var[np.isnan(I_nu)] = np.nan
    return np.sqrt(B_eq_var)


def get_energy_density_from_surface_density(
    surface_density: np.array, pathlength: float, dispersion: np.array
) -> np.array:
    """Calculate the energy density in erg/cm**3 with formular:
        rho_E = 1.36 1/2 * rho_H1 * vel**2
            = 1/2 * sigma/l * dispersion**2

    Args:
        surface_density (np.array): surface density in M_sun/pc**2
        pathlength (float): Thickness of galaxy in pc
        dispersion (np.array): velocity dispersion in m/s

    Returns:
        np.array: energy density in erg/cm**3
    """
    return (
        1
        / 2
        * surface_density
        * M_sun.value
        / (const.parsec) ** 2
        / (pathlength * const.parsec)
        * (dispersion) ** 2
        * 1e7  # erg
        * 1e-6  # cm**-3
    )


def get_energy_density_error(
    surf: np.array, surf_error: np.array, pathlength: float, dispersion: np.array
):
    return np.sqrt(
        (
            surf_error
            * (
                1
                / 2
                * M_sun.value
                / const.parsec ** 2
                / (pathlength * const.parsec)
                * dispersion ** 2
                * 1e7
                * 1.0e-6
            )
        )
        ** 2
        + (
            (0.1 * pathlength * const.parsec)
            * (
                1
                / 2
                * surf
                * M_sun.value
                / const.parsec ** 2
                / (pathlength * const.parsec) ** 2
                * dispersion ** 2
                * 1e7
                * 1.0e-6
            )
        )
        ** 2
        + (
            (0.25 * dispersion)
            * (
                surf
                * M_sun.value
                / const.parsec ** 2
                / (pathlength * const.parsec) ** 2
                * dispersion
                * 1e7
                * 1.0e-6
            )
        )
        ** 2
    )


def get_H2_surface_density(co_intensity: np.array, inclination: int) -> np.array:
    """Calculate the H2 surface density from the CO emission maps from Heracles, this follows Leroy, et al. 2008/2009
    and assumes J: 2 -> 1 emission
    Args:
        co_intensity (np.array): Intensity of emission in [K km/s]
        inclination (int): inclination of the galaxy
    
    Returns:
        np.array: H2 surface density in [M_sun pc²]
    """
    return 5.5 * np.cos(inclination * np.pi / 180) * co_intensity


def get_H2_surface_density_error(co_error: np.array, inclination: int) -> np.array:
    """Calculate the error on the H2 surface density from the CO emission maps from Heracles, this follows Leroy, et al. 2008/2009

    Args:
        co_error (np.array): error on the CO emission in [K km/s]
        inclination (int): inclination of the galaxy

    Returns:
        np.array: H2 surface density error in [M_sun pc²]
    """
    return np.sqrt((co_error * 5.5 * np.cos(inclination * np.pi / 180)) ** 2)


def get_HI_surface_density(
    h1_intensity: np.array, major: float, minor: float, inclination: int
) -> np.array:
    """Calculate the HI surface density from the THINGS maps

    Args:
        h1_intensity (np.array): intensity of the HI emission in [Jy/beam km/s]
        major (float): FWHM of the major axis of the beam
        minor (float): FWHM of the major axis of the beam
        inclination (int): inclination of the galaxy

    Returns:
        np.array: HI surface density in [M_sun pc²]
    """
    return (
        12.06
        * np.cos(inclination * np.pi / 180)
        * (major * minor) ** (-1)
        * h1_intensity
    )


def get_HI_surface_density_error(
    h1_error: np.array, major: float, minor: float, inclination: int
):
    """Calculate the error on the HI surface density from the THINGS maps

    Args:
        h1_error (np.array): error on the HI emission in [Jy/beam km/s]
        major (float): FWHM of the major axis of the beam
        minor (float): FWHM of the major axis of the beam
        inclination (int): inclination of the galaxy

    Returns:
        np.array: HI surface density error in [M_sun pc²]
    """
    return np.sqrt(
        (h1_error * 12.06 * np.cos(inclination * np.pi / 180) * (major * minor) ** (-1))
        ** 2
    )


def get_magnetic_field_energy_density(magnetic_field: np.array) -> np.array:
    """Convert the magnetic field strength into magnetic energy density

    Args:
        magnetic_field (np.array): magnetic field strength in [G]

    Returns:
        np.array: magnetic energy density in [erg/cm³]
    """
    return magnetic_field ** 2 / (8 * np.pi)


def get_magnetic_field_energy_density_error(
    magnetic_field: np.array, magnetic_field_error: np.array
) -> np.array:
    """Convert the magnetic field strength error into the energy density error

    Args:
        magnetic_field (np.array): magnetic field strength in [G]
        magnetic_field_error (np.array): error on the magnetic field strength in [G]

    Returns:
        np.array: error on the magnetic energy density in [erg/cm³]
    """
    return np.sqrt((magnetic_field_error * magnetic_field / (4 * np.pi)) ** 2)


def sfr_error(sfr: np.array, rms: float) -> np.array:
    """Estimated error on the SFR emission (= 5%)

    Args:
        sfr (np.array): the star formation rate from Leroy et al. 2008
        rms (np.array): the rms value of the sfr map

    Returns:
        np.array: the error on the sfr emission
    """
    return np.sqrt((0.05 * sfr) ** 2 + rms ** 2)


def radio_error(flux: np.array, rms: float) -> np.array:
    """Estimated error on the RC emission (= 5%)

    Args:
        flux (np.array): the flux of the LOFAR map
        rms (np.array): the rms value of the LOFAR map

    Returns:
        np.array: the error on the rc emission
    """
    return np.sqrt((0.05 * flux) ** 2 + rms ** 2)


def h1_error(h1: np.array, rms: float) -> np.array:
    """Estimated error on the HI THINGS emission (= 5%)

    Args:
        h1 (np.array): the emission of the THINGS map
        rms (np.array): the rms value of the THINGS map

    Returns:
        np.array: the error on the HI emission
    """
    return np.sqrt((0.05 * h1) ** 2 + rms ** 2)


def co_error(co: np.array, rms: float) -> np.array:
    """Estimated error on the CO HERACLES emission (= 20%)

    Args:
        co (np.array): the emission of the HERACLES map
        rms (np.array): the rms value of the HERACLES map

    Returns:
        np.array: the error on the co emission
    """
    return np.sqrt((0.2 * co) ** 2 + rms ** 2)


def integrated_to_mean(flux: float, ellipse: array) -> float:
    """Convert the integrated flux density of LOFAR to a mean value across the measured ellipse

    Args:
        flux (float): integrated flux density
        ellipse (array): ellipse size in [arcmin, arcmin]

    Returns:
        float: the mean flux density across the ellipse
    """
    major = ellipse[0] * u.arcmin
    minor = ellipse[1] * u.arcmin

    return (flux / (np.pi * major.to(u.rad) * minor.to(u.rad))).value


def beam_in_integration_area(ellipse: array, beamsize: Union[float, int]) -> float:
    """Get the amount of beams that fit into the integration area for estimation of the error

    Args:
        ellipse (array): ellipse size in [arcmin, arcmin]
        beamsize (Union[float, int]): beam size n [arcsec]

    Returns:
        float: the amount of beams
    """
    return (
        (np.pi * ellipse[0] * u.arcmin * ellipse[1] * u.arcmin)
        / (beamsize * u.arcsec).to(u.arcmin)
    ).value


def galaxy_size_in_kpc(ellipse: array, distance: float) -> float:
    """

    Args:
        ellipse (array): ellipse size in [arcmin, arcmin]
        distance (float): distance to the galaxy in [Mpc]

    Returns:
        float: the size of the galaxy in [kpc²]
    """
    major = (
        (ellipse[0] * u.arcmin * distance * u.Mpc)
        .to(u.kpc, u.dimensionless_angles())
        .value
    )
    minor = (
        (ellipse[1] * u.arcmin * distance * u.Mpc)
        .to(u.kpc, u.dimensionless_angles())
        .value
    )
    return np.pi * major * minor

