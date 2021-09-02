import os
import re
import subprocess
import sys
import urllib.error
from typing import Tuple
from astropy.units.quantity import Quantity

import numpy as np
import PIL.Image
import uncertainties as unc
import uncertainties.unumpy as unp
from astropy.convolution import Gaussian2DKernel, convolve_fft
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from astroquery.hips2fits import hips2fits
from astroquery.skyview import SkyView
from numpy.core.records import array
from scipy import stats


def mute():
    """Mute print output from subprocesses in parallel processes"""
    sys.stdout = open(os.devnull, "w")


def get_galaxy_number_from_name(galaxy: str) -> str:
    """ Get the catalogue number from the galaxy name.
    eg. from n5194 get 5194

    Args:
        galaxy (str): Name of the galaxy

    Returns:
        str: Catalogue number as string
    """
    return re.search(r"\d+", galaxy).group()


def get_pretty_name(name: str) -> str:
    """ Get the pretty name of the galaxy.
    eg. from n5194 get NGC 5194

    Args:
        name (str): Name of the galaxy

    Returns:
        str: pretty name
    """
    return "NGC " + re.search(r"\d+", name).group()


def get_magnetic_galaxy_dir(galaxy: str, data_directory: str) -> str:
    """ Get the full path to a specific galaxy directory

    Args:
        galaxy (str): Name of the galaxy
        data_directory (str): dr2 data directory

    Returns:
        str: Path to galaxy directory
    """
    return data_directory + "/magnetic/" + galaxy


def run_command_in_dir(cmd: list, directory: str):
    """ Run a command in a specific dir

    Args:
        cmd (list): Command list to run
        dir (str): Directory
    """
    p = subprocess.Popen(cmd, cwd=directory)
    p.wait()


def query_yes_no(question: str, default: str = "yes") -> bool:
    """Ask a yes/no question via raw_input() and return their answer.

    Args:
        question (str): string that is presented to the user
        default (str, optional): Presumed answer if the user just hits <Enter>.
            It must be "yes", "no" or None (meaning
            an answer is required of the user). Defaults to "yes".

    Raises:
        ValueError: Non valid default answer, can only be "yes" or "no"

    Returns:
        bool: value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == "":
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' " "(or 'y' or 'n').\n")


def get_overlay_image(coord: SkyCoord, size: Quantity, wcs: WCS) -> PIL.Image:
    """Returns an image from the SDSS image query as PIL.Image. This can be used
    to set as RGB background for FITSimages in aplpy or with matplotlib and imshow.

    Args:
        coord (SkyCoord): coordinate object of center of image
        size (Quantity): size of the image in astropy units (degree/arcmin/arcsec/...)
        wcs (WCS): WCS object of the base image

    Returns:
        PIL.Image: rgb image
    """
    image: np.array = None
    try:
        print("Trying SDSS")
        # Check if image is in SDSS
        SkyView.get_images(
            position=f"{coord.ra.degree}, {coord.dec.degree}",
            radius=size,
            survey=["SDSSi"],
        )

        image = PIL.Image.fromarray(
            hips2fits.query_with_wcs("CDS/P/SDSS9/color-alt", wcs=wcs, format="png")
        )
    except urllib.error.HTTPError as e:
        print("Not found in SDSS")

    if image is None:
        try:
            print("Trying DSS")
            # Check if image is in DSS
            SkyView.get_images(
                position=f"{coord.ra.degree}, {coord.dec.degree}",
                radius=size,
                survey=["DSS2 Blue"],
            )

            image = PIL.Image.fromarray(
                hips2fits.query_with_wcs("CDS/P/DSS2/color", wcs=wcs, format="png")
            )
        except urllib.error.HTTPError as e:
            print("Not found in DSS, abort", e)
    return image


def powerlaw(p, x):
    """Power law function for ODR fit

    Args:
        p (array): [a, b]
        x (np.array): x

    Returns:
        np.array
    """
    a, b = p
    return a * np.power(x, b)


def confindence_band(
    x: np.array,
    xd: np.array,
    yd: np.array,
    p: array,
    sigma: np.array,
    conf: float = 0.95,
) -> Tuple[np.array, np.array]:
    """Calculate a confidence intervall for a ODR fit

    Args:
        x (np.array): the data points, which are used for drawing the plot
        xd (np.array): measured data points in x
        yd (np.array): measured data points in y
        p (array): best fit parameters according to ODR
        sigma (np.array): the covariance matrix according to ODR
        conf (float, optional): the confidence level. Defaults to 0.95.

    Returns:
        Tuple[np.array, np.array]: the lower and upper confidence band 
    """
    # calculate parameter confidence interval
    amp, index = unc.correlated_values(p, sigma)

    # calculate regression confidence interval
    py = amp * unp.pow(x, index)
    nom = unp.nominal_values(py)
    std = unp.std_devs(py)

    alpha = 1.0 - conf  # significance
    N = xd.size  # data sample size
    var_n = len(p)  # number of parameters
    # Quantile of Student's t distribution for p=(1-alpha/2)
    q = stats.t.ppf(1.0 - alpha / 2.0, N - var_n - 1)

    # Predicted values (best-fit model)
    yp = powerlaw(p, x)
    # Confidence band
    dy = q * std
    # Upper & lower confidence bands.
    lpb, upb = yp - dy, yp + dy
    return lpb, upb


# Taken from https://github.com/sebastian-schulz/radio-pixel-plots/
def convert_kpc2px(kpc, distance, pixel_per_arcsec):
    # distance is in Mpc, and needs to be convertet to parsec
    kpc_per_arcsec = distance * 1e6 * np.tan(2 * np.pi / (360.0 * 3600.0)) / 1000.0
    return float(kpc * pixel_per_arcsec / kpc_per_arcsec)


def smooth(
    path: str, length: float, distance: float, pix_per_as: float, output_path: str
):
    """Implementation of the smoothing experiment

    Args:
        path (str): the path to the base fits file
        length (float): the diffusion length in [kpc]
        distance (float): the distance to the galaxy in [Mpc]
        pix_per_as (float): the amount of pixel per arcsec
        output_path (str): the path for the smoothed output file
    """
    hdu = fits.open(path)
    data = hdu[0].data
    header = hdu[0].header

    # Slice the Frequency and Stokes axis
    try:
        data = data[0, 0, :, :]
    except IndexError:
        data = data

    sigma = convert_kpc2px(length, distance, pix_per_as) / 1.177

    kernel = Gaussian2DKernel(sigma)
    conv_data = convolve_fft(data, kernel)

    out_hdu = fits.PrimaryHDU(conv_data, header=header)
    fits.HDUList(out_hdu).writeto(output_path, overwrite=True)
    print(f"Written {output_path}")
