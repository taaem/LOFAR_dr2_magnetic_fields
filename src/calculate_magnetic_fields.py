import multiprocessing as mp
from typing import List

import numpy as np
import pyregion
import yaml
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.wcs import WCS

import src.helper as helper
import src.matplotlib_helper as plt_helper
from src.math_functions import (b_field_revised, b_field_revised_error,
                                beam_in_integration_area, beam_size,
                                integrated_to_mean)


def calculate_all_magnetic_field(config: dict):
    """Loop through all galaxies in config and calculate magnetic field strengths

    Args:
        config (dict): Config
    """
    if config["threads"] > 1:
        print("Using parallel processing, output will be supressed...")
    pool = mp.Pool(
        config["threads"], initializer=helper.mute if config["threads"] > 1 else None
    )
    for galaxy in config["galaxies"]:
        print("------- Starting", galaxy["name"], "-------")
        pool.apply_async(
            calculate_magnetic_field,
            args=(galaxy, config),
            callback=lambda name: print("------- Finished", name, "-------"),
        )

    pool.close()
    pool.join()


def calculate_magnetic_field(galaxy_config: dict, config: dict):
    """Calculating different magnetic field maps for a specific galaxy,

    Args:
        galaxy_config (dict): Galaxy Config
        config (dict): Config
    """
    # If generate_diff calculate the base map (eg. no extra options) first
    if galaxy_config["use_thermal"]:
        __calculate_magnetic_field(
            name=galaxy_config["name"],
            data_directory=config["data_directory"],
            alpha_min=config["alpha_min"],
            alpha_max=config["alpha_max"],
            frequency=config["frequency"],
            K_0=config["proton_electron_ratio"],
            pathlength=galaxy_config["pathlength"],
            thermal=False,
            rms=galaxy_config["rms_6"],
            size=galaxy_config["size"],
            inclination=galaxy_config["inclination"],
            spix_integrated=galaxy_config["spix"],
            spix_integrated_error=galaxy_config["spix_error"],
            radio_integrated=galaxy_config["radio_integrated"],
            ellipse=galaxy_config["region_ellipse"],
            levels=galaxy_config["magnetic_levels"],
        use_integrated_spix=galaxy_config["use_integrated"],
        )

    # Calculate the specified map
    __calculate_magnetic_field(
        name=galaxy_config["name"],
        data_directory=config["data_directory"],
        alpha_min=config["alpha_min"],
        alpha_max=config["alpha_max"],
        frequency=config["frequency"],
        K_0=config["proton_electron_ratio"],
        pathlength=galaxy_config["pathlength"],
        thermal=galaxy_config["use_thermal"],
        rms=galaxy_config["rms_6"],
        size=galaxy_config["size"],
        inclination=galaxy_config["inclination"],
        use_integrated_spix=galaxy_config["use_integrated"],
        spix_integrated=galaxy_config["spix"],
        spix_integrated_error=galaxy_config["spix_error"],
        radio_integrated=galaxy_config["radio_integrated"],
        ellipse=galaxy_config["region_ellipse"],
        levels=galaxy_config["magnetic_levels"],
    )
    # If generate_diff calculate the diff to the base map
    if galaxy_config["use_thermal"]:
        calculate_map_difference(
            name=galaxy_config["name"],
            data_directory=config["data_directory"],
            thermal=galaxy_config["use_thermal"],
        )

    return galaxy_config["name"]


def __calculate_magnetic_field(
    name: str,
    data_directory: str,
    alpha_min: float,
    alpha_max: float,
    frequency: float,
    K_0: int,
    pathlength: float,
    thermal: bool,
    rms: float,
    size: float,
    inclination: int,
    use_integrated_spix: bool,
    spix_integrated: float,
    spix_integrated_error: float,
    radio_integrated: float,
    ellipse: List[float],
    levels: List[float],
):
    """Calculate the magnetic field strength based on the revised equipartition formula by Beck & Krause 2005
    Generate plots for usage in paper or thesis. Also save the corresponding fits files for everything so the
    plots can easily be reconstructed.

    Args:
        name (str): Name of the galaxy
        data_directory (str): dr2 data directory
        alpha_min (float, optional): Minimum spectral index value, lower will be clipped. Default: -0.6.
        alpha_max (float, optional): Maximum spectral index value, higher will be clipped. Default: -1.1.
        frequency (float, optional): Frequency of LOFAR data in [Hz]. Default: 144e6.
        K_0 (int, optional): Proton to electron ratio. Defaults: 100.
        pathlength (float, optional): Line of sight through galaxy in [kpc]. Default: 1e3.
        thermal (bool, optional): Switch whether to calculate non thermal maps or not: Default is False.
        rms (float): rms value of the 6" LOFAR map
        size (float): size of the cutout in [arcmin]
        inclination (int): inclination of the galaxy
        use_integrated_spix (bool): use the integrated spectral index instead of the spectral index map
        spix_integrated (float): integrated spectral index
        spix_integrated_error (float): error on the integrated spectral index
        radio_integrated (float): integrated flux density of the 6" LOFAR map
        ellipse (List[float]): major and minor axis in [arcmin, arcmin]
        levels (List[float]): magnetic field contour levels
    """
    print(f"Calculating magnetic fields for {name} with thermal: {thermal}")

    if use_integrated_spix:
        print(f"Using integrated spectral index: {spix_integrated: 4.2f}")
    plt_helper.setup_matploblib(magnetic=True)

    galaxy_number = helper.get_galaxy_number_from_name(name)

    # Setup paths
    region_path = (
        f"{data_directory}/magnetic/{name}/n{galaxy_number}_flux_elliptical_high.reg"
    )
    output_path = get_path_to_magnetic_map(name, data_directory, thermal, "")
    flux_path = f"{data_directory}/magnetic/{name}/n{galaxy_number}_144mhz_{'non_thermal_' if thermal else ''}6as.fits"
    flux_map = fits.open(flux_path)

    spix = spix_integrated
    spix_error = spix_integrated_error
    if not use_integrated_spix:
        spix_path = get_path_to_spix(name, data_directory, thermal, False, f"_6as.fits")
        spix_error_path = get_path_to_spix(
            name, data_directory, thermal, True, f"_6as.fits"
        )
        # Read data
        spix_map = fits.open(spix_path)
        spix_error_map = fits.open(spix_error_path)

        spix = spix_map[0].data
        spix_error = spix_error_map[0].data

    # Slice the Frequency and Stokes axis
    try:
        flux = flux_map[0].data[0, 0, :, :]
    except IndexError:
        flux = flux_map[0].data
    # Get the WCS from the intensity map
    wcs = WCS(flux_map[0].header).celestial

    magnetic_field = b_field_revised(
        alpha=spix,
        beam=beam_size(flux_map[0].header["BMAJ"] * 3600),
        I_nu=flux,
        nu=frequency,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        K_0=K_0,
        pathlength=pathlength,
        inclination=inclination,
    )

    magnetic_field_error = b_field_revised_error(
        alpha=spix,
        alpha_error=spix_error,
        I_nu=flux,
        I_nu_error=np.sqrt((0.05 * flux) ** 2 + rms ** 2),
        beam=beam_size(flux_map[0].header["BMAJ"] * 3600),
        nu=frequency,
        K_0=K_0,
        K_0_error=0.5 * K_0,
        pathlength=pathlength,
        pathlength_error=0.5 * pathlength,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        inclination=inclination,
    )

    coord = SkyCoord.from_name(helper.get_pretty_name(name))
    # Cutout central region
    cutout = Cutout2D(magnetic_field, coord, size * u.arcmin, wcs=wcs,)
    cutout_error = Cutout2D(magnetic_field_error, coord, size * u.arcmin, wcs=wcs,)
    wcs = cutout.wcs
    magnetic_field = cutout.data
    magnetic_field_error = cutout_error.data

    # Generate new fits
    output_hdu = fits.PrimaryHDU(magnetic_field)
    output_hdu.header.update(wcs.to_header())
    output_hdul = fits.HDUList(output_hdu)

    # Load region
    region = pyregion.read_region_as_imagecoord(
        open(region_path).read(), output_hdu.header
    )
    region_mask = pyregion.get_mask(region, output_hdul[0])

    # Convert integrated flux to a mean flux over the area of the galaxy
    radio_integrated_mean = integrated_to_mean(radio_integrated, ellipse)
    radio_integrated_mean_error = integrated_to_mean(
        np.sqrt(
            (0.1 * radio_integrated) ** 2
            + (rms * np.sqrt(beam_in_integration_area(ellipse, 6))) ** 2
        ),
        ellipse,
    )

    magnetic_field_mean = b_field_revised(
        alpha=spix_integrated,
        beam=1,  # Integrated values don't need beam conversion
        I_nu=radio_integrated_mean,
        nu=frequency,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        K_0=K_0,
        pathlength=pathlength,
        inclination=inclination,
    )

    magnetic_field_mean_error = b_field_revised_error(
        alpha=spix_integrated,
        alpha_error=spix_integrated_error,
        I_nu=radio_integrated_mean,
        I_nu_error=radio_integrated_mean_error,
        beam=1,  # Integrated values don't need beam conversion
        nu=frequency,
        K_0=K_0,
        K_0_error=0.5 * K_0,
        pathlength=pathlength,
        pathlength_error=0.5 * pathlength,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        inclination=inclination,
    )

    print("Mean B-Field: {:.8} μG".format(magnetic_field_mean * 1e6))
    print("Mean B-Field Error: {:.8} μG".format(magnetic_field_mean_error * 1e6))
    print("Max B-Field: {:.8} μG".format(np.nanmax(magnetic_field[region_mask]) * 1e6))

    # Copy output to a dictionary to save as .yml
    result_dict = {
        "mean": {
            "value": float(magnetic_field_mean),
            "std": float(magnetic_field_mean_error),
        },
        "surface_mean": {
            "value": float(np.nanmean(magnetic_field[region_mask])),
            "std": float(np.nanstd(magnetic_field[region_mask])),
        },
        "mean_error": {
            "value": float(np.nanmean(magnetic_field_error[region_mask])),
            "std": float(np.nanstd(magnetic_field_error[region_mask])),
        },
        "max": {
            "value": float(np.nanmax(magnetic_field[region_mask])),
            "std": float(
                magnetic_field_error[
                    np.where(magnetic_field == np.nanmax(magnetic_field[region_mask]))
                ]
            ),
        },
    }
    with open(output_path + ".yml", "w") as file:
        yaml.dump(result_dict, file)
        print("Out:", output_path + ".yml")

    plt_helper.plot_magnetic(
        val=magnetic_field,
        wcs=wcs,
        output_path=output_path,
        region=region,
        vmax=float(np.nanmax(magnetic_field[region_mask])),
        inline_title="NGC " + name[1:],
    )

    width = magnetic_field.shape[0]
    height = magnetic_field.shape[1]

    image = helper.get_overlay_image(
        coord, size=size * u.arcmin, wcs=wcs
    )
    plt_helper.plot_magnetic_overlay(
        base=image,
        overlay=magnetic_field * 1e6,
        output_path=output_path + "_overlay",
        wcs=wcs,
        levels=levels,
        inline_title="NGC " + name[1:],
    )

    ###### abs. Error image ######
    plt_helper.setup_matploblib(False)

    plt_helper.plot_magnetic(
        val=magnetic_field_error,
        wcs=wcs,
        output_path=output_path + "_abs_error",
        region=None,
        vmax=np.nanmean(magnetic_field_error[region_mask]) * 3,
        label=r"$\sigma_{B_{\mathrm{eq}}}$ [\si{\micro G}]",
        inline_title="NGC " + name[1:],
    )

    ###### rel Error image ######
    plt_helper.setup_matploblib(False)

    plt_helper.plot_magnetic(
        val=magnetic_field_error / magnetic_field,
        wcs=wcs,
        output_path=output_path + "_rel_error",
        region=region,
        vmax=1,
        label=r"$\sigma_{B_{\mathrm{eq}}}/B_{\mathrm{eq}}$",
        abs_val=False,
        inline_title="NGC " + name[1:],
    )


def calculate_map_difference(name: str, data_directory: str, thermal: bool):
    """Calculate the difference maps for a galaxy

    Args:
        name (str): Name of the galaxy
        data_directory (str): dr2 data directory
        thermal (bool): Switch whether to compare non_thermal map.
    """
    plt_helper.setup_matploblib(magnetic=False)
    base_path = get_path_to_magnetic_map(name, data_directory, False, ".fits")
    compare_path = get_path_to_magnetic_map(name, data_directory, thermal, ".fits")
    output_path = get_path_to_magnetic_map(name, data_directory, thermal, "_diff")

    base_map = fits.open(base_path)
    compare_map = fits.open(compare_path)

    base_field = base_map[0].data
    compare_field = compare_map[0].data

    # Ignore Stokes and frequency axis if provided
    if base_field.shape[1] == 4:
        base_field = base_field[0, 0, :, :]
    if compare_field.shape[1] == 4:
        compare_field = compare_field[0, 0, :, :]

    diff = base_field - compare_field

    plt_helper.plot_magnetic(
        diff,
        wcs=WCS(base_map[0].header).celestial,
        output_path=output_path,
        region=None,
        label=r"$\Delta B$ [\si{\micro G}]",
        vmin=None,
        vmax=None,
        inline_title="NGC " + name[1:],
    )


def get_path_to_magnetic_map(
    name: str, data_directory: str, thermal: bool, file_ending: str = ".fits",
) -> str:
    """Get the path to the magnetic map output, with the specified file ending,
    for output paths the file_ending should be ""

    Args:
        name (str): Name of the galaxy
        data_directory (str): dr2 data directory
        thermal (bool): non thermal data
        file_ending (str, optional): File ending. Defaults to ".fits".

    Returns:
        str: [description]
    """
    return f"{data_directory}/magnetic/{name}/{name}_magnetic{'_non_thermal' if thermal else ''}{file_ending}"


def get_path_to_spix(
    name: str,
    data_directory: str,
    thermal: bool,
    error: bool = False,
    file_ending: str = "_6as.fits",
) -> str:
    """Get the path to the spectral index

    Args:
        name (str): Name of the galaxy
        data_directory (str): dr2 data directory
        thermal (bool): non thermal data
        error (bool): path to error
        file_ending (str, optional): File ending. Defaults to ".fits".
    Returns:
        str: [description]
    """
    return f"{data_directory}/magnetic/{name}/{name}_spix{'_non_thermal' if thermal else ''}{'_error' if error else ''}{file_ending}"

