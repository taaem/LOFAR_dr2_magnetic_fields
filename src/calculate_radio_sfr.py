import multiprocessing as mp
from pathlib import Path

import astropy.units as u
import numpy as np
import pyregion
import yaml
from astropy.io import fits
from astropy.wcs import WCS
from numpy.core.defchararray import array
from scipy.stats import sem

import src.calculate_magnetic_fields
import src.helper as helper
import src.matplotlib_helper as plt_helper
from src import math_functions
from src.exceptions import NotConfiguredException

sfr_label = r"$\Sigma_{\mathrm{SFR}}$ [\si{M_{\odot}.kpc^{-2}.yr^{-1}}]"
sfr_mean_label = r"$\mathrm{SFR}$ [\si{M_{\odot}.yr^{-1}}]"
sfr_mean_unit = r"\si{M_{\odot}.yr^{-1}}"
sfr_sign = r"\Sigma_{\mathrm{SFR}}"
sfr_mean_sign = r"\mathrm{SFR}"
sfr_unit = r"\si{M_{\odot}.kpc^{-2}.yr^{-1}}"

flux_label = r"$I'_{\SI{144}{\mega\hertz}}$ [\si{Jy.beam\tothe{-1}}]"
flux_mean_label = r"$L_{\SI{144}{\mega\hertz}}$ [\si{\watt\per\hertz}]"
flux_mean_unit = r"\si{\watt\per\hertz}"
flux_sign = r"$I'_{\SI{144}{\mega\hertz}}$"
flux_unit = r"\si{Jy.beam\tothe{-1}}"


def calculate_all_radio_sfr(config: dict, skip: bool = False):
    """Calculate star formation rate correlations for all available galaxies

    Args:
        config (dict): Config
    """
    if not skip:
        if config["threads"] > 1:
            print("Using parallel processing, output will be supressed...")
        pool = mp.Pool(
            config["threads"],
            initializer=helper.mute if config["threads"] > 1 else None,
        )
        for galaxy in config["galaxies"]:
            try:
                print("------- Starting", galaxy["name"], "-------")
                pool.apply_async(
                    calculate_radio_sfr,
                    args=(galaxy["name"], config),
                    callback=lambda name: print("------- Finished", name, "-------"),
                )
            except NotConfiguredException:
                print(f"Skipping galaxy {galaxy['name']}, not configured...")

        pool.close()
        pool.join()
    else:
        print(
            "Skipping calculation for galaxies, only combined output will be calculated..."
        )

    plt_helper.setup_matploblib(False)
    holder = {
        "x": np.array([]),
        "x_mean": np.array([]),
        "x_smooth": np.array([]),
        "x_smooth_error": np.array([]),
        "x_std": np.array([]),
        "x_error": np.array([]),
        "y": np.array([]),
        "y_error": np.array([]),
        "y_smooth": np.array([]),
        "y_smooth_error": np.array([]),
        "y_mean": np.array([]),
        "y_std": np.array([]),
        "z": np.array([]),
        "z_smooth": np.array([]),
        "name": [],
    }
    for galaxy in config["galaxies"]:
        if not galaxy["calc_sfr"] or galaxy["skip_combined_radio_sfr"]:
            continue
        holder["name"].append(galaxy["name"])

        # Read Energy density
        path = (
            get_path_to_radio_sfr_dir(galaxy["name"], config["data_directory"])
            + f"/{galaxy['name']}_sfr_rebin_13_5as.fits"
        )
        g_sfr = fits.getdata(path)
        holder["x"] = np.concatenate((holder["x"], g_sfr.flatten()))
        holder["x_error"] = np.concatenate(
            (
                holder["x_error"],
                math_functions.sfr_error(g_sfr, galaxy["sfr"]["rms"]).flatten(),
            )
        )

        holder["x_mean"] = np.append(holder["x_mean"], galaxy["sfr"]["mean"])
        holder["x_std"] = np.append(holder["x_std"], 0.1 * galaxy["sfr"]["mean"])

        if galaxy["smooth_exp"]:
            g_sfr_smooth = fits.getdata(
                get_path_to_radio_sfr_dir(galaxy["name"], config["data_directory"])
                + f"/{galaxy['name']}_sfr_rebin_13_5as_smooth.fits"
            )
            holder["x_smooth"] = np.append(holder["x_smooth"], g_sfr_smooth)
            holder["x_smooth_error"] = np.append(
                holder["x_smooth_error"],
                math_functions.sfr_error(g_sfr_smooth, galaxy["sfr"]["rms"]).flatten(),
            )

        # Read magnetic field
        r_path = f"{get_path_to_radio_sfr_dir(galaxy['name'], config['data_directory'])}/{galaxy['name']}_144mhz_6as_rebin_13_5as.fits"
        r_flux = fits.getdata(r_path)

        # Slice the Frequency and Stokes axis
        try:
            r_flux = r_flux[0, 0, :, :]
        except IndexError:
            r_flux = r_flux

        holder["y"] = np.concatenate((holder["y"], r_flux.flatten()))
        holder["y_error"] = np.concatenate(
            (
                holder["y_error"],
                math_functions.radio_error(
                    r_flux, galaxy["rms_6"]
                ).flatten(),
            )
        )

        radio_mean = (
            galaxy["radio_integrated"]
            * 1e-26
            * 4
            * np.pi
            * (galaxy["distance"] * u.Mpc).to(u.m).value ** 2
        )
        radio_mean_std = (
            np.sqrt((0.1 * galaxy["radio_integrated"]) ** 2 + galaxy["rms_6"] ** 2)
            * 1e-26
            * 4
            * np.pi
            * (galaxy["distance"] * u.Mpc).to(u.m).value ** 2
        )
        holder["y_mean"] = np.append(holder["y_mean"], radio_mean)
        holder["y_std"] = np.append(holder["y_std"], radio_mean_std)

        if galaxy["smooth_exp"]:
            holder["y_smooth"] = np.concatenate((holder["y_smooth"], r_flux.flatten()))
            holder["y_smooth_error"] = np.concatenate(
                (
                    holder["y_smooth_error"],
                    math_functions.radio_error(
                        r_flux, galaxy["rms_6"]
                    ).flatten(),
                )
            )

        g_spix = None
        if galaxy["use_integrated"]:
            g_spix = np.full(r_flux.shape, galaxy["spix"])
        else:
            # Read spectral index
            s_path = src.calculate_magnetic_fields.get_path_to_spix(
                galaxy["name"],
                config["data_directory"],
                galaxy["use_thermal"],
                file_ending="_rebin_13_5as.fits",
            )
            g_spix = fits.getdata(s_path)
        holder["z"] = np.concatenate((holder["z"], g_spix.flatten()))
        if galaxy["smooth_exp"]:
            holder["z_smooth"] = np.concatenate((holder["z_smooth"], g_spix.flatten()))

    # Calculate combined plot
    plt_helper.plot_pixel_power_law(
        x=holder["x"],
        y=holder["y"],
        z=holder["z"],
        x_error=holder["x_error"],
        y_error=holder["y_error"],
        xlabel=sfr_label,
        output_path=config["data_directory"] + "/radio_sfr_combined",
        region_mask=None,
        p0=[1.86e-3 * 6 ** (2), 1],
        x_value=sfr_sign,
        x_unit=sfr_unit,
        density_map=False,
        extra_line_params=[1.86e-3 * 6 ** (2), 1],
        fit_extra_line=False,
        extra_line_label=rf"Condon",
        ylabel=flux_label,
        y_unit=flux_unit,
    )
    plt_helper.plot_pixel_power_law(
        x=holder["x_smooth"],
        y=holder["y_smooth"],
        z=holder["z_smooth"],
        x_error=holder["x_smooth_error"],
        y_error=holder["y_smooth_error"],
        xlabel=sfr_label,
        output_path=config["data_directory"] + "/radio_sfr_combined_smooth",
        region_mask=None,
        p0=[1.86e-3 * 6 ** (2), 1],
        x_value=sfr_sign,
        x_unit=sfr_unit,
        density_map=False,
        extra_line_params=[1.86e-3 * 6 ** (2), 1],
        fit_extra_line=False,
        extra_line_label=rf"Condon",
        ylabel=flux_label,
        y_unit=flux_unit,
    )

    plt_helper.plot_pixel_mean_power_law(
        x=holder["x_mean"],
        y=holder["y_mean"],
        x_std=holder["x_std"],
        y_std=holder["y_std"],
        xlabel=sfr_mean_label,
        output_path=config["data_directory"] + "/radio_sfr_combined_mean",
        p0=[1.33e21, 1],
        x_value=sfr_mean_sign,
        x_unit=sfr_mean_unit,
        extra_line_params=[5.16e21, 1],
        fit_extra_line=False,
        extra_line_label=rf"Condon",
        ylabel=flux_mean_label,
        y_unit=flux_mean_unit,
        no_mean=True,
    )


def calculate_radio_sfr(name: str, config: dict, fig=None):
    # "Check" if the specified galaxy exists
    galaxy_config = next(filter(lambda g: g["name"] == name, config["galaxies"],))
    try:
        if not galaxy_config["calc_sfr"]:
            raise NotConfiguredException()
        # calculate sfr stuff for one galaxy
        __calculate_radio_sfr(
            name=galaxy_config["name"],
            data_directory=config["data_directory"],
            thermal=galaxy_config["use_thermal"],
            p0=galaxy_config["sfr"]["p0"],
            levels=galaxy_config["radio_levels"],
            use_integrated_spix=galaxy_config["use_integrated"],
            spix_integrated=galaxy_config["spix"],
            vmin=galaxy_config["sfr"]["vmin"],
            vmax=galaxy_config["sfr"]["vmax"],
            sfr_rms=galaxy_config["sfr"]["rms"],
            flux_rms=galaxy_config["rms_6"],
            inclination=galaxy_config["inclination"],
            smooth_exp=galaxy_config["smooth_exp"],
        )
    except NotConfiguredException:
        print("Galaxy not configured for SFR...")
    return name


def __calculate_radio_sfr(
    name: str,
    data_directory: str,
    thermal: bool,
    p0: list,
    levels: array,
    use_integrated_spix: bool,
    spix_integrated: float,
    vmin: float,
    vmax: float,
    sfr_rms: float,
    flux_rms: float,
    inclination: int,
    smooth_exp: bool
) -> str:
    """Calculate and plot correlation between radio contnuum emission and star formation rate
    for one galaxy

    Args:
        name (str): Name of galaxy
        data_directory (str): dr2 data directory
        thermal (bool): Use non thermal magnetic field
        p0 (list): inital guess for the fit
        levels (array): contour levels for the radio emission
        use_integrated_spix (bool): use the integrated spectral index instead of the spectral index map
        spix_integrated (float): integrated spectral index
        vmin (float): minimum value of the color scale of the overlay
        vmax (float): maximum value of the color scale of the overlay
        sfr_rms (float): rms value for the sfr map
        flux_rms (float): rms value for the LOFAR 6" map
        inclination (int): inclination of the galaxy
        smooth_exp (bool): perform the smoothing experiment

    Returns:
        str: name of the galaxy
    """
    plt_helper.setup_matploblib(False)

    print(
        f"Calculating correlations between Radio and SFR for galaxy: {name} with thermal: {thermal}"
    )

    radio_path = (
        f"{get_path_to_radio_sfr_dir(name, data_directory)}/{name}_144mhz_6as.fits"
    )
    radio_rebin_path = f"{get_path_to_radio_sfr_dir(name, data_directory)}/{name}_144mhz_6as_rebin_13_5as.fits"
    sfr_path = f"{get_path_to_radio_sfr_dir(name, data_directory)}/{name}_sfr_6as.fits"
    sfr_rebin_path = f"{get_path_to_radio_sfr_dir(name, data_directory)}/{name}_sfr_rebin_13_5as.fits"

    output_path = f"{data_directory}/radio_sfr/{name}/"
    # make sure that the output_dir exists
    Path(output_path).mkdir(parents=True, exist_ok=True)
    output_path += f"{name}_radio_sfr"

    flux_map = fits.open(radio_path)
    flux_rebin_map = fits.open(radio_rebin_path)
    sfr_map = fits.open(sfr_path)
    sfr_rebin_map = fits.open(sfr_rebin_path)

    flux = flux_map[0].data
    flux_rebin = flux_rebin_map[0].data
    sfr = sfr_map[0].data
    sfr_rebin = sfr_rebin_map[0].data

    # Slice the Frequency and Stokes axis
    try:
        flux = flux_map[0].data[0, 0, :, :]
    except IndexError:
        flux = flux_map[0].data

    # Inclination correct the flux
    flux = np.cos(np.pi / 180 * inclination) * flux
    flux_rebin = np.cos(np.pi / 180 * inclination) * flux_rebin

    sfr_error = math_functions.sfr_error(sfr_rebin, sfr_rms)
    flux_error = math_functions.radio_error(flux_rebin, flux_rms)

    sfr_smooth = None
    sfr_smooth_error = None
    if smooth_exp:
        sfr_smooth = fits.getdata(
            f"{get_path_to_radio_sfr_dir(name, data_directory)}/{name}_sfr_rebin_13_5as_smooth.fits"
        )
        sfr_smooth_error = math_functions.sfr_error(sfr_smooth, sfr_rms)

    spix = np.full(flux_rebin.shape, spix_integrated)
    if not use_integrated_spix:
        spix_path = src.calculate_magnetic_fields.get_path_to_spix(
            name, data_directory, thermal, file_ending="_rebin_13_5as.fits"
        )
        spix_map = fits.open(spix_path)
        spix = spix_map[0].data

    print("Generating overlay plot...")
    plt_helper.plot_overlay(
        base=sfr,
        overlay=flux,
        base_label=sfr_label,
        wcs=WCS(sfr_map[0].header),
        output_path=output_path + "_overlay",
        vmin=vmin,
        vmax=vmax,
        levels=levels,
        inline_title="NGC " + name[1:],
        overlay_label=flux_label,
    )
    print("Generating pixel plot and power law fit...")
    plt_helper.plot_pixel_power_law(
        x=sfr_rebin.flatten(),
        x_error=sfr_error.flatten(),
        y=flux_rebin.flatten(),
        y_error=flux_error.flatten(),
        z=spix.flatten(),
        xlabel=sfr_label,
        output_path=output_path + "_pixel",
        p0=p0,
        x_value=sfr_sign,
        x_unit=sfr_unit,
        extra_line_params=[1.86e-3 * 6 ** (2), 1],
        fit_extra_line=False,
        extra_line_label=rf"Condon",
        inline_title="NGC " + name[1:],
        ylabel=flux_label,
        y_unit=flux_unit,
    )

    if smooth_exp:
        print("Generating smoothed pixel plot and power law fit...")
        plt_helper.plot_pixel_power_law(
            x=sfr_smooth.flatten(),
            x_error=sfr_smooth_error.flatten(),
            y=flux_rebin.flatten(),
            y_error=flux_error.flatten(),
            z=spix.flatten(),
            xlabel=sfr_label,
            output_path=output_path + "_pixel_smooth",
            p0=p0,
            x_value=sfr_sign,
            x_unit=sfr_unit,
            extra_line_params=[1.86e-3 * 6 ** (2), 1],
            fit_extra_line=False,
            extra_line_label=rf"Condon",
            inline_title="NGC " + name[1:],
            ylabel=flux_label,
            y_unit=flux_unit,
        )
    return name


def get_path_to_radio_sfr_dir(name: str, data_directory: str) -> str:
    """Get the path to the directory where the star formation data should be stored

    Args:
        name (str): Name of the galaxy
        data_directory (str): dr2 data directory

    Returns:
        str: Path to SFR dir
    """
    return f"{data_directory}/radio_sfr/{name}"
