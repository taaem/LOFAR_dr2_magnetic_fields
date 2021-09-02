import multiprocessing as mp
from pathlib import Path

import numpy as np
import pyregion
import yaml
from astropy.io import fits
from astropy.wcs import WCS
from numpy.core.defchararray import array
from scipy.stats import sem

import src.calculate_magnetic_fields
import src.helper as helper
import src.math_functions as math_functions
import src.matplotlib_helper as plt_helper
from src.exceptions import NotConfiguredException

h1_label = (
    r"$\Sigma_{\mathrm{H\MakeUppercase{\romannumeral 1}}}$ [\si{M_{\odot}.pc^{-2}}]"
)
h1_mean_label = r"$\langle\Sigma_{\mathrm{H\MakeUppercase{\romannumeral 1}}}\rangle$ [\si{M_{\odot}.kpc^{-2}}]"
h1_sign = r"\Sigma_{\mathrm{H\MakeUppercase{\romannumeral 1}}}"
h1_unit = r"\si{M_{\odot}.pc^{-2}}"
h1_mean_unit = r"\si{M_{\odot}.kpc^{-2}}"


def calculate_all_h1(config: dict, skip: bool):
    """Calculate H1 surface density correlations for all available galaxies

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
            print("------- Starting", galaxy["name"], "-------")
            pool.apply_async(
                calculate_h1,
                args=(galaxy["name"], config),
                callback=lambda name: print("------- Finished", name, "-------"),
            )

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
        if not galaxy["calc_h1"] or galaxy["skip_combined_h1"]:
            continue
        holder["name"].append(galaxy["name"])

        # Read H1
        path = (
            get_path_to_h1_dir(galaxy["name"], config["data_directory"])
            + f"/{galaxy['name']}_h1_rebin_6as.fits"
        )
        g_h1 = fits.getdata(path)

        g_h1 = math_functions.get_HI_surface_density(
            g_h1,
            fits.getheader(path)["BMAJ"] * 3600,
            fits.getheader(path)["BMIN"] * 3600,
            galaxy["inclination"],
        )
        g_h1_error = math_functions.get_HI_surface_density_error(
            math_functions.h1_error(g_h1, galaxy["h1"]["rms"]),
            fits.getheader(path)["BMAJ"] * 3600,
            fits.getheader(path)["BMIN"] * 3600,
            galaxy["inclination"],
        )
        holder["x"] = np.concatenate((holder["x"], g_h1.flatten()))
        holder["x_error"] = np.concatenate((holder["x_error"], g_h1_error.flatten()))

        holder["x_mean"] = np.append(holder["x_mean"], np.nanmean(g_h1))
        holder["x_std"] = np.append(
            holder["x_std"], sem(g_h1, axis=None, nan_policy="omit")
        )

        if galaxy["smooth_exp"]:
            g_h1_smooth = fits.getdata(
                get_path_to_h1_dir(galaxy["name"], config["data_directory"])
                + f"/{galaxy['name']}_h1_rebin_6as_smooth.fits"
            )
            holder["x_smooth"] = np.append(holder["x_smooth"], g_h1_smooth)
            holder["x_smooth_error"] = np.append(
                holder["x_smooth_error"],
                math_functions.get_HI_surface_density_error(
                    math_functions.h1_error(g_h1, galaxy["h1"]["rms"]),
                    fits.getheader(path)["BMAJ"] * 3600,
                    fits.getheader(path)["BMIN"] * 3600,
                    galaxy["inclination"],
                ).flatten(),
            )

        # Read magnetic field
        m_path = src.calculate_magnetic_fields.get_path_to_magnetic_map(
            galaxy["name"],
            config["data_directory"],
            galaxy["use_thermal"],
            f"_rebin_6as.fits",
        )
        g_magnetic_field = fits.getdata(m_path) * 1e6
        holder["y"] = np.concatenate((holder["y"], g_magnetic_field.flatten()))

        g_m_error_path = src.calculate_magnetic_fields.get_path_to_magnetic_map(
            galaxy["name"],
            config["data_directory"],
            galaxy["use_thermal"],
            f"_abs_error_rebin_6as.fits",
        )
        g_m_error = fits.getdata(g_m_error_path) * 1e6
        holder["y_error"] = np.concatenate((holder["y_error"], g_m_error.flatten()))

        magnetic_results = yaml.full_load(
            open(
                src.calculate_magnetic_fields.get_path_to_magnetic_map(
                    galaxy["name"],
                    config["data_directory"],
                    galaxy["use_thermal"],
                    f".yml",
                )
            )
        )
        holder["y_mean"] = np.append(
            holder["y_mean"], magnetic_results["mean"]["value"] * 1e6
        )
        holder["y_std"] = np.append(
            holder["y_std"], magnetic_results["mean"]["std"] * 1e6
        )

        if galaxy["smooth_exp"]:
            holder["y_smooth"] = np.concatenate(
                (holder["y_smooth"], g_magnetic_field.flatten())
            )
            holder["y_smooth_error"] = np.concatenate(
                (holder["y_smooth_error"], g_m_error.flatten())
            )

        g_spix = None
        if galaxy["use_integrated"]:
            g_spix = np.full(g_magnetic_field.shape, galaxy["spix"])
        else:
            # Read spectral index
            s_path = src.calculate_magnetic_fields.get_path_to_spix(
                galaxy["name"],
                config["data_directory"],
                galaxy["use_thermal"],
                file_ending=f"_rebin_6as.fits",
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
        xlabel=h1_label,
        output_path=config["data_directory"] + "/h1_combined",
        region_mask=None,
        p0=[1, 1],
        x_value=h1_sign,
        x_unit=h1_unit,
        density_map=True,
        extra_line_params=[1, 0.5],
        fit_extra_line=True,
        extra_line_label=rf"$B_{{\mathrm{{eq}} }} \propto {h1_sign}^{{0.5}}$",
    )
    plt_helper.plot_pixel_power_law(
        x=holder["x_smooth"],
        y=holder["y_smooth"],
        z=holder["z_smooth"],
        x_error=holder["x_smooth_error"],
        y_error=holder["y_smooth_error"],
        xlabel=h1_label,
        output_path=config["data_directory"] + "/h1_combined_smooth",
        region_mask=None,
        p0=[1, 1],
        x_value=h1_sign,
        x_unit=h1_unit,
        density_map=True,
        extra_line_params=[1, 0.5],
        fit_extra_line=True,
        extra_line_label=rf"$B_{{\mathrm{{eq}} }} \propto {h1_sign}^{{0.5}}$",
    )

    plt_helper.plot_pixel_mean_power_law(
        x=holder["x_mean"],
        y=holder["y_mean"],
        x_std=holder["x_std"],
        y_std=holder["y_std"],
        xlabel=h1_mean_label,
        output_path=config["data_directory"] + "/h1_combined_mean",
        p0=[1, 1],
        x_value=h1_sign,
        x_unit=h1_mean_unit,
        extra_line_params=[1, 0.5],
        fit_extra_line=True,
        extra_line_label=rf"$\langle B_{{\mathrm{{eq}} }} \rangle \propto \langle{h1_sign}\rangle^{{0.5}}$",
    )


def calculate_h1(name: str, config: dict, fig=None):
    # "Check" if the specified galaxy exists
    galaxy_config = next(filter(lambda g: g["name"] == name, config["galaxies"],))
    try:
        if not galaxy_config["calc_h1"]:
            raise NotConfiguredException()
        __calculate_h1(
            name=galaxy_config["name"],
            data_directory=config["data_directory"],
            thermal=galaxy_config["use_thermal"],
            p0=galaxy_config["h1"]["p0"],
            levels=galaxy_config["magnetic_levels"],
            inclination=galaxy_config["inclination"],
            use_integrated_spix=galaxy_config["use_integrated"],
            spix_integrated=galaxy_config["spix"],
            vmin=galaxy_config["h1"]["vmin"],
            vmax=galaxy_config["h1"]["vmax"],
            smooth_exp=galaxy_config["smooth_exp"],
        )
    except NotConfiguredException:
        print("Galaxy not configured for HI Density...")
    return name


def __calculate_h1(
    name: str,
    data_directory: str,
    thermal: bool,
    p0: list = None,
    levels: array = None,
    inclination: int = 0,
    use_integrated_spix: bool = False,
    spix_integrated: float = None,
    rms_h1: float = 0,
    vmin: float = None,
    vmax: float = None,
    smooth_exp: bool = False,
):
    """Calculate and plot correlation between magnetic field strength and HI surface density
    for one galaxy

    Args:
        name (str): Name of galaxy
        data_directory (str): dr2 data directory
        thermal (bool): Use non thermal magnetic field
        p0 (list): inital guess for the fit
        levels (array): contour levels for the radio emission        
        inclination (int): inclination of the galaxy
        use_integrated_spix (bool): use the integrated spectral index instead of the spectral index map
        spix_integrated (float): integrated spectral index
        rms_h1 (float): rms value for the HI map
        vmin (float): minimum value of the color scale of the overlay
        vmax (float): maximum value of the color scale of the overlay
        smooth_exp (bool): perform the smoothing experiment
    """
    plt_helper.setup_matploblib(False)

    print(
        f"Calculating correlations between magnetic field and HI surface density for galaxy: {name} with thermal: {thermal}"
    )

    magnetic_path = src.calculate_magnetic_fields.get_path_to_magnetic_map(
        name, data_directory, thermal, ".fits"
    )
    magnetic_rebin_path = src.calculate_magnetic_fields.get_path_to_magnetic_map(
        name, data_directory, thermal, f"_rebin_6as.fits"
    )
    magnetic_error_rebin_path = src.calculate_magnetic_fields.get_path_to_magnetic_map(
        name, data_directory, thermal, f"_abs_error_rebin_6as.fits"
    )
    h1_path = f"{get_path_to_h1_dir(name, data_directory)}/{name}_h1_6as.fits"
    h1_rebin_path = (
        f"{get_path_to_h1_dir(name, data_directory)}/{name}_h1_rebin_6as.fits"
    )

    output_path = get_path_to_h1_dir(name, data_directory)

    # make sure that the output_dir exists
    Path(output_path).mkdir(parents=True, exist_ok=True)
    output_path += f"/{name}_h1"

    magnetic_map = fits.open(magnetic_path)
    magnetic_rebin_map = fits.open(magnetic_rebin_path)
    magnetic_error_rebin_map = fits.open(magnetic_error_rebin_path)
    h1_map = fits.open(h1_path)
    h1_rebin_map = fits.open(h1_rebin_path)

    magnetic = magnetic_map[0].data
    magnetic_rebin = magnetic_rebin_map[0].data
    magnetic_error_rebin = magnetic_error_rebin_map[0].data
    h1_flux_density = h1_map[0].data[0, 0, :, :]
    h1_flux_density_rebin = h1_rebin_map[0].data[0, 0, :, :]
    h1_flux_density_error_rebin = math_functions.h1_error(h1_flux_density_rebin, rms_h1)

    h1_smooth = None
    h1_smooth_error = None
    if smooth_exp:
        h1_smooth = fits.getdata(
            f"{get_path_to_h1_dir(name, data_directory)}/{name}_h1_rebin_6as_smooth.fits"
        )
        h1_smooth_error = math_functions.h1_error(h1_smooth, rms_h1)

    spix = np.full(magnetic_rebin.shape, spix_integrated)
    if not use_integrated_spix:
        spix_path = src.calculate_magnetic_fields.get_path_to_spix(
            name, data_directory, thermal, file_ending=f"_rebin_6as.fits"
        )

        spix_map = fits.open(spix_path)
        spix = spix_map[0].data

    # Convert into HI surface density
    h1_surface_density = math_functions.get_HI_surface_density(
        h1_flux_density,
        h1_map[0].header["BMAJ"] * 3600,
        h1_map[0].header["BMIN"] * 3600,
        inclination,
    )
    h1_surface_density_rebin = math_functions.get_HI_surface_density(
        h1_flux_density_rebin,
        h1_map[0].header["BMAJ"] * 3600,
        h1_map[0].header["BMIN"] * 3600,
        inclination,
    )

    h1_surface_density_error_rebin = math_functions.get_HI_surface_density_error(
        h1_flux_density_error_rebin,
        h1_map[0].header["BMAJ"] * 3600,
        h1_map[0].header["BMIN"] * 3600,
        inclination,
    )

    # Fix broken header
    if h1_map[0].header["CDELT3"] == 0:
        h1_map[0].header["CDELT3"] = 1
        h1_rebin_map[0].header["CDELT3"] = 1

    print("Generating overlay plot...")
    plt_helper.plot_overlay(
        base=h1_surface_density,
        overlay=magnetic * 1e6,
        base_label=h1_label,
        wcs=WCS(h1_map[0].header).celestial,
        output_path=output_path + "_overlay",
        levels=levels,
        inline_title="NGC " + name[1:],
        vmin=vmin,
        vmax=vmax,
    )

    print("Generating pixel plot and power law fit...")
    plt_helper.plot_pixel_power_law(
        x=h1_surface_density_rebin.flatten(),
        x_error=h1_surface_density_error_rebin.flatten(),
        # field strength in µG
        y=magnetic_rebin.flatten() * 1e6,
        y_error=magnetic_error_rebin.flatten() * 1e6,
        z=spix.flatten(),
        xlabel=h1_label,
        output_path=output_path + "_pixel",
        p0=p0,
        x_value=h1_sign,
        x_unit=h1_unit,
        extra_line_params=[1, 0.5],
        fit_extra_line=True,
        extra_line_label=rf"$B_{{\mathrm{{eq}} }} \propto {h1_sign}^{{0.5}}$",
        inline_title="NGC " + name[1:],
    )

    if smooth_exp:
        print("Generating smoothed pixel plot and power law fit...")
        plt_helper.plot_pixel_power_law(
            x=h1_smooth.flatten(),
            x_error=h1_smooth_error.flatten(),
            # field strength in µG
            y=magnetic_rebin.flatten() * 1e6,
            y_error=magnetic_error_rebin.flatten() * 1e6,
            z=spix.flatten(),
            xlabel=h1_label,
            output_path=output_path + "_pixel_smooth",
            p0=p0,
            x_value=h1_sign,
            x_unit=h1_unit,
            extra_line_params=[1, 0.5],
            fit_extra_line=True,
            extra_line_label=rf"$B_{{\mathrm{{eq}} }} \propto {h1_sign}^{{0.5}}$",
            inline_title="NGC " + name[1:],
        )

    return name


def get_path_to_h1_dir(name: str, data_directory: str) -> str:
    """Get the path to the directory where the star formation data should be stored

    Args:
        name (str): Name of the galaxy
        data_directory (str): dr2 data directory

    Returns:
        str: Path to h1 dir
    """
    return f"{data_directory}/h1/{name}"
