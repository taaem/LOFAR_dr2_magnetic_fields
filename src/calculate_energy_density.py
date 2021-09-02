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

energy_density_label = (
    r"$u_{\mathrm{H\MakeUppercase{\romannumeral 1}} + \mathrm{H_2}}$ [\si{erg.cm^{-3}}]"
)
energy_density_mean_label = r"$\langle u_{\mathrm{H\MakeUppercase{\romannumeral 1}} + \mathrm{H_2}} \rangle$ [\si{erg.cm^{-3}}]"
energy_density_sign = r"u_{\mathrm{H\MakeUppercase{\romannumeral 1}} + \mathrm{H_2}}"
energy_density_label_h1 = (
    r"$u_{\mathrm{H\MakeUppercase{\romannumeral 1}}}$ [\si{erg.cm^{-3}}]"
)
energy_density_mean_label_h1 = r"$\langle u_{\mathrm{H\MakeUppercase{\romannumeral 1}}} \rangle$ [\si{erg.cm^{-3}}]"
energy_density_sign_h1 = r"u_{\mathrm{H\MakeUppercase{\romannumeral 1}}}"
energy_density_label_h2 = r"$u_{\mathrm{H_2}}$ [\si{erg.cm^{-3}}]"
energy_density_mean_label_h2 = r"$\langle u_{\mathrm{H_2}} \rangle$ [\si{erg.cm^{-3}}]"
energy_density_sign_h2 = r"u_{\mathrm{H_2}}"
energy_density_unit = r"\si{erg.cm^{-3}}"
magnetic_energy_density_label = r"$u_B$ [\si{erg.cm^{-3}}]"
magnetic_energy_density_mean_label = r"$\langle u_B \rangle$ [\si{erg.cm^{-3}}]"
magnetic_energy_density_unit = r"\si{erg.cm^{-3}}"


def calculate_all_energy_density(config: dict, skip: bool = False):
    """Calculate energy density correlations for all available galaxies

    Args:
        config (dict): Config
        skip (bool): Skip energy calculation
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
                calculate_energy_density,
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
        "h2": np.array([]),
        "h2_error": np.array([]),
        "h2_smooth": np.array([]),
        "h2_smooth_error": np.array([]),
        "h2_mean": np.array([]),
        "h2_std": np.array([]),
        "h1": np.array([]),
        "h1_error": np.array([]),
        "h1_smooth": np.array([]),
        "h1_smooth_error": np.array([]),
        "h1_mean": np.array([]),
        "h1_std": np.array([]),
        "combined": np.array([]),
        "combined_error": np.array([]),
        "combined_smooth": np.array([]),
        "combined_smooth_error": np.array([]),
        "combined_mean": np.array([]),
        "combined_std": np.array([]),
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
        if not galaxy["calc_energy"] or galaxy["skip_combined_energy"]:
            continue
        holder["name"].append(galaxy["name"])

        rebin_res = "6"
        if galaxy["calc_energy"]:
            rebin_res = "13_5"

        # Read Energy density
        holder["h2"] = np.concatenate(
            (
                holder["h2"],
                fits.getdata(
                    f"{get_path_to_energy_density_dir(galaxy['name'], config['data_directory'])}/{galaxy['name']}_energy_density_h2_{rebin_res}as_rebin.fits"
                ).flatten(),
            )
        )
        holder["h2_error"] = np.concatenate(
            (
                holder["h2_error"],
                fits.getdata(
                    f"{get_path_to_energy_density_dir(galaxy['name'], config['data_directory'])}/{galaxy['name']}_energy_density_h2_{rebin_res}as_error_rebin.fits"
                ).flatten(),
            )
        )
        holder["h1"] = np.concatenate(
            (
                holder["h1"],
                fits.getdata(
                    f"{get_path_to_energy_density_dir(galaxy['name'], config['data_directory'])}/{galaxy['name']}_energy_density_h1_{rebin_res}as_rebin.fits"
                ).flatten(),
            )
        )
        holder["h1_error"] = np.concatenate(
            (
                holder["h1_error"],
                fits.getdata(
                    f"{get_path_to_energy_density_dir(galaxy['name'], config['data_directory'])}/{galaxy['name']}_energy_density_h1_{rebin_res}as_error_rebin.fits"
                ).flatten(),
            )
        )
        holder["combined"] = np.concatenate(
            (
                holder["combined"],
                fits.getdata(
                    f"{get_path_to_energy_density_dir(galaxy['name'], config['data_directory'])}/{galaxy['name']}_energy_density_{rebin_res}as_rebin.fits"
                ).flatten(),
            )
        )
        holder["combined_error"] = np.concatenate(
            (
                holder["combined_error"],
                fits.getdata(
                    f"{get_path_to_energy_density_dir(galaxy['name'], config['data_directory'])}/{galaxy['name']}_energy_density_{rebin_res}as_error_rebin.fits"
                ).flatten(),
            )
        )

        if galaxy["smooth_exp"]:
            holder["h2_smooth"] = np.concatenate(
                (
                    holder["h2_smooth"],
                    fits.getdata(
                        f"{get_path_to_energy_density_dir(galaxy['name'], config['data_directory'])}/{galaxy['name']}_energy_density_h2_smooth_{rebin_res}as_rebin.fits"
                    ).flatten(),
                )
            )
            holder["h2_smooth_error"] = np.concatenate(
                (
                    holder["h2_smooth_error"],
                    fits.getdata(
                        f"{get_path_to_energy_density_dir(galaxy['name'], config['data_directory'])}/{galaxy['name']}_energy_density_h2_smooth_{rebin_res}as_error_rebin.fits"
                    ).flatten(),
                )
            )
            holder["h1_smooth"] = np.concatenate(
                (
                    holder["h1_smooth"],
                    fits.getdata(
                        f"{get_path_to_energy_density_dir(galaxy['name'], config['data_directory'])}/{galaxy['name']}_energy_density_h1_smooth_{rebin_res}as_rebin.fits"
                    ).flatten(),
                )
            )
            holder["h1_smooth_error"] = np.concatenate(
                (
                    holder["h1_smooth_error"],
                    fits.getdata(
                        f"{get_path_to_energy_density_dir(galaxy['name'], config['data_directory'])}/{galaxy['name']}_energy_density_h1_smooth_{rebin_res}as_error_rebin.fits"
                    ).flatten(),
                )
            )
            holder["combined_smooth"] = np.concatenate(
                (
                    holder["combined_smooth"],
                    fits.getdata(
                        f"{get_path_to_energy_density_dir(galaxy['name'], config['data_directory'])}/{galaxy['name']}_energy_density_smooth_{rebin_res}as_rebin.fits"
                    ).flatten(),
                )
            )
            holder["combined_smooth_error"] = np.concatenate(
                (
                    holder["combined_smooth_error"],
                    fits.getdata(
                        f"{get_path_to_energy_density_dir(galaxy['name'], config['data_directory'])}/{galaxy['name']}_energy_density_smooth_{rebin_res}as_error_rebin.fits"
                    ).flatten(),
                )
            )

        holder["h2_mean"] = np.append(
            holder["h2_mean"],
            np.nanmean(
                fits.getdata(
                    f"{get_path_to_energy_density_dir(galaxy['name'], config['data_directory'])}/{galaxy['name']}_energy_density_h2_{rebin_res}as_rebin.fits"
                )
            ),
        )
        holder["h2_std"] = np.append(
            holder["h2_std"],
            sem(
                fits.getdata(
                    f"{get_path_to_energy_density_dir(galaxy['name'], config['data_directory'])}/{galaxy['name']}_energy_density_h2_{rebin_res}as_rebin.fits"
                ),
                axis=None,
                nan_policy="omit",
            ),
        )
        holder["h1_mean"] = np.append(
            holder["h1_mean"],
            np.nanmean(
                fits.getdata(
                    f"{get_path_to_energy_density_dir(galaxy['name'], config['data_directory'])}/{galaxy['name']}_energy_density_h1_{rebin_res}as_rebin.fits"
                )
            ),
        )
        holder["h1_std"] = np.append(
            holder["h1_std"],
            sem(
                fits.getdata(
                    f"{get_path_to_energy_density_dir(galaxy['name'], config['data_directory'])}/{galaxy['name']}_energy_density_h1_{rebin_res}as_rebin.fits"
                ),
                axis=None,
                nan_policy="omit",
            ),
        )
        holder["combined_mean"] = np.append(
            holder["combined_mean"],
            np.nanmean(
                fits.getdata(
                    f"{get_path_to_energy_density_dir(galaxy['name'], config['data_directory'])}/{galaxy['name']}_energy_density_{rebin_res}as_rebin.fits"
                )
            ),
        )
        holder["combined_std"] = np.append(
            holder["combined_std"],
            sem(
                fits.getdata(
                    f"{get_path_to_energy_density_dir(galaxy['name'], config['data_directory'])}/{galaxy['name']}_energy_density_{rebin_res}as_rebin.fits"
                ),
                axis=None,
                nan_policy="omit",
            ),
        )

        # Read magnetic field
        m_path = src.calculate_magnetic_fields.get_path_to_magnetic_map(
            galaxy["name"],
            config["data_directory"],
            galaxy["use_thermal"],
            f"_rebin_{rebin_res}as.fits",
        )
        g_magnetic_field = math_functions.get_magnetic_field_energy_density(
            fits.getdata(m_path)
        )
        holder["y"] = np.concatenate((holder["y"], g_magnetic_field.flatten()))

        g_m_error_path = src.calculate_magnetic_fields.get_path_to_magnetic_map(
            galaxy["name"],
            config["data_directory"],
            galaxy["use_thermal"],
            f"_abs_error_rebin_{rebin_res}as.fits",
        )
        g_m_error = math_functions.get_magnetic_field_energy_density_error(
            fits.getdata(m_path), fits.getdata(g_m_error_path)
        )
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
            holder["y_mean"],
            math_functions.get_magnetic_field_energy_density(
                magnetic_results["mean"]["value"]
            ),
        )
        holder["y_std"] = np.append(
            holder["y_std"],
            math_functions.get_magnetic_field_energy_density_error(
                magnetic_results["mean"]["value"], magnetic_results["mean"]["std"]
            ),
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
                file_ending=f"_rebin_{rebin_res}as.fits",
            )
            g_spix = fits.getdata(s_path)
        holder["z"] = np.concatenate((holder["z"], g_spix.flatten()))
        if galaxy["smooth_exp"]:
            holder["z_smooth"] = np.concatenate((holder["z_smooth"], g_spix.flatten()))

    # Calculate combined plot
    plt_helper.plot_pixel_power_law(
        x=holder["h2"],
        x_error=holder["h2_error"],
        y=holder["y"],
        y_error=holder["y_error"],
        z=holder["z"],
        xlabel=energy_density_label_h2,
        ylabel=magnetic_energy_density_label,
        output_path=config["data_directory"] + "/energy_density_h2",
        region_mask=None,
        p0=[1, 1],
        extra_line_params=[1, 1],
        x_value=energy_density_sign_h2,
        y_unit=magnetic_energy_density_unit,
        x_unit=energy_density_unit,
        cutoff=config["energy_cutoff"],
        density_map=False,
    )

    plt_helper.plot_pixel_power_law(
        x=holder["h2_smooth"],
        x_error=holder["h2_smooth_error"],
        y=holder["y_smooth"],
        y_error=holder["y_smooth_error"],
        z=holder["z_smooth"],
        xlabel=energy_density_label_h2,
        ylabel=magnetic_energy_density_label,
        output_path=config["data_directory"] + "/energy_density_h2_smooth",
        region_mask=None,
        p0=[1, 1],
        extra_line_params=[1, 1],
        x_value=energy_density_sign_h2,
        y_unit=magnetic_energy_density_unit,
        x_unit=energy_density_unit,
        cutoff=config["energy_cutoff"],
        density_map=False,
    )

    plt_helper.plot_pixel_mean_power_law(
        x=holder["h2_mean"],
        y=holder["y_mean"],
        x_std=holder["h2_std"],
        y_std=holder["y_std"],
        xlabel=energy_density_mean_label_h2,
        ylabel=magnetic_energy_density_mean_label,
        output_path=config["data_directory"] + "/energy_density_h2_mean",
        p0=[1, 1],
        extra_line_params=[1, 1],
        x_value=energy_density_sign_h2,
        y_unit=magnetic_energy_density_unit,
        x_unit=energy_density_unit,
    )

    # Calculate combined plot
    plt_helper.plot_pixel_power_law(
        x=holder["h1"],
        x_error=holder["h1_error"],
        y=holder["y"],
        y_error=holder["y_error"],
        z=holder["z"],
        xlabel=energy_density_label_h1,
        ylabel=magnetic_energy_density_label,
        output_path=config["data_directory"] + "/energy_density_h1",
        region_mask=None,
        p0=[ 3.452647e-10, 1],
        extra_line_params=[1, 1],
        x_value=energy_density_sign_h1,
        y_unit=magnetic_energy_density_unit,
        x_unit=energy_density_unit,
        cutoff=config["energy_cutoff"],
        density_map=False,
    )

    plt_helper.plot_pixel_power_law(
        x=holder["h1_smooth"],
        x_error=holder["h1_smooth_error"],
        y=holder["y_smooth"],
        y_error=holder["y_smooth_error"],
        z=holder["z_smooth"],
        xlabel=energy_density_label_h1,
        ylabel=magnetic_energy_density_label,
        output_path=config["data_directory"] + "/energy_density_h1_smooth",
        region_mask=None,
        p0=[ 3.452647e-10, 1],
        extra_line_params=[1, 1],
        x_value=energy_density_sign_h1,
        y_unit=magnetic_energy_density_unit,
        x_unit=energy_density_unit,
        cutoff=config["energy_cutoff"],
        density_map=False,
    )

    plt_helper.plot_pixel_mean_power_law(
        x=holder["h1_mean"],
        y=holder["y_mean"],
        x_std=holder["h1_std"],
        y_std=holder["y_std"],
        xlabel=energy_density_mean_label_h1,
        ylabel=magnetic_energy_density_mean_label,
        output_path=config["data_directory"] + "/energy_density_h1_mean",
        p0=[ 3.452647e-10, 1],
        extra_line_params=[1, 1],
        x_value=energy_density_sign_h1,
        y_unit=magnetic_energy_density_unit,
        x_unit=energy_density_unit,
    )

    # Calculate combined plot
    plt_helper.plot_pixel_power_law(
        x=holder["combined"],
        x_error=holder["combined_error"],
        y=holder["y"],
        y_error=holder["y_error"],
        z=holder["z"],
        xlabel=energy_density_label,
        ylabel=magnetic_energy_density_label,
        output_path=config["data_directory"] + "/energy_density_combined",
        region_mask=None,
        p0=[1, 1],
        extra_line_params=[1, 1],
        x_value=energy_density_sign,
        y_unit=magnetic_energy_density_unit,
        x_unit=energy_density_unit,
        cutoff=config["energy_cutoff"],
        density_map=False,
    )

    plt_helper.plot_pixel_power_law(
        x=holder["combined_smooth"],
        x_error=holder["combined_smooth_error"],
        y=holder["y_smooth"],
        y_error=holder["y_smooth_error"],
        z=holder["z_smooth"],
        xlabel=energy_density_label,
        ylabel=magnetic_energy_density_label,
        output_path=config["data_directory"] + "/energy_density_combined_smooth",
        region_mask=None,
        p0=[1, 1],
        extra_line_params=[1, 1],
        x_value=energy_density_sign,
        y_unit=magnetic_energy_density_unit,
        x_unit=energy_density_unit,
        cutoff=config["energy_cutoff"],
        density_map=False,
    )

    plt_helper.plot_pixel_mean_power_law(
        x=holder["combined_mean"],
        y=holder["y_mean"],
        x_std=holder["combined_std"],
        y_std=holder["y_std"],
        xlabel=energy_density_mean_label,
        ylabel=magnetic_energy_density_mean_label,
        output_path=config["data_directory"] + "/energy_density_combined_mean",
        p0=[1, 1],
        extra_line_params=[1, 1],
        x_value=energy_density_sign,
        y_unit=magnetic_energy_density_unit,
        x_unit=energy_density_unit,
    )


def calculate_energy_density(name: str, config: dict):
    # "Check" if the specified galaxy exists
    galaxy_config = next(filter(lambda g: g["name"] == name, config["galaxies"],))
    try:
        if not galaxy_config["calc_energy"]:
            raise NotConfiguredException()
        __calculate_energy_density(
            name=galaxy_config["name"],
            data_directory=config["data_directory"],
            thermal=galaxy_config["use_thermal"],
            pathlength_h1=galaxy_config["h1"]["pathlength"],
            p0=galaxy_config["energy"]["p0"],
            levels=galaxy_config["energy"]["levels"],
            inclination=galaxy_config["inclination"],
            has_h2=galaxy_config["calc_surf"],
            pathlength_co=galaxy_config["co"]["pathlength"]
            if galaxy_config["calc_surf"]
            else None,
            cutoff=galaxy_config["energy"]["cutoff"],
            vmin=galaxy_config["energy"]["vmin"],
            vmax=galaxy_config["energy"]["vmax"],
            use_integrated_spix=galaxy_config["use_integrated"],
            spix_integrated=galaxy_config["spix"],
            rms_h1=galaxy_config["h1"]["rms"],
            rms_co=galaxy_config["co"]["rms"] if galaxy_config["calc_surf"] else 0,
            smooth_exp=galaxy_config["smooth_exp"],
        )
    except NotConfiguredException:
        print("Galaxy not configured for Energy Density...")
    return name


def __calculate_energy_density(
    name: str,
    data_directory: str,
    thermal: bool,
    pathlength_h1: float,
    p0: list,
    levels: array,
    inclination: int,
    has_h2: bool,
    pathlength_co: float,
    cutoff: float,
    vmin: float,
    vmax: float,
    use_integrated_spix: bool,
    spix_integrated: float,
    rms_h1: float,
    rms_co: float,
    smooth_exp: bool,
) -> str:
    """Calculate and plot correlation between magnetic field strength and energy density
    for one galaxy

    Args:
        name (str): Name of galaxy
        data_directory (str): dr2 data directory
        thermal (bool): Use non thermal magnetic field
        pathlength_h1 (float): pathlength of the THINGS map
        p0 (list): inital guess for the fit
        levels (array): contour levels for the radio emission        
        inclination (int): inclination of the galaxy
        has_h2 (bool): False if no heracles map is available
        pathlength_co (float, optional): pathlength of the HERACLES map
        cutoff (float): fit cutoff for the turbulent energy density
        vmin (float): minimum value of the color scale of the overlay
        vmax (float): maximum value of the color scale of the overlay
        use_integrated_spix (bool): use the integrated spectral index instead of the spectral index map
        spix_integrated (float): integrated spectral index
        rms_h1 (float): rms value for the HI map
        rms_co (float): rms value for the heracles map
        smooth_exp (bool): perform the smoothing experiment

    Returns:
        str: name of the galaxy
    """
    plt_helper.setup_matploblib(False)

    rebin_res = "6"
    if has_h2:
        rebin_res = "13_5"

    print(
        f"Calculating correlations between magnetic field and energy density for galaxy: {name} with thermal: {thermal}"
    )

    magnetic_path = src.calculate_magnetic_fields.get_path_to_magnetic_map(
        name, data_directory, thermal, ".fits"
    )
    magnetic_rebin_path = src.calculate_magnetic_fields.get_path_to_magnetic_map(
        name, data_directory, thermal, f"_rebin_{rebin_res}as.fits"
    )
    magnetic_error_rebin_path = src.calculate_magnetic_fields.get_path_to_magnetic_map(
        name, data_directory, thermal, f"_abs_error_rebin_{rebin_res}as.fits"
    )

    h1_path = (
        f"{get_path_to_energy_density_dir(name, data_directory)}/{name}_h1_6as.fits"
    )
    h1_rebin_path = f"{get_path_to_energy_density_dir(name, data_directory)}/{name}_h1_rebin_{rebin_res}as.fits"
    dispersion_path = (
        f"{get_path_to_energy_density_dir(name, data_directory)}/{name}_dis_6as.fits"
    )
    dispersion_rebin_path = f"{get_path_to_energy_density_dir(name, data_directory)}/{name}_dis_rebin_{rebin_res}as.fits"

    output_path = get_path_to_energy_density_dir(name, data_directory)
    # make sure that the output_dir exists
    Path(output_path).mkdir(parents=True, exist_ok=True)
    output_path += f"/{name}_energy_density"

    magnetic_map = fits.open(magnetic_path)
    h1_map = fits.open(h1_path)
    dispersion_map = fits.open(dispersion_path)

    magnetic_rebin_map = fits.open(magnetic_rebin_path)
    magnetic_error_rebin_map = fits.open(magnetic_error_rebin_path)
    h1_rebin_map = fits.open(h1_rebin_path)
    dispersion_rebin_map = fits.open(dispersion_rebin_path)

    magnetic = magnetic_map[0].data
    magnetic_error_rebin = magnetic_error_rebin_map[0].data
    h1_flux_density = h1_map[0].data[0, 0, :, :]
    dispersion = abs(dispersion_map[0].data[0, 0, :, :])

    magnetic_rebin = magnetic_rebin_map[0].data
    h1_flux_density_rebin = h1_rebin_map[0].data[0, 0, :, :]
    dispersion_rebin = abs(dispersion_rebin_map[0].data[0, 0, :, :])

    h1_flux_density_error_rebin = math_functions.h1_error(h1_flux_density_rebin, rms_h1)

    # Use the mean dispersion to estimate the energy density otherwise fluctuations will be too high
    dispersion = np.nanmean(dispersion)
    dispersion_rebin = np.nanmean(dispersion_rebin)

    spix = np.full(magnetic_rebin.shape, spix_integrated)
    if not use_integrated_spix:
        spix_path = src.calculate_magnetic_fields.get_path_to_spix(
            name, data_directory, thermal, file_ending=f"_rebin_{rebin_res}as.fits"
        )

        spix_map = fits.open(spix_path)
        spix = spix_map[0].data

    # Fix broken header
    if h1_map[0].header["CDELT3"] == 0:
        h1_map[0].header["CDELT3"] = 1
        h1_rebin_map[0].header["CDELT3"] = 1

    magnetic_energy_density = math_functions.get_magnetic_field_energy_density(magnetic)
    magnetic_energy_density_rebin = math_functions.get_magnetic_field_energy_density(
        magnetic_rebin
    )
    magnetic_energy_density_error_rebin = math_functions.get_magnetic_field_energy_density_error(
        magnetic_rebin, magnetic_error_rebin
    )

    ## Helper Function
    def make_plots(
        l_energy, l_energy_rebin, l_energy_error_rebin, out, only_h2, only_h1
    ):
        print("Generating overlay plot...")
        energy_string = energy_density_label
        energy_val = energy_density_sign
        if only_h2 and not only_h1:
            energy_string = energy_density_label_h2
            energy_val = energy_density_sign_h2
        elif only_h1 and not only_h2:
            energy_string = energy_density_label_h1
            energy_val = energy_density_sign_h1

        if l_energy is not None:
            plt_helper.plot_overlay(
                base=l_energy,
                overlay=magnetic_energy_density,
                base_label=energy_string,
                overlay_label=magnetic_energy_density_label,
                wcs=WCS(h1_map[0].header).celestial,
                output_path=output_path + "_overlay" + out,
                vmin=vmin,
                vmax=vmax,
                levels=levels,
                inline_title="NGC " + name[1:],
            )

        print("Generating pixel plot and power law fit...")
        plt_helper.plot_pixel_power_law(
            x=l_energy_rebin.flatten(),
            x_error=l_energy_error_rebin.flatten(),
            y=magnetic_energy_density_rebin.flatten(),
            y_error=magnetic_energy_density_error_rebin.flatten(),
            z=spix.flatten(),
            xlabel=energy_string,
            output_path=output_path + "_pixel" + out,
            ylabel=magnetic_energy_density_label,
            p0=p0,
            extra_line_params=[1, 1],
            x_value=energy_val,
            y_unit=magnetic_energy_density_unit,
            x_unit=energy_density_unit,
            cutoff=cutoff,
            inline_title="NGC " + name[1:],
        )

        # Save energy density maps
        if l_energy is not None:
            output_hdu = fits.PrimaryHDU(abs(l_energy))
            output_hdu.header.update(WCS(h1_map[0].header).to_header())
            output_hdul = fits.HDUList(output_hdu)
            output_hdul.writeto(
                f"{get_path_to_energy_density_dir(name, data_directory)}/{name}_energy_density{out}_6as.fits",
                overwrite=True,
            )
            print(
                "Out:",
                f"{get_path_to_energy_density_dir(name, data_directory)}/{name}_energy_density{out}_6as.fits",
            )

        output_hdu = fits.PrimaryHDU(l_energy_rebin)
        output_hdu.header.update(WCS(h1_rebin_map[0].header).to_header())
        output_hdul = fits.HDUList(output_hdu)
        output_hdul.writeto(
            f"{get_path_to_energy_density_dir(name, data_directory)}/{name}_energy_density{out}_{rebin_res}as_rebin.fits",
            overwrite=True,
        )
        print(
            "Out:",
            f"{get_path_to_energy_density_dir(name, data_directory)}/{name}_energy_density{out}_{rebin_res}as_rebin.fits",
        )

        output_hdu = fits.PrimaryHDU(l_energy_error_rebin)
        output_hdu.header.update(WCS(h1_rebin_map[0].header).to_header())
        output_hdul = fits.HDUList(output_hdu)
        output_hdul.writeto(
            f"{get_path_to_energy_density_dir(name, data_directory)}/{name}_energy_density{out}_{rebin_res}as_error_rebin.fits",
            overwrite=True,
        )
        print(
            "Out:",
            f"{get_path_to_energy_density_dir(name, data_directory)}/{name}_energy_density{out}_{rebin_res}as_error_rebin.fits",
        )

    ## End Helper Function

    if smooth_exp:
        h1_smooth = fits.getdata(
            f"{get_path_to_energy_density_dir(name, data_directory)}/{name}_h1_rebin_13_5as_smooth.fits"
        )
        h1_smooth_error = math_functions.h1_error(h1_smooth, rms_h1)
        co_smooth = fits.getdata(
            f"{get_path_to_energy_density_dir(name, data_directory)}/{name}_co_rebin_13_5as_smooth.fits"
        )
        co_smooth_error = math_functions.h1_error(h1_smooth, rms_h1)

        h1_surface_density_smooth = math_functions.get_HI_surface_density(
            h1_smooth,
            h1_map[0].header["BMAJ"] * 3600,
            h1_map[0].header["BMIN"] * 3600,
            inclination,
        )
        h1_surface_density_smooth_error = math_functions.get_HI_surface_density_error(
            h1_smooth_error,
            h1_map[0].header["BMAJ"] * 3600,
            h1_map[0].header["BMIN"] * 3600,
            inclination,
        )
        h2_surface_density_smooth = math_functions.get_H2_surface_density(
            co_smooth, inclination
        )
        h2_surface_density_smooth_error = math_functions.get_H2_surface_density_error(
            co_smooth_error, inclination
        )

        # H2 Energy density
        energy_density_smooth = math_functions.get_energy_density_from_surface_density(
            h2_surface_density_smooth, pathlength_co, dispersion_rebin
        )
        energy_density_smooth_error = np.sqrt(
            math_functions.get_energy_density_error(
                h2_surface_density_smooth,
                h2_surface_density_smooth_error,
                pathlength_co,
                dispersion_rebin,
            )
            ** 2
        )
        make_plots(
            None,
            energy_density_smooth,
            energy_density_smooth_error,
            "_h2_smooth",
            True,
            False,
        )
        # H1 Energy Density
        h1_energy_density_smooth = math_functions.get_energy_density_from_surface_density(
            h1_surface_density_smooth, pathlength_h1, dispersion
        )
        h1_energy_density_smooth_error = math_functions.get_energy_density_error(
            h1_surface_density_smooth,
            h1_surface_density_smooth_error,
            pathlength_h1,
            dispersion_rebin,
        )
        make_plots(
            None,
            h1_energy_density_smooth,
            h1_energy_density_smooth_error,
            "_h1_smooth",
            False,
            True,
        )

        make_plots(
            None,
            h1_energy_density_smooth + energy_density_smooth,
            np.sqrt(
                h1_energy_density_smooth_error ** 2 + energy_density_smooth_error ** 2
            ),
            "_smooth",
            True,
            True,
        )

    energy_density: np.array = np.zeros(h1_flux_density.shape)
    energy_density_rebin: np.array = np.zeros(h1_flux_density_rebin.shape)
    energy_density_error_rebin: np.array = np.zeros(h1_flux_density_rebin.shape)
    if has_h2:
        co_path = (
            f"{get_path_to_energy_density_dir(name, data_directory)}/{name}_co_6as.fits"
        )
        co_rebin_path = f"{get_path_to_energy_density_dir(name, data_directory)}/{name}_co_rebin_{rebin_res}as.fits"
        co_map = fits.open(co_path)
        co_rebin_map = fits.open(co_rebin_path)
        co = co_map[0].data[0, :, :]
        co_rebin = co_rebin_map[0].data[0, :, :]
        co_error_rebin = math_functions.co_error(co_rebin, rms_co)
        h2_surface_density = math_functions.get_H2_surface_density(co, inclination)
        h2_surface_density_rebin = math_functions.get_H2_surface_density(
            co_rebin, inclination
        )
        h2_surface_density_error_rebin = math_functions.get_H2_surface_density_error(
            co_error_rebin, inclination
        )
        energy_density += math_functions.get_energy_density_from_surface_density(
            h2_surface_density, pathlength_co, dispersion
        )
        energy_density_rebin += math_functions.get_energy_density_from_surface_density(
            h2_surface_density_rebin, pathlength_co, dispersion_rebin
        )

        energy_density_error_rebin = np.sqrt(
            energy_density_error_rebin ** 2
            + math_functions.get_energy_density_error(
                h2_surface_density_rebin,
                h2_surface_density_error_rebin,
                pathlength_co,
                dispersion_rebin,
            )
            ** 2
        )

        make_plots(
            energy_density,
            energy_density_rebin,
            energy_density_error_rebin,
            "_h2",
            True,
            False,
        )

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

    h1_energy_density = math_functions.get_energy_density_from_surface_density(
        h1_surface_density, pathlength_h1, dispersion
    )

    h1_energy_density_rebin = math_functions.get_energy_density_from_surface_density(
        h1_surface_density_rebin, pathlength_h1, dispersion_rebin
    )

    h1_energy_density_error_rebin = math_functions.get_energy_density_error(
        h1_surface_density_rebin,
        h1_surface_density_error_rebin,
        pathlength_h1,
        dispersion_rebin,
    )

    make_plots(
        h1_energy_density,
        h1_energy_density_rebin,
        h1_energy_density_error_rebin,
        "_h1",
        False,
        True,
    )

    make_plots(
        h1_energy_density + energy_density,
        h1_energy_density_rebin + energy_density_rebin,
        np.sqrt(h1_energy_density_error_rebin ** 2 + energy_density_error_rebin ** 2),
        "",
        True,
        True,
    )

    return name


def get_path_to_energy_density_dir(name: str, data_directory: str) -> str:
    """Get the path to the directory where the star formation data should be stored

    Args:
        name (str): Name of the galaxy
        data_directory (str): dr2 data directory

    Returns:
        str: Path to energy_density dir
    """
    return f"{data_directory}/energy_density/{name}"
