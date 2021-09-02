import multiprocessing as mp
from pathlib import Path

import numpy as np
import pyregion
from astropy.io import fits
from astropy.wcs import WCS
from numpy.core.defchararray import array
from scipy.stats import sem
import yaml

import src.calculate_magnetic_fields
import src.helper as helper
import src.math_functions as math_functions
import src.matplotlib_helper as plt_helper
from src.exceptions import NotConfiguredException

surf_label = r"$\Sigma_{\mathrm{H\MakeUppercase{\romannumeral 1} + H_2}}$ [\si{M_{\odot}.pc^{-2}}]"
surf_mean_label = r"$\langle\Sigma_{\mathrm{H\MakeUppercase{\romannumeral 1} + H_2}} \rangle$ [\si{M_{\odot}.pc^{-2}}]"
surf_sign = r"\Sigma_{\mathrm{H\MakeUppercase{\romannumeral 1} + H_2}}"
surf_unit = r"\si{M_{\odot}.pc^{-2}}"

h2_label = r"$\Sigma_{\mathrm{H}_2}$ [\si{M_{\odot}.pc^{-2}}]"
h2_mean_label = r"$\langle\Sigma_{\mathrm{H}_2} \rangle$ [\si{M_{\odot}.pc^{-2}}]"
h2_sign = r"\Sigma_{\mathrm{H}_2}"
h2_unit = r"\si{M_{\odot}.pc^{-2}}"


def calculate_all_surface_density(config: dict, skip: bool = False):
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
                calculate_surface_density,
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
        "h2": np.array([]),
        "h2_mean": np.array([]),
        "h2_smooth": np.array([]),
        "h2_smooth_error": np.array([]),
        "h2_std": np.array([]),
        "h2_error": np.array([]),
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
        if not galaxy["calc_surf"] or galaxy["skip_combined_surf"]:
            continue
        holder["name"].append(galaxy["name"])

        # Read surface density
        path = (
            get_path_to_surface_density_dir(galaxy["name"], config["data_directory"])
            + f"/{galaxy['name']}_surface_density_13_5as_rebin.fits"
        )
        g_surf = fits.getdata(path)
        holder["x"] = np.concatenate((holder["x"], g_surf.flatten()))

        e_path = f"{get_path_to_surface_density_dir(galaxy['name'], config['data_directory'])}/{galaxy['name']}_surface_density_13_5as_error_rebin.fits"
        e_surf = fits.getdata(e_path)
        holder["x_error"] = np.concatenate((holder["x_error"], e_surf.flatten()))

        holder["x_mean"] = np.append(holder["x_mean"], np.nanmean(g_surf))
        holder["x_std"] = np.append(
            holder["x_std"], sem(g_surf, axis=None, nan_policy="omit")
        )

        if galaxy["smooth_exp"]:
            g_surf_smooth = fits.getdata(
                get_path_to_surface_density_dir(
                    galaxy["name"], config["data_directory"]
                )
                + f"/{galaxy['name']}_surface_density_13_5as_rebin_smooth.fits"
            )
            e_surf_smooth = fits.getdata(
                get_path_to_surface_density_dir(
                    galaxy["name"], config["data_directory"]
                )
                + f"/{galaxy['name']}_surface_density_13_5as_error_rebin_smooth.fits"
            )
            holder["x_smooth"] = np.append(holder["x_smooth"], g_surf_smooth.flatten())
            holder["x_smooth_error"] = np.append(
                holder["x_smooth_error"], e_surf_smooth.flatten()
            )

        # Read H2 surface density
        path = (
            get_path_to_surface_density_dir(galaxy["name"], config["data_directory"])
            + f"/{galaxy['name']}_h2_surface_density_13_5as_rebin.fits"
        )
        g_surf = fits.getdata(path)
        holder["h2"] = np.concatenate((holder["h2"], g_surf.flatten()))

        e_path = f"{get_path_to_surface_density_dir(galaxy['name'], config['data_directory'])}/{galaxy['name']}_h2_surface_density_13_5as_error_rebin.fits"
        e_surf = fits.getdata(e_path)
        holder["h2_error"] = np.concatenate((holder["h2_error"], e_surf.flatten()))

        holder["h2_mean"] = np.append(holder["h2_mean"], np.nanmean(g_surf))
        holder["h2_std"] = np.append(
            holder["h2_std"], sem(g_surf, axis=None, nan_policy="omit")
        )

        if galaxy["smooth_exp"]:
            g_surf_smooth = fits.getdata(
                get_path_to_surface_density_dir(
                    galaxy["name"], config["data_directory"]
                )
                + f"/{galaxy['name']}_h2_surface_density_13_5as_rebin_smooth.fits"
            )
            e_surf_smooth = fits.getdata(
                get_path_to_surface_density_dir(
                    galaxy["name"], config["data_directory"]
                )
                + f"/{galaxy['name']}_h2_surface_density_13_5as_error_rebin_smooth.fits"
            )
            holder["h2_smooth"] = np.append(
                holder["h2_smooth"], g_surf_smooth.flatten()
            )
            holder["h2_smooth_error"] = np.append(
                holder["h2_smooth_error"], e_surf_smooth.flatten()
            )

        # Read magnetic field
        m_path = src.calculate_magnetic_fields.get_path_to_magnetic_map(
            galaxy["name"],
            config["data_directory"],
            galaxy["use_thermal"],
            f"_rebin_13_5as.fits",
        )
        g_magnetic_field = fits.getdata(m_path) * 1e6
        holder["y"] = np.concatenate((holder["y"], g_magnetic_field.flatten()))

        g_m_error_path = src.calculate_magnetic_fields.get_path_to_magnetic_map(
            galaxy["name"],
            config["data_directory"],
            galaxy["use_thermal"],
            f"_abs_error_rebin_13_5as.fits",
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
                file_ending=f"_rebin_13_5as.fits",
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
        xlabel=surf_label,
        output_path=config["data_directory"] + "/surface_density_combined",
        region_mask=None,
        p0=[1, 1],
        x_value=surf_sign,
        x_unit=surf_unit,
        density_map=False,
        extra_line_params=[1, 0.5],
        fit_extra_line=True,
        extra_line_label=rf"$B_{{\mathrm{{eq}} }} \propto {surf_sign}^{{0.5}}$",
    )

    plt_helper.plot_pixel_power_law(
        x=holder["x_smooth"],
        y=holder["y_smooth"],
        z=holder["z_smooth"],
        x_error=holder["x_smooth_error"],
        y_error=holder["y_smooth_error"],
        xlabel=surf_label,
        output_path=config["data_directory"] + "/surface_density_combined_smooth",
        region_mask=None,
        p0=[1, 1],
        x_value=surf_sign,
        x_unit=surf_unit,
        density_map=False,
        extra_line_params=[1, 0.5],
        fit_extra_line=True,
        extra_line_label=rf"$B_{{\mathrm{{eq}} }} \propto {surf_sign}^{{0.5}}$",
    )

    plt_helper.plot_pixel_mean_power_law(
        x=holder["x_mean"],
        y=holder["y_mean"],
        x_std=holder["x_std"],
        y_std=holder["y_std"],
        xlabel=surf_mean_label,
        output_path=config["data_directory"] + "/surface_density_combined_mean",
        p0=[1, 1],
        x_value=surf_sign,
        x_unit=surf_unit,
        extra_line_params=[1, 0.5],
        fit_extra_line=True,
        extra_line_label=rf"$\langle B_{{\mathrm{{eq}} }} \rangle \propto \langle{surf_sign}\rangle^{{0.5}}$",
    )

    # Calculate combined plot
    plt_helper.plot_pixel_power_law(
        x=holder["h2"],
        y=holder["y"],
        z=holder["z"],
        x_error=holder["h2_error"],
        y_error=holder["y_error"],
        xlabel=h2_label,
        output_path=config["data_directory"] + "/h2_surface_density_combined",
        region_mask=None,
        p0=[1, 1],
        x_value=h2_sign,
        x_unit=h2_unit,
        density_map=False,
        extra_line_params=[1, 0.5],
        fit_extra_line=True,
        extra_line_label=rf"$B_{{\mathrm{{eq}} }} \propto {h2_sign}^{{0.5}}$",
    )

    plt_helper.plot_pixel_power_law(
        x=holder["h2_smooth"],
        y=holder["y_smooth"],
        z=holder["z_smooth"],
        x_error=holder["h2_smooth_error"],
        y_error=holder["y_smooth_error"],
        xlabel=h2_label,
        output_path=config["data_directory"] + "/h2_surface_density_combined_smooth",
        region_mask=None,
        p0=[1, 1],
        x_value=h2_sign,
        x_unit=h2_unit,
        density_map=False,
        extra_line_params=[1, 0.5],
        fit_extra_line=True,
        extra_line_label=rf"$B_{{\mathrm{{eq}} }} \propto {h2_sign}^{{0.5}}$",
    )

    plt_helper.plot_pixel_mean_power_law(
        x=holder["h2_mean"],
        y=holder["y_mean"],
        x_std=holder["h2_std"],
        y_std=holder["y_std"],
        xlabel=h2_mean_label,
        output_path=config["data_directory"] + "/h2_surface_density_combined_mean",
        p0=[1, 1],
        x_value=h2_sign,
        x_unit=h2_unit,
        extra_line_params=[1, 0.5],
        fit_extra_line=True,
        extra_line_label=rf"$\langle B_{{\mathrm{{eq}} }} \rangle \propto \langle{h2_sign}\rangle^{{0.5}}$",
    )


def calculate_surface_density(name: str, config: dict, fig=None):
    # "Check" if the specified galaxy exists
    galaxy_config = next(filter(lambda g: g["name"] == name, config["galaxies"],))
    try:
        if not galaxy_config["calc_surf"]:
            raise NotConfiguredException()
        __calculate_surface_density(
            name=galaxy_config["name"],
            data_directory=config["data_directory"],
            thermal=galaxy_config["use_thermal"],
            p0=galaxy_config["co"]["p0"],
            levels=galaxy_config["magnetic_levels"],
            inclination=galaxy_config["inclination"],
            use_integrated_spix=galaxy_config["use_integrated"],
            spix_integrated=galaxy_config["spix"],
            vmin=galaxy_config["co"]["vmin"],
            vmax=galaxy_config["co"]["vmax"],
            rms_h1=galaxy_config["h1"]["rms"],
            rms_co=galaxy_config["co"]["rms"],
            smooth_exp=galaxy_config["smooth_exp"],
        )
    except NotConfiguredException:
        print("Galaxy not configured for Surface Density...")
    return name


def __calculate_surface_density(
    name: str,
    data_directory: str,
    thermal: bool,
    p0: list = None,
    levels: array = None,
    inclination: int = 0,
    use_integrated_spix: bool = False,
    spix_integrated: float = None,
    vmin: float = 0,
    vmax: float = None,
    rms_h1: float = 0,
    rms_co: float = 0,
    smooth_exp: bool = False,
):
    """Calculate and plot correlation between magnetic field strength and surface density
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
        vmin (float): minimum value of the color scale of the overlay
        vmax (float): maximum value of the color scale of the overlay
        rms_h1 (float): rms value for the HI map
        rms_co (float): rms value for the CO map
        smooth_exp (bool): perform the smoothing experiment
    """
    plt_helper.setup_matploblib(False)

    print(
        f"Calculating correlations between magnetic field and surface density for galaxy: {name} with thermal: {thermal}"
    )

    magnetic_path = src.calculate_magnetic_fields.get_path_to_magnetic_map(
        name, data_directory, thermal, ".fits"
    )
    magnetic_rebin_path = src.calculate_magnetic_fields.get_path_to_magnetic_map(
        name, data_directory, thermal, f"_rebin_13_5as.fits"
    )
    magnetic_error_rebin_path = src.calculate_magnetic_fields.get_path_to_magnetic_map(
        name, data_directory, thermal, f"_abs_error_rebin_13_5as.fits"
    )
    h1_path = (
        f"{get_path_to_surface_density_dir(name, data_directory)}/{name}_h1_6as.fits"
    )
    h1_rebin_path = f"{get_path_to_surface_density_dir(name, data_directory)}/{name}_h1_rebin_13_5as.fits"
    co_path = (
        f"{get_path_to_surface_density_dir(name, data_directory)}/{name}_co_6as.fits"
    )
    co_rebin_path = f"{get_path_to_surface_density_dir(name, data_directory)}/{name}_co_rebin_13_5as.fits"

    output_path = get_path_to_surface_density_dir(name, data_directory)

    # make sure that the output_dir exists
    Path(output_path).mkdir(parents=True, exist_ok=True)
    output_path += f"/{name}_surf"

    magnetic_map = fits.open(magnetic_path)
    magnetic_rebin_map = fits.open(magnetic_rebin_path)
    magnetic_error_rebin_map = fits.open(magnetic_error_rebin_path)
    h1_map = fits.open(h1_path)
    h1_rebin_map = fits.open(h1_rebin_path)
    co_map = fits.open(co_path)
    co_rebin_map = fits.open(co_rebin_path)

    magnetic = magnetic_map[0].data
    magnetic_rebin = magnetic_rebin_map[0].data
    magnetic_error_rebin = magnetic_error_rebin_map[0].data
    h1_flux_density = h1_map[0].data[0, 0, :, :]
    h1_flux_density_rebin = h1_rebin_map[0].data[0, 0, :, :]
    h1_flux_density_error_rebin = math_functions.h1_error(h1_flux_density_rebin, rms_h1)
    co_flux_density = co_map[0].data[0, :, :]
    co_flux_density_rebin = co_rebin_map[0].data[0, :, :]
    co_flux_density_error_rebin = math_functions.co_error(co_flux_density_rebin, rms_co)


    # Smoothing experiment
    surface_density_smooth = None
    surface_density_smooth_error = None
    h2_surface_density_smooth = None
    h2_surface_density_smooth_error = None
    if smooth_exp:
        h1_smooth = fits.getdata(
            f"{get_path_to_surface_density_dir(name, data_directory)}/{name}_h1_rebin_13_5as_smooth.fits"
        )
        h1_smooth_error = math_functions.h1_error(h1_smooth, rms_h1)
        co_smooth = fits.getdata(
            f"{get_path_to_surface_density_dir(name, data_directory)}/{name}_co_rebin_13_5as_smooth.fits"
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
        surface_density_smooth = h1_surface_density_smooth + h2_surface_density_smooth
        surface_density_smooth_error = np.sqrt(
            h1_surface_density_smooth_error ** 2 + h2_surface_density_smooth_error ** 2
        )

    spix = np.full(magnetic_rebin.shape, spix_integrated)
    if not use_integrated_spix:
        spix_path = src.calculate_magnetic_fields.get_path_to_spix(
            name, data_directory, thermal, file_ending=f"_rebin_13_5as.fits"
        )
        spix_map = fits.open(spix_path)
        spix = spix_map[0].data

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
    h2_surface_density = math_functions.get_H2_surface_density(
        co_flux_density, inclination
    )
    h2_surface_density_rebin = math_functions.get_H2_surface_density(
        co_flux_density_rebin, inclination
    )
    h2_surface_density_error_rebin = math_functions.get_H2_surface_density_error(
        co_flux_density_error_rebin, inclination
    )

    surface_density = h1_surface_density + h2_surface_density
    surface_density_rebin = h1_surface_density_rebin + h2_surface_density_rebin
    surface_density_error_rebin = np.sqrt(
        h1_surface_density_error_rebin ** 2 + h2_surface_density_error_rebin ** 2
    )

    # Fix broken header
    if h1_map[0].header["CDELT3"] == 0:
        h1_map[0].header["CDELT3"] = 1
        h1_rebin_map[0].header["CDELT3"] = 1

    print("Generating overlay plot...")
    plt_helper.plot_overlay(
        base=surface_density,
        overlay=magnetic * 1e6,
        base_label=surf_label,
        wcs=WCS(h1_map[0].header).celestial,
        output_path=output_path + "_overlay",
        vmin=vmin,
        vmax=vmax,
        levels=levels,
        inline_title="NGC " + name[1:],
    )
    print("Generating H2 overlay plot...")
    plt_helper.plot_overlay(
        base=h2_surface_density,
        overlay=magnetic * 1e6,
        base_label=h2_label,
        wcs=WCS(h1_map[0].header).celestial,
        output_path=output_path + "_h2_overlay",
        vmin=vmin,
        vmax=vmax,
        levels=levels,
        inline_title="NGC " + name[1:],
    )

    print("Generating pixel plot and power law fit...")
    plt_helper.plot_pixel_power_law(
        x=surface_density_rebin.flatten(),
        x_error=surface_density_error_rebin.flatten(),
        # field strength in µG
        y=magnetic_rebin.flatten() * 1e6,
        y_error=magnetic_error_rebin.flatten() * 1e6,
        z=spix.flatten(),
        xlabel=surf_label,
        output_path=output_path + "_pixel",
        p0=p0,
        x_value=surf_sign,
        x_unit=surf_unit,
        extra_line_params=[1, 0.5],
        fit_extra_line=True,
        extra_line_label=rf"$B_{{\mathrm{{eq}} }} \propto {surf_sign}^{{0.5}}$",
        inline_title="NGC " + name[1:],
    )

    print("Generating H2 pixel plot and power law fit...")
    plt_helper.plot_pixel_power_law(
        x=h2_surface_density_rebin.flatten(),
        x_error=h2_surface_density_error_rebin.flatten(),
        # field strength in µG
        y=magnetic_rebin.flatten() * 1e6,
        y_error=magnetic_error_rebin.flatten() * 1e6,
        z=spix.flatten(),
        xlabel=surf_label,
        output_path=output_path + "_h2_pixel",
        p0=p0,
        x_value=surf_sign,
        x_unit=surf_unit,
        extra_line_params=[1, 0.5],
        fit_extra_line=True,
        extra_line_label=rf"$B_{{\mathrm{{eq}} }} \propto {h2_sign}^{{0.5}}$",
        inline_title="NGC " + name[1:],
    )

    if smooth_exp:
        print("Generating smoothed pixel plot and power law fit...")
        plt_helper.plot_pixel_power_law(
            x=surface_density_smooth.flatten(),
            x_error=surface_density_smooth_error.flatten(),
            # field strength in µG
            y=magnetic_rebin.flatten() * 1e6,
            y_error=magnetic_error_rebin.flatten() * 1e6,
            z=spix.flatten(),
            xlabel=surf_label,
            output_path=output_path + "_pixel_smooth",
            p0=p0,
            x_value=surf_sign,
            x_unit=surf_unit,
            extra_line_params=[1, 0.5],
            fit_extra_line=True,
            extra_line_label=rf"$B_{{\mathrm{{eq}} }} \propto {surf_sign}^{{0.5}}$",
            inline_title="NGC " + name[1:],
        )

        print("Generating smoothed H2 pixel plot and power law fit...")
        plt_helper.plot_pixel_power_law(
            x=h2_surface_density_smooth.flatten(),
            x_error=h2_surface_density_smooth_error.flatten(),
            # field strength in µG
            y=magnetic_rebin.flatten() * 1e6,
            y_error=magnetic_error_rebin.flatten() * 1e6,
            z=spix.flatten(),
            xlabel=surf_label,
            output_path=output_path + "_h2_pixel_smooth",
            p0=p0,
            x_value=surf_sign,
            x_unit=surf_unit,
            extra_line_params=[1, 0.5],
            fit_extra_line=True,
            extra_line_label=rf"$B_{{\mathrm{{eq}} }} \propto {h2_sign}^{{0.5}}$",
            inline_title="NGC " + name[1:],
        )

    # Save surface density maps
    output_hdu = fits.PrimaryHDU(surface_density)
    output_hdu.header.update(WCS(h1_map[0].header).to_header())
    output_hdul = fits.HDUList(output_hdu)
    output_hdul.writeto(
        f"{get_path_to_surface_density_dir(name, data_directory)}/{name}_surface_density_6as.fits",
        overwrite=True,
    )
    print(
        "Out:",
        f"{get_path_to_surface_density_dir(name, data_directory)}/{name}_surface_density_6as.fits",
    )

    output_hdu = fits.PrimaryHDU(surface_density_rebin)
    output_hdu.header.update(WCS(h1_rebin_map[0].header).to_header())
    output_hdul = fits.HDUList(output_hdu)
    output_hdul.writeto(
        f"{get_path_to_surface_density_dir(name, data_directory)}/{name}_surface_density_13_5as_rebin.fits",
        overwrite=True,
    )
    print(
        "Out:",
        f"{get_path_to_surface_density_dir(name, data_directory)}/{name}_surface_density_13_5as_rebin.fits",
    )

    if smooth_exp:
        output_hdu = fits.PrimaryHDU(surface_density_smooth)
        output_hdu.header.update(WCS(h1_rebin_map[0].header).to_header())
        output_hdul = fits.HDUList(output_hdu)
        output_hdul.writeto(
            f"{get_path_to_surface_density_dir(name, data_directory)}/{name}_surface_density_13_5as_rebin_smooth.fits",
            overwrite=True,
        )
        print(
            "Out:",
            f"{get_path_to_surface_density_dir(name, data_directory)}/{name}_surface_density_13_5as_rebin_smooth.fits",
        )

        output_hdu = fits.PrimaryHDU(surface_density_smooth_error)
        output_hdu.header.update(WCS(h1_rebin_map[0].header).to_header())
        output_hdul = fits.HDUList(output_hdu)
        output_hdul.writeto(
            f"{get_path_to_surface_density_dir(name, data_directory)}/{name}_surface_density_13_5as_error_rebin_smooth.fits",
            overwrite=True,
        )
        print(
            "Out:",
            f"{get_path_to_surface_density_dir(name, data_directory)}/{name}_surface_density_13_5as_error_rebin.fits",
        )

    output_hdu = fits.PrimaryHDU(surface_density_error_rebin)
    output_hdu.header.update(WCS(h1_rebin_map[0].header).to_header())
    output_hdul = fits.HDUList(output_hdu)
    output_hdul.writeto(
        f"{get_path_to_surface_density_dir(name, data_directory)}/{name}_surface_density_13_5as_error_rebin.fits",
        overwrite=True,
    )
    print(
        "Out:",
        f"{get_path_to_surface_density_dir(name, data_directory)}/{name}_surface_density_13_5as_error_rebin.fits",
    )

    output_hdu = fits.PrimaryHDU(h2_surface_density_rebin)
    output_hdu.header.update(WCS(h1_rebin_map[0].header).to_header())
    output_hdul = fits.HDUList(output_hdu)
    output_hdul.writeto(
        f"{get_path_to_surface_density_dir(name, data_directory)}/{name}_h2_surface_density_13_5as_rebin.fits",
        overwrite=True,
    )
    print(
        "Out:",
        f"{get_path_to_surface_density_dir(name, data_directory)}/{name}_h2_surface_density_13_5as_rebin.fits",
    )

    output_hdu = fits.PrimaryHDU(h2_surface_density_error_rebin)
    output_hdu.header.update(WCS(h1_rebin_map[0].header).to_header())
    output_hdul = fits.HDUList(output_hdu)
    output_hdul.writeto(
        f"{get_path_to_surface_density_dir(name, data_directory)}/{name}_h2_surface_density_13_5as_error_rebin.fits",
        overwrite=True,
    )
    print(
        "Out:",
        f"{get_path_to_surface_density_dir(name, data_directory)}/{name}_h2_surface_density_13_5as_error_rebin.fits",
    )

    if smooth_exp:
        output_hdu = fits.PrimaryHDU(h2_surface_density_smooth)
        output_hdu.header.update(WCS(h1_rebin_map[0].header).to_header())
        output_hdul = fits.HDUList(output_hdu)
        output_hdul.writeto(
            f"{get_path_to_surface_density_dir(name, data_directory)}/{name}_h2_surface_density_13_5as_rebin_smooth.fits",
            overwrite=True,
        )
        print(
            "Out:",
            f"{get_path_to_surface_density_dir(name, data_directory)}/{name}_h2_surface_density_13_5as_rebin_smooth.fits",
        )

        output_hdu = fits.PrimaryHDU(h2_surface_density_smooth_error)
        output_hdu.header.update(WCS(h1_rebin_map[0].header).to_header())
        output_hdul = fits.HDUList(output_hdu)
        output_hdul.writeto(
            f"{get_path_to_surface_density_dir(name, data_directory)}/{name}_h2_surface_density_13_5as_error_rebin_smooth.fits",
            overwrite=True,
        )
        print(
            "Out:",
            f"{get_path_to_surface_density_dir(name, data_directory)}/{name}_h2_surface_density_13_5as_error_rebin_smooth.fits",
        )
    return name


def get_path_to_surface_density_dir(name: str, data_directory: str) -> str:
    """Get the path to the directory where the star formation data should be stored

    Args:
        name (str): Name of the galaxy
        data_directory (str): dr2 data directory

    Returns:
        str: Path to h1 dir
    """
    return f"{data_directory}/surf/{name}"
