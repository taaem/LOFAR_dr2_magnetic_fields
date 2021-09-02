import array
import shutil
from pathlib import Path

import src.calculate_energy_density as energy_density
import src.calculate_h1 as h1_func
import src.calculate_magnetic_fields as magnetic
import src.calculate_radio_sfr as radio_sfr
import src.calculate_sfr as sfr
import src.calculate_surface_density as surf
import src.helper as helper


def generate_all_scripts(galaxy: dict, config: dict, has_magnetic: bool):
    """Generate all scripts for one galaxy

    Args:
        galaxy (dict): Galaxy Config
        config (dict): Config
    """

    if not has_magnetic:
        copy_base_files(
            galaxy["name"],
            config["data_directory"],
            galaxy["ref_name"],
            galaxy["thermal_name"],
            galaxy["use_integrated"],
        )

        if not galaxy["use_integrated"]:
            regrid_spix(galaxy["name"], config["data_directory"])

        generate_magnetic_script_before(
            galaxy["name"],
            galaxy["rms_6"],
            galaxy["rms_20"],
            config["data_directory"],
        )
        if galaxy["use_thermal"]:
            generate_magnetic_scripts(
                galaxy["name"],
                galaxy["rms_20"],
                galaxy["rms_ref"],
                config["data_directory"],
                galaxy["ref_name"],
                galaxy["ref_freq"],
                galaxy["thermal_name"],
                galaxy["thermal_freq"],
                galaxy["thermal_beam"],
                galaxy["ref_beam"],
            )
    else:
        generate_after_magnetic_script(
            galaxy["name"],
            config["data_directory"],
            galaxy["use_thermal"],
            galaxy["use_integrated"],
        )
        if galaxy["calc_sfr"]:
            generate_sfr_script(
                galaxy["name"],
                config["data_directory"],
                galaxy["use_thermal"],
                galaxy["sfr"]["rms"],
                galaxy["smooth_exp"],
                galaxy["smooth_length"],
                galaxy["distance"],
                0.5,
            )
            generate_radio_sfr_script(
                galaxy["name"],
                config["data_directory"],
                galaxy["use_thermal"],
                galaxy["sfr"]["rms"],
                galaxy[f"rms_6"],
                galaxy["smooth_exp"],
                galaxy["smooth_length"],
                galaxy["distance"],
                0.5,
            )
        if galaxy["calc_h1"]:
            generate_h1_script(
                galaxy["name"],
                config["data_directory"],
                galaxy["use_thermal"],
                galaxy["h1"]["rms"],
                galaxy["smooth_exp"],
                galaxy["smooth_length"],
                galaxy["distance"],
                0.5,
            )
        if galaxy["calc_surf"]:
            generate_surf_script(
                galaxy["name"],
                config["data_directory"],
                galaxy["use_thermal"],
                galaxy["h1"]["rms"],
                galaxy["co"]["rms"],
                galaxy["smooth_exp"],
                galaxy["smooth_length"],
                galaxy["distance"],
                0.5,
            )
        if galaxy["calc_energy"]:
            generate_energy_density_script(
                galaxy["name"],
                config["data_directory"],
                galaxy["use_thermal"],
                galaxy["h1"]["rms"],
                galaxy["calc_surf"],
                galaxy["co"]["rms"] if galaxy["calc_surf"] else None,
                galaxy["smooth_exp"],
                galaxy["smooth_length"],
                galaxy["distance"],
                0.5,
            )


def run_all_scripts(galaxy: dict, config: dict, has_magnetic: bool):
    """Run all Casa scripts for one galaxy

    Args:
        galaxy (dict): Galaxy Config
        config (dict): Config
        has_magnetic (bool): Was the magnetic code already run?
    """
    if not has_magnetic:
        if not galaxy["use_integrated"]:
            helper.run_command_in_dir(
                [config["casa_executable"], "--nogui", "-c", "casa_regrid_spix.py",],
                helper.get_magnetic_galaxy_dir(
                    galaxy["name"], config["data_directory"]
                ),
            )

        helper.run_command_in_dir(
            [config["casa_executable"], "--nogui", "-c", "casa_cmd_before.py",],
            helper.get_magnetic_galaxy_dir(galaxy["name"], config["data_directory"]),
        )

        if galaxy["use_thermal"]:
            helper.run_command_in_dir(
                [config["casa_executable"], "--nogui", "-c", "casa_cmd.py",],
                helper.get_magnetic_galaxy_dir(
                    galaxy["name"], config["data_directory"]
                ),
            )
    else:
        helper.run_command_in_dir(
            [config["casa_executable"], "--nogui", "-c", "casa_cmd_after.py",],
            helper.get_magnetic_galaxy_dir(galaxy["name"], config["data_directory"]),
        )
        if galaxy["calc_sfr"]:
            helper.run_command_in_dir(
                [config["casa_executable"], "--nogui", "-c", "casa_cmd.py",],
                sfr.get_path_to_sfr_dir(galaxy["name"], config["data_directory"]),
            )
            helper.run_command_in_dir(
                [config["casa_executable"], "--nogui", "-c", "casa_cmd.py",],
                radio_sfr.get_path_to_radio_sfr_dir(
                    galaxy["name"], config["data_directory"]
                ),
            )
        if galaxy["calc_h1"]:
            helper.run_command_in_dir(
                [config["casa_executable"], "--nogui", "-c", "casa_cmd.py",],
                h1_func.get_path_to_h1_dir(galaxy["name"], config["data_directory"]),
            )
        if galaxy["calc_surf"]:
            helper.run_command_in_dir(
                [config["casa_executable"], "--nogui", "-c", "casa_cmd.py",],
                surf.get_path_to_surface_density_dir(
                    galaxy["name"], config["data_directory"]
                ),
            )
        if galaxy["calc_energy"]:
            helper.run_command_in_dir(
                [config["casa_executable"], "--nogui", "-c", "casa_cmd.py",],
                energy_density.get_path_to_energy_density_dir(
                    galaxy["name"], config["data_directory"]
                ),
            )


def copy_base_files(
    name: str,
    data_directory: str,
    ref_name: str,
    thermal_name: str,
    use_integrated_spix: bool,
):
    """Copy the basic files into the magnetic galaxy directories

    Args:
        name (str): name of the galaxy
        data_directory (str): path to the data directory
        ref_name (str): name of the fits file for the reference image
        thermal_name (str): name of the fits file for the thermal image
        use_integrated_spix (bool): does this galaxy use the integrated spectral index or a map?
    """
    print(f"Copying files for {name}")
    galaxy_dir = helper.get_magnetic_galaxy_dir(name, data_directory)
    # Make directory if it doesn't exist
    Path(galaxy_dir).mkdir(parents=True, exist_ok=True)

    galaxy_number = helper.get_galaxy_number_from_name(name)

    # Copy Reference image to working directory, if file exists don't abort
    shutil.copy(
        f"{data_directory}/cutouts/6as/n{galaxy_number}_144mhz_6as.fits", galaxy_dir
    )
    shutil.copy(
        f"{data_directory}/cutouts/20as/n{galaxy_number}_144mhz_20as.fits", galaxy_dir,
    )

    if not use_integrated_spix:
        shutil.copy(f"{data_directory}/spix/n{galaxy_number}_spix.fits", galaxy_dir)
        shutil.copy(
            f"{data_directory}/spix/n{galaxy_number}_spix_error.fits", galaxy_dir
        )

    shutil.copy(
        f"{data_directory}/regions/high/n{galaxy_number}_flux_elliptical_high.reg",
        galaxy_dir,
    )
    shutil.copy(
        f"{data_directory}/regions/low/n{galaxy_number}_flux_elliptical_low.reg",
        galaxy_dir,
    )
    if thermal_name:
        shutil.copy(
            f"{data_directory}/ancillary/thermal/{thermal_name}.fits", galaxy_dir,
        )
    if ref_name:
        shutil.copy(
            f"{data_directory}/ancillary/reference/{ref_name}.fits", galaxy_dir,
        )


def regrid_spix(name: str, data_directory: str):
    """Regridding the spectral index onto the coordinate system of the 6" map

    Args:
        name (str): name of the galaxy
        data_directory (str): path to the data directory
    """
    print(f"Generating Casa regrid spix script for {name}")
    galaxy_dir = helper.get_magnetic_galaxy_dir(name, data_directory)

    galaxy_number = helper.get_galaxy_number_from_name(name)

    with open(galaxy_dir + "/casa_regrid_spix.py", "w") as casa_file:
        # Import 20as spectral index map / spectral index error map
        casa_file.write(
            f"importfits('{name}_spix.fits', '{name}_spix.image', overwrite=True) \n"
        )
        casa_file.write(
            f"importfits('{name}_spix_error.fits', '{name}_spix_error.image', overwrite=True) \n"
        )
        # Import 6as intensity map
        casa_file.write(
            f"importfits('n{galaxy_number}_144mhz_6as.fits', 'n{galaxy_number}_144mhz_6as.image', overwrite=True) \n"
        )
        # regrid 20as image onto coordinate system of 6as image
        casa_file.write(
            f"imregrid('{name}_spix.image', 'n{galaxy_number}_144mhz_6as.image', '{name}_spix_6as.image', asvelocity=False, interpolation='cubic', overwrite=True) \n"
        )
        casa_file.write(
            f"imregrid('{name}_spix_error.image', 'n{galaxy_number}_144mhz_6as.image', '{name}_spix_error_6as.image', asvelocity=False, interpolation='cubic', overwrite=True) \n"
        )
        # export new regridded 20as image
        casa_file.write(
            f"exportfits('{name}_spix_6as.image', '{name}_spix_6as.fits', dropdeg=True, overwrite=True) \n"
        )
        # export new regridded 20as image
        casa_file.write(
            f"exportfits('{name}_spix_error_6as.image', '{name}_spix_error_6as.fits', dropdeg=True, overwrite=True) \n"
        )


def generate_magnetic_scripts(
    name: str,
    rms: float,
    rms_ref: float,
    data_directory: str,
    ref_name: str,
    ref_freq: float,
    thermal_name: str,
    thermal_freq: float,
    thermal_beam: array,
    ref_beam: array,
):
    """Generate a Casa script to subtract the thermal emission from the LOFAR cutout
    Alse regenerate the spectral index without the thermal emission

    Args:
        name (str): Name of the galaxy
        rms (float): RMS value for the LOFAR map
        rms_ref (float): RMS value for the Reference 1395MHz map
        data_directory (str): dr2 data directory
        ref_name (str): name of the fits file of the reference map
        ref_freq (float): frequency of the reference map in [MHz]
        thermal_name (str): name of the fits file of the thermal map
        thermal_freq (float): frequency of the thermal map in [MHz]
        thermal_beam (array): beam size of the thermal map in [arcsec, arcsec]
        ref_beam (array): beam size of the reference map in [arcsec, arcsec]
    """
    print(f"Generating Casa scripts for {name}")
    galaxy_dir = helper.get_magnetic_galaxy_dir(name, data_directory)

    galaxy_number = helper.get_galaxy_number_from_name(name)

    with open(galaxy_dir + "/casa_cmd.py", "w") as casa_file:
        casa_file.write(
            f'importfits("{thermal_name}.fits", "{thermal_name}.image", overwrite=True) \n'
        )
        casa_file.write(
            f'importfits("{ref_name}.fits", "{ref_name}.image", overwrite=True) \n'
        )

        casa_file.write("\n")

        # regrid 15as image onto coordinate system of 6as image
        casa_file.write(
            f'imregrid("{thermal_name}.image", "n{galaxy_number}_144mhz_6as.image", "{thermal_name}_regrid_6as.image", asvelocity=False, interpolation="cubic", overwrite=True) \n'
        )
        # regrid 15as image onto coordinate system of 20as image
        casa_file.write(
            f'imregrid("{thermal_name}.image", "n{galaxy_number}_144mhz_20as.image", "{thermal_name}_regrid_20as.image", asvelocity=False, interpolation="cubic", overwrite=True) \n'
        )
        thermal_20as_name = f"{thermal_name}_regrid_20as.image"

        casa_file.write("\n")

        # Subtract thermal emission from image
        casa_file.write(
            f'immath(imagename=["n{galaxy_number}_144mhz_6as.image", "{thermal_name}_regrid_6as.image"], mode="evalexpr", expr="IM0 - 6^2/({thermal_beam[0] * thermal_beam[1]}) * (144/{thermal_freq})^(-0.1) * IM1", outfile="n{galaxy_number}_144mhz_non_thermal_6as.image") \n'
        )

        # Subtract thermal emission from 20as image, since we didn't
        # smooth, calculate beam difference
        casa_file.write(
            f'immath(imagename=["n{galaxy_number}_144mhz_20as.image", "{thermal_20as_name}"], mode="evalexpr", expr="IM0 - 20^2/({thermal_beam[0] * thermal_beam[1]}) * (144/{thermal_freq})^(-0.1) * IM1", outfile="n{galaxy_number}_144mhz_non_thermal_20as.image") \n'
        )

        casa_file.write("\n")

        # Export Fits files
        casa_file.write(
            f'exportfits("n{galaxy_number}_144mhz_non_thermal_6as.image", "n{galaxy_number}_144mhz_non_thermal_6as.fits", overwrite=True) \n'
        )
        casa_file.write(
            f'exportfits("n{galaxy_number}_144mhz_non_thermal_20as.image", "n{galaxy_number}_144mhz_non_thermal_20as.fits", overwrite=True) \n'
        )

        casa_file.write("\n")

        # Regrid and smooth reference image
        casa_file.write(
            f'imregrid("{ref_name}.image", "n{galaxy_number}_144mhz_20as.image", "{ref_name}_regrid_20as.image", asvelocity=False, interpolation="cubic", overwrite=True) \n'
        )

        # Copy mask to non-thermal image
        casa_file.write(f"ia.open('n{galaxy_number}_144mhz_non_thermal_20as.image') \n")
        casa_file.write(
            f"ia.maskhandler('copy', ['n{galaxy_number}_144mhz_20as.image:n{galaxy_number}_144mhz_20as.mask', 'n{galaxy_number}_144mhz_20as.mask']) \n"
        )
        casa_file.write(
            f"ia.maskhandler('set', 'n{galaxy_number}_144mhz_20as.mask') \n"
        )
        casa_file.write(f"ia.done() \n")
        casa_file.write("\n")

        # Mask the 1365MHz image in 3sigma
        casa_file.write(f"ia.open('{ref_name}_regrid_20as.image') \n")
        casa_file.write(
            f"ia.calcmask(mask='{ref_name}_regrid_20as.image > {3 * rms_ref}', name='{ref_name}_regrid_20as.mask') \n"
        )
        casa_file.write(f"ia.done() \n")
        casa_file.write("\n")

        # Calculate the non thermal component of the reference image
        casa_file.write(
            f"immath(imagename=['{ref_name}_regrid_20as.image', '{thermal_20as_name}'], mode='evalexpr', expr='IM0 - ({ref_beam[0] * ref_beam[1]} / {thermal_beam[0] * thermal_beam[1]}) * ({ref_freq}/{thermal_freq})^(-0.1) * IM1', outfile='{ref_name}_regrid_non_thermal_20as.image') \n"
        )
        casa_file.write("\n")

        # Calculate spectral index
        casa_file.write(
            f"immath(imagename=['n{galaxy_number}_144mhz_non_thermal_20as.image', '{ref_name}_regrid_non_thermal_20as.image'], mode='evalexpr', expr='log((IM1 * 20^2/{ref_beam[0] * ref_beam[1]})/IM0)/log({ref_freq}/144)', outfile='n{galaxy_number}_spix_non_thermal_20as.image') \n"
        )
        casa_file.write(
            f"immath(imagename=['n{galaxy_number}_144mhz_non_thermal_20as.image', '{ref_name}_regrid_non_thermal_20as.image'], mode='evalexpr', expr='sqrt(({rms}/IM0)^2 + ({rms_ref}/IM1)^2 + 2*0.05^2)/log({ref_freq}/144)', outfile='n{galaxy_number}_spix_non_thermal_20as_error.image') \n"
        )
        casa_file.write("\n")

        # Regrid for 6as
        casa_file.write(
            f"imregrid('n{galaxy_number}_spix_non_thermal_20as.image', 'n{galaxy_number}_144mhz_6as.image', 'n{galaxy_number}_spix_non_thermal_6as.image', asvelocity=False, interpolation='cubic', overwrite=True) \n"
        )
        casa_file.write(
            f"imregrid('n{galaxy_number}_spix_non_thermal_20as_error.image', 'n{galaxy_number}_144mhz_6as.image', 'n{galaxy_number}_spix_non_thermal_6as_error.image', asvelocity=False, interpolation='cubic', overwrite=True) \n"
        )
        casa_file.write("\n")

        # Export
        casa_file.write(
            f"exportfits('n{galaxy_number}_spix_non_thermal_20as.image', 'n{galaxy_number}_spix_non_thermal_20as.fits', overwrite=True, dropdeg=True) \n"
        )
        casa_file.write(
            f"exportfits('n{galaxy_number}_spix_non_thermal_20as_error.image', 'n{galaxy_number}_spix_non_thermal_error_20as.fits', overwrite=True, dropdeg=True) \n"
        )
        casa_file.write(
            f"exportfits('n{galaxy_number}_spix_non_thermal_6as.image', 'n{galaxy_number}_spix_non_thermal_6as.fits', overwrite=True, dropdeg=True) \n"
        )
        casa_file.write(
            f"exportfits('n{galaxy_number}_spix_non_thermal_6as_error.image', 'n{galaxy_number}_spix_non_thermal_error_6as.fits', overwrite=True, dropdeg=True) \n"
        )
        casa_file.write("\n")


def generate_magnetic_script_before(
    name: str, rms_6: float, rms_20: float, data_directory: str,
):
    """Generate a Casa script to subtract the thermal emission from the LOFAR cutout
    Also regenerate the spectral index without the thermal emission

    Args:
        name (str): Name of the galaxy
        rms_6 (float): RMS value for the 6" LOFAR map
        rms_20 (float): RMS value for the 20" LOFAR map
        data_directory (str): dr2 data directory
    """
    print(f"Generating Casa scripts for {name}")
    galaxy_dir = helper.get_magnetic_galaxy_dir(name, data_directory)

    galaxy_number = helper.get_galaxy_number_from_name(name)

    with open(galaxy_dir + "/casa_cmd_before.py", "w") as casa_file:
        # Import all files
        casa_file.write(
            f'importfits("n{galaxy_number}_144mhz_6as.fits", "n{galaxy_number}_144mhz_6as.image", overwrite=True) \n'
        )
        casa_file.write(
            f'importfits("n{galaxy_number}_144mhz_20as.fits", "n{galaxy_number}_144mhz_20as.image", overwrite=True) \n'
        )

        casa_file.write("\n")

        # Mask the LOFAR 20" image in 3sigma
        casa_file.write(f"ia.open('n{galaxy_number}_144mhz_20as.image') \n")
        casa_file.write(
            f"ia.calcmask(mask='n{galaxy_number}_144mhz_20as.image > {3 * rms_20}', name='n{galaxy_number}_144mhz_20as.mask') \n"
        )
        casa_file.write(f"ia.done() \n")
        casa_file.write("\n")

        # Mask the LOFAR 6" image in 3sigma
        casa_file.write(f"ia.open('n{galaxy_number}_144mhz_6as.image') \n")
        casa_file.write(
            f"ia.calcmask(mask='n{galaxy_number}_144mhz_6as.image > {3 * rms_6}', name='n{galaxy_number}_144mhz_6as.mask') \n"
        )
        casa_file.write(f"ia.done() \n")
        casa_file.write("\n")

        # Export
        casa_file.write(
            f"exportfits('n{galaxy_number}_144mhz_6as.image', 'n{galaxy_number}_144mhz_6as.fits', overwrite=True) \n"
        )
        casa_file.write(
            f"exportfits('n{galaxy_number}_144mhz_20as.image', 'n{galaxy_number}_144mhz_20as.fits', overwrite=True) \n"
        )
        casa_file.write("\n")


def generate_after_magnetic_script(
    name: str, data_directory: str, thermal: bool, has_integrated_spix: bool,
):
    """Generate script to rebin magnetic field to calculate correlations
    and regrid the spectral index onto the magnetic field

    Args:
        name (str): Name of the galaxy
        data_directory (str): dr2 data directory
        cmd_name (str): casa command name
        thermal (bool): thermal data
    """
    galaxy_dir = helper.get_magnetic_galaxy_dir(name, data_directory)

    magnetic_filename = magnetic.get_path_to_magnetic_map(
        name, data_directory, thermal, ""
    )
    magnetic_error_filename = magnetic.get_path_to_magnetic_map(
        name, data_directory, thermal, "_abs_error"
    )
    spix_filename = magnetic.get_path_to_spix(
        name, data_directory, thermal, file_ending=""
    )

    with open(galaxy_dir + "/casa_cmd_after.py", "w") as casa_file:
        casa_file.write(
            f"importfits('{magnetic_filename}.fits', '{magnetic_filename}.image', overwrite=True) \n"
        )
        casa_file.write(
            f"importfits('{magnetic_error_filename}.fits', '{magnetic_error_filename}.image', overwrite=True) \n"
        )
        if not has_integrated_spix:
            casa_file.write(
                f"importfits('{spix_filename}_6as.fits', '{spix_filename}_6as.image', overwrite=True) \n"
            )

        casa_file.write(
            f"imrebin('{magnetic_filename}.image', factor=[4,4], outfile='{magnetic_filename}_rebin_6as.image', overwrite=True) \n"
        )
        casa_file.write(
            f"imrebin('{magnetic_error_filename}.image', factor=[4,4], outfile='{magnetic_error_filename}_rebin_6as.image', overwrite=True) \n"
        )

        # Rebin for 13as SFR maps (imrebin only takes integers, instead of 13as
        # we rebin to 13.5as)
        casa_file.write(
            f"imrebin('{magnetic_filename}.image', factor=[9,9], outfile='{magnetic_filename}_rebin_13_5as.image', overwrite=True) \n"
        )
        casa_file.write(
            f"imrebin('{magnetic_error_filename}.image', factor=[9,9], outfile='{magnetic_error_filename}_rebin_13_5as.image', overwrite=True) \n"
        )
        # Rebin for 11as CO maps (imrebin only takes integers, instead of 11as
        # we rebin to 12as)
        casa_file.write(
            f"imrebin('{magnetic_filename}.image', factor=[8,8], outfile='{magnetic_filename}_rebin_12as.image', overwrite=True) \n"
        )
        casa_file.write(
            f"imrebin('{magnetic_error_filename}.image', factor=[8,8], outfile='{magnetic_error_filename}_rebin_12as.image', overwrite=True) \n"
        )

        if not has_integrated_spix:
            casa_file.write(
                f"imregrid('{spix_filename}_6as.image', '{magnetic_filename}_rebin_6as.image', '{spix_filename}_rebin_6as.image', asvelocity=False, interpolation='cubic', overwrite=True) \n"
            )
            casa_file.write(
                f"exportfits('{spix_filename}_rebin_6as.image', '{spix_filename}_rebin_6as.fits', overwrite=True, dropdeg=True) \n"
            )
            casa_file.write(
                f"imregrid('{spix_filename}_6as.image', '{magnetic_filename}_rebin_13_5as.image', '{spix_filename}_rebin_13_5as.image', asvelocity=False, interpolation='cubic', overwrite=True) \n"
            )
            casa_file.write(
                f"imregrid('{spix_filename}_6as.image', '{magnetic_filename}_rebin_12as.image', '{spix_filename}_rebin_12as.image', asvelocity=False, interpolation='cubic', overwrite=True) \n"
            )
            casa_file.write(
                f"exportfits('{spix_filename}_rebin_13_5as.image', '{spix_filename}_rebin_13_5as.fits', overwrite=True, dropdeg=True) \n"
            )
            casa_file.write(
                f"exportfits('{spix_filename}_rebin_12as.image', '{spix_filename}_rebin_12as.fits', overwrite=True, dropdeg=True) \n"
            )

        casa_file.write(
            f"exportfits('{magnetic_filename}_rebin_6as.image', '{magnetic_filename}_rebin_6as.fits', overwrite=True, dropdeg=True) \n"
        )
        casa_file.write(
            f"exportfits('{magnetic_filename}_rebin_13_5as.image', '{magnetic_filename}_rebin_13_5as.fits', overwrite=True, dropdeg=True) \n"
        )
        casa_file.write(
            f"exportfits('{magnetic_filename}_rebin_12as.image', '{magnetic_filename}_rebin_12as.fits', overwrite=True, dropdeg=True) \n"
        )

        casa_file.write(
            f"exportfits('{magnetic_error_filename}_rebin_6as.image', '{magnetic_error_filename}_rebin_6as.fits', overwrite=True, dropdeg=True) \n"
        )
        casa_file.write(
            f"exportfits('{magnetic_error_filename}_rebin_13_5as.image', '{magnetic_error_filename}_rebin_13_5as.fits', overwrite=True, dropdeg=True) \n"
        )
        casa_file.write(
            f"exportfits('{magnetic_error_filename}_rebin_12as.image', '{magnetic_error_filename}_rebin_12as.fits', overwrite=True, dropdeg=True) \n"
        )


def generate_sfr_script(
    name: str,
    data_directory: str,
    thermal: bool,
    rms: float,
    smooth_exp: bool,
    smooth_length: float,
    distance: float,
    pix_per_as: float,
):
    """Generate a Casa script for regridding the SFR map to the magnetic field map, so the data can
    be compared later

    Args:
        name (str): Name of the galaxy
        data_directory (str): dr2 data directory
        thermal (bool): Use non thermal magnetic field data
        rms (float): RMS value for mask for binned image
        smooth_exp (bool): make the smoothing experiment
        smooth_length (float): the diffusion/smoothing length in [kpc]
        distance (float): distance to the galaxy in [Mpc]
        pix_per_as (float): pixel per arcsecond for the LOFAR map
    """

    galaxy_number = helper.get_galaxy_number_from_name(name)

    sfr_path = sfr.get_path_to_sfr_dir(name, data_directory)
    magnetic_path = magnetic.get_path_to_magnetic_map(
        name, data_directory, thermal, ".fits"
    )
    magnetic_rebin_path = magnetic.get_path_to_magnetic_map(
        name, data_directory, thermal, "_rebin_13_5as.fits"
    )

    # Get name without file endings
    magnetic_filename = magnetic_path.split("/")[-1].split(".", 1)[0]
    magnetic_rebin_filename = magnetic_rebin_path.split("/")[-1].split(".", 1)[0]

    sfr_input_path = (
        f"{data_directory}/ancillary/sfr_fuv+24/ngc{galaxy_number}_fuv+24_res_13as.fits"
    )
    sfr_input_filename = sfr_input_path.split("/")[-1]
    sfr_smoothed_filename = f"n{galaxy_number}_fuv+24_res_13as_smooth.fits"

    # Create SFR dir for this galaxy
    Path(sfr_path).mkdir(parents=True, exist_ok=True)

    try:
        shutil.copyfile(sfr_input_path, sfr_path + "/" + sfr_input_filename)
    except FileExistsError:
        print(f"file already exists...")

    if smooth_exp:
        helper.smooth(
            sfr_path + "/" + sfr_input_filename,
            smooth_length,
            distance,
            pix_per_as,
            sfr_path + "/" + sfr_smoothed_filename,
        )

    with open(sfr_path + "/casa_cmd.py", "w") as casa_file:
        casa_file.write(
            f'importfits("{sfr_input_filename}", "ngc{galaxy_number}_fuv+24_res_13as.image", overwrite=True) \n'
        )
        if smooth_exp:
            casa_file.write(
                f'importfits("{sfr_smoothed_filename}", "ngc{galaxy_number}_fuv+24_res_13as_smooth.image", overwrite=True) \n'
            )
        casa_file.write(
            f'importfits("{magnetic_path}", "{magnetic_filename}.image", overwrite=True) \n'
        )
        casa_file.write(
            f'imregrid("ngc{galaxy_number}_fuv+24_res_13as.image", "{magnetic_filename}.image", "ngc{galaxy_number}_fuv+24_res_6as_regrid.image", asvelocity=False, interpolation="cubic", overwrite=True) \n'
        )
        casa_file.write(
            f'imrebin("ngc{galaxy_number}_fuv+24_res_6as_regrid.image", factor=[9, 9], outfile="ngc{galaxy_number}_fuv+24_res_13_5as_rebin_regrid.image", overwrite=True) \n'
        )
        if smooth_exp:
            casa_file.write(
                f'imregrid("ngc{galaxy_number}_fuv+24_res_13as_smooth.image", "{magnetic_filename}.image", "ngc{galaxy_number}_fuv+24_res_6as_smooth_regrid.image", asvelocity=False, interpolation="cubic", overwrite=True) \n'
            )
            casa_file.write(
                f'imrebin("ngc{galaxy_number}_fuv+24_res_6as_smooth_regrid.image", factor=[9, 9], outfile="ngc{galaxy_number}_fuv+24_res_13_5as_smooth_rebin_regrid.image", overwrite=True) \n'
            )

        # Mask the rebinned image in 3sigma
        casa_file.write(
            f"ia.open('ngc{galaxy_number}_fuv+24_res_13_5as_rebin_regrid.image') \n"
        )
        casa_file.write(
            f"ia.calcmask(mask='\"ngc{galaxy_number}_fuv+24_res_13_5as_rebin_regrid.image\" > {3 * rms}', name='ngc{galaxy_number}_fuv+24_res_13_5as_rebin_regrid.mask') \n"
        )
        casa_file.write(f"ia.done() \n")
        casa_file.write("\n")

        if smooth_exp:
            # Mask the rebinned and smoothed image in 3sigma
            casa_file.write(
                f"ia.open('ngc{galaxy_number}_fuv+24_res_13_5as_smooth_rebin_regrid.image') \n"
            )
            casa_file.write(
                f"ia.calcmask(mask='\"ngc{galaxy_number}_fuv+24_res_13_5as_smooth_rebin_regrid.image\" > {3 * rms}', name='ngc{galaxy_number}_fuv+24_res_13_5as_smooth_rebin_regrid.mask') \n"
            )
            casa_file.write(f"ia.done() \n")
            casa_file.write("\n")

        casa_file.write(
            f'exportfits("ngc{galaxy_number}_fuv+24_res_6as_regrid.image", "n{galaxy_number}_sfr_6as.fits", overwrite=True) \n'
        )
        casa_file.write(
            f'exportfits("ngc{galaxy_number}_fuv+24_res_13_5as_rebin_regrid.image", "n{galaxy_number}_sfr_rebin_13_5as.fits", overwrite=True) \n'
        )
        if smooth_exp:
            casa_file.write(
                f'exportfits("ngc{galaxy_number}_fuv+24_res_6as_smooth_regrid.image", "n{galaxy_number}_sfr_6as_smooth.fits", overwrite=True) \n'
            )
            casa_file.write(
                f'exportfits("ngc{galaxy_number}_fuv+24_res_13_5as_smooth_rebin_regrid.image", "n{galaxy_number}_sfr_rebin_13_5as_smooth.fits", overwrite=True) \n'
            )


def generate_radio_sfr_script(
    name: str,
    data_directory: str,
    thermal: bool,
    rms_sfr: float,
    rms_radio: float,
    smooth_exp: bool,
    smooth_length: float,
    distance: float,
    pix_per_as: float,
):
    """Generate a Casa script for regridding the RC - SFR relation maps


    Args:
        name (str): Name of the galaxy
        data_directory (str): dr2 data directory
        rms (float): RMS value for mask for binned image
        thermal (bool): use thermal corrected magnetic maps
        rms_sfr (float): rms value for the sfr map
        rms_radio (float): rms value for the LOFAR 6" map
        smooth_exp (bool): make the smoothing experiment
        smooth_length (float): the diffusion/smoothing length in [kpc]
        distance (float): distance to the galaxy in [Mpc]
        pix_per_as (float): pixel per arcsecond for the LOFAR map
    """
    galaxy_number = helper.get_galaxy_number_from_name(name)

    sfr_path = radio_sfr.get_path_to_radio_sfr_dir(name, data_directory)
    radio_path = f"{data_directory}/cutouts/6as/n{galaxy_number}_144mhz_6as.fits"

    radio_filename = radio_path.split("/")[-1].split(".")[0]

    sfr_input_path = (
        f"{data_directory}/ancillary/sfr_fuv+24/ngc{galaxy_number}_fuv+24_res_13as.fits"
    )
    sfr_input_filename = sfr_input_path.split("/")[-1]
    Path(sfr_path).mkdir(parents=True, exist_ok=True)
    sfr_smoothed_filename = f"n{galaxy_number}_fuv+24_res_13as_smooth.fits"

    magnetic_path = magnetic.get_path_to_magnetic_map(
        name, data_directory, thermal, ".fits"
    )
    magnetic_filename = magnetic_path.split("/")[-1].split(".")[0]

    try:
        shutil.copyfile(sfr_input_path, sfr_path + "/" + sfr_input_filename)
        shutil.copy(radio_path, sfr_path)
    except FileExistsError:
        print(f"file already exists...")

    if smooth_exp:
        helper.smooth(
            sfr_path + "/" + sfr_input_filename,
            smooth_length,
            distance,
            pix_per_as,
            sfr_path + "/" + sfr_smoothed_filename,
        )

    with open(sfr_path + "/casa_cmd.py", "w") as casa_file:
        casa_file.write(
            f'importfits("{sfr_input_filename}", "ngc{galaxy_number}_fuv+24_res_13as.image", overwrite=True) \n'
        )
        casa_file.write(
            f'importfits("{radio_filename}.fits", "{radio_filename}.image", overwrite=True) \n'
        )
        casa_file.write(
            f'importfits("{magnetic_path}", "{magnetic_filename}.image", overwrite=True) \n'
        )
        if smooth_exp:
            casa_file.write(
                f'importfits("{sfr_smoothed_filename}", "ngc{galaxy_number}_fuv+24_res_13as_smooth.image", overwrite=True) \n'
            )

        casa_file.write(
            f"imregrid('{radio_filename}.image', '{magnetic_filename}.image', '{radio_filename}_regrid.image', asvelocity=False, interpolation='cubic', overwrite=True) \n"
        )
        casa_file.write(
            f'imrebin("{radio_filename}_regrid.image", factor=[9, 9], outfile="{radio_filename}_rebin_13_5as.image", overwrite=True) \n'
        )

        casa_file.write(
            f'imregrid("ngc{galaxy_number}_fuv+24_res_13as.image", "{radio_filename}_regrid.image", "ngc{galaxy_number}_fuv+24_res_6as_regrid.image", asvelocity=False, interpolation="cubic", overwrite=True) \n'
        )
        casa_file.write(
            f'imrebin("ngc{galaxy_number}_fuv+24_res_6as_regrid.image", factor=[9, 9], outfile="ngc{galaxy_number}_fuv+24_res_13_5as_rebin_regrid.image", overwrite=True) \n'
        )
        if smooth_exp:
            casa_file.write(
                f'imregrid("ngc{galaxy_number}_fuv+24_res_13as_smooth.image", "{magnetic_filename}.image", "ngc{galaxy_number}_fuv+24_res_6as_smooth_regrid.image", asvelocity=False, interpolation="cubic", overwrite=True) \n'
            )
            casa_file.write(
                f'imrebin("ngc{galaxy_number}_fuv+24_res_6as_smooth_regrid.image", factor=[9, 9], outfile="ngc{galaxy_number}_fuv+24_res_13_5as_smooth_rebin_regrid.image", overwrite=True) \n'
            )

        # Mask the rebinned image in 3sigma
        casa_file.write(f"ia.open('{radio_filename}_rebin_13_5as.image') \n")
        casa_file.write(
            f"ia.calcmask(mask='\"{radio_filename}_rebin_13_5as.image\" > {3 * rms_radio}', name='{radio_filename}_rebin_13_5as.mask') \n"
        )
        casa_file.write(f"ia.done() \n")
        casa_file.write("\n")

        # Mask the rebinned image in 3sigma
        casa_file.write(
            f"ia.open('ngc{galaxy_number}_fuv+24_res_13_5as_rebin_regrid.image') \n"
        )
        casa_file.write(
            f"ia.calcmask(mask='\"ngc{galaxy_number}_fuv+24_res_13_5as_rebin_regrid.image\" > {3 * rms_sfr}', name='ngc{galaxy_number}_fuv+24_res_13_5as_rebin_regrid.mask') \n"
        )
        casa_file.write(f"ia.done() \n")
        casa_file.write("\n")

        if smooth_exp:
            # Mask the rebinned and smoothed image in 3sigma
            casa_file.write(
                f"ia.open('ngc{galaxy_number}_fuv+24_res_13_5as_smooth_rebin_regrid.image') \n"
            )
            casa_file.write(
                f"ia.calcmask(mask='\"ngc{galaxy_number}_fuv+24_res_13_5as_smooth_rebin_regrid.image\" > {3 * rms_sfr}', name='ngc{galaxy_number}_fuv+24_res_13_5as_smooth_rebin_regrid.mask') \n"
            )
            casa_file.write(f"ia.done() \n")
            casa_file.write("\n")

        casa_file.write(
            f'exportfits("ngc{galaxy_number}_fuv+24_res_6as_regrid.image", "n{galaxy_number}_sfr_6as.fits", overwrite=True) \n'
        )
        casa_file.write(
            f'exportfits("ngc{galaxy_number}_fuv+24_res_13_5as_rebin_regrid.image", "n{galaxy_number}_sfr_rebin_13_5as.fits", overwrite=True) \n'
        )
        casa_file.write(
            f'exportfits("{radio_filename}_rebin_13_5as.image", "{radio_filename}_rebin_13_5as.fits", overwrite=True) \n'
        )
        casa_file.write(
            f'exportfits("{radio_filename}_regrid.image", "{radio_filename}.fits", overwrite=True) \n'
        )
        if smooth_exp:
            casa_file.write(
                f'exportfits("ngc{galaxy_number}_fuv+24_res_6as_smooth_regrid.image", "n{galaxy_number}_sfr_6as_smooth.fits", overwrite=True) \n'
            )
            casa_file.write(
                f'exportfits("ngc{galaxy_number}_fuv+24_res_13_5as_smooth_rebin_regrid.image", "n{galaxy_number}_sfr_rebin_13_5as_smooth.fits", overwrite=True) \n'
            )


def generate_h1_script(
    name: str,
    data_directory: str,
    thermal: bool,
    rms: float,
    smooth_exp: bool,
    smooth_length: float,
    distance: float,
    pix_per_as: float,
):
    """Generate a Casa script for regridding and smoothing the the integrated H1 map to the magnetic field map,
    so the data can be compared later

    Args:
        name (str): Name of the galaxy
        data_directory (str): dr2 data directory
        thermal (bool): Use non thermal magnetic field data
        rms (float): the rms value for the THINGS map
        smooth_exp (bool): make the smoothing experiment
        smooth_length (float): the diffusion/smoothing length in [kpc]
        distance (float): distance to the galaxy in [Mpc]
        pix_per_as (float): pixel per arcsecond for the LOFAR map
    """

    galaxy_number = helper.get_galaxy_number_from_name(name)
    h1_path = h1_func.get_path_to_h1_dir(name, data_directory)

    magnetic_path = magnetic.get_path_to_magnetic_map(
        name, data_directory, thermal, ".fits"
    )
    magnetic_rebin_path = magnetic.get_path_to_magnetic_map(
        name, data_directory, thermal, "_rebin_6as.fits"
    )
    # Get name without file endings
    magnetic_filename = magnetic_path.split("/")[-1]
    magnetic_rebin_filename = magnetic_rebin_path.split("/")[-1].split(".", 1)[0]

    h1_input_path = f"{data_directory}/ancillary/h1_integrated/NGC_{galaxy_number}_RO_MOM0_THINGS.FITS"
    h1_input_filename = h1_input_path.split("/")[-1]
    h1_smoothed_filename = f"NGC_{galaxy_number}_RO_MOM0_THINGS_SMOOTH.FITS"
    Path(h1_path).mkdir(parents=True, exist_ok=True)

    try:
        shutil.copyfile(h1_input_path, h1_path + "/" + h1_input_filename)
    except FileExistsError:
        print(f"file already exists...")

    if smooth_exp:
        helper.smooth(
            h1_path + "/" + h1_input_filename,
            smooth_length,
            distance,
            pix_per_as,
            h1_path + "/" + h1_smoothed_filename,
        )

    with open(h1_path + "/casa_cmd.py", "w") as casa_file:
        casa_file.write(
            f'importfits("{h1_input_filename}", "ngc_{galaxy_number}_ro_mom0_things.image", overwrite=True) \n'
        )
        if smooth_exp:
            casa_file.write(
                f'importfits("{h1_smoothed_filename}", "ngc_{galaxy_number}_ro_mom0_things_smooth.image", overwrite=True) \n'
            )
        casa_file.write(
            f'importfits("{magnetic_path}", "{magnetic_filename}.image", overwrite=True) \n'
        )
        casa_file.write(
            f'importfits("{magnetic_rebin_path}", "{magnetic_rebin_filename}.image", overwrite=True) \n'
        )
        casa_file.write(
            f'imhead(imagename="ngc_{galaxy_number}_ro_mom0_things.image", mode="put", hdkey="bunit", hdvalue="Jy/beam.m/s", verbose=False) \n'
        )
        if smooth_exp:
            casa_file.write(
                f'imhead(imagename="ngc_{galaxy_number}_ro_mom0_things_smooth.image", mode="put", hdkey="bunit", hdvalue="Jy/beam.m/s", verbose=False) \n'
            )
        casa_file.write(
            f'imregrid("ngc_{galaxy_number}_ro_mom0_things.image", "{magnetic_filename}.image", "ngc_{galaxy_number}_ro_mom0_things_6as_regrid.image", asvelocity=False, interpolation="cubic", overwrite=True) \n'
        )
        casa_file.write(
            f'imrebin("ngc_{galaxy_number}_ro_mom0_things_6as_regrid.image", factor=[4, 4], outfile="ngc_{galaxy_number}_ro_mom0_things_6as_regrid_rebin.image", overwrite=True) \n'
        )
        if smooth_exp:
            ####
            casa_file.write(
                f'imregrid("ngc_{galaxy_number}_ro_mom0_things_smooth.image", "{magnetic_filename}.image", "ngc_{galaxy_number}_ro_mom0_things_6as_smooth_regrid.image", asvelocity=False, interpolation="cubic", overwrite=True) \n'
            )
            casa_file.write(
                f'imrebin("ngc_{galaxy_number}_ro_mom0_things_6as_smooth_regrid.image", factor=[4, 4], outfile="ngc_{galaxy_number}_ro_mom0_things_6as_smooth_regrid_rebin.image", overwrite=True) \n'
            )

        # Mask the rebinned image in 3sigma
        casa_file.write(
            f"ia.open('ngc_{galaxy_number}_ro_mom0_things_6as_regrid_rebin.image') \n"
        )
        casa_file.write(
            f"ia.calcmask(mask='\"ngc_{galaxy_number}_ro_mom0_things_6as_regrid_rebin.image\" > {3 * rms}', name='ngc_{galaxy_number}_ro_mom0_things_6as_regrid_rebin.mask') \n"
        )
        casa_file.write(f"ia.done() \n")
        casa_file.write("\n")

        if smooth_exp:
            # Mask the rebinned and smoothed image in 3sigma
            casa_file.write(
                f"ia.open('ngc_{galaxy_number}_ro_mom0_things_6as_smooth_regrid_rebin.image') \n"
            )
            casa_file.write(
                f"ia.calcmask(mask='\"ngc_{galaxy_number}_ro_mom0_things_6as_smooth_regrid_rebin.image\" > {3 * rms}', name='ngc_{galaxy_number}_ro_mom0_things_6as_smooth_regrid_rebin.mask') \n"
            )
            casa_file.write(f"ia.done() \n")
            casa_file.write("\n")

        casa_file.write(
            f'exportfits("ngc_{galaxy_number}_ro_mom0_things_6as_regrid.image", "n{galaxy_number}_h1_6as.fits", overwrite=True) \n'
        )
        casa_file.write(
            f'exportfits("ngc_{galaxy_number}_ro_mom0_things_6as_regrid_rebin.image", "n{galaxy_number}_h1_rebin_6as.fits", overwrite=True) \n'
        )
        if smooth_exp:
            ####
            casa_file.write(
                f'exportfits("ngc_{galaxy_number}_ro_mom0_things_6as_smooth_regrid.image", "n{galaxy_number}_h1_6as_smooth.fits", overwrite=True) \n'
            )
            casa_file.write(
                f'exportfits("ngc_{galaxy_number}_ro_mom0_things_6as_smooth_regrid_rebin.image", "n{galaxy_number}_h1_rebin_6as_smooth.fits", overwrite=True) \n'
            )


def generate_surf_script(
    name: str,
    data_directory: str,
    thermal: bool,
    rms_h1: float,
    rms_co: float,
    smooth_exp: bool,
    smooth_length: float,
    distance: float,
    pix_per_as: float,
):
    """Generate a Casa script for regridding and smoothing the the integrated H1 and CO maps to the magnetic field map,
    so the data can be compared later

    Args:
        name (str): Name of the galaxy
        data_directory (str): dr2 data directory
        thermal (bool): Use non thermal magnetic field data
        rms_h1 (float): rms value of the THINGS map
        rms_co (float): rms value of the HERACLES map
        smooth_exp (bool): make the smoothing experiment
        smooth_length (float): the diffusion/smoothing length in [kpc]
        distance (float): distance to the galaxy in [Mpc]
        pix_per_as (float): pixel per arcsecond for the LOFAR map
    """

    galaxy_number = helper.get_galaxy_number_from_name(name)
    surf_path = surf.get_path_to_surface_density_dir(name, data_directory)

    magnetic_path = magnetic.get_path_to_magnetic_map(
        name, data_directory, thermal, ".fits"
    )
    magnetic_rebin_path = magnetic.get_path_to_magnetic_map(
        name, data_directory, thermal, "_rebin_13_5as.fits"
    )
    # Get name without file endings
    magnetic_filename = magnetic_path.split("/")[-1]
    magnetic_rebin_filename = magnetic_rebin_path.split("/")[-1].split(".", 1)[0]

    h1_input_path = f"{data_directory}/ancillary/h1_integrated/NGC_{galaxy_number}_RO_MOM0_THINGS.FITS"
    h1_input_filename = h1_input_path.split("/")[-1]
    h2_input_path = f"{data_directory}/ancillary/co_integrated/ngc{galaxy_number}_heracles_mom0.fits"
    h2_input_filename = h2_input_path.split("/")[-1]
    Path(surf_path).mkdir(parents=True, exist_ok=True)

    h1_smoothed_filename = f"NGC_{galaxy_number}_RO_MOM0_THINGS_SMOOTH.FITS"
    h2_smoothed_filename = f"ngc{galaxy_number}_heracles_mom0_smooth.fits"

    try:
        shutil.copyfile(h1_input_path, surf_path + "/" + h1_input_filename)
        shutil.copyfile(h2_input_path, surf_path + "/" + h2_input_filename)
    except FileExistsError:
        print(f"file already exists...")

    if smooth_exp:
        helper.smooth(
            surf_path + "/" + h1_input_filename,
            smooth_length,
            distance,
            pix_per_as,
            surf_path + "/" + h1_smoothed_filename,
        )
        helper.smooth(
            surf_path + "/" + h2_input_filename,
            smooth_length,
            distance,
            pix_per_as,
            surf_path + "/" + h2_smoothed_filename,
        )

    with open(surf_path + "/casa_cmd.py", "w") as casa_file:
        casa_file.write(
            f'importfits("{h1_input_filename}", "ngc_{galaxy_number}_ro_mom0_things.image", overwrite=True) \n'
        )
        casa_file.write(
            f'importfits("{h2_input_filename}", "ngc{galaxy_number}_heracles_mom0.image", overwrite=True) \n'
        )
        casa_file.write(
            f'importfits("{magnetic_path}", "{magnetic_filename}.image", overwrite=True) \n'
        )
        casa_file.write(
            f'importfits("{magnetic_rebin_path}", "{magnetic_rebin_filename}.image", overwrite=True) \n'
        )

        if smooth_exp:
            casa_file.write(
                f'importfits("{h1_smoothed_filename}", "ngc_{galaxy_number}_ro_mom0_things_smooth.image", overwrite=True) \n'
            )
            casa_file.write(
                f'importfits("{h2_smoothed_filename}", "ngc{galaxy_number}_heracles_mom0_smooth.image", overwrite=True) \n'
            )

        casa_file.write(
            f'imhead(imagename="ngc_{galaxy_number}_ro_mom0_things.image", mode="put", hdkey="bunit", hdvalue="Jy/beam.m/s", verbose=False) \n'
        )
        if smooth_exp:
            casa_file.write(
                f'imhead(imagename="ngc_{galaxy_number}_ro_mom0_things_smooth.image", mode="put", hdkey="bunit", hdvalue="Jy/beam.m/s", verbose=False) \n'
            )

        # H1
        casa_file.write(
            f'imregrid("ngc_{galaxy_number}_ro_mom0_things.image", "{magnetic_filename}.image", "ngc_{galaxy_number}_ro_mom0_things_6as_regrid.image", asvelocity=False, interpolation="cubic", overwrite=True) \n'
        )
        casa_file.write(
            f'imrebin("ngc_{galaxy_number}_ro_mom0_things_6as_regrid.image", factor=[9, 9], outfile="ngc_{galaxy_number}_ro_mom0_things_13_5as_regrid_rebin.image", overwrite=True) \n'
        )
        if smooth_exp:
            # Smooth
            casa_file.write(
                f'imregrid("ngc_{galaxy_number}_ro_mom0_things_smooth.image", "{magnetic_filename}.image", "ngc_{galaxy_number}_ro_mom0_things_6as_smooth_regrid.image", asvelocity=False, interpolation="cubic", overwrite=True) \n'
            )
            casa_file.write(
                f'imrebin("ngc_{galaxy_number}_ro_mom0_things_6as_smooth_regrid.image", factor=[9, 9], outfile="ngc_{galaxy_number}_ro_mom0_things_13_5as_smooth_regrid_rebin.image", overwrite=True) \n'
            )

        # H2/CO
        casa_file.write(
            f'imregrid("ngc{galaxy_number}_heracles_mom0.image", "{magnetic_filename}.image", "ngc{galaxy_number}_heracles_mom0_regrid_6as.image", asvelocity=False, interpolation="cubic", overwrite=True) \n'
        )
        casa_file.write(
            f'imrebin("ngc{galaxy_number}_heracles_mom0_regrid_6as.image", factor=[9, 9], outfile="ngc{galaxy_number}_heracles_mom0_regrid_rebin_13_5as.image", overwrite=True) \n'
        )
        if smooth_exp:
            # Smooth
            casa_file.write(
                f'imregrid("ngc{galaxy_number}_heracles_mom0_smooth.image", "{magnetic_filename}.image", "ngc{galaxy_number}_heracles_mom0_regrid_6as_smooth.image", asvelocity=False, interpolation="cubic", overwrite=True) \n'
            )
            casa_file.write(
                f'imrebin("ngc{galaxy_number}_heracles_mom0_regrid_6as_smooth.image", factor=[9, 9], outfile="ngc{galaxy_number}_heracles_mom0_regrid_rebin_13_5as_smooth.image", overwrite=True) \n'
            )

        # Mask the rebinned image H1 in 3sigma
        casa_file.write(
            f"ia.open('ngc_{galaxy_number}_ro_mom0_things_13_5as_regrid_rebin.image') \n"
        )
        casa_file.write(
            f"ia.calcmask(mask='\"ngc_{galaxy_number}_ro_mom0_things_13_5as_regrid_rebin.image\" > {3 * rms_h1}', name='ngc_{galaxy_number}_ro_mom0_things_13_5as_regrid_rebin.mask') \n"
        )
        casa_file.write(f"ia.done() \n")
        casa_file.write("\n")
        if smooth_exp:
            # Mask the rebinned and smoothed image H1 in 3sigma
            casa_file.write(
                f"ia.open('ngc_{galaxy_number}_ro_mom0_things_13_5as_smooth_regrid_rebin.image') \n"
            )
            casa_file.write(
                f"ia.calcmask(mask='\"ngc_{galaxy_number}_ro_mom0_things_13_5as_smooth_regrid_rebin.image\" > {3 * rms_h1}', name='ngc_{galaxy_number}_ro_mom0_things_13_5as_smooth_regrid_rebin.mask') \n"
            )
            casa_file.write(f"ia.done() \n")
            casa_file.write("\n")

        # Mask the rebinned image CO in 3sigma
        casa_file.write(
            f"ia.open('ngc{galaxy_number}_heracles_mom0_regrid_rebin_13_5as.image') \n"
        )
        casa_file.write(
            f"ia.calcmask(mask='\"ngc{galaxy_number}_heracles_mom0_regrid_rebin_13_5as.image\" > {3 * rms_co}', name='ngc{galaxy_number}_heracles_mom0_regrid_rebin_13_5as.mask') \n"
        )
        casa_file.write(f"ia.done() \n")
        casa_file.write("\n")
        if smooth_exp:
            # Mask the rebinned and smoothed image CO in 3sigma
            casa_file.write(
                f"ia.open('ngc{galaxy_number}_heracles_mom0_regrid_rebin_13_5as_smooth.image') \n"
            )
            casa_file.write(
                f"ia.calcmask(mask='\"ngc{galaxy_number}_heracles_mom0_regrid_rebin_13_5as_smooth.image\" > {3 * rms_co}', name='ngc{galaxy_number}_heracles_mom0_regrid_rebin_13_5as_smooth.mask') \n"
            )
            casa_file.write(f"ia.done() \n")
            casa_file.write("\n")

        casa_file.write(
            f'exportfits("ngc_{galaxy_number}_ro_mom0_things_6as_regrid.image", "n{galaxy_number}_h1_6as.fits", overwrite=True) \n'
        )
        casa_file.write(
            f'exportfits("ngc_{galaxy_number}_ro_mom0_things_13_5as_regrid_rebin.image", "n{galaxy_number}_h1_rebin_13_5as.fits", overwrite=True) \n'
        )

        if smooth_exp:
            # Smooth HI
            casa_file.write(
                f'exportfits("ngc_{galaxy_number}_ro_mom0_things_6as_smooth_regrid.image", "n{galaxy_number}_h1_6as_smooth.fits", overwrite=True) \n'
            )
            casa_file.write(
                f'exportfits("ngc_{galaxy_number}_ro_mom0_things_13_5as_smooth_regrid_rebin.image", "n{galaxy_number}_h1_rebin_13_5as_smooth.fits", overwrite=True) \n'
            )

        casa_file.write(
            f'exportfits("ngc{galaxy_number}_heracles_mom0_regrid_6as.image", "n{galaxy_number}_co_6as.fits", overwrite=True) \n'
        )
        casa_file.write(
            f'exportfits("ngc{galaxy_number}_heracles_mom0_regrid_rebin_13_5as.image", "n{galaxy_number}_co_rebin_13_5as.fits", overwrite=True) \n'
        )
        if smooth_exp:
            # Smooth CO
            casa_file.write(
                f'exportfits("ngc{galaxy_number}_heracles_mom0_regrid_6as_smooth.image", "n{galaxy_number}_co_6as_smooth.fits", overwrite=True) \n'
            )
            casa_file.write(
                f'exportfits("ngc{galaxy_number}_heracles_mom0_regrid_rebin_13_5as_smooth.image", "n{galaxy_number}_co_rebin_13_5as_smooth.fits", overwrite=True) \n'
            )


def generate_energy_density_script(
    name: str,
    data_directory: str,
    thermal: bool,
    rms_h1: float,
    has_h2: bool,
    rms_co: float,
    smooth_exp: bool,
    smooth_length: float,
    distance: float,
    pix_per_as: float,
):
    """Generate a Casa script for regridding and smoothing the the integrated H1 map to later calculate the energy density

    Args:
        name (str): Name of the galaxy
        data_directory (str): dr2 data directory
        thermal (bool): Use non thermal magnetic field data
        rms (float): rms value of the THINGS map
        has_h2 (bool): has this galaxy a co map?
        rms_co (float): rms value of teh heracles map
        smooth_exp (bool): make the smoothing experiment
        smooth_length (float): the diffusion/smoothing length in [kpc]
        distance (float): distance to the galaxy in [Mpc]
        pix_per_as (float): pixel per arcsecond for the LOFAR map
    """
    rebin_res = "6"
    if has_h2:
        rebin_res = "13_5"

    galaxy_number = helper.get_galaxy_number_from_name(name)
    energy_density_path = energy_density.get_path_to_energy_density_dir(
        name, data_directory
    )
    magnetic_path = magnetic.get_path_to_magnetic_map(
        name, data_directory, thermal, ".fits"
    )
    magnetic_rebin_path = magnetic.get_path_to_magnetic_map(
        name, data_directory, thermal, f"_rebin_{rebin_res}as.fits"
    )

    # Get name without file endings
    magnetic_filename = magnetic_path.split("/")[-1].split(".", 1)[0]
    magnetic_rebin_filename = magnetic_rebin_path.split("/")[-1].split(".", 1)[0]

    h1_input_path = f"{data_directory}/ancillary/h1_integrated/NGC_{galaxy_number}_RO_MOM0_THINGS.FITS"
    h1_input_filename = h1_input_path.split("/")[-1]
    h1_dispersion_input_path = (
        f"{data_directory}/ancillary/h1_dis/NGC_{galaxy_number}_RO_MOM2_THINGS.FITS"
    )
    h1_dispersion_input_filename = h1_dispersion_input_path.split("/")[-1]
    h2_input_path = f"{data_directory}/ancillary/co_integrated/ngc{galaxy_number}_heracles_mom0.fits"
    h2_input_filename = h2_input_path.split("/")[-1]

    Path(energy_density_path).mkdir(parents=True, exist_ok=True)

    h1_smoothed_filename = f"NGC_{galaxy_number}_RO_MOM0_THINGS_SMOOTH.FITS"
    h2_smoothed_filename = f"ngc{galaxy_number}_heracles_mom0_smooth.fits"

    try:
        shutil.copyfile(h1_input_path, energy_density_path + "/" + h1_input_filename)
        shutil.copyfile(
            h1_dispersion_input_path,
            energy_density_path + "/" + h1_dispersion_input_filename,
        )
        if has_h2:
            shutil.copyfile(
                h2_input_path, energy_density_path + "/" + h2_input_filename
            )
    except FileExistsError:
        print(f"file already exists...")

    if smooth_exp:
        helper.smooth(
            energy_density_path + "/" + h1_input_filename,
            smooth_length,
            distance,
            pix_per_as,
            energy_density_path + "/" + h1_smoothed_filename,
        )
        helper.smooth(
            energy_density_path + "/" + h2_input_filename,
            smooth_length,
            distance,
            pix_per_as,
            energy_density_path + "/" + h2_smoothed_filename,
        )

    with open(energy_density_path + "/casa_cmd.py", "w") as casa_file:
        casa_file.write(
            f'importfits("{h1_input_filename}", "ngc_{galaxy_number}_ro_mom0_things.image", overwrite=True) \n'
        )
        if smooth_exp:
            casa_file.write(
                f'importfits("{h1_smoothed_filename}", "ngc_{galaxy_number}_ro_mom0_things_smooth.image", overwrite=True) \n'
            )

        casa_file.write(
            f'importfits("{h1_dispersion_input_filename}", "ngc_{galaxy_number}_ro_mom2_things.image", overwrite=True) \n'
        )
        if has_h2:
            casa_file.write(
                f'importfits("{h2_input_filename}", "ngc{galaxy_number}_heracles_mom0.image", overwrite=True) \n'
            )

            if smooth_exp:
                casa_file.write(
                    f'importfits("{h2_smoothed_filename}", "ngc{galaxy_number}_heracles_mom0_smooth.image", overwrite=True) \n'
                )
        casa_file.write(
            f'importfits("{magnetic_path}", "{magnetic_filename}.image", overwrite=True) \n'
        )
        casa_file.write(
            f'importfits("{magnetic_rebin_path}", "{magnetic_rebin_filename}.image", overwrite=True) \n'
        )
        # integrated HI
        casa_file.write(
            f'imhead(imagename="ngc_{galaxy_number}_ro_mom0_things.image", mode="put", hdkey="bunit", hdvalue="Jy/beam.m/s", verbose=False) \n'
        )
        if smooth_exp:
            casa_file.write(
                f'imhead(imagename="ngc_{galaxy_number}_ro_mom0_things_smooth.image", mode="put", hdkey="bunit", hdvalue="Jy/beam.m/s", verbose=False) \n'
            )

        casa_file.write(
            f'imregrid("ngc_{galaxy_number}_ro_mom0_things.image", "{magnetic_filename}.image", "ngc_{galaxy_number}_ro_mom0_things_6as_regrid.image", asvelocity=False, interpolation="cubic", overwrite=True) \n'
        )
        casa_file.write(
            f'imrebin("ngc_{galaxy_number}_ro_mom0_things_6as_regrid.image", factor=[9, 9], outfile="ngc_{galaxy_number}_ro_mom0_things_{rebin_res}as_regrid_rebin.image", overwrite=True) \n'
        )
        if smooth_exp:
            # Smooth
            casa_file.write(
                f'imregrid("ngc_{galaxy_number}_ro_mom0_things_smooth.image", "{magnetic_filename}.image", "ngc_{galaxy_number}_ro_mom0_things_6as_smooth_regrid.image", asvelocity=False, interpolation="cubic", overwrite=True) \n'
            )
            casa_file.write(
                f'imrebin("ngc_{galaxy_number}_ro_mom0_things_6as_smooth_regrid.image", factor=[9, 9], outfile="ngc_{galaxy_number}_ro_mom0_things_13_5as_smooth_regrid_rebin.image", overwrite=True) \n'
            )

        # Mask the rebinned image in 3sigma
        casa_file.write(
            f"ia.open('ngc_{galaxy_number}_ro_mom0_things_{rebin_res}as_regrid_rebin.image') \n"
        )
        casa_file.write(
            f"ia.calcmask(mask='\"ngc_{galaxy_number}_ro_mom0_things_{rebin_res}as_regrid_rebin.image\" > {3 * rms_h1}', name='ngc_{galaxy_number}_ro_mom0_things_{rebin_res}as_regrid_rebin.mask') \n"
        )
        casa_file.write(f"ia.done() \n")
        casa_file.write("\n")

        if smooth_exp:
            # Mask the rebinned and smoothed image H1 in 3sigma
            casa_file.write(
                f"ia.open('ngc_{galaxy_number}_ro_mom0_things_13_5as_smooth_regrid_rebin.image') \n"
            )
            casa_file.write(
                f"ia.calcmask(mask='\"ngc_{galaxy_number}_ro_mom0_things_13_5as_smooth_regrid_rebin.image\" > {3 * rms_h1}', name='ngc_{galaxy_number}_ro_mom0_things_13_5as_smooth_regrid_rebin.mask') \n"
            )
            casa_file.write(f"ia.done() \n")
            casa_file.write("\n")

        # velocity dispersion
        casa_file.write(
            f'imhead(imagename="ngc_{galaxy_number}_ro_mom2_things.image", mode="put", hdkey="bunit", hdvalue="m/s", verbose=False) \n'
        )
        casa_file.write(
            f'imregrid("ngc_{galaxy_number}_ro_mom2_things.image", "{magnetic_filename}.image", "ngc_{galaxy_number}_ro_mom2_things_6as_regrid.image", asvelocity=True, interpolation="cubic", overwrite=True) \n'
        )
        casa_file.write(
            f'imrebin("ngc_{galaxy_number}_ro_mom2_things_6as_regrid.image", factor=[9, 9], outfile="ngc_{galaxy_number}_ro_mom2_things_{rebin_res}as_regrid_rebin.image", overwrite=True) \n'
        )

        if has_h2:
            # CO Map for H2
            casa_file.write(
                f'imregrid("ngc{galaxy_number}_heracles_mom0.image", "{magnetic_filename}.image", "ngc{galaxy_number}_heracles_mom0_regrid.image", asvelocity=False, interpolation="cubic", overwrite=True) \n'
            )
            casa_file.write(
                f'imrebin("ngc{galaxy_number}_heracles_mom0_regrid.image", factor=[9, 9], outfile="ngc{galaxy_number}_heracles_mom0_regrid_rebin.image", overwrite=True) \n'
            )

            if smooth_exp:
                # Smooth
                casa_file.write(
                    f'imregrid("ngc{galaxy_number}_heracles_mom0_smooth.image", "{magnetic_filename}.image", "ngc{galaxy_number}_heracles_mom0_regrid_6as_smooth.image", asvelocity=False, interpolation="cubic", overwrite=True) \n'
                )
                casa_file.write(
                    f'imrebin("ngc{galaxy_number}_heracles_mom0_regrid_6as_smooth.image", factor=[9, 9], outfile="ngc{galaxy_number}_heracles_mom0_regrid_rebin_13_5as_smooth.image", overwrite=True) \n'
                )

            # Mask the rebinned image CO in 3sigma
            casa_file.write(
                f"ia.open('ngc{galaxy_number}_heracles_mom0_regrid_rebin.image') \n"
            )
            casa_file.write(
                f"ia.calcmask(mask='\"ngc{galaxy_number}_heracles_mom0_regrid_rebin.image\" > {3 * rms_co}', name='ngc{galaxy_number}_heracles_mom0_regrid_rebin.mask') \n"
            )
            casa_file.write(f"ia.done() \n")
            casa_file.write("\n")

            if smooth_exp:
                # Mask the rebinned and smoothed image CO in 3sigma
                casa_file.write(
                    f"ia.open('ngc{galaxy_number}_heracles_mom0_regrid_rebin_13_5as_smooth.image') \n"
                )
                casa_file.write(
                    f"ia.calcmask(mask='\"ngc{galaxy_number}_heracles_mom0_regrid_rebin_13_5as_smooth.image\" > {3 * rms_co}', name='ngc{galaxy_number}_heracles_mom0_regrid_rebin_13_5as_smooth.mask') \n"
                )
                casa_file.write(f"ia.done() \n")
                casa_file.write("\n")

        # Export
        casa_file.write(
            f'exportfits("ngc_{galaxy_number}_ro_mom0_things_6as_regrid.image", "n{galaxy_number}_h1_6as.fits", overwrite=True) \n'
        )
        casa_file.write(
            f'exportfits("ngc_{galaxy_number}_ro_mom0_things_{rebin_res}as_regrid_rebin.image", "n{galaxy_number}_h1_rebin_{rebin_res}as.fits", overwrite=True) \n'
        )

        if smooth_exp:
            # Smooth HI
            casa_file.write(
                f'exportfits("ngc_{galaxy_number}_ro_mom0_things_6as_smooth_regrid.image", "n{galaxy_number}_h1_6as_smooth.fits", overwrite=True) \n'
            )
            casa_file.write(
                f'exportfits("ngc_{galaxy_number}_ro_mom0_things_13_5as_smooth_regrid_rebin.image", "n{galaxy_number}_h1_rebin_13_5as_smooth.fits", overwrite=True) \n'
            )
        # Dispersion
        casa_file.write(
            f'exportfits("ngc_{galaxy_number}_ro_mom2_things_6as_regrid.image", "n{galaxy_number}_dis_6as.fits", overwrite=True) \n'
        )
        casa_file.write(
            f'exportfits("ngc_{galaxy_number}_ro_mom2_things_{rebin_res}as_regrid_rebin.image", "n{galaxy_number}_dis_rebin_{rebin_res}as.fits", overwrite=True) \n'
        )
        if has_h2:
            casa_file.write(
                f'exportfits("ngc{galaxy_number}_heracles_mom0_regrid.image", "n{galaxy_number}_co_6as.fits", overwrite=True) \n'
            )
            casa_file.write(
                f'exportfits("ngc{galaxy_number}_heracles_mom0_regrid_rebin.image", "n{galaxy_number}_co_rebin_{rebin_res}as.fits", overwrite=True) \n'
            )

            if smooth_exp:
                # Smooth CO
                casa_file.write(
                    f'exportfits("ngc{galaxy_number}_heracles_mom0_regrid_6as_smooth.image", "n{galaxy_number}_co_6as_smooth.fits", overwrite=True) \n'
                )
                casa_file.write(
                    f'exportfits("ngc{galaxy_number}_heracles_mom0_regrid_rebin_13_5as_smooth.image", "n{galaxy_number}_co_rebin_13_5as_smooth.fits", overwrite=True) \n'
                )
