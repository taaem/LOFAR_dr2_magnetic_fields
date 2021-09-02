import os
import re
import shutil

import src.calculate_energy_density as energy_func
import src.calculate_h1 as h1_func
import src.calculate_radio_sfr as radio_sfr_func
import src.calculate_sfr as sfr_func
import src.calculate_surface_density as surf_func
import src.helper as helper


def clean_all_galaxies(config: dict, yes_flag: bool) -> None:
    """Clean all galaxy directories in config

    Args:
        config (dict): Configuration
        yes_flag (bool): switch to skip asking before removing
    """
    for galaxy in config["galaxies"]:
        clean_galaxy(galaxy["name"], config["data_directory"], yes_flag)


def clean_galaxy(name: str, data_directory: str, yes_flag: bool) -> None:
    """Clean galaxy directory for specified galaxy

    Args:
        name (str): Name of the galaxy
        data_directory (str): dr2 data directory
        yes_flag (bool): switch to skip asking before removing
    """
    print(f"Cleaning files for galaxy: {name}")
    galaxy_dir = helper.get_magnetic_galaxy_dir(name, data_directory)
    sfr_dir = sfr_func.get_path_to_sfr_dir(name, data_directory)
    radio_sfr_dir = radio_sfr_func.get_path_to_radio_sfr_dir(name, data_directory)
    h1_dir = h1_func.get_path_to_h1_dir(name, data_directory)
    energy_dir = energy_func.get_path_to_energy_density_dir(name, data_directory)
    surf_dir = surf_func.get_path_to_surface_density_dir(name, data_directory)

    # Deleting files in all directories
    delete_files_in_dir(galaxy_dir, yes_flag)
    delete_files_in_dir(sfr_dir, yes_flag)
    delete_files_in_dir(radio_sfr_dir, yes_flag)
    delete_files_in_dir(h1_dir, yes_flag)
    delete_files_in_dir(energy_dir, yes_flag)
    delete_files_in_dir(surf_dir, yes_flag)


def delete_files_in_dir(directory: str, yes_flag: bool):
    """Delete all files in directory if yes_flag is true, skip user input

    Args:
        directory (str): Path to directory
        yes_flag (bool): switch to skip asking before removing
    """
    # Loop over all files and folders in directory
    for element in os.scandir(directory):
        # Only delete .fits, .pdf, .image, .py, .log, .last, .FITS, .yml, .png files and
        # directories
        if re.search(r"\.(fits|pdf|image|py|log|last|png|yml|FITS)$", element.name):
            should_delete = yes_flag
            if not should_delete:
                should_delete = helper.query_yes_no(
                    f"Delete {os.path.join(directory, element)}?", default="no"
                )

            if should_delete:
                print("Deleting ", os.path.join(directory, element))
                # if element is an directory recursivly delete everything
                if os.path.isdir(element):
                    shutil.rmtree(os.path.join(directory, element))
                # if element is file, delete the file
                if os.path.isfile(element):
                    os.remove(os.path.join(directory, element))
