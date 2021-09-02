import glob
import shutil
from pathlib import Path

import src.calculate_energy_density as energy_func
import src.calculate_h1 as h1_func
import src.calculate_sfr as sfr_func
import src.helper as helper
import src.calculate_surface_density as surf_func
import src.calculate_radio_sfr as radio_sfr


def copy_to_out(config: dict):
    Path(config["out_directory"]).mkdir(parents=True, exist_ok=True)

    l = ["energy_density_combined",
         "energy_density_h1",
         "energy_density_h2",
         "h1_combined",
         "h2_surface_density_combined",
         "radio_sfr_combined",
         "sfr_combined",
         "surface_density_combined"]

    for file in l:
        shutil.copy(config["data_directory"] + "/" + file + ".pdf", config["out_directory"])
        shutil.copy(config["data_directory"] + "/" + file + "_mean.pdf", config["out_directory"])
        shutil.copy(config["data_directory"] + "/" + file + "_smooth.pdf", config["out_directory"])

    for galaxy in config["galaxies"]:
        name = galaxy["name"]

        magnetic_dir = helper.get_magnetic_galaxy_dir(name, config["data_directory"])
        sfr_dir = sfr_func.get_path_to_sfr_dir(name, config["data_directory"])
        h1_dir = h1_func.get_path_to_h1_dir(name, config["data_directory"])
        energy_dir = energy_func.get_path_to_energy_density_dir(
            name, config["data_directory"])
        surf_dir = surf_func.get_path_to_surface_density_dir(
            name, config["data_directory"])
        radio_sfr_dir = radio_sfr.get_path_to_radio_sfr_dir(name, config["data_directory"])

        for files in [glob.glob(magnetic_dir + f"/*magnetic{'_non_thermal' if galaxy['use_thermal'] else ''}.pdf"),
                     glob.glob(magnetic_dir + f"/*magnetic{'_non_thermal' if galaxy['use_thermal'] else ''}_overlay.pdf")]:
            for file in files:
                shutil.copy(file, config["out_directory"])

        for files in [glob.glob(sfr_dir + "/*_pixel.pdf"), glob.glob(sfr_dir + "/*_overlay.pdf"), glob.glob(sfr_dir + "/*_pixel_smooth.pdf")]:
            for file in files:
                shutil.copy(file, config["out_directory"])

        for files in [glob.glob(h1_dir + "/*_pixel.pdf"), glob.glob(h1_dir + "/*_overlay.pdf"), glob.glob(h1_dir + "/*_pixel_smooth.pdf")]:
            for file in files:
                shutil.copy(file, config["out_directory"])

        for files in [glob.glob(energy_dir + "/*_pixel.pdf"), glob.glob(energy_dir + "/*_overlay.pdf"), glob.glob(energy_dir + "/*_pixel_smooth.pdf")]:
            for file in files:
                shutil.copy(file, config["out_directory"])

        for files in [glob.glob(surf_dir + "/*_pixel.pdf"), glob.glob(surf_dir + "/*_overlay.pdf"), glob.glob(surf_dir + "/*_pixel_smooth.pdf")]:
            for file in files:
                shutil.copy(file, config["out_directory"])

        for files in [glob.glob(radio_sfr_dir + "/*_pixel.pdf"), glob.glob(radio_sfr_dir + "/*_overlay.pdf"), glob.glob(radio_sfr_dir + "/*_pixel_smooth.pdf")]:
            for file in files:
                shutil.copy(file, config["out_directory"])
