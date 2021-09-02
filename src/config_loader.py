from pathlib import Path

import yaml
from schema import Optional, Schema

config_schema = {
    "data_directory": str, # path to root folder, where all data resides
    "out_directory": str, # path to the directory, where the "copy" action copies to
    "casa_executable": str, # path to the casa executable, when the "casa" run tasks are invoked
    "threads": int, # amount of threads to use on a multicore system, 1 disables multithreading
    "alpha_min": float, # minimum absolute value of the spectral index (eg. 0.6)
    "alpha_max": float, # maximum absolute value of the spectral index (eg. 1.1)
    "frequency": float, # frequency of the LOFAR observations in Hz (eg. 144.e+6)
    "proton_electron_ratio": int, # K_0, the proton to electron ratio (eg. 100)
    "energy_cutoff": float, # cutoff for the fit in for the combined energy plot (currently not used) (set to 0 for disabling)
    "galaxies": [{
        "name": str, # name of the galaxy (eg. n5194)
        Optional("use_thermal", default=False): bool, # specify whether to correct for thermal emission
        Optional("use_integrated", default=False): bool, # specify whether to use the integrated spectral index instead of the spatially resolved map
        Optional("calc_sfr", default=False): bool, # specify whether to calculate the SFR relations (MF-SFR and RC-SFR)
        Optional("calc_h1", default=False): bool, # specify whether to calculate the MF-SD-HI relation
        Optional("calc_surf", default=False): bool, # specify whether to calculate the MF-SD-H2 and MF-SD relations
        Optional("calc_energy", default=False): bool, # specify whether to calculate the MF-TE relation
        Optional("skip_combined_sfr", default=False): bool, # specify whether to skip this galaxy in the combined MF-SFR plot
        Optional("skip_combined_radio_sfr", default=False): bool, # specify whether to skip this galaxy in the combined RC-SFR plot
        Optional("skip_combined_h1", default=False): bool, # specify whether to skip this galaxy in the combined MF-SD-HI plot
        Optional("skip_combined_surf", default=False): bool, # specify whether to skip this galaxy in the combined MF-SD-H2 and MF-SD plot
        Optional("skip_combined_energy", default=False): bool, # specify whether to skip this galaxy in the combined MF-TE plot
        Optional("smooth_exp", default=False): bool, # specify whether to use this galaxy in the smoothing experiment
        Optional("smooth_length", default=0): float, # if the galaxy is used in the smoothing experiment, give the diffusion length in kpc
        "distance": float, # distance to the galaxy in Mpc
        "rms_6": float, # rms value of the 6" LOFAR map
        "rms_20": float, # rms value of the 20" LOFAR map
        Optional("rms_ref", default=0): float, # if thermal corrected, give the rms value of the spectral index reference map
        "spix": float, # value of the integrated spectral index
        "spix_error": float, # error of the integrated spectral index
        "radio_integrated": float, # integrated radio emission at 6"
        "region_ellipse": [int, float], # size of the ellipse of the integrated radio emission in [arcmin, arcmin]
        Optional("ref_beam", default=None): [int, float], # if thermal corrected, beam size of the reference map
        Optional("ref_name", default=None): str, # if thermal corrected, name of the file of the reference map
        Optional("ref_freq", default=0): float, # if thermal corrected, frequency of the reference map
        Optional("thermal_name", default=None): str, # if thermal corrected, name of the file of the thermal emission map
        Optional("thermal_freq", default=0): float, # if thermal corrected, frequency of the thermal emission map
        Optional("thermal_beam", default=None): [int, float], # if thermal corrected, beam size of the thermal emission map
        "pathlength": float, # pathlength along the line-of-sight of the radio emission through the galaxy
        "inclination": int, # inclination of the galaxy plane to the line-of-sight
        "size": float, # size of the cutout of the magnetic map in arcmin
        Optional("magnetic_levels", default=[6, 9, 12, 15, 18, 21]): [int, float], # levels for the magnetic field strength in the contour plots
        Optional("radio_levels", default=[1.e-3, 2.5e-3, 4.e-3, 5.5e-3, 7.e-3, 9.5e-3]): [int, float], # levels for the radio emission in the contour plots
        Optional("sfr", default=None): {
            "rms": float, # rms value of the SFR map
            "p0": [int, float], # initial guess of the parameters of the power law fit
            Optional("vmin", default=0.0005): float, # minimum value for the color coding of the overlay plot
            Optional("vmax", default=0.03): float, # maximum value for the color coding of the overlay plot
            Optional("mean", default=0): float, # integrated SFR
        },
        Optional("h1", default=None): {
            "rms": float, # rms value of the THINGS HI map
            "p0": [int, float], # initial guess of the parameters of the power law fit
            "pathlength": int, # pathlength through the galaxy for the HI emission
            Optional("vmin", default=1): float, # minimum value for the color coding of the overlay plot
            Optional("vmax", default=None): float, # maximum value for the color coding of the overlay plot
        },
        Optional("co", default=None): {
            "rms": float, # rms value of the HERACLES CO map
            "p0": [int, float], # initial guess of the parameters of the power law fit
            "pathlength": int, # pathlength through the galaxy for the CO emission
            Optional("vmin", default=1): float, # minimum value for the color coding of the overlay plot
            Optional("vmax", default=None): float, # maximum value for the color coding of the overlay plot
        },
        Optional("energy", default=None): {
            "p0": [int, float], # initial guess of the parameters of the power law fit
            "cutoff": float, # # cutoff for the fit (set to 0 for disabling) 
            Optional("vmin", default=0.1e-13): float, # minimum value for the color coding of the overlay plot
            Optional("vmax", default=None): float, # maximum value for the color coding of the overlay plot
            "levels": [int, float] # levels for the magnetic energy density in the contour plot
        },
    }]
}


def get_config(path: str) -> config_schema:
    """Load the config from the path, validate and return the dcitionary

    Args:
        path (str): Path the config.yaml

    Returns:
        config_schema: The configuration dictionary
    """    
    config_path = Path(path)
    config = yaml.full_load(open(config_path))
    return Schema(config_schema).validate(config)
