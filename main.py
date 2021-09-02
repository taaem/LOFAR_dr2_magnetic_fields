import argparse
import warnings

import astropy

import src.calculate_energy_density as energy
import src.calculate_h1 as h1_func
import src.calculate_magnetic_fields as magnetic
import src.calculate_radio_sfr as radio_sfr
import src.calculate_sfr as sfr
import src.calculate_surface_density as surf
import src.cleaner as cleaner
import src.copy
import src.generate_casa as casa
from src import config_loader
from src.exceptions import NotConfiguredException

# Ignore FITS warning as they provide no valuable insights
warnings.filterwarnings(
    "ignore",
    message=r".*Set OBSGEO-. to .* from OBSGEO-\[XYZ\]",
    category=astropy.wcs.FITSFixedWarning,
)

parser = argparse.ArgumentParser(
    description="Tool for calculating magnetic fields and further studies from LOFAR data"
)
parser.add_argument(
    "--config",
    "-c",
    dest="config_path",
    help="Path to configuration file",
    default="config/config.yaml",
)
subparsers = parser.add_subparsers(title="commands", dest="command")

# sub-command for interaction with casa
casa_parser = subparsers.add_parser("casa", help="Generate Casa scripts")
casa_parser.add_argument(
    "galaxy",
    nargs="?",
    default="all",
    help="Galaxy (eg. n5194), can use all to cycle through all galaxies. Default: all",
)
casa_parser.add_argument(
    "--run",
    action="store_true",
    help="Switch to actually run the casa scripts, it may be necessary to clean the directory before",
)
casa_parser.add_argument(
    "--has-magnetic",
    "-m",
    action="store_true",
    help="Switch to run correlation casa scripts",
)

# sub-command for cleaning working directories
clean_parser = subparsers.add_parser("clean", help="Clean folders")
clean_parser.add_argument(
    "galaxy",
    nargs="?",
    default="all",
    help="Galaxy (eg. n5194), can use all to cycle through all galaxies. Default: all",
)
clean_parser.add_argument(
    "--yes", "-y", action="store_true", help="Switch remove without asking",
)

# sub-command for copying out files
copy_parser = subparsers.add_parser("copy", help="Copy images to out directory")

# sub-command for calculating magnetic maps
magnetic_parser = subparsers.add_parser("magnetic", help="Calculate magnetic maps")
magnetic_parser.add_argument(
    "galaxy",
    nargs="?",
    default="all",
    help="Galaxy (eg. n5194), can use all to cycle through all galaxies. Default: all",
)

# sub-command for calculating SFR correlation
sfr_parser = subparsers.add_parser("sfr", help="Calculate SFR correlations")
sfr_parser.add_argument(
    "galaxy",
    nargs="?",
    default="all",
    help="Galaxy (eg. n5194), can use all to cycle through all galaxies. Default: all",
)
sfr_parser.add_argument(
    "--skip", "-s", action="store_true", help="Skip calculation",
)

# sub-command for calculating SFR RC relation
radio_sfr_parser = subparsers.add_parser(
    "radio_sfr", help="Calculate SFR to RC correlations"
)
radio_sfr_parser.add_argument(
    "galaxy",
    nargs="?",
    default="all",
    help="Galaxy (eg. n5194), can use all to cycle through all galaxies. Default: all",
)
radio_sfr_parser.add_argument(
    "--skip", "-s", action="store_true", help="Skip calculation",
)

# sub-command for calculating HI surface density
h1_parser = subparsers.add_parser(
    "h1", help="Calculate HI surface density correlations"
)
h1_parser.add_argument(
    "galaxy",
    nargs="?",
    default="all",
    help="Galaxy (eg. n5194), can use all to cycle through all galaxies. Default: all",
)
h1_parser.add_argument(
    "--skip", "-s", action="store_true", help="Skip calculation",
)

# sub-command for calculating surface density
surf_parser = subparsers.add_parser(
    "surf", help="Calculate H2/HI+H2 density correlations"
)
surf_parser.add_argument(
    "galaxy",
    nargs="?",
    default="all",
    help="Galaxy (eg. n5194), can use all to cycle through all galaxies. Default: all",
)
surf_parser.add_argument(
    "--skip", "-s", action="store_true", help="Skip calculation",
)

# sub-command for calculating star formation correlation
energy_parser = subparsers.add_parser(
    "energy", help="Calculate energy density correlations"
)
energy_parser.add_argument(
    "galaxy",
    nargs="?",
    default="all",
    help="Galaxy (eg. n5194), can use all to cycle through all galaxies. Default: all",
)
energy_parser.add_argument(
    "--skip", "-s", action="store_true", help="Skip calculation of energy densities",
)
args = parser.parse_args()

config = config_loader.get_config(args.config_path)

print(f"Loading galaxies from {config['data_directory']}")

# sub-command casa
if args.command == "casa":
    # check if a specific galaxy was requested
    if args.galaxy == "all":
        print("No or all galaxies specifed, iterating through all available galaxies")
        # generate all scripts
        for galaxy_config in config["galaxies"]:
            casa.generate_all_scripts(galaxy_config, config, args.has_magnetic)
            if args.run:
                # Run the scripts
                casa.run_all_scripts(galaxy_config, config, args.has_magnetic)
    else:
        try:
            # get specific configuration for one galaxy
            galaxy_config = next(
                filter(lambda g: g["name"] == args.galaxy, config["galaxies"],)
            )

            # generate script for the specified galaxy
            casa.generate_all_scripts(galaxy_config, config, args.has_magnetic)

            # if --run was specified run for specified galaxies
            if args.run:
                casa.run_all_scripts(galaxy_config, config, args.has_magnetic)
        except StopIteration:
            print("Specified galaxy not config, aborting...")

# sub-command clean
if args.command == "clean":
    # Check if specific galaxy was specified
    if args.galaxy == "all":
        print("No or all galaxies specifed, iterating through all available galaxies")
        # clean all galaxy files
        cleaner.clean_all_galaxies(config, args.yes)
    else:
        try:
            # "Check" if the specified galaxy exists
            galaxy_config = next(
                filter(lambda g: g["name"] == args.galaxy, config["galaxies"],)
            )
            # Clean files for specified galaxy
            cleaner.clean_galaxy(args.galaxy, config["data_directory"], args.yes)
        except StopIteration:
            print("Specified galaxy not config, aborting...")

if args.command == "copy":
    # Copy all files to out directory
    src.copy.copy_to_out(config)

# sub-command magnetic
if args.command == "magnetic":
    # Check if specific galaxy was specified
    if args.galaxy == "all":
        print("No or all galaxies specifed, iterating through all available galaxies")
        # calculate all magnetic fields
        magnetic.calculate_all_magnetic_field(config)
    else:
        try:
            # "Check" if the specified galaxy exists
            galaxy_config = next(
                filter(lambda g: g["name"] == args.galaxy, config["galaxies"],)
            )
            # calculate magnetic fields for one galaxy
            magnetic.calculate_magnetic_field(galaxy_config, config)
        except StopIteration:
            print("Specified galaxy not config, aborting...")

# sub-command sfr
if args.command == "sfr":
    # Check if specific galaxy was specified
    if args.galaxy == "all":
        print("No or all galaxies specifed, iterating through all available galaxies")
        # calculate sfr for all galaxies
        sfr.calculate_all_sfr(config, args.skip)
    else:
        try:
            # calculate sfr for one galaxy
            sfr.calculate_sfr(args.galaxy, config)
        except StopIteration:
            print("Specified galaxy not config, aborting...")

# sub-command radio_sfr
if args.command == "radio_sfr":
    # Check if specific galaxy was specified
    if args.galaxy == "all":
        print("No or all galaxies specifed, iterating through all available galaxies")
        # calculate radio sfr for all galaxies
        radio_sfr.calculate_all_radio_sfr(config, args.skip)
    else:
        try:
            # calculate radio sfr for one galaxy
            radio_sfr.calculate_radio_sfr(args.galaxy, config)
        except StopIteration:
            print("Specified galaxy not config, aborting...")

# sub-command h1
if args.command == "h1":
    # Check if specific galaxy was specified
    if args.galaxy == "all":
        print("No or all galaxies specifed, iterating through all available galaxies")
        # calculate HI for all galaxies
        h1_func.calculate_all_h1(config, args.skip)
    else:
        try:
            # "Check" if the specified galaxy exists
            galaxy_config = next(
                filter(lambda g: g["name"] == args.galaxy, config["galaxies"],)
            )
            if not galaxy_config["h1"]:
                raise NotConfiguredException()
            # calculate HI for one galaxy
            h1_func.calculate_h1(galaxy_config["name"], config)
        except StopIteration:
            print("Specified galaxy not config, aborting...")
        except NotConfiguredException:
            print("Galaxy not configured for operation...")

# sub-command h1
if args.command == "surf":
    # Check if specific galaxy was specified
    if args.galaxy == "all":
        print("No or all galaxies specifed, iterating through all available galaxies")
        # calculate surface densities for all galaxies
        surf.calculate_all_surface_density(config, args.skip)
    else:
        try:
            # calculate surface densities for one galaxy
            surf.calculate_surface_density(args.galaxy, config)
        except StopIteration:
            print("Specified galaxy not config, aborting...")

# sub-command energy
if args.command == "energy":
    # Check if specific galaxy was specified
    if args.galaxy == "all":
        print("No or all galaxies specifed, iterating through all available galaxies")
        # calculate energy density for all galaxies
        energy.calculate_all_energy_density(config, args.skip)
    else:
        try:
            # calculate energy density stuff for one galaxy
            energy.calculate_energy_density(args.galaxy, config)
        except StopIteration:
            print("Specified galaxy not config, aborting...")

