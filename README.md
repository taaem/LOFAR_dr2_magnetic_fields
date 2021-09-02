# Investigating Magnetic Fields in Nearby Galaxies with LOFAR
[![CC BY 4.0][cc-by-shield]][cc-by]
#### by Tim-Leon Klocke

This work is licensed under a [Creative Commons Attribution 4.0 International License][cc-by].

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg

This repo contains the code for my bachelor thesis on magnetic fields in nearby galaxies with LOFAR data.

### Usage

The code is written as a monolithic script, which can perform all sorts of tasks for the examinations of magnetic fields.
The main entrypoint is the `main.py` script, which calls aóut all specific tasks are implemented in the `src` subdirectory.

All tasks can either be invoked without arguments: `python main.py <task> <flags>`, which then cycles through all available galaxies in the `config/config.yml`.

Or the tasks can be invoked with a galaxy name like such: `python main.py <tasks> <flags> <galaxy>`, where `<galaxy>` is the galaxy code eg. `n5194`.

#### `casa`
The starting point is the `casa` task, which is responsible for copying files in the right directories and regridding the FITS image onto the LOFAR coordinate system. As the name suggests the CASA software is used for the regridding process.
The implementation is in the `src/generate_casa.py` file. 

When invoking `python main.py casa` only the files for the calculation and generation of the magnetic field maps are copied and casa scripts are written into the correct directories, which could than be run with CASA itself.

When invoking `python main.py casa --run` the same procedure as before is performed, but as a last step all CASA scripts are run by CASA automatically. The output should be checked afterwards for errors, since the program will not stop if CASA reported an error.

When the CASA scripts where manually changed all changes will be lost, because all files will be overwritten!

The last option of the CASA subtask is the `python main.py casa --run -m` flag, which should only be invoked __after__ the magnetic analysis command was run, otherwise the task will fail! If run correctly, this makes directories, copies files and regrids data for all ancillary data, which is needed for the later correlation analysis. This also performs the smoothing on the galaxies, where this is configured.

#### `magnetic`
This task generates all magnetic field maps and overlays. The results are saved as YAML files in the corresponding directories so the data doesn't need to be copied from the task output, but can be viewed later on. This also generates the magnetic field FITS files.
Output will be in `<data_dir>/magnetic`.

#### `radio_sfr`
Calculate the RC-SFR relation. If only interested in the combined plots the `--skip` flag can be provided and only the combined plots will be regenerated.
Output will be in `<data_dir>/radio_sfr`.

#### `sfr`
Calculate the MF-SFR relation. If only interested in the combined plots the `--skip` flag can be provided and only the combined plots will be regenerated.
Output will be in `<data_dir>/sfr`.

#### `h1`
Calculate the MF-SD-HI relation. If only interested in the combined plots the `--skip` flag can be provided and only the combined plots will be regenerated.
Output will be in `<data_dir>/h1`.

#### `surf`
Calculate the MF-SD-H2 and MF-SD relation. If only interested in the combined plots the `--skip` flag can be provided and only the combined plots will be regenerated.
Output will be in `<data_dir>/surf`. And the MF-SD relation files are called surf and the MF-SD-H2 is called surf_h2.


#### `energy`
Calculate the MF-TE relation. If only interested in the combined plots the `--skip` flag can be provided and only the combined plots will be regenerated.
Output will be in `<data_dir>/energy_density`. There are files for each component seperatly and the combined energy density, but the names should be quite clear.


### Procedure
I am using the following procedure to calculate all data:
```bash
python main.py casa --run
python main.py magnetic
python main.py casa --run -m
python main.py radio_sfr
python main.py sfr
python main.py h1
python main.py surf
python main.py energy
```

### Directory Layout
The data directory should be structured as follows:
- `ancillary/co_integrated` for HERACLES maps
- `ancillary/h1_dis` for THINGS moment 2 maps
- `ancillary/h1_integrated` for THINGS moment 1 maps
- `ancillary/reference` for WSRT-SINGS reference maps for spectral index generation
- `ancillary/sfr_fuv+24` for SFR maps from Leroy 2008
- `ancillary/thermal` for thermal emission maps
- `cutouts/6as` for the 6" LOFAR cutouts
- `cutouts/20as` for the 20" LOFAR cutouts
- `regions/high` for 6" ellipse regions
- `regions/low` for 20" ellipse regions 
- `regions/high` for 6" ellipse regions
- `spix` for spectral indices and errors