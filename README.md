[![GNU GPLv3 License](https://img.shields.io/github/license/JohannesHaubner/TopOPt)](https://choosealicense.com/licenses/gpl-3.0/)
[![Test TopOpt](https://github.com/JohannesHaubner/TopOpt/actions/workflows/test-TopOpt.yml/badge.svg?style=plastic)](https://github.com/JohannesHaubner/TopOpt/actions/workflows/test-TopOpt.yml)

# TopOpt 

Code repository for the manuscript

>J. Haubner, F. Neumann, M. Ulbrich: A Novel Density Based Approach for Topology Optimization of Stokes Flow, SIAM J. Sci. Comput. (2022, accepted for publication).

The implementation is based on 
http://www.dolfin-adjoint.org/en/release/documentation/stokes-topology/stokes-topology.html

## Usage/Examples

### Conda

```
conda env create -f environment.yml --experimental-solver=libmamba
conda activate topopt

cd topopt
python3 topopt.py

conda deactivate topopt
```

It has to be ensured that the [conda-libmamba-solver](https://github.com/conda-incubator/conda-libmamba-solver) is installed.

For practical problems it is furthermore necessary to link IPOPT against HSL when compiling (see comment in http://www.dolfin-adjoint.org/en/release/documentation/stokes-topology/stokes-topology.html).

For running the MMA examples, it is required to clone the github repository https://github.com/arjendeetman/GCMMA-MMA-Python into the folder mma/MMA_Python.

### Docker
Alternatively, also Docker can be used (only build for linux/amd64 platforms):
```
docker pull ghcr.io/johanneshaubner/topopt:latest

cd topopt
docker run -it -v $(pwd):/topopt ghcr.io/johanneshaubner/topopt
```
In the Docker container:

```
python3 topopt/topopt.py
```

## Running Tests

To run tests, run the following command

```bash
pytest
```
## License

[GNU GPLv3](https://choosealicense.com/licenses/gpl-3.0/)

## Authors
- [Johannes Haubner](https://www.github.com/JohannesHaubner)
- [Franziska Neumann](https://www.mos.ed.tum.de/ftm/personen/mitarbeiter/franziska-neumann-msc/)
- [Michael Ulbrich](https://www-m1.ma.tum.de/bin/view/Lehrstuhl/MichaelUlbrich)

## Acknowledgement
We would like to acknowledge [Jørgen Dokken](http://jsdokken.com/) and [Henrik Finsberg](https://finsberg.github.io/) for the help, support and discussions on reproducibility.
