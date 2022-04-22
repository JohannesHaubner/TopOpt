[![MIT License](https://img.shields.io/apm/l/atomic-design-ui.svg?style=plastic)](https://choosealicense.com/licenses/mit/)
[![Test TopOpt](https://github.com/JohannesHaubner/TopOpt/actions/workflows/test-TopOpt.yml/badge.svg?style=plastic)](https://github.com/JohannesHaubner/TopOpt/actions/workflows/test-TopOpt.yml)

# TopOpt

Code repository for the manuscript

>J. Haubner, F. Neumann, M. Ulbrich: Topology Optimization of Stokes Flow via Density Based Approaches, Preprint No. IGDK-2021-05; May 2021. 

The implementation is based on 
http://www.dolfin-adjoint.org/en/release/documentation/stokes-topology/stokes-topology.html

## Usage/Examples

```
conda env create -f environment.yml
conda activate topopt

cd topopt
python3 topopt.py

conda deactivate topopt
```

The following fix might be necessary: https://fenicsproject.discourse.group/t/installed-but-looking-for-libsuperlu-dist-so-6/7016/3

For practical problems it is furthermore necessary to link IPOPT against HSL when compiling (see comment in http://www.dolfin-adjoint.org/en/release/documentation/stokes-topology/stokes-topology.html). The numerical results presented in the manuscript use 
an installation of IPOPT that is linked against HSL.

## Running Tests

To run tests, run the following command

```bash
pytest
```
## License

[MIT](https://choosealicense.com/licenses/mit/)

## Authors
- [Johannes Haubner](https://www.github.com/JohannesHaubner)
- [Franziska Neumann](https://www.mos.ed.tum.de/ftm/personen/mitarbeiter/franziska-neumann-msc/)
- [Michael Ulbrich](https://www-m1.ma.tum.de/bin/view/Lehrstuhl/MichaelUlbrich)
