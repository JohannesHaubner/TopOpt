[![MIT License](https://img.shields.io/apm/l/atomic-design-ui.svg?style=plastic)](https://github.com/tterb/atomic-design-ui/blob/master/LICENSEs)
[![Test topopt](https://github.com/JohannesHaubner/TopOpt/actions/workflows/test-TopOpt.yml/badge.svg?style=plastic)](https://github.com/JohannesHaubner/TopOpt/actions/workflows/test-TopOpt.yml)

# TopOpt

Code repository for the manuscript

>J. Haubner, F. Neumann, M. Ulbrich: Topology Optimization of Stokes Flow via Density Based Approaches, Preprint No. IGDK-2021-05; May 2021. 

The implementation is based on 
http://www.dolfin-adjoint.org/en/release/documentation/stokes-topology/stokes-topology.html

## Authors
- [Johannes Haubner](https://www.github.com/JohannesHaubner)
- [Michael Ulbrich](https://www-m1.ma.tum.de/bin/view/Lehrstuhl/MichaelUlbrich)

## License

[MIT](https://choosealicense.com/licenses/mit/)

## Usage/Examples

```
conda env create -f environment.yml
conda activate topopt

cd topopt
python3 topopt.py

conda deactivate topopt
```

## Running Tests

To run tests, run the following command

```bash
pytest
```
