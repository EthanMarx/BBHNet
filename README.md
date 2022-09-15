# BBHnet
Single source for training, performing inference with, analyzing the performance of BBHnet, a neural network for performing detection of CBC sources in real gravitational-wave data.

## Contributing

See our [contributing guidelines](https://github.com/ML4GW/BBHNet/blob/main/CONTRIBUTING.md).


## Project organization
The repository is divided into `libs`, modular source libraries for performing the relevant signal processing and deep learning tasks, and `projects`, pipelines for data generation, training, inference, and analysis built on top of these libraries. 

BBHnet relies on the following [submodules](https://git-scm.com/book/en/v2/Git-Tools-Submodules):


[`hermes`](https://github.com/ML4GW/hermes) includes libraries for exporting trained models as accelerated executables, building production-ready inference pipelines, and utilizing cloud resources.


[`ml4gw`](https://github.com/ML4GW/hermes) includes torch utilities for common GW data manipulations, like projecting waveforms onto interfermeters


## Installation
### Setting up the repository
Before you do anything, be sure that the `hermes` and `ml4gw` submodule has been initialized after cloning this repo
```
git submodule update
git submodule init
```

### Environment setup
#### 1. The Easy Way - `pinto`
The simplest way to interact with the code in this respository is to install the ML4GW [Pinto command line utility](https://ml4gw.github.io/pinto/), which contains all the same prerequisites (namely Conda and Poetry) that this repo does.
The only difference is that rather than having to keep track of which projects require Conda and which only need Poetry separately, `pinto` exposes commands which build and execute scripts inside of virtual environments automatically, dynamically detecting how and where to install each project.
For more information, consult the Pinto documentation linked to above.

#### 2. The Hard Way - Conda + Poetry
Otherwise, make sure you have Conda and Poetry installed in the manner outlined in Pinto's documentation.
Then create the base Conda environment on which all projects are based

```console
conda env create -f environment.yaml
```

Projects that requires Conda will have a `poetry.toml` file in them containing in part

```toml
[virtualenvs]
create = false
```

For these projects, you can build the necessary virtual environment by running 

```console
conda create -n <environment name> --clone bbhnet-base
```

then using Poetry to install additional dependencies (called from the project's root directory, not the repository's)

```console
poetry install
```

Otherwise, you should just need to run `poetry install` from the project's directory, and Poetry will take care of creating a virtual environment automatically.

### Running projects
Consult each project's documentation for additional installation steps and available commands. If you installed the `pinto` utility, commands can be run
in each project's virtual environment by

```console
pinto run path/to/project my-command --arg 1
```

Otherwise, for Conda projects, you'll have to activate the appropriate Conda environment first then execute the code inside of it.
For Poetry environments, it should be enough to run `poetry run my-command --arg 1` _from the projects directory_ (one downside of Poetry is that everything has to be done locally).
