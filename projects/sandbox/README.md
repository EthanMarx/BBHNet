# BBHnet Sandbox Project

The sandbox project is an end to end pipeline for quickly experimenting with different ideas and configurations.

## Running Pipeline
We utilize [Pinto's](https://github.com/ML4GW/pinto) Pipeline ability to link project executables in serial. 
If you're in the `sandbox` directory, simply executing `pinto run` (equivalently, `pinto run ./`) will launch the pipeline . If you wish to run the pipeline from another location, execute 
`pinto run /path/to/sandbox/`. `Pinto` will look in the specified path for a `pyproject.toml`, and use the configuration settings to execute each project.

## Configuration
Each projects configuration is contained in the `pyproject.toml` file. The `tool.typeo.base` table contains configuration applicable to multiple projects. Configuration dedicated to each individual project are contained in the `tool.typeo.project-executable`. To refer to a setting in the `tool.typeo.base`  table, simply add a line like: `config_variable` = "${base.config_variable}".

As a toy example,

```
[tool.typeo.base]
highpass = 32 # Hz

[tool.typeo.generate_foreground]
highpass = ${base.highpass}

[tool.typeo.generate_background]
highpass = ${base.highpass}
```
