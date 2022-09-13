# BBHnet Sandbox Project

The sandbox project is an end to end pipeline for quickly experimenting with different ideas and configurations.

## Running Pipeline
We utilize [Pinto's][https://github.com/ML4GW/pinto] Pipeline ability to link project executables in serial. 
If you're in the `sandbox` directory, simply executing `pinto run` (equivalently, `pinto run ./`) will launch the pipeline . If you wish to run the pipeline from another location, execute 
`pinto run /path/to/sandbox/`. `Pinto` will look in the specified path for a `pyproject.toml`, and use the configuration settings to execute each project.

## Configuration
Each projects configuration is contained in the `pyproject.toml` file. The `tool.typeo.base` table contains configuration useful for multiple projects. the `tool.typeo.`
To refer to a setting in this table, simply add `config_variable` = "${base.config_variable}" to 
