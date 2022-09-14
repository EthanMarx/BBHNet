# Generating Background

Queries coincident strain data from interferometers during specified data quality segments.

## Installation
If you are inside this directory, building this project can be done with 

```console
pinto build 
```

## Available Commands
`generate-background`

Running 
```
pinto run generate-background -h
```
will list the available arguments.

They can also by found in the pipelines `pyproject.toml` under the `[tool.typeo.scripts.generate_background]` table.

The project can be run with these default arguments via 

```console
pinto run generate-background --typeo ..:generate-background
```
