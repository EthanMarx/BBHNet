# Generating Glitches

Generates a set single interferometer glitches. Omicron excess power algorithm 

## Installation
If you are inside this directory, building this project can be done with 

```console
pinto build 
```

## Available Commands
`generate-glitches`

Running 
```
pinto run generate-glitches -h
```
will list the available arguments.

They can also by found in the pipelines `pyproject.toml` under the `[tool.typeo.scripts.generate-glitches]` table.

The project can be run with these default arguments via 

```console
pinto run generate-glitches --typeo ..:generate-glitches
```
