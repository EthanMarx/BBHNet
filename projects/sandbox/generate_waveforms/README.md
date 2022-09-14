# Generating Waveforms

Generates a set raw time domain polarizations for a specified waveform approximant.

## Installation
If you are inside this directory, building this project can be done with 

```console
pinto build 
```

## Available Commands
`generate-waveforms`

Running 
```
pinto run generate-waveforms -h
```
will list the available arguments.

They can also by found in the pipelines `pyproject.toml` under the `[tool.typeo.scripts.generate_waveforms]` table.

The project can be run with these default arguments via 

```console
pinto run generate-waveforms --typeo ..:generate-waveforms
```
