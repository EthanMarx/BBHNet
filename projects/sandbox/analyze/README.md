# Analyze
Analyze network outputs from timeslides and injections to produce a figure of merit for model comparison.


## Installation
If you are inside this directory, building this project can be done with 

```console
pinto build 
```

## Available Commands
`analyze`

Running 
```
pinto run analyze -h
```
will list the available arguments.

They can also by found in the pipelines `pyproject.toml` under the `[tool.typeo.scripts.analyze]` table.

The project can be run with these default arguments via 

```console
pinto run infer --typeo ..:analyze
```
