# Timeslide Injections

Queries a stretch of coincident background segments, generates sets of circular timeslides for each segment, and additionally creates a separate injection
stream by injecting signals on top of these time-slid segments. 

## Installation
If you are inside this directory, building this project can be done with 

```console
pinto build 
```

## Available Commands
`timeslide_injections`

Running 
```
pinto run timeslide_injections -h
```
will list the available arguments.

They can also by found in the pipelines `pyproject.toml` under the `[tool.typeo.scripts.timeslide_injections]` table.

The project can be run with these default arguments via 

```console
pinto run timeslide_injections --typeo ..:timeslide_injections
```
