# Infer

Serve a BBHnet model for inference using Triton inference server. Uses a set of [`TimeSlides`](https://github.com/ML4GW/BBHNet/blob/main/libs/io/bbhnet/io/timeslides.py) directories (see the [`timeslide_injections`](https://github.com/ML4GW/BBHNet/tree/main/projects/sandbox/timeslide_injections) project for how these are generated)
 containing `injection` and `background` `fields` 
to send requests to that server. 
## Installation
If you are inside this directory, building this project can be done with 

```console
pinto build 
```

## Available Commands
`infer`

Running 
```
pinto run infer -h
```
will list the available arguments.

They can also by found in the pipelines `pyproject.toml` under the `[tool.typeo.scripts.infer]` table.

The project can be run with these default arguments via 

```console
pinto run infer --typeo ..:infer
```
