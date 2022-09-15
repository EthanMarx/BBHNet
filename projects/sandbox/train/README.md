# Train

Train BBHnet using datasets of waveforms, glitches, and background strain. 
Almost all of the actual training is handled by the bbhnet.trainer library, so consult the code there to get a sense for what the actual training loop looks like. 
The code here prepares the dataloaders and preprocessor modules.

## Installation
If you are inside this directory, building this project can be done with 

```console
pinto build 
```

## Available Commands
`train`

Running 
```
pinto run train -h
```
will list the available arguments.

They can also by found in the pipelines `pyproject.toml` under the `[tool.typeo.scripts.train]` table.

The project can be run with these default arguments via 

```console
pinto run train --typeo ..:train
```
