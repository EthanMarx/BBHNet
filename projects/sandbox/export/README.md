# Export

Export a trained BBHnet network architecture to a Triton model repository (or "model store"). This will export the trained network as an ONNX binary to the repo and create a model configuration file for it, with options for setting the level of concurrency through that configuration file.

Additionally, this will create snapshotter and aggregation models to insert at either end of BBHnet to handle input state caching and output online averaging for streaming use cases.

## Installation
If you are inside this directory, building this project can be done with 

```console
pinto build 
```

## Available Commands
`export-model`

Running 
```
pinto run export-model -h
```
will list the available arguments.

They can also by found in the pipelines `pyproject.toml` under the `[tool.typeo.scripts.export-model]` table.

The project can be run with these default arguments via 

```console
pinto run generate-background --typeo ..:export-model
```
