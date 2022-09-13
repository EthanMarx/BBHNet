# Sandbox Pipeline

The sandbox project is an end to end pipeline for quickly experimenting with different ideas and configurations. The pipeline consists of
multiple projects that are executed in serial via Pinto's [Pipeline utility](https://github.com/EthanMarx/pinto/blob/add-pipeline-to-readme/README.md#pipelines).

## Projects
### [`generate_waveforms`](./generate_waveforms)
Generates raw polarizations of BBH waveforms 

### [`generate_glitches`](./generate_glitches)
Uses [Omicron][link] to generate datasets of single interferometer glitches

### [`generate_background`](./generate_background)
Generates coincident, continuous segments of background strain data

### [`train`](./train)
Trains BBHnet with generated data

### [`export`](./export)
Exporting a trained BBHnet instance to a Triton model repository for as-a-service inference

### [`timeslide_injections`](./timeslide_injections)
Generate a set of timeslides and injection streams.

### [`infer`](./infer)
Use a Triton server instance hosting BBHnet to run inference on generated timeslides and injections.

### [`analyze`](./analyze)
Analyze network outputs from timeslides and injections to produce some figure of merit








