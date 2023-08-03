import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional

import torch
from bokeh.server.server import Server

from aframe.architectures import architecturize
from aframe.logging import configure_logging

from . import structures
from .app import VizApp
from .vetoes import VetoParser

logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)


def _normalize_path(path: Path):
    if not path.is_absolute():
        return Path(__file__).resolve().parent / path
    return path


@architecturize
def main(
    architecture: Callable,
    waveform_prob: float,
    glitch_prob: float,
    downweight: float,
    swap_frac: float,
    mute_frac: float,
    basedir: Path,
    datadir: Path,
    veto_definer_file: Path,
    gate_paths: Dict[str, Path],
    ifos: List[str],
    cosmology: Callable,
    source_prior: Callable,
    start: float,
    stop: float,
    sample_rate: float,
    inference_sampling_rate: float,
    fduration: float,
    valid_frac: float,
    background_length: float,
    integration_length: float,
    kernel_length: float,
    highpass: Optional[float] = None,
    device: str = "cpu",
    port: int = 5005,
    logdir: Optional[Path] = None,
    verbose: bool = False,
) -> None:

    logfile = logdir / "vizapp.log" if logdir is not None else None
    configure_logging(logfile, verbose)

    # load in model weights
    model = architecture(len(ifos))
    model.to(device)

    weights = basedir / "training" / "weights.pt"
    model.load_state_dict(
        torch.load(weights, map_location=torch.device(device))
    )

    model.eval()

    # amount of data to plot on either side of events
    padding = 4
    # set batch size based on amount of seconds we will wan't to plot
    batch_size = (
        int((2 * padding + integration_length) * inference_sampling_rate) + 1
    )
    # initialize preprocessor that uses background_length seconds
    # to calculate psd, and whiten data
    preprocessor = structures.BatchWhitener(
        kernel_length,
        sample_rate,
        inference_sampling_rate,
        batch_size,
        fduration,
        highpass=highpass,
    )
    snapshotter = structures.BackgroundSnapshotter(
        psd_length=background_length,
        kernel_length=kernel_length,
        fduration=fduration,
        sample_rate=sample_rate,
        inference_sampling_rate=inference_sampling_rate,
    )

    veto_definer_file = _normalize_path(veto_definer_file)
    for ifo in ifos:
        gate_paths[ifo] = _normalize_path(gate_paths[ifo])

    veto_parser = VetoParser(
        veto_definer_file,
        gate_paths,
        start,
        stop,
        ifos,
    )

    cosmology = cosmology()
    source_prior, _ = source_prior()
    bkapp = VizApp(
        model=model,
        preprocessor=preprocessor,
        snapshotter=snapshotter,
        waveform_prob=waveform_prob,
        glitch_prob=glitch_prob,
        downweight=downweight,
        swap_frac=swap_frac,
        mute_frac=mute_frac,
        base_directory=basedir,
        data_directory=datadir,
        cosmology=cosmology,
        source_prior=source_prior,
        ifos=ifos,
        sample_rate=sample_rate,
        kernel_length=kernel_length,
        background_length=background_length,
        inference_sampling_rate=inference_sampling_rate,
        integration_length=integration_length,
        fduration=fduration,
        padding=padding,
        valid_frac=valid_frac,
        veto_parser=veto_parser,
    )

    server = Server({"/": bkapp}, num_procs=1, port=port, address="0.0.0.0")
    server.start()
    server.run_until_shutdown()


if __name__ == "__main__":
    main()
