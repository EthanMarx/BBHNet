from pathlib import Path
from typing import List, Tuple, TypeVar

import h5py
import numpy as np
import torch
<<<<<<< HEAD
<<<<<<< HEAD
=======
from train.augmentor import AframeBatchAugmentor
from train.data_structures import SnrRescaler, SnrSampler
>>>>>>> 69016be (getting rid of glitches and implementing new modules)
=======
>>>>>>> 2a1717b (moving whitening into augmentor)

import ml4gw.gw as gw

Tensor = TypeVar("Tensor", np.ndarray, torch.Tensor)


def split(X: Tensor, frac: float, axis: int) -> Tuple[Tensor, Tensor]:
    """
    Split an array into two parts along the given axis
    by an amount specified by `frac`. Generic to both
    numpy arrays and torch Tensors.
    """

    size = int(frac * X.shape[axis])
    # Catches fp error that sometimes happens when size should be an exact int
    # Is there a better way to do this?
    if np.abs(frac * X.shape[axis] - size - 1) < 1e-10:
        size += 1

    if isinstance(X, np.ndarray):
        return np.split(X, [size], axis=axis)
    else:
        splits = [size, X.shape[axis] - size]
        return torch.split(X, splits, dim=axis)


def _sort_key(fname: Path):
    return int(fname.stem.split("-")[-2])


def get_background_fnames(data_dir: Path):
    fnames = data_dir.glob("*.hdf5")
    fnames = sorted(fnames, key=_sort_key)
    return list(fnames)


def get_background(fname: Path):
    background = []
    with h5py.File(fname, "r") as f:
        ifos = list(f.keys())
        for ifo in ifos:
            hoft = f[ifo][:]
            background.append(hoft)
    return np.stack(background)


def get_waveforms(
    waveform_dataset: Path,
    ifos: List[str],
    sample_rate: float,
    valid_frac: float,
):
    # perform train/val split of waveforms,
    # and compute fixed validation responses
    with h5py.File(waveform_dataset, "r") as f:
        signals = f["signals"][:]

        if valid_frac is not None:
            signals, valid_signals = split(signals, 1 - valid_frac, 0)

            valid_cross, valid_plus = valid_signals.transpose(1, 0, 2)
            slc = slice(-len(valid_signals), None)
            dec, psi, phi = f["dec"][slc], f["psi"][slc], f["ra"][slc]

            # project the validation waveforms to IFO
            # responses up front since we don't care
            # about sampling sky parameters
            tensors, vertices = gw.get_ifo_geometry(*ifos)
            valid_responses = gw.compute_observed_strain(
                torch.Tensor(dec),
                torch.Tensor(psi),
                torch.Tensor(phi),
                detector_tensors=tensors,
                detector_vertices=vertices,
                sample_rate=sample_rate,
                plus=torch.Tensor(valid_plus),
                cross=torch.Tensor(valid_cross),
            )
            return signals, valid_responses
    return signals, None
