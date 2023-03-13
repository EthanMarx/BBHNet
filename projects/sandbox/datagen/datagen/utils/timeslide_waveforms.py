from pathlib import Path
from textwrap import dedent
from typing import List, Tuple

import h5py
import numpy as np
import torch

from bbhnet.analysis.ledger.injections import LigoResponseSet
from ml4gw.spectral import normalize_psd


def calc_shifts_required(
    segments: List[Tuple[int, int]], Tb: float, shift: float
):
    """
    Based off of the lengths of the segments and the
    amount of data that will need to be sloughed off
    the ends due to shifting, calculate how many shifts
    will be required to achieve Tb seconds worth of background

    Args:
        segments: A list of tuples of the start and stop times of the segments
        Tb: The amount of background data to generate
        shift: The increment to shift the data by

    Returns the number of shifts required to achieve Tb seconds of background
    """

    livetime = sum([stop - start for start, stop in segments])
    n_segments = len(segments)
    shifts_required = 0
    while True:
        max_shift = shift * shifts_required
        total_livetime = (livetime - n_segments * max_shift) * shifts_required
        if total_livetime < Tb:
            shifts_required += 1
            continue
        break

    return shifts_required


def merge_output(datadir: Path):
    files = datadir.glob("tmp-*.h5")
    response_set = LigoResponseSet()
    for f in files:
        fset = LigoResponseSet.read(f)
        response_set.append(fset)
        f.unlink()
    response_set.write(datadir / "timeslide_waveforms.h5")


def load_psds(background: Path, sample_rate: float, df: float):
    with h5py.File(background, "r") as f:
        psds = []
        for ifo in "HL":
            hoft = f[f"{ifo}1"][:]
            psd = normalize_psd(hoft, df, sample_rate)
            psds.append(psd)
    psds = torch.tensor(np.stack(psds), dtype=torch.float64)
    return psds


def calc_segment_injection_times(
    start: float,
    stop: float,
    spacing: float,
    buffer: float,
    waveform_duration: float,
):
    """
    Calculate the times at which to inject signals into a segment

    Args:
        start: The start time of the segment
        stop: The stop time of the segment
        spacing: The spacing between signals
        jitter: The jitter to apply to the signal times
        buffer: The buffer to apply to the start and end of the segment
        waveform_duration: The duration of the waveform

    Returns np.ndarray of injection times
    """

    buffer += waveform_duration // 2
    spacing += waveform_duration
    injection_times = np.arange(start + buffer, stop - buffer, spacing)
    return injection_times


def create_submit_file(
    executable: str,
    condor_dir: Path,
    accounting_group: str,
    accounting_group_user: str,
    request_memory: int,
    request_disk: int,
    arguments: str,
):

    logdir = condor_dir / "logs"
    logdir.mkdir(exist_ok=True, parents=True)
    subfile = dedent(
        f"""\
        universe = vanilla
        executable = {executable}
        arguments = {arguments}
        log = {logdir}/timeslide_waveforms-$(ProcID).log
        output = {logdir}/timeslide_waveforms-$(ProcID).out
        error = {logdir}/timeslide_waveforms-$(ProcID).err
        accounting_group = {accounting_group}
        accounting_group_user = {accounting_group_user}
        request_memory = {request_memory}
        request_disk = {request_disk}
        queue start,stop,shift from {condor_dir}/segments.txt
    """
    )
    return subfile
