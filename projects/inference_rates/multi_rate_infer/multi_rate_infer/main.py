import logging
from pathlib import Path
from typing import List

import numpy as np
from infer.deploy import main as deploy_infer
from typeo import scriptify

from aframe.logging import configure_logging


@scriptify
def main(
    model_repo_dir: Path,
    base_dir: Path,
    data_dir: Path,
    log_dir: Path,
    injection_set_file: Path,
    inference_sampling_rates: List[float],
    image: str,
    model_name: str,
    accounting_group: str,
    accounting_group_user: str,
    Tb: float,
    shifts: List[float],
    sample_rate: float,
    ifos: List[str],
    batch_size: int,
    integration_window_length: float,
    cluster_window_length: float,
    psd_length: float,
    fduration: float,
    throughput: float,
    chunk_size: float,
    sequence_id: int,
    model_version: int = -1,
    verbose: bool = False,
):
    configure_logging(log_dir / "inference_rates_infer.log", verbose=verbose)

    # Assume that the given throughput is appropriate for the first rate,
    # and calculate other throughputs based on that
    inference_sampling_rates = np.array(inference_sampling_rates)
    rate_ratios = inference_sampling_rates[0] / inference_sampling_rates
    throughputs = throughput * rate_ratios

    for rate, throughput in zip(inference_sampling_rates, throughputs):
        output_dir = base_dir / f"{rate}Hz" / "infer"
        output_dir.mkdir(exist_ok=True, parents=True)
        current_model_dir = model_repo_dir / f"{rate}Hz"

        logging.info(f"Starting inference at {rate} Hz")

        deploy_infer(
            model_repo_dir=current_model_dir,
            output_dir=output_dir,
            data_dir=data_dir,
            log_dir=log_dir,
            injection_set_file=injection_set_file,
            image=image,
            model_name=model_name,
            accounting_group=accounting_group,
            accounting_group_user=accounting_group_user,
            Tb=Tb,
            shifts=shifts,
            sample_rate=sample_rate,
            inference_sampling_rate=rate,
            ifos=ifos,
            batch_size=batch_size,
            integration_window_length=integration_window_length,
            cluster_window_length=cluster_window_length,
            psd_length=psd_length,
            fduration=fduration,
            throughput=throughput,
            chunk_size=chunk_size,
            sequence_id=sequence_id,
            model_version=model_version,
            verbose=verbose,
        )

        logging.info(f"Finished inference at {rate} Hz")
