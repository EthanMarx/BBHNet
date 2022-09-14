import configparser
import logging
from pathlib import Path
from typing import List, Optional

import gwdatafind
import h5py
import numpy as np
from gwpy.segments import Segment, SegmentList
from gwpy.timeseries import TimeSeries
from omicron.cli.process import main as omicron_main
from tqdm import tqdm

from bbhnet.logging import configure_logging
from hermes.typeo import typeo


# TODO: retire?
# struggling to recall the usecase for this function
# in the context of this project
def veto(times: list, segmentlist: SegmentList):

    """
    Remove events from a list of times based on a segmentlist
    A time ``t`` will be vetoed if ``start <= t <= end`` for any veto
    segment in the list.

    Args:
        times:
            The times of event triggers to veto
        segmentlist:
            The list of veto segments to use

    Returns:
        List of booleans; True the triggers to keep
    """

    # find args that sort times and create sorted times array
    sorted_args = np.argsort(times)
    sorted_times = times[sorted_args]

    # initiate array of args to keep;
    # refers to original args of unsorted times array;
    # begin with all args being kept

    keep_bools = np.ones(times.shape[0], dtype=bool)

    # initiate loop variables; extract first segment
    j = 0
    a, b = segmentlist[j]
    i = 0

    while i < sorted_times.size:
        t = sorted_times[i]

        # if before start, not in vetoed segment; move to next trigger now
        if t < a:

            # original arg is the ith sorted arg
            i += 1
            continue

        # if after end, find the next segment and check this trigger again
        if t > b:
            j += 1
            try:
                a, b = segmentlist[j]
                continue
            except IndexError:
                break

        # otherwise it must be in veto segment; move on to next trigger
        original_arg = sorted_args[i]
        keep_bools[original_arg] = False
        i += 1

    return keep_bools


def generate_glitch_dataset(
    ifo: str,
    snr_thresh: float,
    start: float,
    stop: float,
    half_window: float,
    sample_rate: float,
    channel: str,
    frame_type: str,
    trig_file: str,
    vetoes: SegmentList = None,
):

    """
    Analyzes omicron triggers and queries strain data
    for those that satisfy SNR threshold

    Arguments:
        ifo:
            ifo to generate glitch triggers for
        snr_thresh:
            Lower SNR threshold
        start:
            Start gpstime
        stop:
            Stop gpstime
        half_window:
            Half window around trigger time to query data for
        sample_rate:
            Frequency which strain data is sampled
        channel:
            Strain data channel used to read data
        frame_type:
            Frame type for data discovery w/ gwdatafind
        trig_file:
            Text file of triggers produced by omicron
            (first column contains gps times, 3rd column contains snrs)
        vetoes: SegmentList object of times to ignore

    Returns:
        Array of glitch strain data
        Array of glitch SNRS
    """

    glitches = []
    snrs = []

    # load in triggers
    with h5py.File(trig_file) as f:
        triggers = f["triggers"][()]

        # restrict triggers to within gps start and stop times
        # and apply snr threshold
        times = triggers["time"][()]
        mask = (times > start) & (times < stop)
        mask &= triggers["snr"][()] > snr_thresh
        triggers = triggers[mask]

    # if passed, apply vetos
    if vetoes is not None:
        keep_bools = veto(times, vetoes)
        times = times[keep_bools]
        snrs = snrs[keep_bools]

    # re-set 'start' and 'stop' so we aren't querying unnecessary data
    start = np.min(triggers["time"]) - 2 * half_window
    stop = np.max(triggers["time"]) + 2 * half_window

    logging.info(
        f"Querying {stop - start} seconds of data for {len(triggers)} triggers"
    )

    # use gwdatafind to create frame cache
    frames = gwdatafind.find_urls(
        site=ifo.strip("1"),
        frametype=f"{ifo}_{frame_type}",
        gpsstart=int(start),
        gpsend=int(stop),
        urltype="file",
        on_gaps="ignore",
    )

    # read frames all at once, then crop
    # this time series for each trigger.
    # Although its initially slower to read all the frames at once,
    # I think this is overall faster than querying a 4096 frame for
    # every trigger.

    ts = TimeSeries.read(
        frames, channel=f"{ifo}:{channel}", start=start, end=stop, pad=0
    )

    # for each trigger
    for trigger in tqdm(triggers):
        time = trigger["time"]

        try:
            glitch_ts = ts.crop(time - half_window, time + half_window)

            glitch_ts = glitch_ts.resample(sample_rate)

            snrs.append(trigger["snr"])
            glitches.append(glitch_ts)

        except ValueError:
            logging.warning(f"Data not available for trigger at time: {time}")
            continue

    glitches = np.array(glitches)
    snrs = np.array(snrs)
    return glitches, snrs


def omicron_main_wrapper(
    start: int,
    stop: int,
    q_min: float,
    q_max: float,
    f_min: float,
    f_max: float,
    sample_rate: float,
    cluster_dt: float,
    chunk_duration: int,
    segment_duration: int,
    overlap: int,
    mismatch_max: float,
    snr_thresh: float,
    frame_type: str,
    channel: str,
    state_flag: str,
    ifo: str,
    run_dir: Path,
):

    """Parses args into a format compatible for pyomicron,
    and then runs pyomicron main which will create
    and launch condor dags

    Args: See main function

    Returns:
        Path to location of omicron runs
    """

    # pyomicron expects some arguments passed via
    # a config file. Create that config file

    config = configparser.ConfigParser()
    section = "GW"
    config.add_section(section)

    config.set(section, "q-range", f"{q_min} {q_max}")
    config.set(section, "frequency-range", f"{f_min} {f_max}")
    config.set(section, "frametype", f"{ifo}_{frame_type}")
    config.set(section, "channels", f"{ifo}:{channel}")
    config.set(section, "cluster-dt", str(cluster_dt))
    config.set(section, "sample-frequency", str(sample_rate))
    config.set(section, "chunk-duration", str(chunk_duration))
    config.set(section, "segment-duration", str(segment_duration))
    config.set(section, "overlap-duration", str(overlap))
    config.set(section, "mismatch-max", str(mismatch_max))
    config.set(section, "snr-threshold", str(snr_thresh))

    config.add_section("OUTPUTS")
    config.set("OUTPUTS", "format", "hdf5")

    # in an online setting, can also pass state-vector,
    # and bits to check for science mode
    config.set(section, "state-flag", f"{ifo}:{state_flag}")

    config_file_path = run_dir / f"omicron_{ifo}.ini"

    # write config file
    with open(config_file_path, "w") as configfile:
        config.write(configfile)

    # parse args into format expected by omicron
    omicron_args = [
        section,
        "--config-file",
        str(config_file_path),
        "--gps",
        f"{start}",
        f"{stop}",
        "--ifo",
        ifo,
        "-c",
        "request_disk=100",
        "--output-dir",
        str(run_dir),
        "--skip-ligolw_add",
        "--skip-gzip",
    ]

    # create and launch omicron dag
    omicron_main(omicron_args)

    return run_dir


@typeo
def main(
    snr_thresh: float,
    start: int,
    stop: int,
    q_min: float,
    q_max: float,
    f_min: float,
    cluster_dt: float,
    chunk_duration: int,
    segment_duration: int,
    overlap: int,
    mismatch_max: float,
    half_window: float,
    datadir: Path,
    logdir: Path,
    channel: str,
    frame_type: str,
    sample_rate: float,
    state_flag: str,
    ifos: List[str],
    veto_files: Optional[dict[str, str]] = None,
    force_generation: bool = False,
    verbose: bool = False,
):

    """Generates a strain dataset of single interferometer glitches

    Glitches are identified with the Omicron
    (https://virgo.docs.ligo.org/virgoapp/Omicron/)
    excess power algorithm. The Pyomicron
    (https://github.com/gwpy/pyomicron/) wrapper is used
    to simplify the generation of condor jobs
    for performing the Omicron analysis.

    Once Omicron triggers are identified, data is queried
    for these triggers and saved in an h5 file.

    For more detailed information on Omicron arguments,
    see the pyomicron documentation.

    Arguments:

    snr_thresh:
        Lower SNR threshold for declaring a glitch
    start:
        Start gpstime
    stop:
        Stop gpstime
    q_min:
        Minimum quality factor for omicron tiles
    q_max:
        Maximum quality factor for omicron tiles
    f_min:
        Lowest frequency for omicron tiles
    cluster_dt:
        Time window for omicron to cluster neighboring triggers
    chunk_duration:
        Duration of data (seconds) for PSD estimation
    segment_duration:
        Duration of data (seconds) for FFT
    overlap:
        Overlap (seconds) between neighbouring segments and chunks
    mismatch_max:
        Maximum distance between (Q, f) tiles
    half_window:
        Half window around trigger time to query data for
    datadir:
        Location where h5 data file is stored
    logdir:
        Location where log file is stored
    channel:
        Strain channel for reading data
    frame_type:
        Frame type for data discovery w/ gw_data_find
    sample_rate:
        Frequency which strain data is sampled
    state_flag:
        Data quality segment name
    ifos:
        Ifos for which to query data
    veto_files:
        Dictionary where key is ifo and value is path
        to file containing vetoe segments
    """

    logdir.mkdir(exist_ok=True, parents=True)
    datadir.mkdir(exist_ok=True, parents=True)

    configure_logging(logdir / "generate_glitches.log", verbose)

    # output file
    glitch_file = datadir / "glitches.h5"

    if glitch_file.exists() and not force_generation:
        logging.info(
            "Glitch data already exists and forced generation is off. "
            "Not generating glitches"
        )
        return

    # nyquist
    f_max = sample_rate / 2

    for ifo in ifos:
        run_dir = datadir / ifo
        run_dir.mkdir(exist_ok=True)

        # launch omicron dag for ifo
        omicron_main_wrapper(
            start,
            stop,
            q_min,
            q_max,
            f_min,
            f_max,
            sample_rate,
            cluster_dt,
            chunk_duration,
            segment_duration,
            overlap,
            mismatch_max,
            snr_thresh,
            frame_type,
            channel,
            state_flag,
            ifo,
            run_dir,
        )

        # load in vetoes and convert to gwpy SegmentList object
        if veto_files is not None:
            veto_file = veto_files[ifo]

            logging.info(f"Applying vetoes to {ifo} times")

            # load in vetoes
            vetoes = np.loadtxt(veto_file)

            # convert arrays to gwpy Segment objects
            vetoes = [Segment(seg[0], seg[1]) for seg in vetoes]

            # create SegmentList object
            vetoes = SegmentList(vetoes).coalesce()
        else:
            vetoes = None

        # get the path to the omicron triggers
        trigger_dir = run_dir / "triggers" / f"{ifo}:{channel}"
        trigger_file = list(trigger_dir.glob("*.h5"))[0]

        # generate glitches
        glitches, snrs = generate_glitch_dataset(
            ifo,
            snr_thresh,
            start,
            stop,
            half_window,
            sample_rate,
            channel,
            frame_type,
            trigger_file,
            vetoes=vetoes,
        )

        if np.isnan(glitches).any():
            raise ValueError("The glitch data contains NaN values")

        with h5py.File(glitch_file, "a") as f:
            f.create_dataset(f"{ifo}_glitches", data=glitches)
            f.create_dataset(f"{ifo}_snrs", data=snrs)

    return glitch_file


if __name__ == "__main__":
    main()
