from pathlib import Path
from typing import List

from gwpy.timeseries import TimeSeries, TimeSeriesDict
from typeo import scriptify


@scriptify
def main(
    start: float,
    stop: float,
    channel: str,
    ifos: List[str],
    sample_rate: float,
    write_path: Path,
):
    """Generates background data for training and testing aframe

    Args:
        start:
            Starting GPS time of the timeseries to be fetched
        stop:
            Ending GPS time of the timeseries to be fetched
        writepath:
            Path, including file name, that the data will be saved to
        channel:
            Channel from which to fetch the timeseries
        ifos:
            List of interferometers to query data from. Expected to be given
            by prefix; e.g. "H1" for Hanford
        sample_rate:
            Sample rate to which the timeseries will be resampled, specified
            in Hz

    Returns: The `Path` of the output file
    """

    # authenticate()

    data = TimeSeriesDict()
    for ifo in ifos:
        data[ifo] = TimeSeries.get(f"{ifo}:{channel}", start, stop)

    data = data.resample(sample_rate)
    data.write(write_path)
    return write_path


if __name__ == "__main__":
    main()
