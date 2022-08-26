import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import h5py
import numpy as np

from bbhnet.analysis.distributions.distribution import Distribution


@dataclass
class ClusterDistribution(Distribution):
    """
    Distribution representing a clustering of sampled points.
    The result of clustering is that there
    is no event closer in time to any other event
    by (t_clust / 2) seconds, and each remaining event is the
    highest network output
    (hence lowest FAR) in a t_clust second window centered on itself.

    Args:
        t_clust: The length of the clustering window
    """

    t_clust: float

    def __post_init__(self) -> None:
        super().__post_init__()
        self.events = np.array([])
        self.event_times = np.array([])
        self.shifts = np.array([])

    def _load(self, path: Path):
        """Load distribution information from an HDF5 file"""
        with h5py.File(path, "r") as f:
            self.events = f["events"][:]
            self.event_times = f["event_times"][:]
            self.shifts = f["shifts"][:]
            self.fnames = list(f["fnames"][:])
            self.Tb = f["Tb"]
            t_clust = f.attrs["t_clust"]
            if t_clust != self.t_clust:
                raise ValueError(
                    "t_clust of Distribution object {t_clust}"
                    "does not match t_clust of data in h5 file {self.t_clust}"
                )

    def write(self, path: Path):
        """Write the distribution's data to an HDF5 file"""
        with h5py.File(path, "w") as f:
            f["events"] = self.events
            f["event_times"] = self.event_times
            f["shifts"] = self.shifts
            f["fnames"] = list(map(str, self.fnames))
            f["Tb"] = self.Tb
            f.attrs.update({"t_clust": self.t_clust})

    @classmethod
    def from_file(cls, dataset: str, path: Path):
        """Create a new distribution with data loaded from an HDF5 file"""
        with h5py.File(path, "r") as f:
            t_clust = f.attrs["t_clust"]
        obj = cls(dataset, t_clust)
        obj._load(path)
        return obj

    def update(self, x: np.ndarray, t: np.ndarray, shift: float):
        """
        Update the histogram using the values from `x`,
        and update the background time `Tb` using the
        values from `t`. It is assumed that `t` is contiguous
        in time and evenly sampled.
        """
        # TODO: with this clustering method,
        # if there are monotonically increasing
        # network outputs on time scales greater than
        # the window length, all of those events will be clustered
        # to the loudest event:
        # e.g.
        # t = [1,2,3,4,5]
        # x = [1,2,3,4,5]
        # t_clust = 1
        # will all cluster to the event at t = 5

        # update livetime before we start
        # manipulating the t array
        self.Tb += t[-1] - t[0] + t[1] - t[0]

        # infer sample rate from time array
        sample_rate = 1 / (t[1] - t[0])

        # samples per half cluster window
        half_clust_size = int(sample_rate * self.t_clust / 2)

        # indices to remove after clustering
        remove_indices = []

        for i in range(len(t)):

            left = max(0, i - half_clust_size)
            right = min(len(t), i + half_clust_size)

            # if there are any events
            # within half window with a greater
            # network ouput, remove this event
            if any(x[left:right] > x[i]):
                remove_indices.append(i)

        events = np.delete(x, remove_indices)
        times = np.delete(t, remove_indices)

        # update events, event times,
        # livetime and timeshifts
        self.events = np.append(self.events, events)
        self.event_times = np.append(self.event_times, times)
        self.shifts = np.append(
            self.shifts, np.repeat(shift, len(times), axis=0)
        )

        # re sort by event_time
        sorted_args = np.argsort(self.event_times)

        self.events = self.events[sorted_args]
        self.event_times = self.event_times[sorted_args]
        self.shifts = self.shifts[sorted_args]

    def nb(self, threshold: Union[float, np.ndarray]):
        """
        Counts the number of events above the indicated
        `threshold`
        """
        events = np.array(self.events)
        if isinstance(threshold, np.ndarray):
            nb = [np.sum(events >= thresh) for thresh in threshold]
        else:
            nb = np.sum(events >= threshold)

        logging.debug(
            "Threshold {} has {} events greater than it "
            "in distribution {}".format(threshold, nb, self)
        )
        return np.array(nb)
