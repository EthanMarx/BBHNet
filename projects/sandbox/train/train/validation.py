import logging
import pickle
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np
import torch
from train.utils import split

from ml4gw.dataloading import InMemoryDataset
from ml4gw.gw import compute_network_snr, reweight_snrs
from ml4gw.spectral import normalize_psd

if TYPE_CHECKING:
    from train.waveform_injection import BBHNetWaveformInjection


class Metric(torch.nn.Module):
    """
    Abstract class for representing a metric which
    needs to be evaluated at particular threshold(s).
    Inherits from `torch.nn.Module` so that parameters
    for calculating the metric of interest can be
    saved as `buffer`s and moved to appropriate devices
    via `Metric.to`. Child classes should override `call`
    for actually computing the metric values of interest.
    """

    def __init__(self, thresholds) -> None:
        super().__init__()
        self.thresholds = thresholds
        self.values = [0.0 for _ in thresholds]

    def update(self, metrics):
        try:
            metric = metrics[self.name]
        except KeyError:
            metric = {}
            metrics[self.name] = {}

        for threshold, value in zip(self.thresholds, self.values):
            try:
                metric[threshold].append(value)
            except KeyError:
                metric[threshold] = [value]

    def call(self, backgrounds, glitches, signals):
        raise NotImplementedError

    def forward(self, backgrounds, glitches, signals):
        values = self.call(backgrounds, glitches, signals)
        self.values = [v for v in values.cpu().numpy()]
        return values

    def __str__(self):
        tab = " " * 8
        string = ""
        for threshold, value in zip(self.thresholds, self.values):
            string += f"\n{tab}{self.param} = {threshold}: {value:0.4f}"
        return self.name + " @:" + string

    def __getitem__(self, threshold):
        try:
            idx = self.thresholds.index(threshold)
        except ValueError:
            raise KeyError(str(threshold))
        return self.values[idx]

    def __contains__(self, threshold):
        return threshold in self.thresholds


class MultiThresholdAUROC(Metric):
    name = "AUROC"
    param = "max_fpr"

    def call(self, signal_preds, background_preds):
        x = torch.cat([signal_preds, background_preds])
        y = torch.zeros_like(x)
        thresholds = torch.Tensor(self.thresholds).to(y.device)
        y[: len(signal_preds)] = 1

        # shuffle the samples so that constant
        # network outputs don't show up as perfect
        idx = torch.randperm(len(y))
        x = x[idx]
        y = y[idx]

        # now sort the labels by their corresponding prediction
        idx = torch.argsort(x, descending=True)
        y = y[idx]

        tpr = torch.cumsum(y, -1) / y.sum()
        fpr = torch.cumsum(1 - y, -1) / (1 - y).sum()
        dfpr = fpr.diff()
        dtpr = tpr.diff()

        mask = fpr[:-1, None] <= thresholds
        dfpr = dfpr[:, None] * mask
        integral = (tpr[:-1, None] + dtpr[:, None] * 0.5) * dfpr
        return integral.sum(0)


class BackgroundAUROC(MultiThresholdAUROC):
    def __init__(
        self, kernel_size: int, stride: int, thresholds: List[float]
    ) -> None:
        super().__init__(thresholds)
        self.kernel_size = kernel_size
        self.stride = stride

    def call(self, background, _, signal):
        background = background.unsqueeze(0)
        background = torch.nn.functional.max_pool1d(
            background, kernel_size=self.kernel_size, stride=self.stride
        )
        background = background[0]
        return super().call(signal, background)


class BackgroundRecall(Metric):
    """
    Computes the recall of injected signals (fraction
    of total injected signals recovered) at the
    detection statistic threshold given by each of the
    top `k` background "events."

    Background predictions are max-pooled along the time
    dimension using the indicated `kernel_size` and
    `stride` to keep from counting multiple events from
    the same phenomenon.

    Args:
        kernel_size:
            Size of the window, in samples, over which
            to max pool background predictions.
        stride:
            Number of samples between consecutive
            background max pool windows.
        k:
            Max number of top background events against
            whose thresholds to evaluate signal recall.
    """

    name = "recall vs. background"
    param = "k"

    def __init__(self, kernel_size: int, stride: int, k: int = 5) -> None:
        super().__init__([i + 1 for i in range(k)])
        self.kernel_size = kernel_size
        self.stride = stride
        self.k = k

    def call(self, background, _, signal):
        background = background.unsqueeze(0)
        background = torch.nn.functional.max_pool1d(
            background, kernel_size=self.kernel_size, stride=self.stride
        )
        background = background[0]
        topk = torch.topk(background, self.k).values
        recall = (signal.unsqueeze(1) >= topk).sum(0) / len(signal)
        return recall


class WeightedEfficiency(Metric):
    name = "Weighted efficiency"

    def __init__(self, weights: List[np.ndarray]) -> None:
        super().__init__(weights)

    def call(self, background_events, foreground_events):
        threshold = background_events.max()
        mask = foreground_events >= threshold
        effs = self.weights[mask].sum() / self.weights.sum()
        return effs


class GlitchRecall(Metric):
    """
    Computes the recall of injected signals (fraction
    of total injected signals recovered) at the detection
    statistic threshold given by each of the glitch
    specificity values (fraction of glitches rejected)
    specified.

    Args:
        specs:
            Glitch specificity values against which to
            compute detection statistic thresholds.
            Represents the fraction of glitches that would
            be rejected at a given threshold.
    """

    name = "recall vs. glitches"
    param = "specificity"

    def __init__(self, specs: Sequence[float]) -> None:
        for i in specs:
            assert 0 <= i <= 1
        super().__init__(specs)
        self.register_buffer("specs", torch.Tensor(specs))

    def call(self, _, glitches, signal):
        qs = torch.quantile(glitches.unsqueeze(1), self.specs)
        recall = (signal.unsqueeze(1) >= qs).sum(0) / len(signal)
        return recall


@dataclass
class Recorder:
    """Callable which handles metric evaluation and model checkpointing

    Given a model, its most recent train loss measurement,
    and tensors of predictions on background, glitch, and
    signal datasets, this evaluates a series of `Metrics`
    and records them alongside the training loss in a
    dictionary for recording training progress. The weights
    of the best performing model according to the monitored
    metric will be saved in `logdir` as `weights.pt`. Will also
    optionally check for early stopping and perform periodic
    checkpointing of the model weights.

    Args:
        logdir:
            Directory to save artifacts to, including best-performing
            model weights, training history, and an optional subdirectory
            for periodic checkpointing.
        monitor:
            Metric which will be used for deciding which model
            weights are "best-performing"
        threshold:
            Threshold value for the monitored metric at which
            to evaluate the model's performance.
        additional:
            Any additional metrics to evaluate during training
        early_stop:
            Number of epochs to go without an improvement in
            the monitored metric at the given threshold before
            training should be terminated. If left as `None`,
            never prematurely terminate training.
        checkpoint_every:
            Indicates a frequency at which to checkpoint model
            weights in terms of number of epochs. If left as
            `None`, model weights won't be checkpointed and
            only the weights which produced the best score
            on the monitored metric at the indicated threshold
            will be saved.
    """

    logdir: Path
    monitor: Metric
    threshold: float
    additional: Optional[Sequence[Metric]] = None
    early_stop: Optional[int] = None
    checkpoint_every: Optional[int] = None

    def __post_init__(self):
        if self.threshold not in self.monitor:
            raise ValueError(
                "Metric {} has no threshold {}".format(
                    self.monitor.name, self.threshold
                )
            )
        self.history = {"train_loss": []}

        if self.checkpoint_every is not None:
            (self.logdir / "checkpoints").mkdir(parents=True, exist_ok=True)

        self.best = -1  # best monitored metric value so far
        self._i = 0  # epoch counter
        self._since_last = 0  # epochs since last best monitored metric

    def checkpoint(
        self, model: torch.nn.Module, metrics: Dict[str, float]
    ) -> bool:
        self._i += 1
        with open(self.logdir / "history.pkl", "wb") as f:
            pickle.dump(self.history, f)

        if (
            self.checkpoint_every is not None
            and not self._i % self.checkpoint_every
        ):
            epoch = str(self._i).zfill(4)
            fname = self.logdir / "checkpoints" / f"epoch_{epoch}.pt"
            torch.save(model.state_dict(), fname)

        if self.monitor[self.threshold] > self.best:
            fname = self.logdir / "weights.pt"
            torch.save(model.state_dict(), fname)
            self._since_last = 0
            self.best = self.monitor[self.threshold]
        elif self.early_stop is not None:
            self._since_last += 1
            if self._since_last >= self.early_stop:
                return True
        return False

    def __call__(
        self,
        model: torch.nn.Module,
        train_loss: float,
        background: torch.Tensor,
        signal: torch.Tensor,
    ) -> bool:
        self.history["train_loss"].append(train_loss)
        self.monitor(background, signal)
        self.monitor.update(self.history)

        msg = f"Summary:\nTrain loss: {train_loss:0.3e}"
        msg += f"\nValidation {self.monitor}"
        if self.additional is not None:
            for metric in self.additional:
                metric(background, signal)
                metric.update(self.history)
                msg += f"\nValidation {metric}"
        logging.info(msg)

        return self.checkpoint(model, self.history)


def inject_waveforms(
    background: Tuple[np.ndarray, np.ndarray],
    waveforms: np.ndarray,
    signal_times: np.ndarray,
) -> np.ndarray:

    """
    Inject a set of signals into background data

    Args:
        background:
            A tuple (t, data) of np.ndarray arrays.
            The first tuple is an array of times.
            The second tuple is the background strain values
            sampled at those times.
        waveforms:
            An np.ndarary of shape (n_waveforms, waveform_size)
            that contains the waveforms to inject
        signal_times: np.ndarray,:
            An array of times where signals will be injected. Corresponds to
            first sample of waveforms.
    Returns
        A dictionary where the key is an interferometer and the value
        is a timeseries with the signals injected
    """

    times, data = background[0].copy(), background[1].copy()
    if len(times) != len(data):
        raise ValueError(
            "The times and background arrays must be the same length"
        )

    sample_rate = 1 / (times[1] - times[0])
    # create matrix of indices of waveform_size for each waveform
    num_waveforms, waveform_size = waveforms.shape
    idx = np.arange(waveform_size)[None] - int(waveform_size // 2)
    idx = np.repeat(idx, len(waveforms), axis=0)

    # offset the indices of each waveform corresponding to their time offset
    time_diffs = signal_times - times[0]
    idx_diffs = (time_diffs * sample_rate).astype("int64")
    idx += idx_diffs[:, None]

    # flatten these indices and the signals out to 1D
    # and then add them in-place all at once
    idx = idx.reshape(-1)
    waveforms = waveforms.reshape(-1)
    data[idx] += waveforms

    return data


def repeat(X: torch.Tensor, max_num: int) -> torch.Tensor:
    """
    Repeat a 3D tensor `X` along its 0 dimension until
    it has length `max_num`.
    """

    repeats = ceil(max_num / len(X))
    X = X.repeat(repeats, 1, 1)
    return X[:max_num]


def make_glitches(
    glitches: Sequence[np.ndarray],
    background: torch.Tensor,
    glitch_frac: float,
) -> torch.Tensor:
    if len(glitches) != background.size(1):
        raise ValueError(
            "Number of glitch tensors {} doesn't match number "
            "of interferometers {}".format(len(glitches), background.size(1))
        )

    h1_glitches, l1_glitches = map(torch.Tensor, glitches)
    num_h1, num_l1 = len(h1_glitches), len(l1_glitches)
    num_glitches = num_h1 + num_l1
    num_coinc = int(glitch_frac**2 * num_glitches / (1 + glitch_frac**2))
    if num_coinc > min(num_h1, num_l1):
        raise ValueError(
            f"There are more coincident glitches ({num_coinc}) that there "
            "are glitches in one of the ifo glitch datasets. Hanford: "
            "{num_h1}, Livingston: {num_l1}"
        )

    h1_coinc, h1_glitches = split(h1_glitches, num_coinc / num_h1, 0)
    l1_coinc, l1_glitches = split(l1_glitches, num_coinc / num_l1, 0)
    coinc = torch.stack([h1_coinc, l1_coinc], axis=1)
    num_h1, num_l1 = len(h1_glitches), len(l1_glitches)
    num_glitches = num_h1 + num_l1 + num_coinc

    # if we need to create duplicates of some of our
    # background to make this work, figure out how many
    background = repeat(background, num_glitches)

    # now insert the glitches
    kernel_size = background.size(2)
    start = h1_glitches.shape[-1] // 2 - kernel_size // 2
    slc = slice(start, start + kernel_size)

    background[:num_h1, 0] = h1_glitches[:, slc]
    background[num_h1:-num_coinc, 1] = l1_glitches[:, slc]
    background[-num_coinc:] = coinc[:, :, slc]

    return background


class Validator:
    def __init__(
        self,
        recorder: Callable,
        background: np.ndarray,
        glitches: Sequence[np.ndarray],
        injector: "BBHNetWaveformInjection",
        snr_thresh: float,
        highpass: float,
        kernel_length: float,
        stride: float,
        spacing: float,
        glitch_frac: float,
        sample_rate: float,
        batch_size: int,
        integration_window_size: float,
        cluster_window_size: float,
        device: str,
    ) -> None:
        """Callable class for evaluating model validation scores

        Computes model outputs on timeslides and signals
        injected into those timeslides at call time and passes them
        to a `recorder` for evaluation and checkpointing.

        Args:
            recorder:
                Callable which accepts the model being evaluated,
                most recent training loss, and the predictions on
                background, glitch, and signal data and returns a
                boolean indicating whether training should terminate
                or not.
            background:
                2D timeseries of interferometer strain data. Will be
                split into windows of length `kernel_length`, sampled
                every `stride` seconds. Glitch and signal data will be
                augmented onto this dataset
            glitches:
                Each element of `glitches` should be a 2D array
                containing glitches from each interferometer, with
                the 0th axis used to enumerate individual glitches
                and the 1st axis corresponding to time.
            injector:
                A `BBHNetWaveformInjection` object for sampling
                waveforms. Preferring this to an array of waveforms
                for the time being so that we can potentially do
                on-the-fly SNR reweighting during validation. For now,
                waveforms are sampled with no SNR reweighting.
            snr_thresh:
                Lower snr threshold for waveforms. Waveforms that have snrs
                below this threshold will be rescaled to this threshold.
            highpass:
                Low frequency cutoff used when evaluating waveform snr.
            kernel_length:
                The length of windows to sample from the background
                in seconds.
            stride:
                The number of seconds between sampled background
                windows.
            sample_rate:
                The rate at which all relevant data arrays have
                been sampled in Hz
            batch_size:
                Number of samples over which to compute model
                predictions at call time
            glitch_frac:
                Rate at which interferometer channels are
                replaced with glitches during training. Used
                to compute the fraction of validation glitches
                which should be sampled coincidentally.
            device:
                Device on which to perform model evaluation.
        """
        self.device = device
        self.spacing = spacing
        self.batch_size = batch_size
        self.integration_window_size = integration_window_size
        self.cluster_window_size = cluster_window_size

        # move all our validation metrics to the appropriate device
        recorder.monitor.to(device)
        if recorder.additional is not None:
            for metric in recorder.additional:
                metric.to(device)
        self.recorder = recorder

        self.kernel_size = int(kernel_length * sample_rate)
        self.stride_size = int(stride * sample_rate)

        # sample waveforms and rescale snrs below threshold
        waveforms, _ = injector.sample(-1)
        df = 1 / (waveforms.shape[-1] / sample_rate)
        psds = []
        for back in background:
            psd = normalize_psd(back, df, sample_rate)
            psds.append(psd)
        psds = torch.tensor(np.stack(psds), dtype=torch.float64)

        snrs = compute_network_snr(waveforms, psds, sample_rate, highpass)
        mask = snrs < snr_thresh
        logging.info(
            f"Rescaling {mask.sum()} out of {len(snrs)} "
            "waveforms below snr threshold"
        )
        snrs[mask] = snr_thresh
        self.waveforms = reweight_snrs(
            waveforms, snrs, psds, sample_rate, highpass
        )

        # crop ends of waveforms where there is limited signal power
        # so we can cram more waveforms in a given background interval
        start = waveforms.shape[-1] // 2 - int(self.kernel_size)
        stop = start + int(1.5 * self.kernel_size)

        self.background = background
        self.waveforms = waveforms[:, :, start:stop].numpy()
        self.waveform_duration = self.waveforms.shape[-1] / sample_rate

        self.glitch_background = make_glitches(
            glitches, background, glitch_frac
        )

    def integrate_and_cluster(self, y: np.ndarray) -> np.ndarray:
        """
        Convolve predictions with boxcar filter
        to get local integration, slicing off of
        the last values so that timeseries represents
        integration of _past_ data only.
        "Full" convolution means first few samples are
        integrated with 0s, so will have a lower magnitude
        than they technically should.
        """
        window_size = int(self.integration_window_length * self.sample_rate)
        window = np.ones((window_size,)) / window_size
        y = np.convolve(y, window, mode="full")[: -window_size + 1]

        # subtract off the time required for
        # the coalescence to exit the filter
        # padding and enter the input kernel
        # to the neural network
        t0 = -self.fduration / 2

        # now subtract off the time required
        # for the integration window to
        # hit its maximum value
        t0 -= self.integration_window_length

        window_size = int(self.cluster_window_length * self.sample_rate / 2)
        i = np.argmax(y[:window_size])
        events, times = [], []
        while i < len(y):
            val = y[i]
            window = y[i + 1 : i + 1 + window_size]
            if any(val <= window):
                i += np.argmax(window) + 1
            else:
                events.append(val)
                t = t0 + i / self.sample_rate
                times.append(t)
                i += window_size + 1

        events = np.array(events)
        times = np.array(times)
        return events, times

    def recover(
        self,
        detection_statistics: np.ndarray,
        event_times: np.ndarray,
        injection_times: np.ndarray,
    ):
        """
        Recover the detection statistics corresponding to injected events
        """
        diffs = np.abs(injection_times[:, None] - event_times)
        closest = diffs.argmin(axis=-1)
        recovered = detection_statistics[closest]
        return recovered

    def evaluate_glitches(self, model, X):

        dataset = torch.utils.data.TensorDataset(X)
        loader = torch.utils.data.DataLoader(
            dataset,
            pin_memory=True,
            batch_size=self.batch_size,
            pin_memory_device=self.device,
        )

        preds = []
        for (X,) in loader:
            X = X.to(self.device)
            y_hat = model(X)[:, 0]
            preds.append(y_hat)
        return torch.cat(preds)

    def timeslide_injections(
        self,
        model: torch.nn.Module,
    ):
        idx, shift, Tb = 0, 0, 0
        background_events, foreground_events = [], []

        # iteratively create timeslides and injections
        # onto those timeslides until we've exhausted all
        # of our validation waveforms
        while idx < len(self.waveforms):
            start = int(shift * self.sample_rate)
            X = np.stack(
                [self.background[0, :-start], self.background[1, start:]]
            )
            times = np.arange(X.shape[-1]) / self.sample_rate

            injection_times = np.arange(
                self.waveform_duration,
                X.shape[-1] - self.waveform_duration,
                self.spacing,
            )
            num_waveforms = len(injection_times)
            waveforms = self.waveforms[idx : idx + num_waveforms]
            X_inj = inject_waveforms((times, X), waveforms, injection_times)

            X = InMemoryDataset(
                X,
                self.kernel_size,
                coincident=True,
                batch_size=self.batch_size,
                shuffle=False,
                stride=self.stride,
            )
            X_inj = InMemoryDataset(
                X_inj,
                self.kernel_size,
                coincident=True,
                batch_size=self.batch_size,
                shuffle=False,
                stride=self.stride,
            )

            y, y_inj = [], []
            for x, x_inj in zip(X, X_inj):
                y.append(model(x)[:, 0])
                y_inj.append(model(x_inj)[:, 0])

            y = torch.cat(y)
            y_inj = torch.cat(y_inj)

            y, _ = self.integrate_and_cluster(
                y, self.integration_window_size, self.cluster_window_size
            )
            y_inj, times = self.integrate_and_cluster(
                y_inj, self.integration_window_size, self.cluster_window_size
            )
            y_inj = self.recover(y_inj, times, injection_times)

            background_events.append(y)
            foreground_events.append(y_inj)
            Tb += X.shape[-1] / self.sample_rate
            shift += 1
            idx += num_waveforms
        return background_events, foreground_events, Tb

    @torch.no_grad()
    def __call__(self, model: torch.nn.Module, train_loss: float):
        background_events, foreground_events = self.timeslide_injections(
            model, self.background, self.waveforms, self.times
        )
        glitch_events = self.evaluate_glitches(self.glitch_background, model)
        return self.recorder(
            model,
            train_loss,
            background_events,
            glitch_events,
            foreground_events,
        )
