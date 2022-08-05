from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from gwpy.segments import (
    DataQualityDict,
    DataQualityFlag,
    Segment,
    SegmentList,
)
from gwpy.timeseries import TimeSeries
from timeslide_injections import main

from bbhnet.io.timeslides import TimeSlide


@pytest.fixture(params=["priors/nonspin_BBH.prior"])
def prior_file(request):
    return str(Path(__file__).resolve().parent / request.param)


@pytest.fixture(params=[60])
def spacing(request):
    return request.param


@pytest.fixture(params=[10])
def jitter(request):
    return request.param


@pytest.fixture(params=[8])
def buffer(request):
    return request.param


@pytest.fixture(params=[3])
def n_slides(request):
    return request.param


@pytest.fixture(params=[10, 4096])
def file_length(request):
    return request.param


@pytest.fixture(params=[32])
def fmin(request):
    return request.param


@pytest.fixture(params=[2048])
def sample_rate(request):
    return request.param


@pytest.fixture(params=[[0, 1]])
def shifts(request):
    return request.param


@pytest.fixture(params=[["H1", "L1"]])
def ifos(request):
    return request.param


@pytest.fixture(params=["HOFT_C01"])
def frame_type(request):
    return request.param


@pytest.fixture(params=["DCS-CALIB_STRAIN_CLEAN_C01"])
def channel(request):
    return request.param


@pytest.fixture(params=["DCS-ANALYSIS_READY_C01:1"])
def state_flag(request):
    return request.param


def test_timeslide_injections_no_segments(
    logdir,
    outdir,
    prior_file,
    spacing,
    jitter,
    buffer,
    n_slides,
    shifts,
    file_length,
    ifos,
    fmin,
    sample_rate,
    frame_type,
    channel,
):

    start = 1123456789
    stop = 1123457789

    times = np.arange(start, stop, 1 / sample_rate)
    n_samples = len(times)
    ts = TimeSeries(np.ones(n_samples), times=times)

    mock_ts = patch("gwpy.timeseries.TimeSeries.read", return_value=ts)

    mock_datafind = patch("gwdatafind.find_urls", return_value=None)

    with mock_datafind, mock_ts:
        outdir = main(
            start,
            stop,
            logdir,
            outdir,
            prior_file,
            spacing,
            jitter,
            buffer,
            n_slides,
            shifts,
            ifos,
            file_length,
            fmin,
            sample_rate,
            frame_type,
            channel,
        )

    timeslides = outdir.iterdir()
    timeslides = [slide for slide in timeslides if slide.is_dir()]
    timeslides = list(timeslides)
    assert len(timeslides) == n_slides

    for slide in timeslides:
        injection_ts = TimeSlide(slide, field="injection")
        background_ts = TimeSlide(slide, field="background")

        assert len(injection_ts.segments) == len(background_ts.segments)
        assert (injection_ts.path / "params.h5").exists()

        for segment in background_ts.segments:
            shifted_seg = segment.make_shift(slide)

            *y, t = segment.load(*ifos)
            *y_shifted, t_shifted = shifted_seg.load(*ifos)

            # time array refers to fixed ifo
            # so shouldnt change
            assert (t == t_shifted).all()

            # make we are shifting!
            for i, shift in enumerate(shifts):
                assert (
                    np.roll(y[i], int(shift * sample_rate)) == y_shifted[i]
                ).all()


def test_timeslide_injections_with_segments(
    logdir,
    outdir,
    prior_file,
    spacing,
    jitter,
    buffer,
    n_slides,
    shifts,
    file_length,
    ifos,
    fmin,
    sample_rate,
    frame_type,
    channel,
):

    start = 1000000000
    stop = 1000001000

    times = np.arange(start, stop, 1 / sample_rate)
    n_samples = len(times)
    ts = TimeSeries(np.arange(0, n_samples, 1), times=times)

    # create same segments for each ifo
    segments = DataQualityDict()
    segment_list = SegmentList(
        [
            Segment([start + 200, start + 300]),
            Segment([start + 400, start + 500]),
        ]
    )

    for i, ifo in enumerate(ifos):
        segments[f"{ifo}:{state_flag}"] = DataQualityFlag(active=segment_list)

    mock_ts = patch("gwpy.timeseries.TimeSeries.read", return_value=ts)
    mock_datafind = patch("gwdatafind.find_urls", return_value=None)
    mock_segments = patch(
        "gwpy.segments.DataQualityDict.query_dqsegdb", return_value=segments
    )

    with mock_datafind, mock_ts, mock_segments:
        outdir = main(
            start,
            stop,
            logdir,
            outdir,
            prior_file,
            spacing,
            jitter,
            buffer,
            n_slides,
            shifts,
            ifos,
            file_length,
            fmin,
            sample_rate,
            frame_type,
            channel,
            state_flag=state_flag,
        )

    # get all the timeslide directories
    timeslides = outdir.iterdir()
    timeslides = [slide.name for slide in timeslides if slide.is_dir()]
    timeslides = list(timeslides)

    # create timeslide
    injection_ts = TimeSlide(outdir / "dt-0.0-0.0", field="injection")
    background_ts = TimeSlide(outdir / "dt-0.0-0.0", field="background")

    assert len(injection_ts.segments) == len(background_ts.segments)

    # make sure there is as many directories
    # as requested slides
    assert len(timeslides) == n_slides

    for slide in timeslides:

        # unpack float values of shifts
        # from timeslide dir name
        shifts = list(map(float, slide.split("-")[1:]))

        # should be able to find same segment
        # in different time slide
        for segment in injection_ts.segments:
            segment.make_shift(slide)

        for segment in background_ts.segments:
            shifted_seg = segment.make_shift(slide)

            *y, t = segment.load(*ifos)
            *y_shifted, t_shifted = shifted_seg.load(*ifos)

            # time array refers to fixed ifo
            # so shouldnt change
            assert (t == t_shifted).all()

            # make we are shifting!
            for i, shift in enumerate(shifts):
                assert (
                    np.roll(y[i], int(shift * sample_rate)) == y_shifted[i]
                ).all()
