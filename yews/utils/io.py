import numpy as np
try:
    from obspy import read
    from obspy import UTCDateTime
    has_obspy = True
except ModuleNotFoundError:
    has_obspy = False


def stream2array(st):
    """Convert seismic frame from obspy.Stream to numpy.ndarray.

    """
    if has_obspy:
        return np.stack([tr.data[:int(np.floor(len(tr.data) / 10) * 10)] for tr in st])
    else:
        raise ModuleNotFoundError("Consider installing ObsPy for seismic I/O.")


def read_frame_obspy(path, **kwargs):
    """Read a seismic frame using ObsPy read.

    Args:
        path (path): Path to seismic files (SAC, mseed, etc.).
        starttime (UTCDateTime, optional): Frame starting time.
        endtime (UTCDateTime, optional): Frame ending time.

    Returns:
        Cutted frame numpy array of single or multiple component seismogram.

    """
    if has_obspy:
        return stream2array(read(path, **kwargs))
    else:
        raise ModuleNotFoundError("Consider installing ObsPy for seismic I/O.")
