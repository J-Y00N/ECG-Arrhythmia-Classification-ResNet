"""Inter-patient evaluation of a residual CNN on MIT-BIH heartbeats."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("ecg-arrhythmia-classification")
except PackageNotFoundError:  # a source tree that was never pip-installed
    __version__ = "0.0.0+unknown"