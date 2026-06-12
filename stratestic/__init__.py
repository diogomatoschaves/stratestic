from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("stratestic-private")
except PackageNotFoundError:  # not installed (e.g. running from a checkout)
    __version__ = "0.0.0"
