from importlib.metadata import PackageNotFoundError, version

# the import package is "stratestic" in both distributions; the public one
# is published as "stratestic" and the private fork as "stratestic-private"
for _distribution in ("stratestic", "stratestic-private"):
    try:
        __version__ = version(_distribution)
        break
    except PackageNotFoundError:
        __version__ = "0.0.0"  # not installed (e.g. running from a checkout)

del _distribution
