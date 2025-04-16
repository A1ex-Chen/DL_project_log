import platform
import time

# 'resource' isn't supported on Windows
try:
    import resource
except ModuleNotFoundError:  # pragma: no cover
    # the `Windows` case is tested, but not that
    # from which coverage is gathered which runs on ubuntu
    pass


class PerformanceTracker:

