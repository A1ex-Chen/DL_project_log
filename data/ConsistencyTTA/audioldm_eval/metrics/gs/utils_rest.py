try:
    import gudhi
except ImportError as e:
    import six

    error = e.__class__(
        "You are likely missing your GUDHI installation, "
        "you should visit http://gudhi.gforge.inria.fr/python/latest/installation.html "
        "for further instructions.\nIf you use conda, you can use\nconda install -c conda-forge gudhi"
    )
    six.raise_from(error, e)

import numpy as np
from scipy.spatial.distance import cdist  # , pdist, squareform
import matplotlib.pyplot as plt









