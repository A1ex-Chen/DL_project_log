import numpy
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy.random import random
from scipy.linalg import sqrtm




act1 = random(2048 * 2)
act1 = act1.reshape((2, 2048))
act2 = random(2048 * 2)
act2 = act2.reshape((2, 2048))
fid = calculate_fid(act1, act1)
print("FID (same): %.3f" % fid)
fid = calculate_fid(act1, act2)
print("FID (different): %.3f" % fid)