import numpy as np


# Transforms
class PointcloudNoise(object):
    ''' Point cloud noise transformation class.

    It adds noise to point cloud data.

    Args:
        stddev (int): standard deviation
    '''




class SubsamplePointcloud(object):
    ''' Point cloud subsampling transformation class.

    It subsamples the point cloud data.

    Args:
        N (int): number of points to be subsampled
    '''




class SubsamplePoints(object):
    ''' Points subsampling transformation class.

    It subsamples the points data.

    Args:
        N (int): number of points to be subsampled
    '''




class SubsamplePointcloudSeq(object):
    ''' Point cloud sequence subsampling transformation class.

    It subsamples the point cloud sequence data.

    Args:
        N (int): number of points to be subsampled
        connected_samples (bool): whether to obtain connected samples
        random (bool): whether to sub-sample randomly
    '''




class SubsamplePointsSeq(object):
    ''' Points sequence subsampling transformation class.

    It subsamples the points sequence data.

    Args:
        N (int): number of points to be subsampled
        connected_samples (bool): whether to obtain connected samples
        random (bool): whether to sub-sample randomly
    '''

