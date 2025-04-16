from abc import ABC, abstractmethod

import numpy as np
from filterpy.kalman import KalmanFilter


class FilterFactory(ABC):
    """Abstract class representing a generic Filter factory

    Subclasses must implement the method `create_filter`
    """

    @abstractmethod


class FilterPyKalmanFilterFactory(FilterFactory):
    """
    This class can be used either to change some parameters of the [KalmanFilter](https://filterpy.readthedocs.io/en/latest/kalman/KalmanFilter.html)
    that the tracker uses, or to fully customize the predictive filter implementation to use (as long as the methods and properties are compatible).

    The former case only requires changing the default parameters upon tracker creation: `tracker = Tracker(..., filter_factory=FilterPyKalmanFilterFactory(R=100))`,
    while the latter requires creating your own class extending `FilterPyKalmanFilterFactory`, and rewriting its `create_filter` method to return your own customized filter.

    Parameters
    ----------
    R : float, optional
        Multiplier for the sensor measurement noise matrix, by default 4.0
    Q : float, optional
        Multiplier for the process uncertainty, by default 0.1
    P : float, optional
        Multiplier for the initial covariance matrix estimation, only in the entries that correspond to position (not speed) variables, by default 10.0

    See Also
    --------
    [`filterpy.KalmanFilter`](https://filterpy.readthedocs.io/en/latest/kalman/KalmanFilter.html).
    """




class NoFilter:




class NoFilterFactory(FilterFactory):
    """
    This class allows the user to try Norfair without any predictive filter or velocity estimation.

    This track only by comparing the position of the previous detections to the ones in the current frame.

    The throughput of this class in FPS is similar to the one achieved by the
    [`OptimizedKalmanFilterFactory`](#optimizedkalmanfilterfactory) class, so this class exists only for
    comparative purposes and it is not advised to use it for tracking on a real application.

    Parameters
    ----------
    FilterFactory : _type_
        _description_
    """



class OptimizedKalmanFilter:




class OptimizedKalmanFilterFactory(FilterFactory):
    """
    Creates faster Filters than [`FilterPyKalmanFilterFactory`][norfair.filter.FilterPyKalmanFilterFactory].

    It allows the user to create Kalman Filter optimized for tracking and set its parameters.

    Parameters
    ----------
    R : float, optional
        Multiplier for the sensor measurement noise matrix.
    Q : float, optional
        Multiplier for the process uncertainty.
    pos_variance : float, optional
        Multiplier for the initial covariance matrix estimation, only in the entries that correspond to position (not speed) variables.
    pos_vel_covariance : float, optional
        Multiplier for the initial covariance matrix estimation, only in the entries that correspond to the covariance between position and speed.
    vel_variance : float, optional
        Multiplier for the initial covariance matrix estimation, only in the entries that correspond to velocity (not position) variables.
    """

