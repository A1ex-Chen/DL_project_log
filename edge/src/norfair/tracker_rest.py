from logging import warning
from typing import Any, Callable, Hashable, List, Optional, Sequence, Tuple, Union

import numpy as np
from rich import print

from norfair.camera_motion import CoordinatesTransformation

from .distances import (
    AVAILABLE_VECTORIZED_DISTANCES,
    ScalarDistance,
    get_distance_by_name,
)
from .filter import FilterFactory, OptimizedKalmanFilterFactory
from .utils import validate_points


class Tracker:
    """
    The class in charge of performing the tracking of the detections produced by a detector.

    Parameters
    ----------
    distance_function : Union[str, Callable[[Detection, TrackedObject], float]]
        Function used by the tracker to determine the distance between newly detected objects and the objects that are currently being tracked.
        This function should take 2 input arguments, the first being a [Detection][norfair.tracker.Detection], and the second a [TrackedObject][norfair.tracker.TrackedObject].
        It has to return a `float` with the distance it calculates.
        Some common distances are implemented in [distances][], as a shortcut the tracker accepts the name of these [predefined distances][norfair.distances.get_distance_by_name].
        Scipy's predefined distances are also accepted. A `str` with one of the available metrics in
        [`scipy.spatial.distance.cdist`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html).
    distance_threshold : float
        Defines what is the maximum distance that can constitute a match.
        Detections and tracked objects whose distances are above this threshold won't be matched by the tracker.
    hit_counter_max : int, optional
        Each tracked objects keeps an internal hit counter which tracks how often it's getting matched to a detection,
        each time it gets a match this counter goes up, and each time it doesn't it goes down.

        If it goes below 0 the object gets destroyed. This argument defines how large this inertia can grow,
        and therefore defines how long an object can live without getting matched to any detections, before it is displaced as a dead object, if no ReID distance function is implemented it will be destroyed.
    initialization_delay : Optional[int], optional
         Determines how large the object's hit counter must be in order to be considered as initialized, and get returned to the user as a real object.
         It must be smaller than `hit_counter_max` or otherwise the object would never be initialized.

         If set to 0, objects will get returned to the user as soon as they are detected for the first time,
         which can be problematic as this can result in objects appearing and immediately dissapearing.

         Defaults to `hit_counter_max / 2`
    pointwise_hit_counter_max : int, optional
        Each tracked object keeps track of how often the points it's tracking have been getting matched.
        Points that are getting matched (`pointwise_hit_counter > 0`) are said to be live, and points which aren't (`pointwise_hit_counter = 0`)
        are said to not be live.

        This is used to determine things like which individual points in a tracked object get drawn by [`draw_tracked_objects`][norfair.drawing.draw_tracked_objects] and which don't.
        This argument defines how large the inertia for each point of a tracker can grow.
    detection_threshold : float, optional
        Sets the threshold at which the scores of the points in a detection being fed into the tracker must dip below to be ignored by the tracker.
    filter_factory : FilterFactory, optional
        This parameter can be used to change what filter the [`TrackedObject`][norfair.tracker.TrackedObject] instances created by the tracker will use.
        Defaults to [`OptimizedKalmanFilterFactory()`][norfair.filter.OptimizedKalmanFilterFactory]
    past_detections_length : int, optional
        How many past detections to save for each tracked object.
        Norfair tries to distribute these past detections uniformly through the object's lifetime so they're more representative.
        Very useful if you want to add metric learning to your model, as you can associate an embedding to each detection and access them in your distance function.
    reid_distance_function: Optional[Callable[["TrackedObject", "TrackedObject"], float]]
        Function used by the tracker to determine the ReID distance between newly detected trackers and unmatched trackers by the distance function.

        This function should take 2 input arguments, the first being tracked objects in the initialization phase of type [`TrackedObject`][norfair.tracker.TrackedObject],
        and the second being tracked objects that have been unmatched of type [`TrackedObject`][norfair.tracker.TrackedObject]. It returns a `float` with the distance it
        calculates.
    reid_distance_threshold: float
        Defines what is the maximum ReID distance that can constitute a match.

        Tracked objects whose distance is above this threshold won't be merged, if they are the oldest tracked object will be maintained
        with the position of the new tracked object.
    reid_hit_counter_max: Optional[int]
        Each tracked object keeps an internal ReID hit counter which tracks how often it's getting recognized by another tracker,
        each time it gets a match this counter goes up, and each time it doesn't it goes down. If it goes below 0 the object gets destroyed.
        If used, this argument (`reid_hit_counter_max`) defines how long an object can live without getting matched to any detections, before it is destroyed.
    """



    @property

    @property





class _TrackedObjectFactory:
    global_count = 0






class TrackedObject:
    """
    The objects returned by the tracker's `update` function on each iteration.

    They represent the objects currently being tracked by the tracker.

    Users should not instantiate TrackedObjects manually;
    the Tracker will be in charge of creating them.

    Attributes
    ----------
    estimate : np.ndarray
        Where the tracker predicts the point will be in the current frame based on past detections.
        A numpy array with the same shape as the detections being fed to the tracker that produced it.
    id : Optional[int]
        The unique identifier assigned to this object by the tracker. Set to `None` if the object is initializing.
    global_id : Optional[int]
        The globally unique identifier assigned to this object. Set to `None` if the object is initializing
    last_detection : Detection
        The last detection that matched with this tracked object.
        Useful if you are storing embeddings in your detections and want to do metric learning, or for debugging.
    last_distance : Optional[float]
        The distance the tracker had with the last object it matched with.
    age : int
        The age of this object measured in number of frames.
    live_points :
        A boolean mask with shape `(n_points,)`. Points marked as `True` have recently been matched with detections.
        Points marked as `False` haven't and are to be considered stale, and should be ignored.

        Functions like [`draw_tracked_objects`][norfair.drawing.draw_tracked_objects] use this property to determine which points not to draw.
    initializing_id : int
        On top of `id`, objects also have an `initializing_id` which is the id they are given internally by the `Tracker`;
        this id is used solely for debugging.

        Each new object created by the `Tracker` starts as an uninitialized `TrackedObject`,
        which needs to reach a certain match rate to be converted into a full blown `TrackedObject`.
        `initializing_id` is the id temporarily assigned to `TrackedObject` while they are getting initialized.
    """



    @property

    @property

    @property

    @property


    @property








class Detection:
    """Detections returned by the detector must be converted to a `Detection` object before being used by Norfair.

    Parameters
    ----------
    points : np.ndarray
        Points detected. Must be a rank 2 array with shape `(n_points, n_dimensions)` where n_dimensions is 2 or 3.
    scores : np.ndarray, optional
        An array of length `n_points` which assigns a score to each of the points defined in `points`.

        This is used to inform the tracker of which points to ignore;
        any point with a score below `detection_threshold` will be ignored.

        This useful for cases in which detections don't always have every point present, as is often the case in pose estimators.
    data : Any, optional
        The place to store any extra data which may be useful when calculating the distance function.
        Anything stored here will be available to use inside the distance function.

        This enables the development of more interesting trackers which can do things like assign an appearance embedding to each
        detection to aid in its tracking.
    label : Hashable, optional
        When working with multiple classes the detection's label can be stored to be used as a matching condition when associating
        tracked objects with new detections. Label's type must be hashable for drawing purposes.
    embedding : Any, optional
        The embedding for the reid_distance.
    """

