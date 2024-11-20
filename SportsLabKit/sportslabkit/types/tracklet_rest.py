from __future__ import annotations

import hashlib
import uuid
from typing import Any

import pandas as pd

from sportslabkit.dataframe.bboxdataframe import BBoxDataFrame
from sportslabkit.logger import logger




class Tracklet:
    """Tracklet class to be u

    Stores observations of different types without making predictions about the next state.
    New observation types can be registered, and the tracker can be extended with more functionality if needed.

    Observations are stored in a dictionary, where the key is the name of the observation type and the value is a list of observations. The length of the list is equal to the number of steps the tracker has been alive. The first element of the list is the first observation, and the last element is the most recent observation.

    States are stored in a dictionary, where the key is the name of the state and the value is the most recent state. The state is an indication of the current state of the tracker.

    Args:
        max_staleness (int, optional): The maximum number of steps a tracker can be stale for before it is considered dead. Defaults to 5.
    Attributes:
        id (int): unique id of the tracker
        steps_alive (int): number of steps the tracker was alive
        staleness (float): number of steps since the last positive update
        global_step (int): number of steps since the start of the tracking process
        max_staleness (float): number of steps after which a tracker is considered dead
    """



















    # FIXME: Maybe refactor this to be override_current_observation?






    @property

    # def _update_observation(self, detection: Union[Detection, None], **kwargs) -> None:
    #     """Update all registered observation types with values from a detection object or keyword arguments.

    #     This method updates the values of all registered observation types for the tracker. If a detection object is provided, its attributes will be used to update the corresponding observations. Additionally, any keyword arguments passed to this method can be used to update the observation values, taking precedence over the values from the detection object.

    #     Args:
    #         detection (Union[Detection, None]): A Detection object containing the new values for the registered observation types,or None if no detection is available.
    #         **kwargs: Additional keyword arguments containing observation values to update, which will overwrite the values from the detection object if there are any overlaps.

    #     Raises:
    #         KeyError: If an observation key in the kwargs does not match any registered observation types.

    #     Example:
    #         # Assuming tracker is an instance of the SingleObjectTracker class and detection is a Detection object
    #         tracker._update_observation(detection, velocity=0.8)
    #     """
    #     new_observations = (
    #         {
    #             "box": detection.box,
    #             "score": detection.score,
    #             "class_id": detection.class_id,
    #             "feature": detection.feature,
    #         }
    #         if detection is not None
    #         else {}
    #     )
    #     new_observations.update(kwargs)

    #     for key in self._observations:
    #         if key in new_observations:
    #             self.update_observation(key, new_observations[key])
    #         else:
    #             self.update_observation(key, None)
    # def update(
    #     self,
    #     detection: Union[Detection, None],
    #     states: Optional[Dict[str, Any]] = None,
    # ) -> None:
    #     """Update the tracker with a new detection and optional additional observation values.

    #     Args:
    #         detection (Union[Detection, None]): Detection object to update the tracker with, or None if no detection is available.
    #         global_step (Optional[int], optional): The global step counter for the tracking process. Defaults to None.
    #         **kwargs: Additional keyword arguments containing observation values to update, which will overwrite the values from the detection object if there are any overlaps.

    #     Note:
    #         If there is no detection (i.e., detection is None), the tracker will still update the observation with None values.
    #         Additional observation values provided through keyword arguments will still be updated even if detection is None.

    #     Example:
    #         # Assuming tracker is an instance of the SingleObjectTracker class and detection is a Detection object
    #         tracker.update(detection, global_step=5, velocity=0.8)
    #     """

    #     self.steps_alive += 1
    #     if global_step is not None:
    #         self.global_step = int(global_step)
    #     else:
    #         self.global_step += 1

    #     if detection is not None:
    #         self.steps_positive += 1
    #         self.staleness = 0.0
    #         self.update_observation(detection, **kwargs)
    #     else:
    #         self.staleness += 1
    #         self.update_observation(None, **kwargs)