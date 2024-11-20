from typing import Optional, Sequence, Union

import numpy as np

from norfair.tracker import Detection, TrackedObject
from norfair.utils import warn_once

from .color import ColorLike, Palette, parse_color
from .drawer import Drawable, Drawer
from .utils import _build_text




# HACK add an alias to prevent error in the function below
# the deprecated draw_tracked_objects accepts a parameter called
# "draw_points" which overwrites the function "draw_points" from above
# since draw_tracked_objects needs to call this function, an alias
# is defined that can be used to call draw_points
_draw_points_alias = draw_points




# TODO: We used to have this function to debug
# migrate it to use Drawer and clean it up
# if possible maybe merge this functionality to the function above

# def draw_debug_metrics(
#     frame: np.ndarray,
#     objects: Sequence["TrackedObject"],
#     text_size: Optional[float] = None,
#     text_thickness: Optional[int] = None,
#     color: Optional[Tuple[int, int, int]] = None,
#     only_ids=None,
#     only_initializing_ids=None,
#     draw_score_threshold: float = 0,
#     color_by_label: bool = False,
#     draw_labels: bool = False,
# ):
#     """Draw objects with their debug information

#     It is recommended to set the input variable `objects` to `your_tracker_object.objects`
#     so you can also debug objects wich haven't finished initializing, and you get a more
#     complete view of what your tracker is doing on each step.
#     """
#     frame_scale = frame.shape[0] / 100
#     if text_size is None:
#         text_size = frame_scale / 10
#     if text_thickness is None:
#         text_thickness = int(frame_scale / 5)
#     radius = int(frame_scale * 0.5)

#     for obj in objects:
#         if (
#             not (obj.last_detection.scores is None)
#             and not (obj.last_detection.scores > draw_score_threshold).any()
#         ):
#             continue
#         if only_ids is not None:
#             if obj.id not in only_ids:
#                 continue
#         if only_initializing_ids is not None:
#             if obj.initializing_id not in only_initializing_ids:
#                 continue
#         if color_by_label:
#             text_color = Color.random(abs(hash(obj.label)))
#         elif color is None:
#             text_color = Color.random(obj.initializing_id)
#         else:
#             text_color = color
#         draw_position = _centroid(
#             obj.estimate[obj.last_detection.scores > draw_score_threshold]
#             if obj.last_detection.scores is not None
#             else obj.estimate
#         )

#         for point in obj.estimate:
#             cv2.circle(
#                 frame,
#                 tuple(point.astype(int)),
#                 radius=radius,
#                 color=text_color,
#                 thickness=-1,
#             )

#         # Distance to last matched detection
#         if obj.last_distance is None:
#             last_dist = "-"
#         elif obj.last_distance > 999:
#             last_dist = ">"
#         else:
#             last_dist = "{:.2f}".format(obj.last_distance)

#         # Distance to currently closest detection
#         if obj.current_min_distance is None:
#             current_min_dist = "-"
#         else:
#             current_min_dist = "{:.2f}".format(obj.current_min_distance)

#         # No support for multiline text in opencv :facepalm:
#         lines_to_draw = [
#             "{}|{}".format(obj.id, obj.initializing_id),
#             "a:{}".format(obj.age),
#             "h:{}".format(obj.hit_counter),
#             "ld:{}".format(last_dist),
#             "cd:{}".format(current_min_dist),
#         ]
#         if draw_labels:
#             lines_to_draw.append("l:{}".format(obj.label))

#         for i, line in enumerate(lines_to_draw):
#             draw_position = (
#                 int(draw_position[0]),
#                 int(draw_position[1] + i * text_size * 7 + 15),
#             )
#             cv2.putText(
#                 frame,
#                 line,
#                 draw_position,
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 text_size,
#                 text_color,
#                 text_thickness,
#                 cv2.LINE_AA,
#             )