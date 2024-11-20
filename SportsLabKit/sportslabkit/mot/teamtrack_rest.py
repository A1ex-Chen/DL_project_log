import cv2
import numpy as np
import torch

from sportslabkit.logger import logger
from sportslabkit.matching import MotionVisualMatchingFunction, SimpleMatchingFunction
from sportslabkit.metrics import CosineCMM, IoUCMM
from sportslabkit.mot.base import MultiObjectTracker


class TeamTracker(MultiObjectTracker):
    """TeamTrack"""



        # obs_len = self.motion_model.model.input_channels // 2
        # if x.shape[1] < obs_len:
        #     x = torch.cat([x] + [x[:, 0, :].unsqueeze(1)] * (obs_len - x.shape[1]), dim=1)
        # else:
        #     x = x[:, -obs_len:]
        # X.append(x)

    # if self.multi_target_motion_model and len(X) > 0:
    #     X = torch.stack(X, dim=2)
    #     with torch.no_grad():
    #         Y = self.motion_model(X).numpy().squeeze(0)
    #     for i, tracklet in enumerate(tracklets):
    #         tracklet.update_state("pitch_coordinates", Y[i])




    @property

    @property