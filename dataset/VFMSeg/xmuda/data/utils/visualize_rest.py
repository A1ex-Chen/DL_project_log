import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict

from xmuda.data.utils.turbo_cmap import interpolate_or_clip, turbo_colormap_data


# all classes
NUSCENES_COLOR_PALETTE = [
    (255, 158, 0),  # car
    (255, 158, 0),  # truck
    (255, 158, 0),  # bus
    (255, 158, 0),  # trailer
    (255, 158, 0),  # construction_vehicle
    (0, 0, 230),  # pedestrian
    (255, 61, 99),  # motorcycle
    (255, 61, 99),  # bicycle
    (0, 0, 0),  # traffic_cone
    (0, 0, 0),  # barrier
    (200, 200, 200),  # background
]

# classes after merging (as used in xMUDA)
NUSCENES_COLOR_PALETTE_SHORT = [
    (255, 158, 0),  # vehicle
    (0, 0, 230),  # pedestrian
    (255, 61, 99),  # bike
    (0, 0, 0),  # traffic boundary
    (200, 200, 200),  # background
]

NUSCENES_LIDARSEG_COLOR_PALETTE_DICT = OrderedDict([
    ('ignore', (0, 0, 0)),  # Black
    ('barrier', (112, 128, 144)),  # Slategrey
    ('bicycle', (220, 20, 60)),  # Crimson
    ('bus', (255, 127, 80)),  # Coral
    ('car', (255, 158, 0)),  # Orange
    ('construction_vehicle', (233, 150, 70)),  # Darksalmon
    ('motorcycle', (255, 61, 99)),  # Red
    ('pedestrian', (0, 0, 230)),  # Blue
    ('traffic_cone', (47, 79, 79)),  # Darkslategrey
    ('trailer', (255, 140, 0)),  # Darkorange
    ('truck', (255, 99, 71)),  # Tomato
    ('driveable_surface', (0, 207, 191)),  # nuTonomy green
    ('other_flat', (175, 0, 75)),
    ('sidewalk', (75, 0, 75)),
    ('terrain', (112, 180, 60)),
    ('manmade', (222, 184, 135)),  # Burlywood
    ('vegetation', (0, 175, 0))  # Green
])

NUSCENES_LIDARSEG_COLOR_PALETTE = list(NUSCENES_LIDARSEG_COLOR_PALETTE_DICT.values())

NUSCENES_LIDARSEG_COLOR_PALETTE_SHORT = [
    NUSCENES_LIDARSEG_COLOR_PALETTE_DICT['car'],  # vehicle
    NUSCENES_LIDARSEG_COLOR_PALETTE_DICT['driveable_surface'],
    NUSCENES_LIDARSEG_COLOR_PALETTE_DICT['sidewalk'],
    NUSCENES_LIDARSEG_COLOR_PALETTE_DICT['terrain'],
    NUSCENES_LIDARSEG_COLOR_PALETTE_DICT['manmade'],
    NUSCENES_LIDARSEG_COLOR_PALETTE_DICT['vegetation'],
    NUSCENES_LIDARSEG_COLOR_PALETTE_DICT['ignore']
]


# all classes
A2D2_COLOR_PALETTE_SHORT = [
    (255, 0, 0),  # car
    (255, 128, 0),  # truck
    (182, 89, 6),  # bike
    (204, 153, 255),  # person
    (255, 0, 255),  # road
    (150, 150, 200),  # parking
    (180, 150, 200),  # sidewalk
    (241, 230, 255),  # building
    (147, 253, 194),  # nature
    (255, 246, 143),  # other-objects
    (0, 0, 0)  # ignore
]

# colors as defined in https://github.com/PRBonn/semantic-kitti-api/blob/master/config/semantic-kitti.yaml
SEMANTIC_KITTI_ID_TO_BGR = {  # bgr
  0: [0, 0, 0],
  1: [0, 0, 255],
  10: [245, 150, 100],
  11: [245, 230, 100],
  13: [250, 80, 100],
  15: [150, 60, 30],
  16: [255, 0, 0],
  18: [180, 30, 80],
  20: [255, 0, 0],
  30: [30, 30, 255],
  31: [200, 40, 255],
  32: [90, 30, 150],
  40: [255, 0, 255],
  44: [255, 150, 255],
  48: [75, 0, 75],
  49: [75, 0, 175],
  50: [0, 200, 255],
  51: [50, 120, 255],
  52: [0, 150, 255],
  60: [170, 255, 150],
  70: [0, 175, 0],
  71: [0, 60, 135],
  72: [80, 240, 150],
  80: [150, 240, 255],
  81: [0, 0, 255],
  99: [255, 255, 50],
  252: [245, 150, 100],
  256: [255, 0, 0],
  253: [200, 40, 255],
  254: [30, 30, 255],
  255: [90, 30, 150],
  257: [250, 80, 100],
  258: [180, 30, 80],
  259: [255, 0, 0],
}
SEMANTIC_KITTI_COLOR_PALETTE = [SEMANTIC_KITTI_ID_TO_BGR[id] if id in SEMANTIC_KITTI_ID_TO_BGR.keys() else [0, 0, 0]
                                for id in range(list(SEMANTIC_KITTI_ID_TO_BGR.keys())[-1] + 1)]


# classes after merging (as used in xMUDA)
SEMANTIC_KITTI_COLOR_PALETTE_SHORT_BGR = [
    [245, 150, 100],  # car
    [180, 30, 80],  # truck
    [150, 60, 30],  # bike
    [30, 30, 255],  # person
    [255, 0, 255],  # road
    [255, 150, 255],  # parking
    [75, 0, 75],  # sidewalk
    [0, 200, 255],  # building
    [0, 175, 0],  # nature
    [255, 255, 50],  # other-objects
    [0, 0, 0],  # ignore
]
SEMANTIC_KITTI_COLOR_PALETTE_SHORT = [(c[2], c[1], c[0]) for c in SEMANTIC_KITTI_COLOR_PALETTE_SHORT_BGR]

VIRTUAL_KITTI_COLOR_PALETTE = [
    [0, 175, 0],  # vegetation_terrain
    [255, 200, 0],  # building
    [255, 0, 255],  # road
    [50, 255, 255],  # other-objects
    [80, 30, 180],  # truck
    [100, 150, 245],  # car
    [0, 0, 0],  # ignore
]

WAYMO_COLOR_PALETTE = [
    (200, 200, 200),  # unknown
    (255, 158, 0),  # vehicle
    (0, 0, 230),  # pedestrian
    (50, 255, 255),  # sign
    (255, 61, 99),  # cyclist
    (0, 0, 0),  # ignore
]










        # plt.savefig('/Labs/Scripts/3DPC/exp_xmuda_journal/out/nuscenes_lidarseg/usa_singapore/uda/xmuda/images/seg_res-n015-2018-08-02-17-16-37+0800__CAM_FRONT__1533201470412460.jpg')
