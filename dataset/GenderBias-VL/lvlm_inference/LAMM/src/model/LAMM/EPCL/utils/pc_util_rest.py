# Copyright (c) Facebook, Inc. and its affiliates.

""" Utility functions for processing point clouds.

Author: Charles R. Qi and Or Litany
"""

import os
import sys
import torch

# Point cloud IO
import numpy as np
from plyfile import PlyData, PlyElement

# Mesh IO
import trimesh

# ----------------------------------------
# Point Cloud Sampling
# ----------------------------------------




# ----------------------------------------
# Simple Point manipulations
# ----------------------------------------























    scene = trimesh.scene.Scene()
    for box in scene_bbox:
        scene.add_geometry(convert_box_to_trimesh_fmt(box))

    mesh_list = trimesh.util.concatenate(scene.dump())
    # save to ply file
    trimesh.io.export.export_mesh(mesh_list, out_filename, file_type="ply")

    return


def write_oriented_bbox(scene_bbox, out_filename, colors=None):
    """Export oriented (around Z axis) scene bbox to meshes
    Args:
        scene_bbox: (N x 7 numpy array): xyz pos of center and 3 lengths (dx,dy,dz)
            and heading angle around Z axis.
            Y forward, X right, Z upward. heading angle of positive X is 0,
            heading angle of positive Y is 90 degrees.
        out_filename: (string) filename
    """



    if colors is not None:
        if colors.shape[0] != len(scene_bbox):
            colors = [colors for _ in range(len(scene_bbox))]
            colors = np.array(colors).astype(np.uint8)
        assert colors.shape[0] == len(scene_bbox)
        assert colors.shape[1] == 4

    scene = trimesh.scene.Scene()
    for idx, box in enumerate(scene_bbox):
        box_tr = convert_oriented_box_to_trimesh_fmt(box)
        if colors is not None:
            box_tr.visual.main_color[:] = colors[idx]
            box_tr.visual.vertex_colors[:] = colors[idx]
            for facet in box_tr.facets:
                box_tr.visual.face_colors[facet] = colors[idx]
        scene.add_geometry(box_tr)

    mesh_list = trimesh.util.concatenate(scene.dump())
    # save to ply file
    trimesh.io.export.export_mesh(mesh_list, out_filename, file_type="ply")

    return


def write_oriented_bbox_camera_coord(scene_bbox, out_filename):
    """Export oriented (around Y axis) scene bbox to meshes
    Args:
        scene_bbox: (N x 7 numpy array): xyz pos of center and 3 lengths (dx,dy,dz)
            and heading angle around Y axis.
            Z forward, X rightward, Y downward. heading angle of positive X is 0,
            heading angle of negative Z is 90 degrees.
        out_filename: (string) filename
    """



    scene = trimesh.scene.Scene()
    for box in scene_bbox:
        scene.add_geometry(convert_oriented_box_to_trimesh_fmt(box))

    mesh_list = trimesh.util.concatenate(scene.dump())
    # save to ply file
    trimesh.io.export.export_mesh(mesh_list, out_filename, file_type="ply")

    return


def write_lines_as_cylinders(pcl, filename, rad=0.005, res=64):
    """Create lines represented as cylinders connecting pairs of 3D points
    Args:
        pcl: (N x 2 x 3 numpy array): N pairs of xyz pos
        filename: (string) filename for the output mesh (ply) file
        rad: radius for the cylinder
        res: number of sections used to create the cylinder
    """
    scene = trimesh.scene.Scene()
    for src, tgt in pcl:
        # compute line
        vec = tgt - src
        M = trimesh.geometry.align_vectors([0, 0, 1], vec, False)
        vec = tgt - src  # compute again since align_vectors modifies vec in-place!
        M[:3, 3] = 0.5 * src + 0.5 * tgt
        height = np.sqrt(np.dot(vec, vec))
        scene.add_geometry(
            trimesh.creation.cylinder(
                radius=rad, height=height, sections=res, transform=M
            )
        )
    mesh_list = trimesh.util.concatenate(scene.dump())
    trimesh.io.export.export_mesh(mesh_list, "%s.ply" % (filename), file_type="ply")