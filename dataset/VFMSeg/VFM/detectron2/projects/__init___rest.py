# Copyright (c) Facebook, Inc. and its affiliates.
import importlib
from pathlib import Path
from importlib import abc
_PROJECTS = {
    "point_rend": "PointRend",
    "deeplab": "DeepLab",
    "panoptic_deeplab": "Panoptic-DeepLab",
    "ifc": "IFC",
    "hipie": "HIPIE"
}
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent / "projects"

if _PROJECT_ROOT.is_dir():
    # This is true only for in-place installation (pip install -e, setup.py develop),
    # where setup(package_dir=) does not work: https://github.com/pypa/setuptools/issues/230

    class _D2ProjectsFinder(abc.MetaPathFinder):

    import sys

    sys.meta_path.append(_D2ProjectsFinder())