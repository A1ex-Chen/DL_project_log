import os
import re
import typing

from modelkit.assets import errors
from modelkit.assets.versioning.major_minor import MajorMinorAssetsVersioningSystem
from modelkit.assets.versioning.simple_date import SimpleDateAssetsVersioningSystem
from modelkit.assets.versioning.versioning import AssetsVersioningSystem

GENERIC_ASSET_NAME_RE = (
    r"(([A-Z]:\\)|/)?[a-zA-Z0-9]([a-zA-Z0-9\-\_\.\/\\]*[a-zA-Z0-9])?"
)
GENERIC_ASSET_VERSION_RE = r"(?P<version>[0-9A-Za-z\.\-\_]+?)"


REMOTE_ASSET_RE = (
    rf"^(?P<name>{GENERIC_ASSET_NAME_RE})"
    rf"(:{GENERIC_ASSET_VERSION_RE})?"
    rf"(\[(?P<sub_part>(\/?{GENERIC_ASSET_NAME_RE})+)\])?$"
)


class AssetSpec:
    versioning: AssetsVersioningSystem






    @classmethod

    @classmethod

    @staticmethod
