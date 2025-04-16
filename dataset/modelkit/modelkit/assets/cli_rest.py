import glob
import os
import re
import sys
import tempfile

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn
from rich.table import Table
from rich.tree import Tree

try:
    from modelkit.assets.drivers.gcs import GCSStorageDriver, GCSStorageDriverSettings

    has_gcs = True
except ModuleNotFoundError:
    has_gcs = False
try:
    from modelkit.assets.drivers.s3 import S3StorageDriver, S3StorageDriverSettings

    has_s3 = True
except ModuleNotFoundError:
    has_s3 = False
from modelkit.assets.errors import ObjectDoesNotExistError
from modelkit.assets.manager import AssetsManager
from modelkit.assets.remote import DriverNotInstalledError, StorageProvider
from modelkit.assets.settings import AssetSpec


@click.group("assets")


storage_url_re = (
    r"(?P<storage_prefix>[\w]*)://(?P<bucket_name>[\w\-]+)/(?P<object_name>.+)"
)








@assets_cli.command("new")
@click.argument("asset_path")
@click.argument("asset_spec")
@click.option("--storage-prefix", envvar="MODELKIT_STORAGE_PREFIX")
@click.option("--dry-run", is_flag=True)




@assets_cli.command("update")
@click.argument("asset_path")
@click.argument("asset_spec")
@click.option(
    "--bump-major",
    is_flag=True,
    help="[minor-major] Push a new major version (1.0, 2.0, etc.)",
)
@click.option("--storage-prefix", envvar="MODELKIT_STORAGE_PREFIX")
@click.option("--dry-run", is_flag=True)




@assets_cli.command("list")
@click.option("--storage-prefix", envvar="MODELKIT_STORAGE_PREFIX")


@assets_cli.command("fetch")
@click.argument("asset")
@click.option("--download", is_flag=True)