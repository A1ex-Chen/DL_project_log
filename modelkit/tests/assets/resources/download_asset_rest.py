#!/usr/bin/env python3
import os
import sys

import click

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(root_dir)

from modelkit.assets.remote import StorageProvider  # NOQA  # isort:skip
from modelkit.assets.manager import AssetsManager  # NOQA  # isort:skip


@click.command()
@click.argument("assets_dir")
@click.argument("driver_path")
@click.argument("asset_name")


if __name__ == "__main__":
    download_asset()