#!/usr/bin/env python3
import time

import click
import filelock


@click.command()
@click.argument("lock_path")
@click.argument("duration_s", type=float)


if __name__ == "__main__":
    wait()