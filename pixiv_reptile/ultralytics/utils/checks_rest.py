# Ultralytics YOLO 🚀, AGPL-3.0 license

import contextlib
import glob
import inspect
import math
import os
import platform
import re
import shutil
import subprocess
import time
from importlib import metadata
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import requests
import torch

from ultralytics.utils import (
    ASSETS,
    AUTOINSTALL,
    IS_COLAB,
    IS_JUPYTER,
    IS_KAGGLE,
    IS_PIP_PACKAGE,
    LINUX,
    LOGGER,
    ONLINE,
    PYTHON_VERSION,
    ROOT,
    TORCHVISION_VERSION,
    USER_CONFIG_DIR,
    Retry,
    SimpleNamespace,
    ThreadingLocked,
    TryExcept,
    clean_url,
    colorstr,
    downloads,
    emojis,
    is_github_action_running,
    url2file,
)
















@ThreadingLocked()




@TryExcept()































    s = " ".join(f'"{x}"' for x in pkgs)  # console string
    if s:
        if install and AUTOINSTALL:  # check environment variable
            n = len(pkgs)  # number of packages updates
            LOGGER.info(f"{prefix} Ultralytics requirement{'s' * (n > 1)} {pkgs} not found, attempting AutoUpdate...")
            try:
                t = time.time()
                assert ONLINE, "AutoUpdate skipped (offline)"
                LOGGER.info(attempt_install(s, cmds))
                dt = time.time() - t
                LOGGER.info(
                    f"{prefix} AutoUpdate success ✅ {dt:.1f}s, installed {n} package{'s' * (n > 1)}: {pkgs}\n"
                    f"{prefix} ⚠️ {colorstr('bold', 'Restart runtime or rerun command for updates to take effect')}\n"
                )
            except Exception as e:
                LOGGER.warning(f"{prefix} ❌ {e}")
                return False
        else:
            return False

    return True


def check_torchvision():
    """
    Checks the installed versions of PyTorch and Torchvision to ensure they're compatible.

    This function checks the installed versions of PyTorch and Torchvision, and warns if they're incompatible according
    to the provided compatibility table based on:
    https://github.com/pytorch/vision#installation.

    The compatibility table is a dictionary where the keys are PyTorch versions and the values are lists of compatible
    Torchvision versions.
    """

    # Compatibility table
    compatibility_table = {
        "2.3": ["0.18"],
        "2.2": ["0.17"],
        "2.1": ["0.16"],
        "2.0": ["0.15"],
        "1.13": ["0.14"],
        "1.12": ["0.13"],
    }

    # Extract only the major and minor versions
    v_torch = ".".join(torch.__version__.split("+")[0].split(".")[:2])
    if v_torch in compatibility_table:
        compatible_versions = compatibility_table[v_torch]
        v_torchvision = ".".join(TORCHVISION_VERSION.split("+")[0].split(".")[:2])
        if all(v_torchvision != v for v in compatible_versions):
            print(
                f"WARNING ⚠️ torchvision=={v_torchvision} is incompatible with torch=={v_torch}.\n"
                f"Run 'pip install torchvision=={compatible_versions[0]}' to fix torchvision or "
                "'pip install -U torch torchvision' to update both.\n"
                "For a full compatibility table see https://github.com/pytorch/vision#installation"
            )


def check_suffix(file="yolov8n.pt", suffix=".pt", msg=""):
    """Check file(s) for acceptable suffix."""
    if file and suffix:
        if isinstance(suffix, str):
            suffix = (suffix,)
        for f in file if isinstance(file, (list, tuple)) else [file]:
            s = Path(f).suffix.lower().strip()  # file suffix
            if len(s):
                assert s in suffix, f"{msg}{f} acceptable suffix is {suffix}, not {s}"


def check_yolov5u_filename(file: str, verbose: bool = True):
    """Replace legacy YOLOv5 filenames with updated YOLOv5u filenames."""
    if "yolov3" in file or "yolov5" in file:
        if "u.yaml" in file:
            file = file.replace("u.yaml", ".yaml")  # i.e. yolov5nu.yaml -> yolov5n.yaml
        elif ".pt" in file and "u" not in file:
            original_file = file
            file = re.sub(r"(.*yolov5([nsmlx]))\.pt", "\\1u.pt", file)  # i.e. yolov5n.pt -> yolov5nu.pt
            file = re.sub(r"(.*yolov5([nsmlx])6)\.pt", "\\1u.pt", file)  # i.e. yolov5n6.pt -> yolov5n6u.pt
            file = re.sub(r"(.*yolov3(|-tiny|-spp))\.pt", "\\1u.pt", file)  # i.e. yolov3-spp.pt -> yolov3-sppu.pt
            if file != original_file and verbose:
                LOGGER.info(
                    f"PRO TIP 💡 Replace 'model={original_file}' with new 'model={file}'.\nYOLOv5 'u' models are "
                    f"trained with https://github.com/ultralytics/ultralytics and feature improved performance vs "
                    f"standard YOLOv5 models trained with https://github.com/ultralytics/yolov5.\n"
                )
    return file


def check_model_file_from_stem(model="yolov8n"):
    """Return a model filename from a valid model stem."""
    if model and not Path(model).suffix and Path(model).stem in downloads.GITHUB_ASSETS_STEMS:
        return Path(model).with_suffix(".pt")  # add suffix, i.e. yolov8n -> yolov8n.pt
    else:
        return model


def check_file(file, suffix="", download=True, hard=True):
    """Search/download file (if necessary) and return path."""
    check_suffix(file, suffix)  # optional
    file = str(file).strip()  # convert to string and strip spaces
    file = check_yolov5u_filename(file)  # yolov5n -> yolov5nu
    if (
        not file
        or ("://" not in file and Path(file).exists())  # '://' check required in Windows Python<3.10
        or file.lower().startswith("grpc://")
    ):  # file exists or gRPC Triton images
        return file
    elif download and file.lower().startswith(("https://", "http://", "rtsp://", "rtmp://", "tcp://")):  # download
        url = file  # warning: Pathlib turns :// -> :/
        file = url2file(file)  # '%2F' to '/', split https://url.com/file.txt?auth
        if Path(file).exists():
            LOGGER.info(f"Found {clean_url(url)} locally at {file}")  # file already exists
        else:
            downloads.safe_download(url=url, file=file, unzip=False)
        return file
    else:  # search
        files = glob.glob(str(ROOT / "**" / file), recursive=True) or glob.glob(str(ROOT.parent / file))  # find file
        if not files and hard:
            raise FileNotFoundError(f"'{file}' does not exist")
        elif len(files) > 1 and hard:
            raise FileNotFoundError(f"Multiple files match '{file}', specify exact path: {files}")
        return files[0] if len(files) else []  # return file


def check_yaml(file, suffix=(".yaml", ".yml"), hard=True):
    """Search/download YAML file (if necessary) and return path, checking suffix."""
    return check_file(file, suffix, hard=hard)


def check_is_path_safe(basedir, path):
    """
    Check if the resolved path is under the intended directory to prevent path traversal.

    Args:
        basedir (Path | str): The intended directory.
        path (Path | str): The path to check.

    Returns:
        (bool): True if the path is safe, False otherwise.
    """
    base_dir_resolved = Path(basedir).resolve()
    path_resolved = Path(path).resolve()

    return path_resolved.exists() and path_resolved.parts[: len(base_dir_resolved.parts)] == base_dir_resolved.parts


def check_imshow(warn=False):
    """Check if environment supports image displays."""
    try:
        if LINUX:
            assert not IS_COLAB and not IS_KAGGLE
            assert "DISPLAY" in os.environ, "The DISPLAY environment variable isn't set."
        cv2.imshow("create_self_data", np.zeros((8, 8, 3), dtype=np.uint8))  # show a small 8-pixel image
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        return True
    except Exception as e:
        if warn:
            LOGGER.warning(f"WARNING ⚠️ Environment does not support cv2.imshow() or PIL Image.show()\n{e}")
        return False


def check_yolo(verbose=True, device=""):
    """Return a human-readable YOLO software and hardware summary."""
    import psutil

    from ultralytics.utils.torch_utils import select_device

    if IS_JUPYTER:
        if check_requirements("wandb", install=False):
            os.system("pip uninstall -y wandb")  # uninstall wandb: unwanted account creation prompt with infinite hang
        if IS_COLAB:
            shutil.rmtree("sample_data", ignore_errors=True)  # remove colab /sample_data directory

    if verbose:
        # System info
        gib = 1 << 30  # bytes per GiB
        ram = psutil.virtual_memory().total
        total, used, free = shutil.disk_usage("/")
        s = f"({os.cpu_count()} CPUs, {ram / gib:.1f} GB RAM, {(total - free) / gib:.1f}/{total / gib:.1f} GB disk)"
        with contextlib.suppress(Exception):  # clear display if ipython is installed
            from IPython import display

            display.clear_output()
    else:
        s = ""

    select_device(device=device, newline=False)
    LOGGER.info(f"Setup complete ✅ {s}")


def collect_system_info():
    """Collect and print relevant system information including OS, Python, RAM, CPU, and CUDA."""

    import psutil

    from ultralytics.utils import ENVIRONMENT, IS_GIT_DIR
    from ultralytics.utils.torch_utils import get_cpu_info

    ram_info = psutil.virtual_memory().total / (1024**3)  # Convert bytes to GB
    check_yolo()
    LOGGER.info(
        f"\n{'OS':<20}{platform.platform()}\n"
        f"{'Environment':<20}{ENVIRONMENT}\n"
        f"{'Python':<20}{PYTHON_VERSION}\n"
        f"{'Install':<20}{'git' if IS_GIT_DIR else 'pip' if IS_PIP_PACKAGE else 'other'}\n"
        f"{'RAM':<20}{ram_info:.2f} GB\n"
        f"{'CPU':<20}{get_cpu_info()}\n"
        f"{'CUDA':<20}{torch.version.cuda if torch and torch.cuda.is_available() else None}\n"
    )

    for r in parse_requirements(package="ultralytics"):
        try:
            current = metadata.version(r.name)
            is_met = "✅ " if check_version(current, str(r.specifier), hard=True) else "❌ "
        except metadata.PackageNotFoundError:
            current = "(not installed)"
            is_met = "❌ "
        LOGGER.info(f"{r.name:<20}{is_met}{current}{r.specifier}")

    if is_github_action_running():
        LOGGER.info(
            f"\nRUNNER_OS: {os.getenv('RUNNER_OS')}\n"
            f"GITHUB_EVENT_NAME: {os.getenv('GITHUB_EVENT_NAME')}\n"
            f"GITHUB_WORKFLOW: {os.getenv('GITHUB_WORKFLOW')}\n"
            f"GITHUB_ACTOR: {os.getenv('GITHUB_ACTOR')}\n"
            f"GITHUB_REPOSITORY: {os.getenv('GITHUB_REPOSITORY')}\n"
            f"GITHUB_REPOSITORY_OWNER: {os.getenv('GITHUB_REPOSITORY_OWNER')}\n"
        )


def check_amp(model):
    """
    This function checks the PyTorch Automatic Mixed Precision (AMP) functionality of a YOLOv8 model. If the checks
    fail, it means there are anomalies with AMP on the system that may cause NaN losses or zero-mAP results, so AMP will
    be disabled during training.

    Args:
        model (nn.Module): A YOLOv8 model instance.

    Example:
        ```python
        from ultralytics import YOLO
        from ultralytics.utils.checks import check_amp

        model = YOLO('yolov8n.pt').model.cuda()
        check_amp(model)
        ```

    Returns:
        (bool): Returns True if the AMP functionality works correctly with YOLOv8 model, else False.
    """
    device = next(model.parameters()).device  # get model device
    if device.type in {"cpu", "mps"}:
        return False  # AMP only used on CUDA devices


    im = ASSETS / "bus.jpg"  # image to check
    prefix = colorstr("AMP: ")
    LOGGER.info(f"{prefix}running Automatic Mixed Precision (AMP) checks with YOLOv8n...")
    warning_msg = "Setting 'amp=True'. If you experience zero-mAP or NaN losses you can disable AMP with amp=False."
    try:
        from ultralytics import YOLO

        assert amp_allclose(YOLO("yolov8n.pt"), im)
        LOGGER.info(f"{prefix}checks passed ✅")
    except ConnectionError:
        LOGGER.warning(f"{prefix}checks skipped ⚠️, offline and unable to download YOLOv8n. {warning_msg}")
    except (AttributeError, ModuleNotFoundError):
        LOGGER.warning(
            f"{prefix}checks skipped ⚠️. "
            f"Unable to load YOLOv8n due to possible Ultralytics package modifications. {warning_msg}"
        )
    except AssertionError:
        LOGGER.warning(
            f"{prefix}checks failed ❌. Anomalies were detected with AMP on your system that may lead to "
            f"NaN losses or zero-mAP results, so AMP will be disabled during training."
        )
        return False
    return True


def git_describe(path=ROOT):  # path must be a directory
    """Return human-readable git description, i.e. v5.0-5-g3e25f1e https://git-scm.com/docs/git-describe."""
    with contextlib.suppress(Exception):
        return subprocess.check_output(f"git -C {path} describe --tags --long --always", shell=True).decode()[:-1]
    return ""


def print_args(args: Optional[dict] = None, show_file=True, show_func=False):
    """Print function arguments (optional args dict)."""


    x = inspect.currentframe().f_back  # previous frame
    file, _, func, _, _ = inspect.getframeinfo(x)
    if args is None:  # get args automatically
        args, _, _, frm = inspect.getargvalues(x)
        args = {k: v for k, v in frm.items() if k in args}
    try:
        file = Path(file).resolve().relative_to(ROOT).with_suffix("")
    except ValueError:
        file = Path(file).stem
    s = (f"{file}: " if show_file else "") + (f"{func}: " if show_func else "")
    LOGGER.info(colorstr(s) + ", ".join(f"{k}={strip_auth(v)}" for k, v in args.items()))


def cuda_device_count() -> int:
    """
    Get the number of NVIDIA GPUs available in the environment.

    Returns:
        (int): The number of NVIDIA GPUs available.
    """
    try:
        # Run the nvidia-smi command and capture its output
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=count", "--format=csv,noheader,nounits"], encoding="utf-8"
        )

        # Take the first line and strip any leading/trailing white space
        first_line = output.strip().split("\n")[0]

        return int(first_line)
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        # If the command fails, nvidia-smi is not found, or output is not an integer, assume no GPUs are available
        return 0


def cuda_is_available() -> bool:
    """
    Check if CUDA is available in the environment.

    Returns:
        (bool): True if one or more NVIDIA GPUs are available, False otherwise.
    """
    return cuda_device_count() > 0


# Define constants
IS_PYTHON_MINIMUM_3_10 = check_python("3.10", hard=False)
IS_PYTHON_3_12 = PYTHON_VERSION.startswith("3.12")