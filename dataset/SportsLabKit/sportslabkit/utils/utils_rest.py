"""Genereal utils."""
import hashlib
import itertools
import json
import os
import re
import shutil
import sys
import tempfile
from ast import literal_eval
from collections import deque
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import cv2
import cv2 as cv
import dateutil.parser
import gdown
import git
import numpy as np
import requests
import torch
from numpy.typing import NDArray
from omegaconf import OmegaConf
from PIL import Image
from vidgear.gears import WriteGear

from sportslabkit.logger import logger, tqdm
from sportslabkit.types.types import PathLike


OmegaConf.register_new_resolver("now", lambda x: datetime.now().strftime(x), replace=True)


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout




















class MovieIterator:
    def __init__(self, path: str):
        """Very simple iterator class for movie files.

        Args:
            path (str): Path to movie file

        Attributes:
            video_fps (int): Frames per second
            video_frame_count (int): Total number of frames
            vcInput (cv.VideoCapture): OpenCV VideoCapture object
            img_width (int): Width of frame
            img_height (int): Height of frame
            path (str): Path to movie file

        Raises:
            FileNotFoundError: If file does not exist

        """
        if not os.path.isfile(path):
            raise FileNotFoundError

        path = str(path)
        vcInput = cv.VideoCapture(path)
        self.vcInput = vcInput
        self.video_fps: int = round(vcInput.get(cv.CAP_PROP_FPS))
        self.video_frame_count = round(vcInput.get(cv.CAP_PROP_FRAME_COUNT))
        self.img_width = round(vcInput.get(cv.CAP_PROP_FRAME_WIDTH))
        self.img_height = round(vcInput.get(cv.CAP_PROP_FRAME_HEIGHT))
        self.path = path
        self._index = 0

    def __len__(self) -> int:
        return self.video_frame_count

    def __iter__(self) -> "MovieIterator":
        return self

    def __next__(self) -> NDArray[np.uint8]:
        if self._index < len(self):
            ret, img = self.vcInput.read()
            if ret:
                self._index += 1
                return cv.cvtColor(img, cv.COLOR_BGR2RGB)
            logger.debug("Unexpected end.")  # <- Not sure why this happens
        raise StopIteration


class ImageIterator:
    def __init__(self, path: str):
        """Very simple iterator class for image files.

        Args:
            path (str): Path to image file
        """
        assert os.path.isdir(path), f"{path} is not a directory."
        self.path = path

        imgs = []
        valid_images = [".jpg", ".gif", ".png", ".tga"]
        for f in os.listdir(path):
            ext = os.path.splitext(f)[1]
            if ext.lower() not in valid_images:
                continue
            imgs.append(cv.imread(os.path.join(path, f)))
        self.imgs = imgs
        self._index = 0

    def __len__(self) -> int:
        return len(self.imgs)

    def __iter__(self) -> "ImageIterator":
        return self

    def __next__(self) -> NDArray[np.uint8]:
        if self._index < len(self):
            img = self.imgs[self._index]
            self._index += 1
            return img
        raise StopIteration




















def read_image(img):
    """Reads an image from a file, URL, a numpy array, or a torch tensor.
    Args:
        img (str, Path, Image.Image, np.ndarray, or torch.Tensor): The image to read.
    Returns:
        np.ndarray: The image as a numpy array.
    """
    if isinstance(img, str):
        if img.startswith("http"):
            img = requests.get(img, stream=True).raw
            img = Image.open(img)
        else:
            img = Path(img)
    if isinstance(img, Path):
        img = Image.open(img)
    if isinstance(img, Image.Image):
        img = np.array(img)
    if isinstance(img, torch.Tensor):
        img = img.numpy()
    if not isinstance(img, np.ndarray):
        raise TypeError(f"Unsupported input type: {type(img)}")
    if len(img.shape) != 3:
        raise ValueError(f"Unsupported input shape: {img.shape}")
    if img.shape[2] not in [1, 3]:
        raise ValueError(f"Unsupported input shape: {img.shape}")

    return img


def auto_string_parser(value: str) -> Any:
    """Auxiliary function to parse string values.

    Args:
        value (str): String value to parse.

    Returns:
        value (any): Parsed string value.
    """
    # automatically parse values to correct type
    if value.isdigit():
        return int(value)
    if value.replace(".", "", 1).isdigit():
        return float(value)
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    if value.lower() == "nan":
        return np.nan
    if value.lower() == "inf":
        return np.inf
    if value.lower() == "-inf":
        return -np.inf

    try:
        return literal_eval(value)
    except (ValueError, SyntaxError):
        pass
    try:
        return dateutil.parser.parse(value)
    except (ValueError, TypeError):
        pass
    return value


def count_iter_items(iterable: Iterable) -> int:
    """Consume an iterable not reading it into memory; return the number of items.

    Args:
        iterable (Iterable): Iterable object

    Returns:
        int: Number of items
    """
    counter = itertools.count()
    deque(zip(iterable, counter), maxlen=0)  # (consume at C speed)
    return next(counter)


def load_config(yaml_path: str) -> OmegaConf:
    """Load config from yaml file.

    Args:
        yaml_path (str): Path to yaml file

    Returns:
        OmegaConf: Config object loaded from yaml file
    """
    assert os.path.exists(yaml_path)
    cfg = OmegaConf.load(yaml_path)

    cfg.outdir = cfg.outdir  # prevent multiple interpolations
    os.makedirs(cfg.outdir, exist_ok=True)

    # TODO: add validation
    return cfg


def write_config(yaml_path: str, cfg: OmegaConf) -> None:
    """Write config to yaml file.

    Args:
        yaml_path (str): Path to yaml file
        cfg (OmegaConf): Config object
    """
    assert os.path.exists(yaml_path)
    OmegaConf.save(cfg, yaml_path)


def pil2cv(image: Image.Image) -> NDArray[np.uint8]:
    """Convert PIL image to OpenCV image.

    Args:
        image (Image.Image): PIL image

    Returns:
        NDArray[np.uint8]: Numpy Array (OpenCV image)
    """
    new_image = np.array(image, dtype=np.uint8)
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv.cvtColor(new_image, cv.COLOR_RGB2BGR)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv.cvtColor(new_image, cv.COLOR_RGBA2BGRA)
    return new_image


def cv2pil(image: NDArray[np.uint8], convert_bgr2rgb=True) -> Image.Image:
    """Convert OpenCV image to PIL image.

    Args:
        image (NDArray[np.uint8]): Numpy Array (OpenCV image)

    Returns:
        Image.Image: PIL image
    """
    new_image = image.copy()
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        if convert_bgr2rgb:
            new_image = cv.cvtColor(new_image, cv.COLOR_BGR2RGB)
    elif new_image.shape[2] == 4:  # 透過
        if convert_bgr2rgb:
            new_image = cv.cvtColor(new_image, cv.COLOR_BGRA2RGBA)
    new_image = Image.fromarray(new_image)
    return new_image


def get_fps(path):
    path = str(path)
    cap = cv2.VideoCapture(path)
    return cap.get(cv2.CAP_PROP_FPS)


def make_video(
    frames: Iterable[NDArray[np.uint8]],
    outpath: PathLike,
    vcodec: str = "libx264",
    pix_fmt: str = "yuv420p",
    preset: str = "medium",
    crf: int | None = None,
    ss: int | None = None,
    t: int | None = None,
    c: str | None = None,
    height: int | None = -1,
    width: int | None = -1,
    input_framerate: int | None = None,
    logging: bool = False,
    custom_ffmpeg: str | None = None,
) -> None:
    """Make video from a list of opencv format frames.

    Args:
        frames (Iterable[NDArray[np.uint8]]): List of opencv format frames
        outpath (str): Path to output video file
        vcodec (str): Video codec.
        preset (str): Video encoding preset. A preset is a collection of options
            that will provide a certain encoding speed to compression ratio. A
            slower preset will provide better compression (compression is quality
            per filesize). Use the slowest preset that you have patience for.
            The available presets in descending order of speed are:

            - ultrafast
            - superfast
            - veryfast
            - faster
            - fast
            - medium (default preset)
            - slow
            - slower
            - veryslow

            Defaults to `medium`.

        crf (int): Constant Rate Factor. Use the crf (Constant Rate Factor)
            parameter to control the output quality. The lower crf, the higher
            the quality (range: 0-51). Visually lossless compression corresponds
            to -crf 18. Use the preset parameter to control the speed of the
            compression process. Defaults to `23`.
        ss (int): Start-time of the clip in seconds. Defaults to `0`.
        t (Optional[int]): Duration of the clip in seconds. Defaults to None.
        c (bool): copies the first video, audio, and subtitle bitstream from the input to the output file without re-encoding them. Defaults to `False`.
        height (int): Video height. Defaults to `None`.
        width (int): Video width. Defaults to `None`.
        input_framerate (int): Input framerate. Defaults to `25`.
        logging (bool): Logging. Defaults to `False`.
    Todo:
        * add FPS option
        * functionality to use PIL image
        * reconsider compression (current compression is not good)
    """

    scale_filter = f"scale={width}:{height}"
    output_params = {
        k: v
        for k, v in {
            "-vcodec": vcodec,
            "-pix_fmt": pix_fmt,
            # encoding quality
            "-crf": crf,
            "-preset": preset,
            # size
            "-vf": scale_filter,
            # Trimming
            "-c": c,
            "-ss": ss,
            "-t": t,
            # frame rate
            "-input_framerate": input_framerate,
        }.items()
        if v is not None
    }

    logger.debug(f"output_params: {output_params}")

    if not Path(outpath).parent.exists():
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
    writer = WriteGear(
        output=outpath,
        compression_mode=True,
        logging=logging,
        custom_ffmpeg=custom_ffmpeg,
        **output_params,
    )

    # loop over
    for frame in tqdm(frames, desc="Writing video", level="INFO"):
        writer.write(frame, rgb_mode=True)  # activate RGB Mode

    writer.close()


class MovieIterator:





class ImageIterator:





def merge_dict_of_lists(d1: dict, d2: dict) -> dict:
    """Merge two dicts of lists.

    Parameters
    ----------
    d1 : dict
        The first dict to merge.
    d2 : dict
        The second dict to merge.

    Returns
    -------
    dict
        The merged dict.
    """
    keys = set(d1.keys()).union(d2.keys())
    ret = {k: list(d1.get(k, [])) + list(d2.get(k, [])) for k in keys}
    return ret


def get_git_root():
    """Get the root of the git repository."""
    git_repo = git.Repo(__file__, search_parent_directories=True)
    git_root = git_repo.git.rev_parse("--show-toplevel")
    return Path(git_root)


def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download&confirm=1"

    session = requests.Session()

    response = session.get(URL, params={"id": id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {"id": id, "confirm": token}
        response = session.get(URL, params=params, stream=True)
    print(URL, id)
    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value

    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


def increment_path(path: str | Path, exist_ok: bool = False, mkdir: bool = False) -> Path:
    """Increments a path (appends a suffix) if it already exists.

    Args:
        path (Union[str, Path]): The path to increment.
        exist_ok (bool, optional): If set to True, no increment will be done. Defaults to False.
        mkdir (bool, optional): If set to True, the directory will be created. Defaults to False.

    Returns:
        Path: The incremented path.
    """
    path = Path(path)

    if exist_ok:
        return path

    suffix = 1
    new_path = path
    while new_path.exists():
        new_path = Path(f"{path}_{suffix}")
        suffix += 1

    if mkdir:
        new_path.mkdir(parents=True, exist_ok=True)

    return new_path


def load_keypoints(keypoint_json):
    """Loads source and target keypoints from a JSON file.

    Args:
        keypoint_json (str): Path to JSON file containing keypoints.

    Returns:
        source_keypoints (np.ndarray): Source keypoints.
        target_keypoints (np.ndarray): Target keypoints.
    """
    with open(keypoint_json) as f:
        data = json.load(f)

    source_keypoints = []
    target_keypoints = []

    for key, value in data.items():
        source_kp = value
        target_kp = literal_eval(key)
        source_keypoints.append(source_kp)
        target_keypoints.append(target_kp)

    source_keypoints = np.array(source_keypoints)
    target_keypoints = np.array(target_keypoints)
    return source_keypoints, target_keypoints

def sanitize_url_name(url: str) -> str:
    """Sanitize the URL to create a safe filename for caching.

    Args:
        url (str): The URL to sanitize.

    Returns:
        str: A sanitized version of the URL suitable for use as a filename.
    """
    parsed_url = urlparse(url)
    filename = Path(parsed_url.path).name

    # Remove any character that isn't a word character, whitespace, or dash
    sanitized_name = re.sub(r'[^\w\s-]', '', filename).strip().replace(' ', '_')

    return sanitized_name


def fetch_or_cache_model(
    url: str,
    dst: PathLike | None = None,
    hash_prefix: str | None = None,
    progress: bool = True
) -> str:
    """Fetches a model from a URL or uses a cached version if it exists.

    Args:
        url (str): URL of the object to download.
        dst (PathLike | None, optional): Full path where object will be saved. Defaults to None.
        hash_prefix (str | None, optional): Hash prefix to validate downloaded file. Defaults to None.
        progress (bool, optional): Whether to show download progress. Defaults to True.

    Returns:
        str: The path to the downloaded or cached file.
    """
    CACHE_DIR = Path("~/.cache/sportslabkit").expanduser()
    hashed_url = hashlib.sha256(url.encode()).hexdigest()

    if Path(url).exists():
        return url

    if dst is None:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        dst = CACHE_DIR / f"{sanitize_url_name(url)}_{hashed_url}"

    if os.path.exists(dst):
        return str(dst)

    if url.startswith("https://drive.google.com"):
        if os.path.exists(dst):
            return str(dst)
        gdown.download(str(url), str(dst), quiet=False, fuzzy=True)
        assert os.path.exists(dst)
        return str(dst)


    # Create a temporary directory
    with tempfile.NamedTemporaryFile(delete=False, suffix=".partial") as f:
        tmp_dst = f.name

    try:
        req = Request(url, headers={"User-Agent": "sportslabkit"})
        u = urlopen(req)
        meta = u.info()
        file_size = int(meta.get_all("Content-Length")[0]) if meta.get_all("Content-Length") else None

        if hash_prefix is not None:
            sha256 = hashlib.sha256()

        with tqdm(
            total=file_size,
            disable=not progress,
            desc=f"Downloading file to {dst}",
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            with open(tmp_dst, "wb") as f:
                while True:
                    buffer = u.read(8192)
                    if len(buffer) == 0:
                        break
                    f.write(buffer)
                    if hash_prefix is not None:
                        sha256.update(buffer)
                    pbar.update(len(buffer))

        if hash_prefix is not None:
            digest = sha256.hexdigest()
            if digest[: len(hash_prefix)] != hash_prefix:
                raise RuntimeError(f'Invalid hash value (expected "{hash_prefix}", got "{digest}")')

        shutil.move(tmp_dst, dst)

    finally:
        if os.path.exists(tmp_dst):
            os.remove(tmp_dst)

    return str(dst)


# Due to memory consumption concerns, the function below has been replaced by the function that uses vidgear above.
# ===
# def make_video(images: list, fps: int, outpath: str = 'video.mp4'):
#     """The main def for creating a temporary video out of the
#     PIL Image list passed, according to the FPS passed
#     Parameters
#     ----------
#     image_list : list
#         A list of PIL Images in sequential order you want the video to be generated
#     fps : int
#         The FPS of the video
#     """

#     def convert(img):
#         if isinstance(img, Image.Image):
#             return pil2cv(img)
#         elif isinstance(img, np.ndarray):
#             return img
#         else:
#             raise ValueError(type(img))

#     h, w = convert(images[0]).shape[:2]
#     fourcc = cv.VideoWriter_fourcc('M','J','P','G')
#     video = cv.VideoWriter(filename=outpath+'.mp4', fourcc=fourcc, fps=fps, frameSize=(w, h))

#     for img in tqdm(images, total=len(images)):
#         video.write(img)
#     video.release()
#     os.system(f"ffmpeg -i {outpath+'.mp4'} -vcodec libx264 -acodec aac -y {outpath}")
#     print(f"Find your images and video at {outpath}")