import torch
from yolov6.utils.general import check_version

torch_1_10_plus = check_version(torch.__version__, minimum='1.10.0')
