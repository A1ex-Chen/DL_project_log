import torch
from transformers import AutoConfig, StoppingCriteria





class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.tokenizer = tokenizer
        self.start_len = None
        self.input_ids = input_ids

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if self.start_len is None:
            self.start_len = self.input_ids.shape[1]
        else:
            outputs = self.tokenizer.batch_decode(output_ids[:, self.start_len:], skip_special_tokens=True)[0]
            for keyword in self.keywords:
                if keyword in outputs:
                    return True
        return False


import cv2
import base64
import torch
import pickle
import numpy as np
import torch.distributed as dist

from io import BytesIO
from PIL import Image
from transformers import AutoConfig
from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.data.transforms import RandomResizedCropAndInterpolation
from torchvision import transforms



# aug functions
































# level to args
















func_dict = {
    'Identity': identity_func,
    'AutoContrast': autocontrast_func,
    'Equalize': equalize_func,
    'Rotate': rotate_func,
    'Solarize': solarize_func,
    'Color': color_func,
    'Contrast': contrast_func,
    'Brightness': brightness_func,
    'Sharpness': sharpness_func,
    'ShearX': shear_x_func,
    'TranslateX': translate_x_func,
    'TranslateY': translate_y_func,
    'Posterize': posterize_func,
    'ShearY': shear_y_func,
}

translate_const = 10
MAX_LEVEL = 10
replace_value = (128, 128, 128)
arg_dict = {
    'Identity': none_level_to_args,
    'AutoContrast': none_level_to_args,
    'Equalize': none_level_to_args,
    'Rotate': rotate_level_to_args(MAX_LEVEL, replace_value),
    'Solarize': solarize_level_to_args(MAX_LEVEL),
    'Color': enhance_level_to_args(MAX_LEVEL),
    'Contrast': enhance_level_to_args(MAX_LEVEL),
    'Brightness': enhance_level_to_args(MAX_LEVEL),
    'Sharpness': enhance_level_to_args(MAX_LEVEL),
    'ShearX': shear_level_to_args(MAX_LEVEL, replace_value),
    'TranslateX': translate_level_to_args(
        translate_const, MAX_LEVEL, replace_value
    ),
    'TranslateY': translate_level_to_args(
        translate_const, MAX_LEVEL, replace_value
    ),
    'Posterize': posterize_level_to_args(MAX_LEVEL),
    'ShearY': shear_level_to_args(MAX_LEVEL, replace_value),
}


class RandomAugment(object):

    def __init__(self, N=2, M=10, isPIL=False, augs=[]):
        self.N = N
        self.M = M
        self.isPIL = isPIL
        if augs:
            self.augs = augs
        else:
            self.augs = list(arg_dict.keys())

    def get_random_ops(self):
        sampled_ops = np.random.choice(self.augs, self.N)
        return [(op, 0.5, self.M) for op in sampled_ops]

    def __call__(self, img):
        if self.isPIL:
            img = np.array(img)
        ops = self.get_random_ops()
        for name, prob, level in ops:
            if np.random.random() > prob:
                continue
            args = arg_dict[name](level)
            img = func_dict[name](img, *args)
        return img























import cv2
import base64
import torch
import pickle
import numpy as np
import torch.distributed as dist

from io import BytesIO
from PIL import Image
from transformers import AutoConfig
from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.data.transforms import RandomResizedCropAndInterpolation
from torchvision import transforms


def auto_upgrade(config):
    cfg = AutoConfig.from_pretrained(config)
    if 'llava' in config and cfg.model_type != 'llava':
        print("You are using newer LLaVA code base, while the checkpoint of v0 is from older code base.")
        print("You must upgrade the checkpoint to the new code base (this can be done automatically).")
        confirm = input("Please confirm that you want to upgrade the checkpoint. [Y/N]")
        if confirm.lower() in ["y", "yes"]:
            print("Upgrading checkpoint...")
            assert len(cfg.architectures) == 1
            setattr(cfg.__class__, "model_type", "llava")
            cfg.architectures[0] = 'LlavaLlamaForCausalLM'
            cfg.save_pretrained(config)
            print("Checkpoint upgraded.")
        else:
            print("Checkpoint upgrade aborted.")
            exit(1)

# aug functions


def identity_func(img):
    return img


def autocontrast_func(img, cutoff=0):
    '''
        same output as PIL.ImageOps.autocontrast
    '''
    n_bins = 256


    channels = [tune_channel(ch) for ch in cv2.split(img)]
    out = cv2.merge(channels)
    return out


def equalize_func(img):
    '''
        same output as PIL.ImageOps.equalize
        PIL's implementation is different from cv2.equalize
    '''
    n_bins = 256


    channels = [tune_channel(ch) for ch in cv2.split(img)]
    out = cv2.merge(channels)
    return out


def rotate_func(img, degree, fill=(0, 0, 0)):
    '''
    like PIL, rotate by degree, not radians
    '''
    H, W = img.shape[0], img.shape[1]
    center = W / 2, H / 2
    M = cv2.getRotationMatrix2D(center, degree, 1)
    out = cv2.warpAffine(img, M, (W, H), borderValue=fill)
    return out


def solarize_func(img, thresh=128):
    '''
        same output as PIL.ImageOps.posterize
    '''
    table = np.array([el if el < thresh else 255 - el for el in range(256)])
    table = table.clip(0, 255).astype(np.uint8)
    out = table[img]
    return out


def color_func(img, factor):
    '''
        same output as PIL.ImageEnhance.Color
    '''
    # implementation according to PIL definition, quite slow
    #  degenerate = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[:, :, np.newaxis]
    #  out = blend(degenerate, img, factor)
    #  M = (
    #      np.eye(3) * factor
    #      + np.float32([0.114, 0.587, 0.299]).reshape(3, 1) * (1. - factor)
    #  )[np.newaxis, np.newaxis, :]
    M = (
        np.float32([
            [0.886, -0.114, -0.114],
            [-0.587, 0.413, -0.587],
            [-0.299, -0.299, 0.701]]) * factor
        + np.float32([[0.114], [0.587], [0.299]])
    )
    out = np.matmul(img, M).clip(0, 255).astype(np.uint8)
    return out


def contrast_func(img, factor):
    """
        same output as PIL.ImageEnhance.Contrast
    """
    mean = np.sum(np.mean(img, axis=(0, 1)) * np.array([0.114, 0.587, 0.299]))
    table = np.array([(
        el - mean) * factor + mean
        for el in range(256)
    ]).clip(0, 255).astype(np.uint8)
    out = table[img]
    return out


def brightness_func(img, factor):
    '''
        same output as PIL.ImageEnhance.Contrast
    '''
    table = (np.arange(256, dtype=np.float32) * factor).clip(0, 255).astype(np.uint8)
    out = table[img]
    return out


def sharpness_func(img, factor):
    '''
    The differences the this result and PIL are all on the 4 boundaries, the center
    areas are same
    '''
    kernel = np.ones((3, 3), dtype=np.float32)
    kernel[1][1] = 5
    kernel /= 13
    degenerate = cv2.filter2D(img, -1, kernel)
    if factor == 0.0:
        out = degenerate
    elif factor == 1.0:
        out = img
    else:
        out = img.astype(np.float32)
        degenerate = degenerate.astype(np.float32)[1:-1, 1:-1, :]
        out[1:-1, 1:-1, :] = degenerate + factor * (out[1:-1, 1:-1, :] - degenerate)
        out = out.astype(np.uint8)
    return out


def shear_x_func(img, factor, fill=(0, 0, 0)):
    H, W = img.shape[0], img.shape[1]
    M = np.float32([[1, factor, 0], [0, 1, 0]])
    out = cv2.warpAffine(img, M, (W, H), borderValue=fill, flags=cv2.INTER_LINEAR).astype(np.uint8)
    return out


def translate_x_func(img, offset, fill=(0, 0, 0)):
    '''
        same output as PIL.Image.transform
    '''
    H, W = img.shape[0], img.shape[1]
    M = np.float32([[1, 0, -offset], [0, 1, 0]])
    out = cv2.warpAffine(img, M, (W, H), borderValue=fill, flags=cv2.INTER_LINEAR).astype(np.uint8)
    return out


def translate_y_func(img, offset, fill=(0, 0, 0)):
    '''
        same output as PIL.Image.transform
    '''
    H, W = img.shape[0], img.shape[1]
    M = np.float32([[1, 0, 0], [0, 1, -offset]])
    out = cv2.warpAffine(img, M, (W, H), borderValue=fill, flags=cv2.INTER_LINEAR).astype(np.uint8)
    return out


def posterize_func(img, bits):
    '''
        same output as PIL.ImageOps.posterize
    '''
    out = np.bitwise_and(img, np.uint8(255 << (8 - bits)))
    return out


def shear_y_func(img, factor, fill=(0, 0, 0)):
    H, W = img.shape[0], img.shape[1]
    M = np.float32([[1, 0, 0], [factor, 1, 0]])
    out = cv2.warpAffine(img, M, (W, H), borderValue=fill, flags=cv2.INTER_LINEAR).astype(np.uint8)
    return out


def cutout_func(img, pad_size, replace=(0, 0, 0)):
    replace = np.array(replace, dtype=np.uint8)
    H, W = img.shape[0], img.shape[1]
    rh, rw = np.random.random(2)
    pad_size = pad_size // 2
    ch, cw = int(rh * H), int(rw * W)
    x1, x2 = max(ch - pad_size, 0), min(ch + pad_size, H)
    y1, y2 = max(cw - pad_size, 0), min(cw + pad_size, W)
    out = img.copy()
    out[x1:x2, y1:y2, :] = replace
    return out


# level to args
def enhance_level_to_args(MAX_LEVEL):
    return level_to_args


def shear_level_to_args(MAX_LEVEL, replace_value):

    return level_to_args


def translate_level_to_args(translate_const, MAX_LEVEL, replace_value):

    return level_to_args


def cutout_level_to_args(cutout_const, MAX_LEVEL, replace_value):

    return level_to_args


def solarize_level_to_args(MAX_LEVEL):
    return level_to_args


def none_level_to_args(level):
    return ()


def posterize_level_to_args(MAX_LEVEL):
    return level_to_args


def rotate_level_to_args(MAX_LEVEL, replace_value):

    return level_to_args


func_dict = {
    'Identity': identity_func,
    'AutoContrast': autocontrast_func,
    'Equalize': equalize_func,
    'Rotate': rotate_func,
    'Solarize': solarize_func,
    'Color': color_func,
    'Contrast': contrast_func,
    'Brightness': brightness_func,
    'Sharpness': sharpness_func,
    'ShearX': shear_x_func,
    'TranslateX': translate_x_func,
    'TranslateY': translate_y_func,
    'Posterize': posterize_func,
    'ShearY': shear_y_func,
}

translate_const = 10
MAX_LEVEL = 10
replace_value = (128, 128, 128)
arg_dict = {
    'Identity': none_level_to_args,
    'AutoContrast': none_level_to_args,
    'Equalize': none_level_to_args,
    'Rotate': rotate_level_to_args(MAX_LEVEL, replace_value),
    'Solarize': solarize_level_to_args(MAX_LEVEL),
    'Color': enhance_level_to_args(MAX_LEVEL),
    'Contrast': enhance_level_to_args(MAX_LEVEL),
    'Brightness': enhance_level_to_args(MAX_LEVEL),
    'Sharpness': enhance_level_to_args(MAX_LEVEL),
    'ShearX': shear_level_to_args(MAX_LEVEL, replace_value),
    'TranslateX': translate_level_to_args(
        translate_const, MAX_LEVEL, replace_value
    ),
    'TranslateY': translate_level_to_args(
        translate_const, MAX_LEVEL, replace_value
    ),
    'Posterize': posterize_level_to_args(MAX_LEVEL),
    'ShearY': shear_level_to_args(MAX_LEVEL, replace_value),
}


class RandomAugment(object):





def build_transform(is_train, randaug=True, input_size=224, interpolation='bicubic'):
    if is_train:
        t = [
            RandomResizedCropAndInterpolation(
                input_size, scale=(0.5, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
        ]
        if randaug:
            t.append(
                RandomAugment(
                    2, 7, isPIL=True,
                    augs=[
                        'Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness',
                        'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate',
                    ]))
        t += [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),
        ]
        t = transforms.Compose(t)
    else:
        t = transforms.Compose([
            transforms.Resize((input_size, input_size),
                              interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD)
        ])

    return t


def img2b64(img_path):
    img = Image.open(img_path)  # path to file
    img_buffer = BytesIO()
    img.save(img_buffer, format=img.format)
    byte_data = img_buffer.getvalue()
    base64_str = base64.b64encode(byte_data)  # bytes
    base64_str = base64_str.decode("utf-8")  # str
    return base64_str


def str2b64(str):
    return base64.b64encode(str.encode('utf-8')).decode('utf-8')


def b642str(b64):
    return base64.b64decode(b64).decode('utf-8')


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.LongTensor([tensor.numel()]).to("cuda")
    size_list = [torch.LongTensor([0]).to("cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.ByteTensor(size=(max_size,)).to("cuda"))
    if local_size != max_size:
        padding = torch.ByteTensor(size=(max_size - local_size,)).to("cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def mean(lst):
    return sum(lst) / len(lst)


def stop_gradient_by_name(name: str):

    return apply_fn