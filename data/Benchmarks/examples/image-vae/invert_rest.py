import torchvision.transforms.functional as F
from PIL import Image, ImageOps


class Invert(object):
    """Inverts the color channels of an PIL Image
    while leaving intact the alpha channel.
    """


