from PIL import Image

from util import list_all_images


if __name__ == '__main__':
    ls: list[str] = list_all_images(root='real_images/imagenet_64', image_exts=['png'])
    for i, elem in enumerate(ls):
        convert(src=elem, dest=elem.replace('imagenet_64', 'imagenet_64_jpg').replace('png', 'jpg'))