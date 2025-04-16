@staticmethod
def _get_c2_imagenet_pretrained(name):
    prefix = ModelCatalog.S3_C2_DETECTRON_PREFIX
    name = name[len('ImageNetPretrained/'):]
    name = ModelCatalog.C2_IMAGENET_MODELS[name]
    url = '/'.join([prefix, name])
    return url
