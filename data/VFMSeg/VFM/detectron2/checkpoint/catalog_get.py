@staticmethod
def get(name):
    if name.startswith('Caffe2Detectron/COCO'):
        return ModelCatalog._get_c2_detectron_baseline(name)
    if name.startswith('ImageNetPretrained/'):
        return ModelCatalog._get_c2_imagenet_pretrained(name)
    raise RuntimeError('model not present in the catalog: {}'.format(name))
