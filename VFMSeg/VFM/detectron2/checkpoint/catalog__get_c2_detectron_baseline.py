@staticmethod
def _get_c2_detectron_baseline(name):
    name = name[len('Caffe2Detectron/COCO/'):]
    url = ModelCatalog.C2_DETECTRON_MODELS[name]
    if 'keypoint_rcnn' in name:
        dataset = ModelCatalog.C2_DATASET_COCO_KEYPOINTS
    else:
        dataset = ModelCatalog.C2_DATASET_COCO
    if '35998355/rpn_R-50-C4_1x' in name:
        type = 'rpn'
    else:
        type = 'generalized_rcnn'
    url = ModelCatalog.C2_DETECTRON_PATH_FORMAT.format(prefix=ModelCatalog.
        S3_C2_DETECTRON_PREFIX, url=url, type=type, dataset=dataset)
    return url
