@classmethod
def from_config(cls, cfg, is_train: bool=True):
    augs = utils.build_augmentation(cfg, is_train)
    if cfg.INPUT.CROP.ENABLED and is_train:
        augs.insert(0, T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE))
        recompute_boxes = cfg.MODEL.MASK_ON
    else:
        recompute_boxes = False
    ret = {'is_train': is_train, 'augmentations': augs, 'image_format': cfg
        .INPUT.FORMAT, 'use_instance_mask': cfg.MODEL.MASK_ON,
        'instance_mask_format': cfg.INPUT.MASK_FORMAT, 'use_keypoint': cfg.
        MODEL.KEYPOINT_ON, 'recompute_boxes': recompute_boxes}
    if cfg.MODEL.KEYPOINT_ON:
        ret['keypoint_hflip_indices'] = utils.create_keypoint_hflip_indices(cfg
            .DATASETS.TRAIN)
    if cfg.MODEL.LOAD_PROPOSALS:
        ret['precomputed_proposal_topk'] = (cfg.DATASETS.
            PRECOMPUTED_PROPOSAL_TOPK_TRAIN if is_train else cfg.DATASETS.
            PRECOMPUTED_PROPOSAL_TOPK_TEST)
    return ret
