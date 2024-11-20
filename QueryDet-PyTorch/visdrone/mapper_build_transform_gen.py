def build_transform_gen(cfg, is_train):
    if is_train:
        sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
    else:
        sample_style = 'choice'
    logger = logging.getLogger(__name__)
    tfm_gens = []
    if is_train:
        tfm_gens.append(T.RandomFlip(horizontal=True, vertical=False))
        tfm_gens.append(T.ResizeShortestEdge(short_edge_length=cfg.VISDRONE
            .SHORT_LENGTH, max_size=cfg.VISDRONE.MAX_LENGTH, sample_style=
            sample_style))
    else:
        tfm_gens.append(T.ResizeShortestEdge(short_edge_length=[cfg.
            VISDRONE.TEST_LENGTH], max_size=cfg.VISDRONE.TEST_LENGTH,
            sample_style=sample_style))
    return tfm_gens
