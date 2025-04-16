@classmethod
def from_config(cls, cfg):
    enc_cfg = cfg['MODEL']['ENCODER']
    dec_cfg = cfg['MODEL']['DECODER']
    openimage_switch = {'grounding': dec_cfg['OPENIMAGE']['GROUNDING'].get(
        'ENABLED', False), 'mask': dec_cfg['OPENIMAGE'].get('ENABLED', False)}
    task_switch = {'bbox': dec_cfg.get('DETECTION', False), 'mask': dec_cfg
        .get('MASK', True), 'spatial': dec_cfg['SPATIAL'].get('ENABLED', 
        False), 'grounding': dec_cfg['GROUNDING'].get('ENABLED', False),
        'openimage': openimage_switch, 'visual': dec_cfg['VISUAL'].get(
        'ENABLED', False), 'audio': dec_cfg['AUDIO'].get('ENABLED', False)}
    extra = {'task_switch': task_switch}
    backbone = build_backbone(cfg)
    lang_encoder = build_language_encoder(cfg)
    sem_seg_head = build_xdecoder_head(cfg, backbone.output_shape(),
        lang_encoder, extra=extra)
    loss_weights = {}
    matcher = None
    losses = {}
    weight_dict = {}
    grd_weight = {}
    top_x_layers = {}
    criterion = None
    train_dataset_name = None
    phrase_prob = None
    deep_supervision = None
    no_object_weight = None
    interactive_mode = 'best'
    interactive_iter = 20
    dilation = 3
    dilation_kernel = torch.ones((1, 1, dilation, dilation), device=torch.
        cuda.current_device())
    return {'backbone': backbone, 'sem_seg_head': sem_seg_head, 'criterion':
        criterion, 'losses': losses, 'num_queries': dec_cfg[
        'NUM_OBJECT_QUERIES'], 'object_mask_threshold': dec_cfg['TEST'][
        'OBJECT_MASK_THRESHOLD'], 'overlap_threshold': dec_cfg['TEST'][
        'OVERLAP_THRESHOLD'], 'metadata': None, 'size_divisibility':
        dec_cfg['SIZE_DIVISIBILITY'],
        'sem_seg_postprocess_before_inference': dec_cfg['TEST'][
        'SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE'] or dec_cfg['TEST'][
        'PANOPTIC_ON'] or dec_cfg['TEST']['INSTANCE_ON'], 'pixel_mean': cfg
        ['INPUT']['PIXEL_MEAN'], 'pixel_std': cfg['INPUT']['PIXEL_STD'],
        'task_switch': task_switch, 'phrase_prob': phrase_prob,
        'semantic_on': dec_cfg['TEST']['SEMANTIC_ON'], 'instance_on':
        dec_cfg['TEST']['INSTANCE_ON'], 'panoptic_on': dec_cfg['TEST'][
        'PANOPTIC_ON'], 'test_topk_per_image': cfg['MODEL']['DECODER'][
        'TEST']['DETECTIONS_PER_IMAGE'], 'train_dataset_name':
        train_dataset_name, 'interactive_mode': interactive_mode,
        'interactive_iter': interactive_iter, 'dilation_kernel':
        dilation_kernel}
