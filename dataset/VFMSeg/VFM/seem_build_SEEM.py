def build_SEEM(pretrained_pth, config_pth, gpu_index):
    """
    build args
    """
    opt = load_opt_from_config_files(config_pth)
    opt = init_distributed(opt, gpu_index)
    """
    build model
    """
    model = BaseModel(opt, build_model(opt)).from_pretrained(pretrained_pth
        ).eval().cuda()
    with torch.no_grad():
        model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(
            COCO_PANOPTIC_CLASSES + ['background'], is_eval=True)
    return model
