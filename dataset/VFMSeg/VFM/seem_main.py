def main():
    """
    build args
    """
    args = parse_option()
    opt = load_opt_from_config_files(args.conf_files)
    opt = init_distributed(opt)
    cur_model = 'None'
    pretrained_pth = '/Labs/Scripts/3DPC/xMUDA/VFM_ckpt/seem_focall_v1.pt'
    cur_model = 'Focal-L'
    """
    build model
    """
    model = BaseModel(opt, build_model(opt)).from_pretrained(pretrained_pth
        ).eval().cuda()
    with torch.no_grad():
        model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(
            COCO_PANOPTIC_CLASSES + ['background'], is_eval=True)
    """
    audio
    """
    audio = None
    task = []
    image = Image.open(
        '/Labs/Scripts/3DPC/Datasets/3DPC/nuScenes/samples/CAM_FRONT/n015-2018-08-02-17-16-37+0800__CAM_FRONT__1533201470412460.jpg'
        )
    res = inference(model, audio, image, task)
