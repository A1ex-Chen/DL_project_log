def build_SAM(sam_ckpt=None, device=0):
    if sam_ckpt == None:
        path = '/Labs/Scripts/3DPC/xMUDA/VFM_ckpt/'
        sam_ckpt = path + 'sam_vit_h_4b8939.pth'
    model_type = 'vit_h'
    sam = sam_model_registry[model_type](checkpoint=sam_ckpt)
    sam.cuda(device)
    mask_generator = SamAutomaticMaskGenerator(sam)
    return mask_generator
