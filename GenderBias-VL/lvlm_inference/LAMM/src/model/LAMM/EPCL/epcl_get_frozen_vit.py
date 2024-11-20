def get_frozen_vit(vit_model='ViT-B/32', device='cpu'):
    clip_vit = CLIP.load(vit_model, device=device)[0].visual
    for _, v in clip_vit.named_parameters():
        v.requires_grad = False
    return clip_vit
