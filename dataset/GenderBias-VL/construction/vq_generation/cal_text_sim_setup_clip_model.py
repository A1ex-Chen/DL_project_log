def setup_clip_model(device):
    clip_model, preprocess = clip.load('ViT-L/14', device='cuda:1')
    print(device)
    clip_model.to(device=device)
    clip_model.eval()
    return clip_model, preprocess
