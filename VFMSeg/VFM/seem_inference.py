@torch.no_grad()
def inference(model, image, tasks, info, audio=None, refimg=None, reftxt=
    None, audio_pth=None, video_pth=None):
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        return interface_seem(model, audio, image, tasks, info, refimg,
            reftxt, audio_pth, video_pth)
