@torch.no_grad()
def inference(image, task, *args, **kwargs):
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        if 'Video' in task:
            return interactive_infer_video(model, audio, image, task, *args,
                **kwargs)
        else:
            return interactive_infer_image(model, audio, image, task, *args,
                **kwargs)
