def smart_hub_load(repo='ultralytics/yolov5', model='yolov5s', **kwargs):
    if check_version(torch.__version__, '1.9.1'):
        kwargs['skip_validation'] = True
    if check_version(torch.__version__, '1.12.0'):
        kwargs['trust_repo'] = True
    try:
        return torch.hub.load(repo, model, **kwargs)
    except Exception:
        return torch.hub.load(repo, model, force_reload=True, **kwargs)
