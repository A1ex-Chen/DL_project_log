def smart_DDP(model):
    assert not check_version(torch.__version__, '1.12.0', pinned=True
        ), 'torch==1.12.0 torchvision==0.13.0 DDP training is not supported due to a known issue. Please upgrade or downgrade torch to use DDP. See https://github.com/ultralytics/yolov5/issues/8395'
    if check_version(torch.__version__, '1.11.0'):
        return DDP(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK,
            static_graph=True)
    return DDP(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK)
