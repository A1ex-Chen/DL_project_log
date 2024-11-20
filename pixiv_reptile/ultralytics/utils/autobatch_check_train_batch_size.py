def check_train_batch_size(model, imgsz=640, amp=True, batch=-1):
    """
    Compute optimal YOLO training batch size using the autobatch() function.

    Args:
        model (torch.nn.Module): YOLO model to check batch size for.
        imgsz (int): Image size used for training.
        amp (bool): If True, use automatic mixed precision (AMP) for training.

    Returns:
        (int): Optimal batch size computed using the autobatch() function.
    """
    with torch.cuda.amp.autocast(amp):
        return autobatch(deepcopy(model).train(), imgsz, fraction=batch if 
            0.0 < batch < 1.0 else 0.6)
