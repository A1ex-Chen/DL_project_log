def check_train_batch_size(model, imgsz=640, amp=True):
    with torch.cuda.amp.autocast(amp):
        return autobatch(deepcopy(model).train(), imgsz)
