def infer_model_precision(model):
    counter = Counter()
    for param in model.parameters():
        counter[param.dtype] += 1
    if counter[torch.float16] > 0:
        return Precision.FP16
    else:
        return Precision.FP32
