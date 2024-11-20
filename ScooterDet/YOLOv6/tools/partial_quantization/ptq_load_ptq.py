def load_ptq(model, calib_path, device):
    model_ptq = quant_model_init(model, device)
    model_ptq.load_state_dict(torch.load(calib_path)['model'].state_dict())
    return model_ptq
