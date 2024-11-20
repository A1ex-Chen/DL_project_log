@torch.no_grad()
def apply_ema(optimizer, model, ema_model, decay, ema_update_type):
    if not decay:
        return
    ema_model_weight_list = list(ema_model.parameters())
    if ema_update_type == 'fp32':
        model_weight_list = list(amp.master_params(optimizer))
    elif ema_update_type == 'fp16':
        model_weight_list = list(model.parameters())
    for ema, source in zip(ema_model_weight_list, model_weight_list):
        ema.copy_(decay * ema + (1 - decay) * source)
