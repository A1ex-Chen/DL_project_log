def init_multi_tensor_ema(optimizer, model, ema_model, ema_update_type):
    ema_model_weight_list = list(ema_model.parameters())
    if ema_update_type == 'fp32':
        model_weight_list = list(amp.master_params(optimizer))
    elif ema_update_type == 'fp16':
        model_weight_list = list(model.parameters())
    overflow_buf_for_ema = torch.cuda.IntTensor([0])
    return ema_model_weight_list, model_weight_list, overflow_buf_for_ema
