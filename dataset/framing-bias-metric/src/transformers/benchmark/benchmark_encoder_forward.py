def encoder_forward():
    with torch.no_grad():
        outputs = inference_model(input_ids)
    return outputs
