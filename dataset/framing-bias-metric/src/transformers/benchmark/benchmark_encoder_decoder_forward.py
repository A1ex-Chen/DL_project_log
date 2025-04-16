def encoder_decoder_forward():
    with torch.no_grad():
        outputs = inference_model(input_ids, decoder_input_ids=input_ids)
    return outputs
