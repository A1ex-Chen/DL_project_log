def trace_model(model, batch_size=256, device=torch.device('cpu')):
    model.eval()
    audio_length = model.audio_cfg.audio_length
    example_audio = torch.ones((batch_size, audio_length), device=device)
    example_text = torch.zeros((batch_size, model.context_length), dtype=
        torch.int, device=device)
    model = torch.jit.trace_module(model, inputs=dict(forward=(
        example_audio, example_text), encode_text=(example_text,),
        encode_image=(example_audio,)))
    model.audio_cfg.audio_length = audio_length
    return model
