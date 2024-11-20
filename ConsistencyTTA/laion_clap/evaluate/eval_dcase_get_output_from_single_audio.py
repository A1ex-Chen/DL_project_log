def get_output_from_single_audio(audio, text, model, device):
    audio_features = model(audio, None, device)
    audio_features = F.normalize(audio_features, dim=-1)
    text_features = model(None, text, device=device)
    text_features = F.normalize(text_features, dim=-1)
    audio_features_mlp = model.audio_transform(audio_features)
    text_features_mlp = model.text_transform(text_features)
    return (audio_features, text_features, audio_features_mlp,
        text_features_mlp, model.logit_scale_a.exp(), model.logit_scale_t.exp()
        )
