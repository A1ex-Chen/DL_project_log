def check_eval_mode(self):
    for model, name in zip([self.text_encoder, self.vae, self.fn_STFT, self
        .unet], ['text_encoder', 'vae', 'fn_STFT', 'unet']):
        assert model.training == False, f'The {name} is not in eval mode.'
        for param in model.parameters():
            assert param.requires_grad == False, f'The {name} is not frozen.'
