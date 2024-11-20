def test_shadow_params_not_updated(self):
    unet, ema_unet = self.get_models()
    ema_unet.step(unet.parameters())
    orig_params = list(unet.parameters())
    for s_param, param in zip(ema_unet.shadow_params, orig_params):
        assert torch.allclose(s_param, param)
    for _ in range(4):
        ema_unet.step(unet.parameters())
    for s_param, param in zip(ema_unet.shadow_params, orig_params):
        assert torch.allclose(s_param, param)
