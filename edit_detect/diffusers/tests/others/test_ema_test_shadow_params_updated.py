def test_shadow_params_updated(self):
    unet, ema_unet = self.get_models()
    unet_pseudo_updated_step_one = self.simulate_backprop(unet)
    ema_unet.step(unet_pseudo_updated_step_one.parameters())
    orig_params = list(unet_pseudo_updated_step_one.parameters())
    for s_param, param in zip(ema_unet.shadow_params, orig_params):
        assert ~torch.allclose(s_param, param)
    for _ in range(4):
        ema_unet.step(unet.parameters())
    for s_param, param in zip(ema_unet.shadow_params, orig_params):
        assert ~torch.allclose(s_param, param)
