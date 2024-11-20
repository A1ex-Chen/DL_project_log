def test_zero_decay(self):
    unet, ema_unet = self.get_models(decay=0.0)
    unet_step_one = self.simulate_backprop(unet)
    ema_unet.step(unet_step_one.parameters())
    step_one_shadow_params = ema_unet.shadow_params
    unet_step_two = self.simulate_backprop(unet_step_one)
    ema_unet.step(unet_step_two.parameters())
    step_two_shadow_params = ema_unet.shadow_params
    for step_one, step_two in zip(step_one_shadow_params,
        step_two_shadow_params):
        assert torch.allclose(step_one, step_two)
