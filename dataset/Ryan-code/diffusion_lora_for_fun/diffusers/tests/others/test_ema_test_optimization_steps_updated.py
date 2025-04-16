def test_optimization_steps_updated(self):
    unet, ema_unet = self.get_models()
    ema_unet.step(unet.parameters())
    assert ema_unet.optimization_step == 1
    for _ in range(2):
        ema_unet.step(unet.parameters())
    assert ema_unet.optimization_step == 3
