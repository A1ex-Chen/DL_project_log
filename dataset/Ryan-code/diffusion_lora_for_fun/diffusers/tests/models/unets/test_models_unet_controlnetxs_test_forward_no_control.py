@is_flaky
def test_forward_no_control(self):
    unet = self.get_dummy_unet()
    controlnet = self.get_dummy_controlnet_from_unet(unet)
    model = UNetControlNetXSModel.from_unet(unet, controlnet)
    unet = unet.to(torch_device)
    model = model.to(torch_device)
    input_ = self.dummy_input
    control_specific_input = ['controlnet_cond', 'conditioning_scale']
    input_for_unet = {k: v for k, v in input_.items() if k not in
        control_specific_input}
    with torch.no_grad():
        unet_output = unet(**input_for_unet).sample.cpu()
        unet_controlnet_output = model(**input_, apply_control=False
            ).sample.cpu()
    assert np.abs(unet_output.flatten() - unet_controlnet_output.flatten()
        ).max() < 0.0003
