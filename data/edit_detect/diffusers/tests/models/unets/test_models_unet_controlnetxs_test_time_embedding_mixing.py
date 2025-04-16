def test_time_embedding_mixing(self):
    unet = self.get_dummy_unet()
    controlnet = self.get_dummy_controlnet_from_unet(unet)
    controlnet_mix_time = self.get_dummy_controlnet_from_unet(unet,
        time_embedding_mix=0.5, learn_time_embedding=True)
    model = UNetControlNetXSModel.from_unet(unet, controlnet)
    model_mix_time = UNetControlNetXSModel.from_unet(unet, controlnet_mix_time)
    unet = unet.to(torch_device)
    model = model.to(torch_device)
    model_mix_time = model_mix_time.to(torch_device)
    input_ = self.dummy_input
    with torch.no_grad():
        output = model(**input_).sample
        output_mix_time = model_mix_time(**input_).sample
    assert output.shape == output_mix_time.shape
