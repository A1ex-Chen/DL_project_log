def test_modify_padding_mode(self):

    def set_pad_mode(network, mode='circular'):
        for _, module in network.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                module.padding_mode = mode
    for scheduler_cls in [DDIMScheduler, LCMScheduler]:
        components, _, _ = self.get_dummy_components(scheduler_cls)
        pipe = self.pipeline_class(**components)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        _pad_mode = 'circular'
        set_pad_mode(pipe.vae, _pad_mode)
        set_pad_mode(pipe.unet, _pad_mode)
        _, _, inputs = self.get_dummy_inputs()
        _ = pipe(**inputs).images
