def prepare_for_ipex(self, promt, dtype=torch.float32, height=None, width=
    None, guidance_scale=7.5):
    self.unet = self.unet.to(memory_format=torch.channels_last)
    self.vae.decoder = self.vae.decoder.to(memory_format=torch.channels_last)
    self.text_encoder = self.text_encoder.to(memory_format=torch.channels_last)
    if self.safety_checker is not None:
        self.safety_checker = self.safety_checker.to(memory_format=torch.
            channels_last)
    unet_input_example, vae_decoder_input_example = self.get_input_example(
        promt, height, width, guidance_scale)
    if dtype == torch.bfloat16:
        self.unet = ipex.optimize(self.unet.eval(), dtype=torch.bfloat16,
            inplace=True)
        self.vae.decoder = ipex.optimize(self.vae.decoder.eval(), dtype=
            torch.bfloat16, inplace=True)
        self.text_encoder = ipex.optimize(self.text_encoder.eval(), dtype=
            torch.bfloat16, inplace=True)
        if self.safety_checker is not None:
            self.safety_checker = ipex.optimize(self.safety_checker.eval(),
                dtype=torch.bfloat16, inplace=True)
    elif dtype == torch.float32:
        self.unet = ipex.optimize(self.unet.eval(), dtype=torch.float32,
            inplace=True, weights_prepack=True, auto_kernel_selection=False)
        self.vae.decoder = ipex.optimize(self.vae.decoder.eval(), dtype=
            torch.float32, inplace=True, weights_prepack=True,
            auto_kernel_selection=False)
        self.text_encoder = ipex.optimize(self.text_encoder.eval(), dtype=
            torch.float32, inplace=True, weights_prepack=True,
            auto_kernel_selection=False)
        if self.safety_checker is not None:
            self.safety_checker = ipex.optimize(self.safety_checker.eval(),
                dtype=torch.float32, inplace=True, weights_prepack=True,
                auto_kernel_selection=False)
    else:
        raise ValueError(
            " The value of 'dtype' should be 'torch.bfloat16' or 'torch.float32' !"
            )
    with torch.cpu.amp.autocast(enabled=dtype == torch.bfloat16
        ), torch.no_grad():
        unet_trace_model = torch.jit.trace(self.unet, unet_input_example,
            check_trace=False, strict=False)
        unet_trace_model = torch.jit.freeze(unet_trace_model)
    self.unet.forward = unet_trace_model.forward
    with torch.cpu.amp.autocast(enabled=dtype == torch.bfloat16
        ), torch.no_grad():
        ave_decoder_trace_model = torch.jit.trace(self.vae.decoder,
            vae_decoder_input_example, check_trace=False, strict=False)
        ave_decoder_trace_model = torch.jit.freeze(ave_decoder_trace_model)
    self.vae.decoder.forward = ave_decoder_trace_model.forward
