def __loadModels(self):
    self.embedding_dim = self.text_encoder.config.hidden_size
    models_args = {'device': self.torch_device, 'max_batch_size': self.
        max_batch_size, 'embedding_dim': self.embedding_dim, 'inpaint':
        self.inpaint}
    if 'clip' in self.stages:
        self.models['clip'] = make_CLIP(self.text_encoder, **models_args)
    if 'unet' in self.stages:
        self.models['unet'] = make_UNet(self.unet, **models_args)
    if 'vae' in self.stages:
        self.models['vae'] = make_VAE(self.vae, **models_args)
