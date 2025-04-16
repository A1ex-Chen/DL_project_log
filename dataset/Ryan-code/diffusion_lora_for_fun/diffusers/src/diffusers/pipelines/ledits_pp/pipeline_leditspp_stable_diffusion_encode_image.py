@torch.no_grad()
def encode_image(self, image, dtype=None, height=None, width=None,
    resize_mode='default', crops_coords=None):
    image = self.image_processor.preprocess(image=image, height=height,
        width=width, resize_mode=resize_mode, crops_coords=crops_coords)
    resized = self.image_processor.postprocess(image=image, output_type='pil')
    if max(image.shape[-2:]) > self.vae.config['sample_size'] * 1.5:
        logger.warning(
            'Your input images far exceed the default resolution of the underlying diffusion model. The output images may contain severe artifacts! Consider down-sampling the input using the `height` and `width` parameters'
            )
    image = image.to(dtype)
    x0 = self.vae.encode(image.to(self.device)).latent_dist.mode()
    x0 = x0.to(dtype)
    x0 = self.vae.config.scaling_factor * x0
    return x0, resized
