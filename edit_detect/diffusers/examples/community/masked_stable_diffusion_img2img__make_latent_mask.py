def _make_latent_mask(self, latents, mask):
    if mask is not None:
        latent_mask = []
        if not isinstance(mask, list):
            tmp_mask = [mask]
        else:
            tmp_mask = mask
        _, l_channels, l_height, l_width = latents.shape
        for m in tmp_mask:
            if not isinstance(m, PIL.Image.Image):
                if len(m.shape) == 2:
                    m = m[..., np.newaxis]
                if m.max() > 1:
                    m = m / 255.0
                m = self.image_processor.numpy_to_pil(m)[0]
            if m.mode != 'L':
                m = m.convert('L')
            resized = self.image_processor.resize(m, l_height, l_width)
            if self.debug_save:
                resized.save('latent_mask.png')
            latent_mask.append(np.repeat(np.array(resized)[np.newaxis, :, :
                ], l_channels, axis=0))
        latent_mask = torch.as_tensor(np.stack(latent_mask)).to(latents)
        latent_mask = latent_mask / latent_mask.max()
    return latent_mask
