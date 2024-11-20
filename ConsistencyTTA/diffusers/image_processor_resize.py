def resize(self, images: PIL.Image.Image) ->PIL.Image.Image:
    """
        Resize a PIL image. Both height and width will be downscaled to the next integer multiple of `vae_scale_factor`
        """
    w, h = images.size
    w, h = (x - x % self.vae_scale_factor for x in (w, h))
    images = images.resize((w, h), resample=PIL_INTERPOLATION[self.resample])
    return images
