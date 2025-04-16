def tiled_decode(self, latents, current_height, current_width):
    core_size = self.unet.config.sample_size // 4
    core_stride = core_size
    pad_size = self.unet.config.sample_size // 4 * 3
    decoder_view_batch_size = 1
    views = self.get_views(current_height, current_width, stride=
        core_stride, window_size=core_size)
    views_batch = [views[i:i + decoder_view_batch_size] for i in range(0,
        len(views), decoder_view_batch_size)]
    latents_ = F.pad(latents, (pad_size, pad_size, pad_size, pad_size),
        'constant', 0)
    image = torch.zeros(latents.size(0), 3, current_height, current_width).to(
        latents.device)
    count = torch.zeros_like(image).to(latents.device)
    with self.progress_bar(total=len(views_batch)) as progress_bar:
        for j, batch_view in enumerate(views_batch):
            len(batch_view)
            latents_for_view = torch.cat([latents_[:, :, h_start:h_end + 
                pad_size * 2, w_start:w_end + pad_size * 2] for h_start,
                h_end, w_start, w_end in batch_view])
            image_patch = self.vae.decode(latents_for_view / self.vae.
                config.scaling_factor, return_dict=False)[0]
            h_start, h_end, w_start, w_end = views[j]
            h_start, h_end, w_start, w_end = (h_start * self.
                vae_scale_factor, h_end * self.vae_scale_factor, w_start *
                self.vae_scale_factor, w_end * self.vae_scale_factor)
            p_h_start, p_h_end, p_w_start, p_w_end = (pad_size * self.
                vae_scale_factor, image_patch.size(2) - pad_size * self.
                vae_scale_factor, pad_size * self.vae_scale_factor, 
                image_patch.size(3) - pad_size * self.vae_scale_factor)
            image[:, :, h_start:h_end, w_start:w_end] += image_patch[:, :,
                p_h_start:p_h_end, p_w_start:p_w_end]
            count[:, :, h_start:h_end, w_start:w_end] += 1
            progress_bar.update()
    image = image / count
    return image
