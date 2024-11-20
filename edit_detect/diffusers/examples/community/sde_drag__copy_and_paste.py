def _copy_and_paste(self, latent, source_new, target_new, adapt_radius,
    max_height, max_width, image_scale, noise_scale, generator):

    def adaption_r(source, target, adapt_radius, max_height, max_width):
        r_x_lower = min(adapt_radius, source[0], target[0])
        r_x_upper = min(adapt_radius, max_width - source[0], max_width -
            target[0])
        r_y_lower = min(adapt_radius, source[1], target[1])
        r_y_upper = min(adapt_radius, max_height - source[1], max_height -
            target[1])
        return r_x_lower, r_x_upper, r_y_lower, r_y_upper
    for source_, target_ in zip(source_new, target_new):
        r_x_lower, r_x_upper, r_y_lower, r_y_upper = adaption_r(source_,
            target_, adapt_radius, max_height, max_width)
        source_feature = latent[:, :, source_[1] - r_y_lower:source_[1] +
            r_y_upper, source_[0] - r_x_lower:source_[0] + r_x_upper].clone()
        latent[:, :, source_[1] - r_y_lower:source_[1] + r_y_upper, source_
            [0] - r_x_lower:source_[0] + r_x_upper
            ] = image_scale * source_feature + noise_scale * torch.randn(latent
            .shape[0], 4, r_y_lower + r_y_upper, r_x_lower + r_x_upper,
            device=self.device, generator=generator)
        latent[:, :, target_[1] - r_y_lower:target_[1] + r_y_upper, target_
            [0] - r_x_lower:target_[0] + r_x_upper] = source_feature * 1.1
    return latent
