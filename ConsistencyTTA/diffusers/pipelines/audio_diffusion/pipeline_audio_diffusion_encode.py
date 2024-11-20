@torch.no_grad()
def encode(self, images: List[Image.Image], steps: int=50) ->np.ndarray:
    """Reverse step process: recover noisy image from generated image.

        Args:
            images (`List[PIL Image]`): list of images to encode
            steps (`int`): number of encoding steps to perform (defaults to 50)

        Returns:
            `np.ndarray`: noise tensor of shape (batch_size, 1, height, width)
        """
    assert isinstance(self.scheduler, DDIMScheduler)
    self.scheduler.set_timesteps(steps)
    sample = np.array([np.frombuffer(image.tobytes(), dtype='uint8').
        reshape((1, image.height, image.width)) for image in images])
    sample = sample / 255 * 2 - 1
    sample = torch.Tensor(sample).to(self.device)
    for t in self.progress_bar(torch.flip(self.scheduler.timesteps, (0,))):
        prev_timestep = (t - self.scheduler.num_train_timesteps // self.
            scheduler.num_inference_steps)
        alpha_prod_t = self.scheduler.alphas_cumprod[t]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep
            ] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        model_output = self.unet(sample, t)['sample']
        pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
        sample = (sample - pred_sample_direction) * alpha_prod_t_prev ** -0.5
        sample = (sample * alpha_prod_t ** 0.5 + beta_prod_t ** 0.5 *
            model_output)
    return sample
