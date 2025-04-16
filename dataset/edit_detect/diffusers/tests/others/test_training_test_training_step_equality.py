@slow
def test_training_step_equality(self):
    device = 'cpu'
    ddpm_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_start=
        0.0001, beta_end=0.02, beta_schedule='linear', clip_sample=True)
    ddim_scheduler = DDIMScheduler(num_train_timesteps=1000, beta_start=
        0.0001, beta_end=0.02, beta_schedule='linear', clip_sample=True)
    assert ddpm_scheduler.config.num_train_timesteps == ddim_scheduler.config.num_train_timesteps
    set_seed(0)
    clean_images = [torch.randn((4, 3, 32, 32)).clip(-1, 1).to(device) for
        _ in range(4)]
    noise = [torch.randn((4, 3, 32, 32)).to(device) for _ in range(4)]
    timesteps = [torch.randint(0, 1000, (4,)).long().to(device) for _ in
        range(4)]
    model, optimizer = self.get_model_optimizer(resolution=32)
    model.train().to(device)
    for i in range(4):
        optimizer.zero_grad()
        ddpm_noisy_images = ddpm_scheduler.add_noise(clean_images[i], noise
            [i], timesteps[i])
        ddpm_noise_pred = model(ddpm_noisy_images, timesteps[i]).sample
        loss = torch.nn.functional.mse_loss(ddpm_noise_pred, noise[i])
        loss.backward()
        optimizer.step()
    del model, optimizer
    model, optimizer = self.get_model_optimizer(resolution=32)
    model.train().to(device)
    for i in range(4):
        optimizer.zero_grad()
        ddim_noisy_images = ddim_scheduler.add_noise(clean_images[i], noise
            [i], timesteps[i])
        ddim_noise_pred = model(ddim_noisy_images, timesteps[i]).sample
        loss = torch.nn.functional.mse_loss(ddim_noise_pred, noise[i])
        loss.backward()
        optimizer.step()
    del model, optimizer
    self.assertTrue(torch.allclose(ddpm_noisy_images, ddim_noisy_images,
        atol=1e-05))
    self.assertTrue(torch.allclose(ddpm_noise_pred, ddim_noise_pred, atol=
        1e-05))
