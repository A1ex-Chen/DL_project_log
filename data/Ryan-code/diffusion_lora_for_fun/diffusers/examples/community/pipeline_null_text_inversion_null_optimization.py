def null_optimization(self, latents, context, num_inner_steps, epsilon):
    uncond_embeddings, cond_embeddings = context.chunk(2)
    uncond_embeddings_list = []
    latent_cur = latents[-1]
    bar = tqdm(total=num_inner_steps * self.num_inference_steps)
    for i in range(self.num_inference_steps):
        uncond_embeddings = uncond_embeddings.clone().detach()
        uncond_embeddings.requires_grad = True
        optimizer = Adam([uncond_embeddings], lr=0.01 * (1.0 - i / 100.0))
        latent_prev = latents[len(latents) - i - 2]
        t = self.scheduler.timesteps[i]
        with torch.no_grad():
            noise_pred_cond = self.get_noise_pred_single(latent_cur, t,
                cond_embeddings)
        for j in range(num_inner_steps):
            noise_pred_uncond = self.get_noise_pred_single(latent_cur, t,
                uncond_embeddings)
            noise_pred = noise_pred_uncond + 7.5 * (noise_pred_cond -
                noise_pred_uncond)
            latents_prev_rec = self.prev_step(noise_pred, t, latent_cur)
            loss = nnf.mse_loss(latents_prev_rec, latent_prev)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_item = loss.item()
            bar.update()
            if loss_item < epsilon + i * 2e-05:
                break
        for j in range(j + 1, num_inner_steps):
            bar.update()
        uncond_embeddings_list.append(uncond_embeddings[:1].detach())
        with torch.no_grad():
            context = torch.cat([uncond_embeddings, cond_embeddings])
            latent_cur = self.get_noise_pred(latent_cur, t, context)
    bar.close()
    return uncond_embeddings_list
