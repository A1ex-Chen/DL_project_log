def sample(no_samples=args.num_sampling):
    shape = no_samples, H // 4, W // 4
    q_samples = torch.zeros(size=shape).long().to(device)
    for i in range(H // 4):
        for j in range(W // 4):
            out = model.pixelcnn_prior(q_samples)
            proba = F.softmax(out, dim=1)
            q_samples[:, i, j] = torch.multinomial(proba[:, :, i, j], 1
                ).squeeze().float()
    latents_shape = q_samples.shape
    encoding_inds = q_samples.view(-1, 1)
    encoding_one_hot = torch.zeros(encoding_inds.size(0), args.
        num_embeddings, device=device)
    encoding_one_hot.scatter_(1, encoding_inds, 1)
    quantized_latents = torch.matmul(encoding_one_hot, model.codebook.
        codebook.weight)
    quantized_latents = quantized_latents.view(latents_shape + (args.
        embedding_dim,))
    z_q_samples = quantized_latents.permute(0, 3, 1, 2).contiguous()
    x_samples = model.decoder(z_q_samples)
    return x_samples.detach().cpu()
