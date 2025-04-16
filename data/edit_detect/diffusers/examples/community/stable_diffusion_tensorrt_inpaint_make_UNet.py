def make_UNet(model, device, max_batch_size, embedding_dim, inpaint=False,
    unet_dim=4):
    return UNet(model, fp16=True, device=device, max_batch_size=
        max_batch_size, embedding_dim=embedding_dim, unet_dim=unet_dim)
