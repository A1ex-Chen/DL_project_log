def test(net, testloader, device: str='cpu'):
    """Validate the network on the entire test set."""
    loss = 0.0
    net.eval()
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(testloader):
            images = images.to(device)
            recon_images, mu, logvar = net(images)
            recon_loss = F.mse_loss(recon_images, images)
            kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss += recon_loss + kld_loss
    loss = loss.detach().item()
    if images.shape[1] == 1:
        images = torch.repeat_interleave(images, repeats=3, dim=1)
        recon_images = torch.repeat_interleave(recon_images, repeats=3, dim=1)
    fid = compute_fid(images, recon_images, device)
    return loss, fid
