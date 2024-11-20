def compute_seq_mse(seqs: torch.Tensor, images: torch.Tensor, reconsts:
    torch.Tensor, device: Union[str, torch.device]='cuda'):
    seq_mse_mean: list = []
    seq_mse_var: list = []
    for i, (seq, image, reconst) in tqdm(enumerate(zip(seqs, images, reconsts))
        ):
        assert len(torch.unique(image, dim=0)) == 1
        mse_seq = mse_series(x0=image.to(device), pred_orig_images=seq.to(
            device).transpose(0, 1)).cpu()
        seq_mse_mean.append(mse_seq.mean(dim=0))
        seq_mse_var.append(mse_seq.var(dim=0))
    seq_mse_mean = torch.stack(seq_mse_mean, dim=0)
    seq_mse_var = torch.stack(seq_mse_var, dim=0)
    return seq_mse_mean, seq_mse_var
