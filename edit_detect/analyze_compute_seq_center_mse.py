def compute_seq_center_mse(seqs: torch.Tensor, images: torch.Tensor,
    reconsts: torch.Tensor, device: Union[str, torch.device]='cuda'):
    seq_center_mse_mean: list = []
    seq_center_mse_var: list = []
    for i, (seq, image, reconst) in tqdm(enumerate(zip(seqs, images, reconsts))
        ):
        assert len(torch.unique(image, dim=0)) == 1
        center_mse_seq = center_mse_series(pred_orig_images=seq.to(device).
            transpose(0, 1)).cpu()
        seq_center_mse_mean.append(center_mse_seq.mean(dim=0))
        seq_center_mse_var.append(center_mse_seq.var(dim=0))
    seq_center_mse_mean = torch.stack(seq_center_mse_mean, dim=0)
    seq_center_mse_var = torch.stack(seq_center_mse_var, dim=0)
    return seq_center_mse_mean, seq_center_mse_var
