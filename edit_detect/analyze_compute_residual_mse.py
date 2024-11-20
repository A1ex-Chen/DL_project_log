def compute_residual_mse(seqs: torch.Tensor, images: torch.Tensor, device:
    Union[str, torch.device]='cuda'):
    seq_mse_mean: list = []
    seq_mse_var: list = []
    for i, (seq, image) in tqdm(enumerate(zip(seqs, images))):
        assert len(torch.unique(image, dim=0)) == 1
        seq_mse_mean.append(seq.mean(dim=0))
        seq_mse_var.append(seq.var(dim=0))
    seq_mse_mean = torch.stack(seq_mse_mean, dim=0)
    seq_mse_var = torch.stack(seq_mse_var, dim=0)
    return seq_mse_mean, seq_mse_var
