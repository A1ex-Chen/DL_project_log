def batch_update(self, images: torch.Tensor, noisy_images: torch.Tensor,
    reconsts: torch.Tensor, noises: torch.Tensor, timestep: int, label: str,
    seqs: torch.Tensor=None, residuals: torch.Tensor=None, traj_residuals:
    torch.Tensor=None):
    n: int = len(images)
    labels: List[str] = [label] * n
    tss: List[int] = [timestep] * n
    if seqs is None:
        seqs = [None] * n
    if residuals is None:
        residuals = [None] * n
    if traj_residuals is None:
        traj_residuals = [None] * n
    for i, (image, noisy_image, noise, reconst, seq, residual,
        traj_residual, lab, ts) in enumerate(zip(images, noisy_images,
        noises, reconsts, seqs, residuals, traj_residuals, labels, tss)):
        if seq is not None:
            self.update_seq(values=seq)
        self.update_reconst(values=reconst)
        self.update_noise(values=noise)
        self.update_image(values=image)
        self.update_noisy_image(values=noisy_image)
        if residual is not None:
            self.update_residual(values=residual)
        if traj_residual is not None:
            self.update_traj_residual(values=traj_residual)
        self.update_ts(values=torch.LongTensor([ts]))
        self.update_label(values=torch.LongTensor([lab]))
