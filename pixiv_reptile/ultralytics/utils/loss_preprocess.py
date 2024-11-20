def preprocess(self, targets, batch_size, scale_tensor):
    """Preprocesses the target counts and matches with the input batch size to output a tensor."""
    if targets.shape[0] == 0:
        out = torch.zeros(batch_size, 0, 6, device=self.device)
    else:
        i = targets[:, 0]
        _, counts = i.unique(return_counts=True)
        counts = counts.to(dtype=torch.int32)
        out = torch.zeros(batch_size, counts.max(), 6, device=self.device)
        for j in range(batch_size):
            matches = i == j
            n = matches.sum()
            if n:
                bboxes = targets[matches, 2:]
                bboxes[..., :4].mul_(scale_tensor)
                out[j, :n] = torch.cat([targets[matches, 1:2], bboxes], dim=-1)
    return out
