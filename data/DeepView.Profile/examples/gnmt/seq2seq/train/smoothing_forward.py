def forward(self, x, target):
    logprobs = torch.nn.functional.log_softmax(x, dim=-1, dtype=torch.float32)
    non_pad_mask = target != self.padding_idx
    nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
    nll_loss = nll_loss.squeeze(1)[non_pad_mask]
    smooth_loss = -logprobs.mean(dim=-1)[non_pad_mask]
    loss = self.confidence * nll_loss + self.smoothing * smooth_loss
    return loss.sum()
