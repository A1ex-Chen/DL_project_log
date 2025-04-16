def forward(self, x, target):
    logprobs = torch.nn.functional.log_softmax(x, dim=-1)
    nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
    nll_loss = nll_loss.squeeze(1)
    smooth_loss = -logprobs.mean(dim=-1)
    loss = self.confidence * nll_loss + self.smoothing * smooth_loss
    return loss.mean()
