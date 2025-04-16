def forward(self, x, target):
    if self.training:
        x = x.float()
        target = target.float()
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs * target
        nll_loss = nll_loss.sum(-1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()
    else:
        return torch.nn.functional.cross_entropy(x, target)
