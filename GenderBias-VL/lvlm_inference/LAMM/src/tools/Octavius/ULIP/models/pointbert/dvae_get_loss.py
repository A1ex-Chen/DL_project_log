def get_loss(self, ret, gt):
    loss_recon = self.recon_loss(ret, gt)
    logits = ret[-1]
    softmax = F.softmax(logits, dim=-1)
    mean_softmax = softmax.mean(dim=1)
    log_qy = torch.log(mean_softmax)
    log_uniform = torch.log(torch.tensor([1.0 / self.num_tokens], device=gt
        .device))
    loss_klv = F.kl_div(log_qy, log_uniform.expand(log_qy.size(0), log_qy.
        size(1)), None, None, 'batchmean', log_target=True)
    return loss_recon, loss_klv
