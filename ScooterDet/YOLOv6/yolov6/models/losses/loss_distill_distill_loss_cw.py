def distill_loss_cw(self, s_feats, t_feats, temperature=1):
    N, C, H, W = s_feats[0].shape
    loss_cw = F.kl_div(F.log_softmax(s_feats[0].view(N, C, H * W) /
        temperature, dim=2), F.log_softmax(t_feats[0].view(N, C, H * W).
        detach() / temperature, dim=2), reduction='sum', log_target=True) * (
        temperature * temperature) / (N * C)
    N, C, H, W = s_feats[1].shape
    loss_cw += F.kl_div(F.log_softmax(s_feats[1].view(N, C, H * W) /
        temperature, dim=2), F.log_softmax(t_feats[1].view(N, C, H * W).
        detach() / temperature, dim=2), reduction='sum', log_target=True) * (
        temperature * temperature) / (N * C)
    N, C, H, W = s_feats[2].shape
    loss_cw += F.kl_div(F.log_softmax(s_feats[2].view(N, C, H * W) /
        temperature, dim=2), F.log_softmax(t_feats[2].view(N, C, H * W).
        detach() / temperature, dim=2), reduction='sum', log_target=True) * (
        temperature * temperature) / (N * C)
    return loss_cw
