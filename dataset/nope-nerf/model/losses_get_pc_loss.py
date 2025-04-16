def get_pc_loss(self, Xt, Yt):
    match_method = self.cfg['match_method']
    if match_method == 'dense':
        loss1 = self.comp_point_point_error(Xt[0].permute(1, 0), Yt[0].
            permute(1, 0))
        loss2 = self.comp_point_point_error(Yt[0].permute(1, 0), Xt[0].
            permute(1, 0))
        loss = loss1 + loss2
    return loss
