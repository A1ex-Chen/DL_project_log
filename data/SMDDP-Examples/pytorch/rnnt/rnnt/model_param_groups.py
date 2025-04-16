def param_groups(self, lr):
    chain_params = lambda *layers: chain(*[l.parameters() for l in layers])
    return [{'params': chain_params(self.encoder), 'lr': lr * self.
        enc_lr_factor}, {'params': chain_params(self.prediction), 'lr': lr *
        self.pred_lr_factor}, {'params': chain_params(self.joint_enc, self.
        joint_pred, self.joint_net), 'lr': lr * self.joint_lr_factor}]
