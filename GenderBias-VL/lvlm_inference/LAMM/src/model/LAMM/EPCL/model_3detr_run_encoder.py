def run_encoder(self, point_clouds):
    if self.pre_encoder is None:
        return self.encoder(point_clouds)
    xyz, features = self._break_up_pc(point_clouds)
    pre_enc_xyz, pre_enc_features, pre_enc_inds = self.pre_encoder(xyz,
        features)
    pre_enc_features = pre_enc_features.permute(0, 2, 1)
    if self.use_task_emb:
        task_emb = self.get_prompt(batch_size=pre_enc_features.shape[0],
            te_token=self.te_tok, te_encoder=self.te_encoder, device=
            pre_enc_features.device)
    else:
        task_emb = None
    enc_xyz, enc_features, enc_inds = self.encoder(pre_enc_features, xyz=
        pre_enc_xyz, task_emb=task_emb)
    return enc_xyz, enc_features, enc_inds
