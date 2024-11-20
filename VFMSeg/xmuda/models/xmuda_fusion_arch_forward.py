def forward(self, data_batch):
    img = data_batch['img']
    img_indices = data_batch['img_indices']
    x = self.net_2d(img)
    feats_2d = []
    for i in range(x.shape[0]):
        feats_2d.append(x.permute(0, 2, 3, 1)[i][img_indices[i][:, 0],
            img_indices[i][:, 1]])
    feats_2d = torch.cat(feats_2d, 0)
    feats_3d = self.net_3d(data_batch['x'])
    x = torch.cat([feats_2d, feats_3d], 1)
    feats_fuse = self.fuse(x)
    x = self.linear(feats_fuse)
    preds = {'feats': feats_fuse, 'seg_logit': x}
    if self.dual_head:
        preds['seg_logit_2d'] = self.linear_2d(feats_2d)
        preds['seg_logit_3d'] = self.linear_3d(feats_3d)
    return preds
