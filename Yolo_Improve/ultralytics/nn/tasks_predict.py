def predict(self, x, profile=False, visualize=False, txt_feats=None,
    augment=False, embed=None):
    """
        Perform a forward pass through the model.

        Args:
            x (torch.Tensor): The input tensor.
            profile (bool, optional): If True, profile the computation time for each layer. Defaults to False.
            visualize (bool, optional): If True, save feature maps for visualization. Defaults to False.
            txt_feats (torch.Tensor): The text features, use it if it's given. Defaults to None.
            augment (bool, optional): If True, perform data augmentation during inference. Defaults to False.
            embed (list, optional): A list of feature vectors/embeddings to return.

        Returns:
            (torch.Tensor): Model's output tensor.
        """
    txt_feats = (self.txt_feats if txt_feats is None else txt_feats).to(device
        =x.device, dtype=x.dtype)
    if len(txt_feats) != len(x):
        txt_feats = txt_feats.repeat(len(x), 1, 1)
    ori_txt_feats = txt_feats.clone()
    y, dt, embeddings = [], [], []
    for m in self.model:
        if m.f != -1:
            x = y[m.f] if isinstance(m.f, int) else [(x if j == -1 else y[j
                ]) for j in m.f]
        if profile:
            self._profile_one_layer(m, x, dt)
        if isinstance(m, C2fAttn):
            x = m(x, txt_feats)
        elif isinstance(m, WorldDetect):
            x = m(x, ori_txt_feats)
        elif isinstance(m, ImagePoolingAttn):
            txt_feats = m(x, txt_feats)
        else:
            x = m(x)
        y.append(x if m.i in self.save else None)
        if visualize:
            feature_visualization(x, m.type, m.i, save_dir=visualize)
        if embed and m.i in embed:
            embeddings.append(nn.functional.adaptive_avg_pool2d(x, (1, 1)).
                squeeze(-1).squeeze(-1))
            if m.i == max(embed):
                return torch.unbind(torch.cat(embeddings, 1), dim=0)
    return x
