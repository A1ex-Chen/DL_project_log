def _predict_once(self, x, profile=False, visualize=False, embed=None):
    """
        Perform a forward pass through the network.

        Args:
            x (torch.Tensor): The input tensor to the model.
            profile (bool):  Print the computation time of each layer if True, defaults to False.
            visualize (bool): Save the feature maps of the model if True, defaults to False.
            embed (list, optional): A list of feature vectors/embeddings to return.

        Returns:
            (torch.Tensor): The last output of the model.
        """
    y, dt, embeddings = [], [], []
    for m in self.model:
        if m.f != -1:
            x = y[m.f] if isinstance(m.f, int) else [(x if j == -1 else y[j
                ]) for j in m.f]
        if profile:
            self._profile_one_layer(m, x, dt)
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
