def _profile_one_layer(self, m, x, dt):
    """
        Profile the computation time and FLOPs of a single layer of the model on a given input. Appends the results to
        the provided list.

        Args:
            m (nn.Module): The layer to be profiled.
            x (torch.Tensor): The input data to the layer.
            dt (list): A list to store the computation time of the layer.

        Returns:
            None
        """
    c = m == self.model[-1] and isinstance(x, list)
    flops = thop.profile(m, inputs=[x.copy() if c else x], verbose=False)[0
        ] / 1000000000.0 * 2 if thop else 0
    t = time_sync()
    for _ in range(10):
        m(x.copy() if c else x)
    dt.append((time_sync() - t) * 100)
    if m == self.model[0]:
        LOGGER.info(
            f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  module")
    LOGGER.info(f'{dt[-1]:10.2f} {flops:10.2f} {m.np:10.0f}  {m.type}')
    if c:
        LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")
