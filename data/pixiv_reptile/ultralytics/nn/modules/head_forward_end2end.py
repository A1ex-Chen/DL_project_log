def forward_end2end(self, x):
    """
        Performs forward pass of the v10Detect module.

        Args:
            x (tensor): Input tensor.

        Returns:
            (dict, tensor): If not in training mode, returns a dictionary containing the outputs of both one2many and one2one detections.
                           If in training mode, returns a dictionary containing the outputs of one2many and one2one detections separately.
        """
    x_detach = [xi.detach() for xi in x]
    one2one = [torch.cat((self.one2one_cv2[i](x_detach[i]), self.
        one2one_cv3[i](x_detach[i])), 1) for i in range(self.nl)]
    for i in range(self.nl):
        x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
    if self.training:
        return {'one2many': x, 'one2one': one2one}
    y = self._inference(one2one)
    y = self.postprocess(y.permute(0, 2, 1), self.max_det, self.nc)
    return y if self.export else (y, {'one2many': x, 'one2one': one2one})
