def __call__(self, x: Dict[str, object]):
    with torch.no_grad():
        feed_list = [torch.from_numpy(v).cuda() for k, v in x.items()]
        y_pred = self._model.handle(*feed_list)
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred,
        y_pred = [t.cpu().numpy() for t in y_pred]
        y_pred = dict(zip(self._output_names, y_pred))
    return y_pred
