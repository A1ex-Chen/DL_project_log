def enc_pred(self, x, x_lens, y, y_lens, pred_stream, state=None):
    pred_stream.wait_stream(torch.cuda.current_stream())
    f, x_lens = self.encode(x, x_lens)
    with torch.cuda.stream(pred_stream):
        y = label_collate(y)
        g, _ = self.predict(y, state)
    torch.cuda.current_stream().wait_stream(pred_stream)
    return f, g, x_lens
