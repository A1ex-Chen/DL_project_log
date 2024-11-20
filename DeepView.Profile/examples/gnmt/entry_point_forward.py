def forward(self, src, src_len, tgt, tgt_len):
    out = self.gnmt(src, src_len, tgt[:-1])
    T, B = out.size(0), out.size(1)
    tgt_labels = tgt[1:]
    loss = self.loss_fn(out.view(T * B, -1), tgt_labels.contiguous().view(-1))
    return loss / B
