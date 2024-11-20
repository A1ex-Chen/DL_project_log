def forward(self, x):
    outputs = self._forward(x)
    if self.training:
        return outputs
    else:
        if self.export:
            bs = -1
            no = outputs[0].shape[1]
        else:
            bs, no = outputs[0].shape[0], outputs[0].shape[1]
        self.hw = [out.shape[-2:] for out in outputs]
        outs = []
        for i in range(len(outputs)):
            h, w = self.hw[i]
            out = outputs[i]
            outs.append(out.view(bs, no, h * w))
        out = torch.cat(outs, dim=-1).permute(0, 2, 1)
        out = self.decode_outputs(out, dtype=x[0].type())
        return out,
