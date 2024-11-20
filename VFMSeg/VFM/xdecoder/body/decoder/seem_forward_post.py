def forward_post(self, tgt):
    tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
    tgt = tgt + self.dropout(tgt2)
    tgt = self.norm(tgt)
    return tgt
