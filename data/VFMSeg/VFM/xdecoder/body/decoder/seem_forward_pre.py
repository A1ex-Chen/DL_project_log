def forward_pre(self, tgt):
    tgt2 = self.norm(tgt)
    tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
    tgt = tgt + self.dropout(tgt2)
    return tgt
