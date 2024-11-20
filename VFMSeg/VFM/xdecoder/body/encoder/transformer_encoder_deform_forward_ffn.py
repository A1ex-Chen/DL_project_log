def forward_ffn(self, src):
    src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
    src = src + self.dropout3(src2)
    src = self.norm2(src)
    return src
