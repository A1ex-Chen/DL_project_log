def forward(self, x, mask):
    x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x,
        mask=mask))
    x = self.output_sublayer(x, self.feed_forward)
    return self.dropout(x)
