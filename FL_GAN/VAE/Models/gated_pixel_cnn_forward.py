def forward(self, x):
    v_stack = self.conv_vstack(x)
    h_stack = self.conv_hstack(x)
    for layer in self.conv_layers:
        v_stack, h_stack = layer(v_stack, h_stack)
    out = self.conv_out(F.elu(h_stack))
    return out
