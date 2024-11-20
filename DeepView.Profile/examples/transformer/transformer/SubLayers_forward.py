def forward(self, x):
    residual = x
    output = x.transpose(1, 2)
    output = self.w_2(F.relu(self.w_1(output)))
    output = output.transpose(1, 2)
    output = self.dropout(output)
    output = self.layer_norm(output + residual)
    return output
