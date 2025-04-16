def linear_hook(self, input, output):
    batch_size = input[0].size(0) if input[0].dim() == 2 else 1
    weight_ops = self.weight.nelement() * (2 if multiply_adds else 1)
    bias_ops = self.bias.nelement()
    flops = batch_size * (weight_ops + bias_ops)
    list_linear.append(flops)
