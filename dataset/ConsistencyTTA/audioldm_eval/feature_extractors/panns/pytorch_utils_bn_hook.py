def bn_hook(self, input, output):
    list_bn.append(input[0].nelement() * 2)
