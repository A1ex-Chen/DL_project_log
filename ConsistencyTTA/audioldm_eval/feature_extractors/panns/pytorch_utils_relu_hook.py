def relu_hook(self, input, output):
    list_relu.append(input[0].nelement() * 2)
