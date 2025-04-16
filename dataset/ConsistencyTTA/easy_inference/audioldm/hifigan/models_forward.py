def forward(self, x):
    x = self.conv_pre(x)
    for i in range(self.num_upsamples):
        x = F.leaky_relu(x, LRELU_SLOPE)
        x = self.ups[i](x)
        xs = None
        for j in range(self.num_kernels):
            if xs is None:
                xs = self.resblocks[i * self.num_kernels + j](x)
            else:
                xs += self.resblocks[i * self.num_kernels + j](x)
        x = xs / self.num_kernels
    x = F.leaky_relu(x)
    x = self.conv_post(x)
    x = torch.tanh(x)
    return x
