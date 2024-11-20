def forward(self, x):
    x = self.encoder(x)
    atten = nn.Softmax()(self.attention(x))
    x = nn.ReLU()(self.reduce(atten * x))
    x = self.decoder(x)
    return x
