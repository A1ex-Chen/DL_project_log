def forward(self, images):
    encoder_outs = self.encoder(images)
    enc_down_32, enc_down_16, enc_down_8, enc_down_4 = encoder_outs
    out = self.avgpool(enc_down_32)
    out = torch.flatten(out, 1)
    out = self.fc(out)
    return out
