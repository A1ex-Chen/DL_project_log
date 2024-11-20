def forward(self, image):
    out = self.encoder.forward_first_conv(image)
    out = self.se_layer0(out)
    out = F.max_pool2d(out, kernel_size=3, stride=2, padding=1)
    out = self.encoder.forward_layer1(out)
    out = self.se_layer1(out)
    skip1 = self.skip_layer1(out)
    out = self.encoder.forward_layer2(out)
    out = self.se_layer2(out)
    skip2 = self.skip_layer2(out)
    out = self.encoder.forward_layer3(out)
    out = self.se_layer3(out)
    skip3 = self.skip_layer3(out)
    out = self.encoder.forward_layer4(out)
    out = self.se_layer4(out)
    out = self.context_module(out)
    outs = [out, skip3, skip2, skip1]
    return self.decoder(enc_outs=outs)