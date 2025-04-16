def forward(self, x):
    x = self.embed(x)
    x = x.transpose(1, 2)
    conv_results = []
    conv_results.append(self.conv1(x).view(-1, self.hparams.n_filters))
    conv_results.append(self.conv2(x).view(-1, self.hparams.n_filters))
    conv_results.append(self.conv3(x).view(-1, self.hparams.n_filters))
    x = torch.cat(conv_results, 1)
    logits = self.classifier(x)
    return logits
