def forward(self, input, mixup_lambda=None, device=None):
    """
        Input: (batch_size, data_length)"""
    x = self.spectrogram_extractor(input)
    x = self.logmel_extractor(x)
    x = x.transpose(1, 3)
    x = self.bn0(x)
    x = x.transpose(1, 3)
    if self.training:
        x = self.spec_augmenter(x)
    if self.training and mixup_lambda is not None:
        x = do_mixup(x, mixup_lambda)
    x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
    x = F.dropout(x, p=0.2, training=self.training)
    x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
    x = F.dropout(x, p=0.2, training=self.training)
    x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
    x = F.dropout(x, p=0.2, training=self.training)
    x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
    x = F.dropout(x, p=0.2, training=self.training)
    x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
    x = F.dropout(x, p=0.2, training=self.training)
    x = torch.mean(x, dim=3)
    latent_x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
    latent_x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
    latent_x = latent_x1 + latent_x2
    latent_x = latent_x.transpose(1, 2)
    latent_x = F.relu_(self.fc1(latent_x))
    latent_output = interpolate(latent_x, 32)
    x1, _ = torch.max(x, dim=2)
    x2 = torch.mean(x, dim=2)
    x = x1 + x2
    x = F.dropout(x, p=0.5, training=self.training)
    x = F.relu_(self.fc1(x))
    embedding = F.dropout(x, p=0.5, training=self.training)
    clipwise_output = torch.sigmoid(self.fc_audioset(x))
    output_dict = {'clipwise_output': clipwise_output, 'embedding':
        embedding, 'fine_grained_embedding': latent_output}
    return output_dict
