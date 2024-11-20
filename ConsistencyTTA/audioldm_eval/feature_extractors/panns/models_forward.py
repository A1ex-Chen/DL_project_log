def forward(self, input, mixup_lambda=None):
    """
        Input: (batch_size, data_length)"""
    x = self.spectrogram_extractor(input)
    x = self.logmel_extractor(x)
    frames_num = x.shape[2]
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
    x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
    x = F.dropout(x, p=0.2, training=self.training)
    x = torch.mean(x, dim=3)
    x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
    x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
    x = x1 + x2
    x = F.dropout(x, p=0.5, training=self.training)
    x = x.transpose(1, 2)
    x = F.relu_(self.fc1(x))
    x = x.transpose(1, 2)
    x = F.dropout(x, p=0.5, training=self.training)
    clipwise_output, _, segmentwise_output = self.att_block(x)
    segmentwise_output = segmentwise_output.transpose(1, 2)
    framewise_output = interpolate(segmentwise_output, self.interpolate_ratio)
    framewise_output = pad_framewise_output(framewise_output, frames_num)
    output_dict = {'framewise_output': framewise_output, 'clipwise_output':
        clipwise_output}
    return output_dict
