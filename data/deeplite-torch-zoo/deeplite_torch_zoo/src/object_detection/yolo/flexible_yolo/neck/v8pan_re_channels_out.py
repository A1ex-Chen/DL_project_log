def re_channels_out(self):
    for idx, channel_out in enumerate(self.channels_outs):
        self.channels_outs[idx] = self.get_width(channel_out)
