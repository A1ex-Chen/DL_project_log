def re_channels_out(self):
    for idx, channel_out in enumerate(self.channels_out):
        self.channels_out[idx] = self.get_width(channel_out)
