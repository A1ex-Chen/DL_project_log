@property
def shape(self):
    return (self.number_of_frames, self.frame_height, self.frame_width,
        self.frame_channels)
