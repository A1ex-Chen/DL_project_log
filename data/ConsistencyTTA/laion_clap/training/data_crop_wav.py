def crop_wav(self, x):
    crop_size = self.audio_cfg['crop_size']
    crop_pos = random.randint(0, len(x) - crop_size - 1)
    return x[crop_pos:crop_pos + crop_size]
