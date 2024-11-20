def preproc_func(self, audio, audio_shape):
    audio_, audio_shape_ = self.feat_proc([audio, audio_shape])
    if self.dist_lamb:
        audio_ = audio_.half()
    return audio_, audio_shape_
