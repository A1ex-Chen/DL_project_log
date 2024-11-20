def encode_audio(self, audio, device):
    return self.audio_branch(audio, mixup_lambda=None, device=device)
