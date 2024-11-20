def __getitem__(self, index):
    caption, audio_path = self.captions[index], self.audio_paths[index]
    gt_waveform = torch_tools.read_wav_file(audio_path, self.seg_length[1],
        self.sample_rate)
    indice = self.indices[index]
    gen_wav_path = f'{self.generated_path}/output_{indice}.wav'
    gen_waveform = torch_tools.read_wav_file(gen_wav_path, self.seg_length[
        0], self.sample_rate)
    gen_mel = self.mel[index, :, :, :].unsqueeze(dim=0
        ) if self.mel is not None else None
    return caption, gt_waveform, gen_waveform, gen_mel
