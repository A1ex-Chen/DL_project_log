def __getitem__(self, index):
    """Load waveform, text, and target of an audio clip

        Parameters
        ----------
            index: int
                the index number
        Return
        ------
            output: dict {
                "hdf5_path": str,
                "index_in_hdf5": int,
                "audio_name": str,
                "waveform": list (audio_length,),
                "target": list (class_num, ),
                "text": torch.tensor (context_length,)
            }
                the output dictionary
        """
    s_index = self.queue[index]
    audio_name = self.fp['audio_name'][s_index].decode()
    hdf5_path = self.fp['hdf5_path'][s_index].decode().replace('../workspace',
        '/home/la/kechen/Research/ke_zsasp/workspace')
    r_idx = self.fp['index_in_hdf5'][s_index]
    target = self.fp['target'][s_index].astype(np.float32)
    text = self.prompt_text(target)
    with h5py.File(hdf5_path, 'r') as f:
        waveform = int16_to_float32(f['waveform'][r_idx])[:self.audio_cfg[
            'clip_samples']]
    assert len(waveform) == self.audio_cfg['clip_samples'
        ], 'The sample length is not match'
    mel_spec = get_mel(torch.from_numpy(waveform), self.audio_cfg)[None, :, :]
    mel_spec = torch.cat([mel_spec, mel_spec.clone(), mel_spec.clone(),
        mel_spec.clone()], dim=0).cpu().numpy()
    longer = random.choice([True, False])
    if longer == False:
        mel_spec[1:, :, :] = 0.0
    data_dict = {'hdf5_path': hdf5_path, 'index_in_hdf5': r_idx,
        'audio_name': audio_name, 'waveform': waveform, 'class_label':
        target, 'text': text, 'longer': longer, 'mel_fusion': mel_spec}
    return data_dict
