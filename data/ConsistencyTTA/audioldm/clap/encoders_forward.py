def forward(self, batch, key=None):
    if self.model.training == True and not self.training_mode:
        print(
            'The pretrained CLAP model should always be in eval mode. Reloading model just in case you change the parameters.'
            )
        self.model, self.model_cfg = create_model(self.amodel, self.tmodel,
            self.pretrained, precision=self.precision, device='cuda',
            enable_fusion=self.enable_fusion, fusion_type=self.fusion_type)
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()
    if self.embed_mode == 'audio':
        with torch.no_grad():
            audio_dict_list = []
            assert self.sampling_rate == 16000, 'We only support 16000 sampling rate'
            if self.random_mute:
                batch = self._random_mute(batch)
            batch = torchaudio.functional.resample(batch, orig_freq=self.
                sampling_rate, new_freq=48000)
            for waveform in self.batch_to_list(batch):
                audio_dict = {}
                audio_dict = get_audio_features(audio_dict, waveform, 
                    480000, data_truncating='fusion', data_filling=
                    'repeatpad', audio_cfg=self.model_cfg['audio_cfg'])
                audio_dict_list.append(audio_dict)
            embed = self.model.get_audio_embedding(audio_dict_list)
    elif self.embed_mode == 'text':
        with torch.no_grad():
            text_data = self.tokenizer(batch)
            embed = self.model.get_text_embedding(text_data)
    embed = embed.unsqueeze(1)
    self.unconditional_token = self.model.get_text_embedding(self.tokenizer
        (['', '']))[0:1]
    for i in range(embed.size(0)):
        if self.make_decision(self.unconditional_prob):
            embed[i] = self.unconditional_token
    return embed.detach()
