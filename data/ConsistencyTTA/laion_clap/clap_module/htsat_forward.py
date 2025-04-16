def forward(self, x: torch.Tensor, mixup_lambda=None, infer_mode=False,
    device=None):
    if self.enable_fusion and x['longer'].sum() == 0:
        if self.training:
            x['longer'][torch.randint(0, x['longer'].shape[0], (1,))] = True
        else:
            x = x['mel_fusion'].to(device=device, non_blocking=True)
            x = x.transpose(1, 3)
            x = self.bn0(x)
            x = x.transpose(1, 3)
            x = self.reshape_wav2img(x)
            output_dict = self.forward_features(x, longer_idx=[])
            return output_dict
    if not self.enable_fusion:
        x = x['waveform'].to(device=device, non_blocking=True)
        x = self.spectrogram_extractor(x)
        x = self.logmel_extractor(x)
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        if self.training:
            x = self.spec_augmenter(x)
        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)
        x = self.reshape_wav2img(x)
        output_dict = self.forward_features(x)
    else:
        longer_list = x['longer'].to(device=device, non_blocking=True)
        x = x['mel_fusion'].to(device=device, non_blocking=True)
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        longer_list_idx = torch.where(longer_list)[0]
        if self.fusion_type in ['daf_1d', 'aff_1d', 'iaff_1d']:
            new_x = x[:, 0:1, :, :].clone().contiguous()
            if len(longer_list_idx) > 0:
                fusion_x_local = x[longer_list_idx, 1:, :, :].clone(
                    ).contiguous()
                FB, FC, FT, FF = fusion_x_local.size()
                fusion_x_local = fusion_x_local.view(FB * FC, FT, FF)
                fusion_x_local = torch.permute(fusion_x_local, (0, 2, 1)
                    ).contiguous()
                fusion_x_local = self.mel_conv1d(fusion_x_local)
                fusion_x_local = fusion_x_local.view(FB, FC, FF,
                    fusion_x_local.size(-1))
                fusion_x_local = torch.permute(fusion_x_local, (0, 2, 1, 3)
                    ).contiguous().flatten(2)
                if fusion_x_local.size(-1) < FT:
                    fusion_x_local = torch.cat([fusion_x_local, torch.zeros
                        ((FB, FF, FT - fusion_x_local.size(-1)), device=
                        device)], dim=-1)
                else:
                    fusion_x_local = fusion_x_local[:, :, :FT]
                new_x = new_x.squeeze(1).permute((0, 2, 1)).contiguous()
                new_x[longer_list_idx] = self.fusion_model(new_x[
                    longer_list_idx], fusion_x_local)
                x = new_x.permute((0, 2, 1)).contiguous()[:, None, :, :]
            else:
                x = new_x
        elif self.fusion_type in ['daf_2d', 'aff_2d', 'iaff_2d', 'channel_map'
            ]:
            x = x
        if self.training:
            x = self.spec_augmenter(x)
        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)
        x = self.reshape_wav2img(x)
        output_dict = self.forward_features(x, longer_idx=longer_list_idx)
    return output_dict
