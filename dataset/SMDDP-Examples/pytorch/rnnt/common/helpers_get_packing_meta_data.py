def get_packing_meta_data(self, max_f_len, feat_lens, txt_lens, async_cp=
    False, idx=0):
    dict_meta_data = {'batch_offset': None, 'g_len': None, 'max_f_len':
        None, 'packed_batch': None}
    if self.apex_transducer_joint is not None:
        g_len = txt_lens + 1
        if self.apex_transducer_joint == 'pack':
            batch_offset = torch.cumsum(g_len * ((feat_lens + self.
                enc_stack_time_factor - 1) // self.enc_stack_time_factor),
                dim=0)
            if async_cp:
                self.list_packed_batch_cpu[idx].copy_(batch_offset[-1].
                    detach(), non_blocking=True)
            dict_meta_data = {'batch_offset': batch_offset, 'g_len': g_len,
                'max_f_len': (max_f_len + self.enc_stack_time_factor - 1) //
                self.enc_stack_time_factor, 'packed_batch': batch_offset[-1
                ].item() if not async_cp else None}
        elif self.apex_transducer_joint == 'not_pack':
            dict_meta_data['g_len'] = g_len
            dict_meta_data['packed_batch'] = 0
    return dict_meta_data
