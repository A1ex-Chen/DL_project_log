def __getitem__(self, index):
    ins_id, enc_input, enc_cls, mf, out, gt, gt_mr = self.record[index]
    data_item = {'ins_id': ins_id, 'encoder_input_ids': enc_input,
        'encoder_class': enc_cls, 'gt': gt, 'gt_mr': gt_mr, 'mention_flag': mf}
    if out is not None:
        data_item['cap'] = out
    return data_item
