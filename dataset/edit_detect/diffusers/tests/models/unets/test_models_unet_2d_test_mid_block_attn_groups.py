def test_mid_block_attn_groups(self):
    init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
    init_dict['add_attention'] = True
    init_dict['attn_norm_num_groups'] = 4
    model = self.model_class(**init_dict)
    model.to(torch_device)
    model.eval()
    self.assertIsNotNone(model.mid_block.attentions[0].group_norm,
        'Mid block Attention group norm should exist but does not.')
    self.assertEqual(model.mid_block.attentions[0].group_norm.num_groups,
        init_dict['attn_norm_num_groups'],
        'Mid block Attention group norm does not have the expected number of groups.'
        )
    with torch.no_grad():
        output = model(**inputs_dict)
        if isinstance(output, dict):
            output = output.to_tuple()[0]
    self.assertIsNotNone(output)
    expected_shape = inputs_dict['sample'].shape
    self.assertEqual(output.shape, expected_shape,
        'Input and output shapes do not match')
