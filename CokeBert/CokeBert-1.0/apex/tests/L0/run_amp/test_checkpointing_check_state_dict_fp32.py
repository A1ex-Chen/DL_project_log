def check_state_dict_fp32(self, state_dict):
    for key in state_dict:
        if 'num_batches_tracked' in key:
            continue
        param = state_dict[key]
        self.assertEqual(param.type(), FLOAT,
            'Parameter in state_dict not FLOAT')
