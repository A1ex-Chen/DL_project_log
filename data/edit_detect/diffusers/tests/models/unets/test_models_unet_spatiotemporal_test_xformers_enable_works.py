@unittest.skipIf(torch_device != 'cuda' or not is_xformers_available(),
    reason=
    'XFormers attention is only available with CUDA and `xformers` installed')
def test_xformers_enable_works(self):
    init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
    model = self.model_class(**init_dict)
    model.enable_xformers_memory_efficient_attention()
    assert model.mid_block.attentions[0].transformer_blocks[0
        ].attn1.processor.__class__.__name__ == 'XFormersAttnProcessor', 'xformers is not enabled'
