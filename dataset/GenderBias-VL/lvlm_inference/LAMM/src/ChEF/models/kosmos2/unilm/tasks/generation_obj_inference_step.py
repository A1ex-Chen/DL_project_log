def inference_step(self, generator, models, sample, prefix_tokens=None,
    constraints=None):
    with torch.no_grad():
        if getattr(self.args, 'add_bos_token', False):
            bos_token = self.source_dictionary.bos()
        else:
            bos_token = self.source_dictionary.eos()
        if constraints is not None:
            raise NotImplementedError(
                'Constrained decoding with the language_modeling task is not supported'
                )
        if prefix_tokens is None and sample['net_input']['src_tokens'
            ].nelement():
            prefix_tokens = sample['net_input']['src_tokens']
        return generator.generate(models, sample, prefix_tokens=
            prefix_tokens, bos_token=bos_token)
