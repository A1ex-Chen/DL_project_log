def prepare_init_args_and_inputs_for_common(self):
    init_dict = {'num_attention_heads': 2, 'attention_head_dim': 4,
        'num_layers': 2, 'embedding_dim': 8, 'num_embeddings': 7,
        'additional_embeddings': 4}
    inputs_dict = self.dummy_input
    return init_dict, inputs_dict
