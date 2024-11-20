def test_special_attn_proc(self):


    class AttnEasyProc(torch.nn.Module):

        def __init__(self, num):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(num))
            self.is_run = False
            self.number = 0
            self.counter = 0

        def __call__(self, attn, hidden_states, encoder_hidden_states=None,
            attention_mask=None, number=None):
            batch_size, sequence_length, _ = hidden_states.shape
            attention_mask = attn.prepare_attention_mask(attention_mask,
                sequence_length, batch_size)
            query = attn.to_q(hidden_states)
            encoder_hidden_states = (encoder_hidden_states if 
                encoder_hidden_states is not None else hidden_states)
            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)
            query = attn.head_to_batch_dim(query)
            key = attn.head_to_batch_dim(key)
            value = attn.head_to_batch_dim(value)
            attention_probs = attn.get_attention_scores(query, key,
                attention_mask)
            hidden_states = torch.bmm(attention_probs, value)
            hidden_states = attn.batch_to_head_dim(hidden_states)
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)
            hidden_states += self.weight
            self.is_run = True
            self.counter += 1
            self.number = number
            return hidden_states
    init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
    init_dict['block_out_channels'] = 16, 32
    init_dict['attention_head_dim'] = 8, 16
    model = self.model_class(**init_dict)
    model.to(torch_device)
    processor = AttnEasyProc(5.0)
    model.set_attn_processor(processor)
    model(**inputs_dict, cross_attention_kwargs={'number': 123}).sample
    assert processor.counter == 8
    assert processor.is_run
    assert processor.number == 123
