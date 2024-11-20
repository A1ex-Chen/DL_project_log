def call_og_forward():
    return self._original_forward(input_ids=input_ids, attention_mask=
        attention_mask, head_mask=head_mask, past_key_values=
        past_key_values, inputs_embeds=inputs_embeds, labels=labels,
        use_cache=use_cache, output_attentions=output_attentions,
        output_hidden_states=output_hidden_states, return_dict=return_dict)
