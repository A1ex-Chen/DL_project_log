def embed_sentences_checkpointed(self, input_ids, attention_mask,
    sent_encoder, checkpoint_batch_size=-1):
    if checkpoint_batch_size < 0 or input_ids.shape[0] < checkpoint_batch_size:
        return sent_encoder(input_ids, attention_mask=attention_mask)[1]
    else:
        device = input_ids.device
        input_shape = input_ids.size()
        token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=
            device)
        head_mask = [None] * sent_encoder.config.num_hidden_layers
        extended_attention_mask: torch.Tensor = (sent_encoder.
            get_extended_attention_mask(attention_mask, input_shape, device))

        def partial_encode(*inputs):
            encoder_outputs = sent_encoder.encoder(inputs[0],
                attention_mask=inputs[1], head_mask=head_mask)
            sequence_output = encoder_outputs[0]
            pooled_output = sent_encoder.pooler(sequence_output)
            return pooled_output
        embedding_output = sent_encoder.embeddings(input_ids=input_ids,
            position_ids=None, token_type_ids=token_type_ids, inputs_embeds
            =None)
        pooled_output_list = []
        for b in range(math.ceil(input_ids.shape[0] / checkpoint_batch_size)):
            b_embedding_output = embedding_output[b * checkpoint_batch_size
                :(b + 1) * checkpoint_batch_size]
            b_attention_mask = extended_attention_mask[b *
                checkpoint_batch_size:(b + 1) * checkpoint_batch_size]
            pooled_output = checkpoint.checkpoint(partial_encode,
                b_embedding_output, b_attention_mask)
            pooled_output_list.append(pooled_output)
        return torch.cat(pooled_output_list, dim=0)
