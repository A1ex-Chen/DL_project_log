def compute_null_conditioning():
    null_conditioning_list = []
    for a_tokenizer, a_text_encoder in zip(tokenizers, text_encoders):
        null_conditioning_list.append(a_text_encoder(tokenize_captions([''],
            tokenizer=a_tokenizer).to(accelerator.device),
            output_hidden_states=True).hidden_states[-2])
    return torch.concat(null_conditioning_list, dim=-1)
