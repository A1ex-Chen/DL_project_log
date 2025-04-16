def decode_text_latents(self, text_latents, device):
    output_token_list, seq_lengths = self.text_decoder.generate_captions(
        text_latents, self.text_tokenizer.eos_token_id, device=device)
    output_list = output_token_list.cpu().numpy()
    generated_text = [self.text_tokenizer.decode(output[:int(length)],
        skip_special_tokens=True) for output, length in zip(output_list,
        seq_lengths)]
    return generated_text
