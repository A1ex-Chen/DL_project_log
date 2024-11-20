def forward_encoder(self, samples):
    questions = samples['text_input']
    questions = self.tokenizer(questions, padding='longest', truncation=
        True, max_length=self.max_txt_len, return_tensors='pt').to(self.device)
    questions.input_ids[:, 0] = self.tokenizer.enc_token_id
    samples.update({'tokenized_text': questions})
    image_embeds = self.visual_encoder.forward_features(samples['image'])
    encoder_output = self.text_encoder.forward_automask(tokenized_text=
        samples['tokenized_text'], visual_embeds=image_embeds)
    return encoder_output, image_embeds
