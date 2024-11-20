def forward(self, samples, is_train=True):
    sentences = samples['text_input']
    sentences = self.tokenizer(sentences, padding='longest', truncation=
        True, max_length=self.max_txt_len, return_tensors='pt').to(self.device)
    samples.update({'tokenized_text': sentences})
    targets = samples['label']
    image_embeds = self.visual_encoder.forward_features(samples['image'])
    encoder_output = self.text_encoder.forward_automask(samples[
        'tokenized_text'], image_embeds)
    prediction = self.cls_head(encoder_output.last_hidden_state[:, 0, :])
    if is_train:
        if self.use_distill:
            with torch.no_grad():
                self._momentum_update()
                image_embeds_m = self.visual_encoder_m(samples['image'])
                encoder_output_m = self.text_encoder_m.forward_automask(samples
                    ['tokenized_text'], image_embeds_m)
                prediction_m = self.cls_head_m(encoder_output_m.
                    last_hidden_state[:, 0, :])
            alpha = self.alpha * self._rampup_factor(epoch=samples['epoch'],
                iters=samples['iters'], num_iters_per_epoch=samples[
                'num_iters_per_epoch'])
            loss = (1 - alpha) * F.cross_entropy(prediction, targets
                ) - alpha * torch.sum(F.log_softmax(prediction, dim=1) * F.
                softmax(prediction_m, dim=1), dim=1).mean()
        else:
            loss = F.cross_entropy(prediction, targets)
        return BlipOutputWithLogits(loss=loss, intermediate_output=
            BlipIntermediateOutput(image_embeds=image_embeds,
            image_embeds_m=image_embeds_m, encoder_output=encoder_output,
            encoder_output_m=encoder_output_m), logits=prediction, logits_m
            =prediction_m)
    else:
        return {'predictions': prediction, 'targets': targets}
