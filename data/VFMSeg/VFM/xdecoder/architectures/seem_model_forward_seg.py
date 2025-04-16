def forward_seg(self, batched_inputs):
    images = [x['image'].to(self.device) for x in batched_inputs]
    images = [((x - self.pixel_mean) / self.pixel_std) for x in images]
    images = ImageList.from_tensors(images, self.size_divisibility)
    self.sem_seg_head.predictor.lang_encoder.get_text_embeddings(self.
        train_class_names, is_eval=False)
    extra = {}
    if 'instances' in batched_inputs[0]:
        targets = self.prepare_targets(batched_inputs, images)
        if self.task_switch['grounding']:
            grounding_tokens = [x['grounding_query_embs'] for x in targets]
            grounding_tokens = nn.utils.rnn.pad_sequence(grounding_tokens,
                padding_value=-1)
            non_zero_query_mask = grounding_tokens.sum(dim=-1
                ) == -grounding_tokens.shape[-1]
            grounding_tokens[non_zero_query_mask] = 0
            extra['grounding_tokens'] = grounding_tokens
            extra['grounding_nonzero_mask'] = non_zero_query_mask.t()
        if self.task_switch['spatial']:
            pos_masks = [x['spatial_query']['rand_shape'].to(self.device) for
                x in batched_inputs]
            neg_masks = [(x['spatial_query']['rand_shape'].to(self.device) &
                False) for x in batched_inputs]
            fp_masks = torch.stack([(x['spatial_query']['rand_shape'].to(
                self.device) & False) for x in batched_inputs])
            extra.update({'spatial_query_pos_mask': pos_masks,
                'spatial_query_neg_mask': neg_masks, 'false_positive_mask':
                fp_masks})
    features = self.backbone(images.tensor)
    mask_features, _, multi_scale_features = (self.sem_seg_head.
        pixel_decoder.forward_features(features))
    if self.task_switch['spatial']:
        with torch.no_grad():
            rand_iter_num = random.randint(0, 2)
            for i in range(rand_iter_num):
                outputs = self.sem_seg_head.predictor(multi_scale_features,
                    mask_features, extra=extra, task='spatial')
                extra.update(outputs)
                extra.update(self.prepare_next_spaital_mask(extra,
                    batched_inputs))
    outputs = self.sem_seg_head.predictor(multi_scale_features,
        mask_features, extra=extra, task='seg')
    extra = {'lang_logit': self.sem_seg_head.predictor.lang_encoder.
        logit_scale, 'class_embeddings': getattr(self.sem_seg_head.
        predictor.lang_encoder, '{}_text_embeddings'.format('default')),
        'false_positive_mask': extra['false_positive_mask']}
    self.criterion.losses = self.losses['seg']
    losses = self.criterion(outputs, targets, extra)
    del outputs
    return losses
