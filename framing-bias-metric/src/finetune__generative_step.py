def _generative_step(self, batch: dict) ->dict:
    t0 = time.time()
    generated_ids = self.model.generate(batch['input_ids'], attention_mask=
        batch['attention_mask'], use_cache=True, decoder_start_token_id=
        self.decoder_start_token_id, num_beams=self.eval_beams, max_length=
        self.eval_max_length, do_sample=self.do_sample, top_p=self.top_p,
        top_k=self.top_k, length_penalty=self.length_penalty, temperature=
        self.temperature, num_return_sequences=self.num_return_sequences)
    gen_time = (time.time() - t0) / batch['input_ids'].shape[0]
    preds: List[str] = self.ids_to_clean_text(generated_ids)
    target: List[str] = self.ids_to_clean_text(batch['labels'])
    loss_tensors = self._step(batch)
    base_metrics = {name: loss for name, loss in zip(self.loss_names,
        loss_tensors)}
    rouge: Dict = self.calc_generative_metrics(preds, target)
    summ_len = np.mean(lmap(len, generated_ids))
    base_metrics.update(gen_time=gen_time, gen_len=summ_len, preds=preds,
        target=target, **rouge)
    return base_metrics
