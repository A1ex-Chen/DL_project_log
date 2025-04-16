def do_generate(self, image_list: list, prompt: str, max_new_tokens, **kwargs):
    self.generator.ppl = False
    self.generator.max_len_b = max_new_tokens
    sample = self.make_batches(image_list, [prompt])[0]
    translations = self.task.inference_step(self.generator, [self.model],
        sample, constraints=None)
    results = []
    for i, (id, hypos) in enumerate(zip(sample['ids'].tolist(), translations)):
        src_tokens_i = utils.strip_pad(sample['net_input']['src_tokens'][i],
            self.task.target_dictionary.pad())
        results.append((id, src_tokens_i, hypos))
    outputs = []
    for id_, src_tokens, hypos in sorted(results, key=lambda x: x[0]):
        src_str = self.task.source_dictionary.string(src_tokens, self.cfg.
            common_eval.post_process)
        for hypo in hypos[:min(len(hypos), self.cfg.generation.nbest)]:
            hypo_tokens, hypo_str, alignment = post_process_prediction(
                hypo_tokens=hypo['tokens'].int().cpu(), src_str=src_str,
                alignment=hypo['alignment'], align_dict=None, tgt_dict=self
                .task.target_dictionary, remove_bpe=self.cfg.common_eval.
                post_process, extra_symbols_to_ignore=
                get_symbols_to_strip_from_output(self.generator))
            hypo_str = hypo_str.replace(src_str, '')
            outputs.append(hypo_str)
    return outputs[0]
