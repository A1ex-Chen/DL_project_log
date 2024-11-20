def convert_models(self, tatoeba_ids, dry_run=False):
    entries_to_convert = [x for x in self.registry if x[0] in tatoeba_ids]
    converted_paths = convert_all_sentencepiece_models(entries_to_convert,
        dest_dir=self.model_card_dir)
    for path in converted_paths:
        long_pair = remove_prefix(path.name, 'opus-mt-').split('-')
        assert len(long_pair) == 2
        new_p_src = self.get_two_letter_code(long_pair[0])
        new_p_tgt = self.get_two_letter_code(long_pair[1])
        hf_model_id = f'opus-mt-{new_p_src}-{new_p_tgt}'
        new_path = path.parent.joinpath(hf_model_id)
        os.rename(str(path), str(new_path))
        self.write_model_card(hf_model_id, dry_run=dry_run)
