def normalizer(self, proto):
    list_normalizers = [normalizers.Replace('``', '"'), normalizers.Replace
        ("''", '"')]
    if not self.original_tokenizer.keep_accents:
        list_normalizers.append(normalizers.NFKD())
        list_normalizers.append(normalizers.StripAccents())
    if self.original_tokenizer.do_lower_case:
        list_normalizers.append(normalizers.Lowercase())
    precompiled_charsmap = proto.normalizer_spec.precompiled_charsmap
    list_normalizers.append(normalizers.Precompiled(precompiled_charsmap))
    return normalizers.Sequence(list_normalizers)
