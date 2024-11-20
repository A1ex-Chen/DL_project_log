def resolve_lang_code(self, r) ->Tuple[List[str], str, str]:
    """R is a row in ported"""
    short_pair = r.short_pair
    src, tgt = short_pair.split('-')
    src_tags, src_multilingual = self.get_tags(src, r.src_name)
    assert isinstance(src_tags, list)
    tgt_tags, tgt_multilingual = self.get_tags(tgt, r.tgt_name)
    assert isinstance(tgt_tags, list)
    return dedup(src_tags + tgt_tags), src_multilingual, tgt_multilingual
