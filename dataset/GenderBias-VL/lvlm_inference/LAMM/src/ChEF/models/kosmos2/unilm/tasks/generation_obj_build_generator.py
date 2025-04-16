def build_generator(self, models, args, seq_gen_cls=None,
    extra_gen_cls_kwargs=None, prefix_allowed_tokens_fn=None):
    from ..models.vl.vlm_generator import SequenceGenerator
    sampling = getattr(args, 'sampling', False)
    sampling_topk = getattr(args, 'sampling_topk', -1)
    sampling_topp = getattr(args, 'sampling_topp', -1.0)
    diverse_beam_groups = getattr(args, 'diverse_beam_groups', -1)
    diverse_beam_strength = getattr(args, 'diverse_beam_strength', 0.5)
    match_source_len = getattr(args, 'match_source_len', False)
    diversity_rate = getattr(args, 'diversity_rate', -1)
    constrained = getattr(args, 'constraints', False)
    if prefix_allowed_tokens_fn is None:
        prefix_allowed_tokens_fn = getattr(args, 'prefix_allowed_tokens_fn',
            None)
    if sum(int(cond) for cond in [sampling, diverse_beam_groups > 0,
        match_source_len, diversity_rate > 0]) > 1:
        raise ValueError('Provided Search parameters are mutually exclusive.')
    assert sampling_topk < 0 or sampling, '--sampling-topk requires --sampling'
    assert sampling_topp < 0 or sampling, '--sampling-topp requires --sampling'
    if sampling:
        search_strategy = search.Sampling(self.target_dictionary,
            sampling_topk, sampling_topp)
    elif diverse_beam_groups > 0:
        search_strategy = search.DiverseBeamSearch(self.target_dictionary,
            diverse_beam_groups, diverse_beam_strength)
    elif match_source_len:
        search_strategy = search.LengthConstrainedBeamSearch(self.
            target_dictionary, min_len_a=1, min_len_b=0, max_len_a=1,
            max_len_b=0)
    elif diversity_rate > -1:
        search_strategy = search.DiverseSiblingsSearch(self.
            target_dictionary, diversity_rate)
    elif constrained:
        search_strategy = search.LexicallyConstrainedBeamSearch(self.
            target_dictionary, args.constraints)
    elif prefix_allowed_tokens_fn:
        search_strategy = search.PrefixConstrainedBeamSearch(self.
            target_dictionary, prefix_allowed_tokens_fn)
    else:
        search_strategy = search.BeamSearch(self.target_dictionary)
    extra_gen_cls_kwargs = extra_gen_cls_kwargs or {}
    if seq_gen_cls is None:
        seq_gen_cls = SequenceGenerator
    return seq_gen_cls(models, self.target_dictionary, beam_size=getattr(
        args, 'beam', 5), max_len_a=getattr(args, 'max_len_a', 0),
        max_len_b=getattr(args, 'max_len_b', 200), min_len=getattr(args,
        'min_len', 1), normalize_scores=not getattr(args, 'unnormalized', 
        False), len_penalty=getattr(args, 'lenpen', 1), unk_penalty=getattr
        (args, 'unkpen', 0), temperature=getattr(args, 'temperature', 1.0),
        match_source_len=getattr(args, 'match_source_len', False),
        no_repeat_ngram_size=getattr(args, 'no_repeat_ngram_size', 0),
        search_strategy=search_strategy, **extra_gen_cls_kwargs)
