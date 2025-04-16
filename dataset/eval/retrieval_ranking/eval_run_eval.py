def run_eval(args, system, examples):
    """ """
    random.seed(42)
    if args.oracle_candidates:
        print('**********************************')
        print('oracle_candidates are included')
        print('**********************************')
    if args.debug:
        examples = examples.select(range(0, 10))
    run_results = []
    for example in tqdm(examples):
        system.set_text(example['context'], contextual=args.contextual,
            scorer=args.scorer, max_seq_length=args.max_seq_length)
        answers = example['answers']['text']
        if args.oracle_candidates:
            logger.debug('ADD ORACLES: %s', answers)
            if args.contextual:
                gt_sentence, gt_sentence_idx = extract_context_for_oracle(
                    system.sentences, answers[0])
                system.add_oracles(set(answers), gt_sentence, gt_sentence_idx)
            else:
                system.add_oracles(set(answers))
        query = example['query'].strip().lower()
        search_result = system.search(query, top_n=10, window_size=int(args
            .context_window))
        run_results.append({'file': example['title'], 'query': query,
            'ground_truths': example['answers']['text'],
            'number_of_candidates': len(system.phrases + system.list_oracle
            ), 'included_in_candidates': any(x.strip().lower() in [(phrase[
            0] if system.contextual else phrase) for phrase in system.
            phrases + system.list_oracle] for x in answers), 'result':
            search_result})
    eval_results = get_metrics(run_results)
    return eval_results, run_results
