def generate_summaries_or_translations(examples: List[str], out_file: str,
    model_name: str, batch_size: int=8, device: str=DEFAULT_DEVICE, fp16=
    False, task='summarization', prefix=None, **generate_kwargs) ->Dict:
    """Save model.generate results to <out_file>, and return how long it took."""
    fout = Path(out_file).open('w', encoding='utf-8')
    model_name = str(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    if fp16:
        model = model.half()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    logger.info(f'Inferred tokenizer type: {tokenizer.__class__}')
    start_time = time.time()
    use_task_specific_params(model, task)
    if prefix is None:
        prefix = prefix or getattr(model.config, 'prefix', '') or ''
    for examples_chunk in tqdm(list(chunks(examples, batch_size))):
        examples_chunk = [(prefix + text) for text in examples_chunk]
        batch = tokenizer(examples_chunk, return_tensors='pt', truncation=
            True, padding='longest').to(device)
        summaries = model.generate(input_ids=batch.input_ids,
            attention_mask=batch.attention_mask, **generate_kwargs)
        dec = tokenizer.batch_decode(summaries, skip_special_tokens=True,
            clean_up_tokenization_spaces=False)
        for hypothesis in dec:
            fout.write(hypothesis + '\n')
            fout.flush()
    fout.close()
    runtime = int(time.time() - start_time)
    n_obs = len(examples)
    return dict(n_obs=n_obs, runtime=runtime, seconds_per_sample=round(
        runtime / n_obs, 4))
