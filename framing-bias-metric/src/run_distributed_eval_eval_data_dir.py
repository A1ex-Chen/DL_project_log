def eval_data_dir(data_dir, save_dir: str, model_name: str, bs: int=8,
    max_source_length: int=1024, type_path='val', n_obs=None, fp16=False,
    task='summarization', local_rank=None, num_return_sequences=1,
    dataset_kwargs: Dict=None, prefix='', **generate_kwargs) ->Dict:
    """Run evaluation on part of the data for one gpu and save to {save_dir}/rank_{rank}_output.json"""
    model_name = str(model_name)
    assert local_rank is not None
    torch.distributed.init_process_group(backend='nccl', rank=local_rank)
    save_dir = Path(save_dir)
    save_path = save_dir.joinpath(f'rank_{local_rank}_output.json')
    torch.cuda.set_device(local_rank)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).cuda()
    if fp16:
        model = model.half()
    use_task_specific_params(model, task)
    num_beams = generate_kwargs.pop('num_beams', model.config.num_beams)
    if num_return_sequences > num_beams:
        num_beams = num_return_sequences
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    logger.info(f'Inferred tokenizer type: {tokenizer.__class__}')
    if max_source_length is None:
        max_source_length = tokenizer.model_max_length
    if prefix is None:
        prefix = prefix or getattr(model.config, 'prefix', '') or ''
    ds = Seq2SeqDataset(tokenizer, data_dir, max_source_length,
        max_target_length=1024, type_path=type_path, n_obs=n_obs, prefix=
        prefix, **dataset_kwargs)
    sampler = ds.make_sortish_sampler(bs, distributed=True,
        add_extra_examples=False, shuffle=True)
    data_loader = DataLoader(ds, sampler=sampler, batch_size=bs, collate_fn
        =ds.collate_fn)
    results = []
    for batch in tqdm(data_loader):
        summaries = model.generate(input_ids=batch['input_ids'].to(model.
            device), attention_mask=batch['attention_mask'].to(model.device
            ), num_return_sequences=num_return_sequences, num_beams=
            num_beams, **generate_kwargs)
        preds = tokenizer.batch_decode(summaries, skip_special_tokens=True,
            clean_up_tokenization_spaces=False)
        ids = batch['ids']
        if num_return_sequences > 1:
            preds = chunks(preds, num_return_sequences)
        for i, pred in enumerate(preds):
            results.append(dict(pred=pred, id=ids[i].item()))
    save_json(results, save_path)
    return results, sampler.num_replicas
