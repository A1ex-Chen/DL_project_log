def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments,
        TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith('.json'):
        model_args, data_args, training_args = parser.parse_json_file(json_file
            =os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = (parser.
            parse_args_into_dataclasses())
    logging.basicConfig(format=
        '%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt=
        '%m/%d/%Y %H:%M:%S', handlers=[logging.StreamHandler(sys.stdout)])
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    logger.warning(
        f'Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}'
         +
        f'distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}'
        )
    logger.info(f'Training/evaluation parameters {training_args}')
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir
        ) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)
            ) > 0:
            raise ValueError(
                f'Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome.'
                )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f'Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change the `--output_dir` or add `--overwrite_output_dir` to train from scratch.'
                )
    set_seed(training_args.seed)
    if data_args.dataset_name is not None:
        if data_args.dataset_config_name:
            raw_datasets = load_dataset(data_args.dataset_name, data_args.
                dataset_config_name, cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None)
        else:
            raw_datasets = load_dataset(data_args.dataset_name, cache_dir=
                model_args.cache_dir, use_auth_token=True if model_args.
                use_auth_token else None)
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files['train'] = data_args.train_file
            extension = data_args.train_file.split('.')[-1]
        if data_args.validation_file is not None:
            data_files['validation'] = data_args.validation_file
            extension = data_args.validation_file.split('.')[-1]
        if data_args.test_file is not None:
            data_files['test'] = data_args.test_file
            extension = data_args.test_file.split('.')[-1]
        raw_datasets = load_dataset(extension, data_files=data_files, field
            ='data', cache_dir=model_args.cache_dir, use_auth_token=True if
            model_args.use_auth_token else None)
    config = AutoConfig.from_pretrained(model_args.config_name if
        model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir, revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None)
    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name if
        model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir, use_fast=True, revision=model_args.
        model_revision, use_auth_token=True if model_args.use_auth_token else
        None)
    model = AutoModelForQuestionAnswering.from_pretrained(model_args.
        model_name_or_path, from_tf=bool('.ckpt' in model_args.
        model_name_or_path), config=config, cache_dir=model_args.cache_dir,
        revision=model_args.model_revision, use_auth_token=True if
        model_args.use_auth_token else None)
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise ValueError(
            'This example script only works for models that have a fast tokenizer. Checkout the big table of models at https://huggingface.co/transformers/index.html#supported-frameworks to find the model types that meet this requirement'
            )
    if training_args.do_train:
        column_names = raw_datasets['train'].column_names
    elif training_args.do_eval:
        column_names = raw_datasets['validation'].column_names
    else:
        column_names = raw_datasets['test'].column_names
    question_column_name = ('query' if 'query' in column_names else
        column_names[0])
    context_column_name = ('context' if 'context' in column_names else
        column_names[1])
    answer_column_name = ('answers' if 'answers' in column_names else
        column_names[2])
    pad_on_right = tokenizer.padding_side == 'right'
    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f'The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for themodel ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}.'
            )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def prepare_train_features(examples):
        examples[question_column_name] = [q.lstrip() for q in examples[
            question_column_name]]
        tokenized_examples = tokenizer(examples[question_column_name if
            pad_on_right else context_column_name], examples[
            context_column_name if pad_on_right else question_column_name],
            truncation='only_second' if pad_on_right else 'only_first',
            max_length=max_seq_length, stride=data_args.doc_stride,
            return_overflowing_tokens=True, return_offsets_mapping=True,
            padding='max_length' if data_args.pad_to_max_length else False)
        sample_mapping = tokenized_examples.pop('overflow_to_sample_mapping')
        offset_mapping = tokenized_examples.pop('offset_mapping')
        tokenized_examples['start_positions'] = []
        tokenized_examples['end_positions'] = []
        for i, offsets in enumerate(offset_mapping):
            input_ids = tokenized_examples['input_ids'][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)
            sequence_ids = tokenized_examples.sequence_ids(i)
            sample_index = sample_mapping[i]
            answers = examples[answer_column_name][sample_index]
            if len(answers['answer_start']) == 0:
                tokenized_examples['start_positions'].append(cls_index)
                tokenized_examples['end_positions'].append(cls_index)
            else:
                start_char = answers['answer_start'][0]
                end_char = start_char + len(answers['text'][0])
                token_start_index = 0
                while sequence_ids[token_start_index] != (1 if pad_on_right
                     else 0):
                    token_start_index += 1
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != (1 if pad_on_right else
                    0):
                    token_end_index -= 1
                if not (offsets[token_start_index][0] <= start_char and 
                    offsets[token_end_index][1] >= end_char):
                    tokenized_examples['start_positions'].append(cls_index)
                    tokenized_examples['end_positions'].append(cls_index)
                else:
                    while token_start_index < len(offsets) and offsets[
                        token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples['start_positions'].append(
                        token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples['end_positions'].append(
                        token_end_index + 1)
        return tokenized_examples
    if training_args.do_train:
        if 'train' not in raw_datasets:
            raise ValueError('--do_train requires a train dataset')
        train_dataset = raw_datasets['train']
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.
                max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        with training_args.main_process_first(desc=
            'train dataset map pre-processing'):
            train_dataset = train_dataset.map(prepare_train_features,
                batched=True, num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names, load_from_cache_file=not
                data_args.overwrite_cache, desc=
                'Running tokenizer on train dataset')
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.
                max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    def prepare_validation_features(examples):
        examples[question_column_name] = [q.lstrip() for q in examples[
            question_column_name]]
        tokenized_examples = tokenizer(examples[question_column_name if
            pad_on_right else context_column_name], examples[
            context_column_name if pad_on_right else question_column_name],
            truncation='only_second' if pad_on_right else 'only_first',
            max_length=max_seq_length, stride=data_args.doc_stride,
            return_overflowing_tokens=True, return_offsets_mapping=True,
            padding='max_length' if data_args.pad_to_max_length else False)
        sample_mapping = tokenized_examples.pop('overflow_to_sample_mapping')
        tokenized_examples['example_id'] = []
        for i in range(len(tokenized_examples['input_ids'])):
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if pad_on_right else 0
            sample_index = sample_mapping[i]
            tokenized_examples['example_id'].append(examples['id'][
                sample_index])
            tokenized_examples['offset_mapping'][i] = [(o if sequence_ids[k
                ] == context_index else None) for k, o in enumerate(
                tokenized_examples['offset_mapping'][i])]
        return tokenized_examples
    if training_args.do_eval:
        if 'validation' not in raw_datasets:
            raise ValueError('--do_eval requires a validation dataset')
        eval_examples = raw_datasets['validation']
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_examples), data_args.
                max_eval_samples)
            eval_examples = eval_examples.select(range(max_eval_samples))
        with training_args.main_process_first(desc=
            'validation dataset map pre-processing'):
            eval_dataset = eval_examples.map(prepare_validation_features,
                batched=True, num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names, load_from_cache_file=not
                data_args.overwrite_cache, desc=
                'Running tokenizer on validation dataset')
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.
                max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
    if training_args.do_predict:
        if 'test' not in raw_datasets:
            raise ValueError('--do_predict requires a test dataset')
        predict_examples = raw_datasets['test']
        if data_args.max_predict_samples is not None:
            predict_examples = predict_examples.select(range(data_args.
                max_predict_samples))
        with training_args.main_process_first(desc=
            'prediction dataset map pre-processing'):
            predict_dataset = predict_examples.map(prepare_validation_features,
                batched=True, num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names, load_from_cache_file=not
                data_args.overwrite_cache, desc=
                'Running tokenizer on prediction dataset')
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.
                max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples)
                )
    data_collator = (default_data_collator if data_args.pad_to_max_length else
        DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8 if
        training_args.fp16 else None))

    def post_processing_function(examples, features, predictions, stage='eval'
        ):
        predictions, predictions_with_indices = postprocess_qa_predictions(
            examples=examples, features=features, predictions=predictions,
            version_2_with_negative=data_args.version_2_with_negative,
            n_best_size=data_args.n_best_size, max_answer_length=data_args.
            max_answer_length, null_score_diff_threshold=data_args.
            null_score_diff_threshold, output_dir=training_args.output_dir,
            log_level=log_level, prefix=stage)
        if data_args.version_2_with_negative:
            formatted_predictions = [{'id': k, 'prediction_text': v,
                'no_answer_probability': 0.0} for k, v in predictions.items()]
        elif 'phrase_sense_disambiguation' not in data_args.dataset_name.lower(
            ):
            formatted_predictions = [{'id': k, 'prediction_text': v} for k,
                v in predictions.items()]
        else:
            formatted_predictions = [{'id': k, 'prediction_text': v,
                'answer_start': predictions_with_indices[k]['offsets'][0]} for
                k, v in predictions.items()]
        references = [{'id': ex['id'], 'answers': ex[answer_column_name]} for
            ex in examples]
        return EvalPrediction(predictions=formatted_predictions, label_ids=
            references)
    metric = load_metric('squad_v2' if data_args.version_2_with_negative else
        'squad', trust_remote_code=True)

    def compute_metrics(p: EvalPrediction):
        if 'phrase_sense_disambiguation' not in data_args.dataset_name.lower():
            return metric.compute(predictions=p.predictions, references=p.
                label_ids)
        formatted_predictions = [{'id': pred['id'], 'prediction_text': pred
            ['prediction_text']} for pred in p.predictions]
        em_with_index_count, f1_with_index_count = 0, 0
        em_count, f1_count = 0, 0
        for pred, label in zip(p.predictions, p.label_ids):
            assert pred['id'] == label['id']
            answer_text = label['answers']['text'][0]
            answer_start = label['answers']['answer_start'][0]
            if pred['prediction_text'].lower() == answer_text.lower():
                em_count += 1
                if pred['answer_start'] == answer_start:
                    em_with_index_count += 1
                else:
                    print({'id': pred['id'], 'pred_text': [pred[
                        'prediction_text'], pred['answer_start']],
                        'answer_text': [answer_text, answer_start]})
            f1_count += compute_f1(answer_text, pred['prediction_text'])
            pred_range = range(pred['answer_start'], pred['answer_start'] +
                len(pred['prediction_text']))
            answer_range = range(answer_start, answer_start + len(answer_text))
            if len(set(pred_range).intersection(set(answer_range))) > 0:
                f1_with_index_count += compute_f1(answer_text, pred[
                    'prediction_text'])
        em_score = round(em_count / len(p.label_ids), 4) * 100
        f1_score = round(f1_count / len(p.label_ids), 4) * 100
        em_with_index_score = round(em_with_index_count / len(p.label_ids), 4
            ) * 100
        f1_with_index_score = round(f1_with_index_count / len(p.label_ids), 4
            ) * 100
        print('Computed results: {}'.format({'exact_match': em_score,
            'exact_match_with_index': em_with_index_score, 'f1': f1_score,
            'f1_with_index': f1_with_index_score}))
        print('Reference results: {}'.format(metric.compute(predictions=
            formatted_predictions, references=p.label_ids)))
        return {'exact_match': em_with_index_score, 'f1': f1_with_index_score}

    def normalize_answer(s):
        """Lower text and remove punctuation, articles and extra whitespace."""

        def remove_articles(text):
            regex = re.compile('\\b(a|an|the)\\b', re.UNICODE)
            return re.sub(regex, ' ', text)

        def white_space_fix(text):
            return ' '.join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()
        return white_space_fix(remove_articles(remove_punc(lower(s))))

    def get_tokens(s):
        if not s:
            return []
        return normalize_answer(s).split()

    def compute_exact(a_gold, a_pred):
        return int(normalize_answer(a_gold) == normalize_answer(a_pred))

    def compute_f1(a_gold, a_pred):
        gold_toks = get_tokens(a_gold)
        pred_toks = get_tokens(a_pred)
        common = collections.Counter(gold_toks) & collections.Counter(pred_toks
            )
        num_same = sum(common.values())
        if len(gold_toks) == 0 or len(pred_toks) == 0:
            return int(gold_toks == pred_toks)
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(pred_toks)
        recall = 1.0 * num_same / len(gold_toks)
        f1 = 2 * precision * recall / (precision + recall)
        return f1
    trainer = QuestionAnsweringTrainer(model=model, args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        eval_examples=eval_examples if training_args.do_eval else None,
        tokenizer=tokenizer, data_collator=data_collator,
        post_process_function=post_processing_function, compute_metrics=
        compute_metrics)
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        metrics = train_result.metrics
        max_train_samples = (data_args.max_train_samples if data_args.
            max_train_samples is not None else len(train_dataset))
        metrics['train_samples'] = min(max_train_samples, len(train_dataset))
        trainer.log_metrics('train', metrics)
        trainer.save_metrics('train', metrics)
        trainer.save_state()
    if training_args.do_eval:
        logger.info('*** Evaluate ***')
        metrics = trainer.evaluate()
        max_eval_samples = (data_args.max_eval_samples if data_args.
            max_eval_samples is not None else len(eval_dataset))
        metrics['eval_samples'] = min(max_eval_samples, len(eval_dataset))
        trainer.log_metrics('eval', metrics)
        trainer.save_metrics('eval', metrics)
    if training_args.do_predict:
        logger.info('*** Predict ***')
        results = trainer.predict(predict_dataset, predict_examples)
        metrics = results.metrics
        max_predict_samples = (data_args.max_predict_samples if data_args.
            max_predict_samples is not None else len(predict_dataset))
        metrics['predict_samples'] = min(max_predict_samples, len(
            predict_dataset))
        trainer.log_metrics('predict', metrics)
        trainer.save_metrics('predict', metrics)
    kwargs = {'finetuned_from': model_args.model_name_or_path, 'tasks':
        'question-answering'}
    if data_args.dataset_name is not None:
        kwargs['dataset_tags'] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs['dataset_args'] = data_args.dataset_config_name
            kwargs['dataset'
                ] = f'{data_args.dataset_name} {data_args.dataset_config_name}'
        else:
            kwargs['dataset'] = data_args.dataset_name
    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)
