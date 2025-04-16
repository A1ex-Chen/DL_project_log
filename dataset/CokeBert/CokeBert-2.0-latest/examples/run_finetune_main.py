def main():
    args = coke_training_args()
    num_labels = 80
    processors = FewrelProcessor
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device('cuda' if torch.cuda.is_available() and not
            args.no_cuda else 'cpu')
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        n_gpu = 1
        torch.distributed.init_process_group(backend='nccl')
    logger.info(
        'device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}'
        .format(device, n_gpu, bool(args.local_rank != -1), args.fp16))
    if args.gradient_accumulation_steps < 1:
        raise ValueError(
            'Invalid gradient_accumulation_steps parameter: {}, should be >= 1'
            .format(args.gradient_accumulation_steps))
    args.train_batch_size = int(args.train_batch_size / args.
        gradient_accumulation_steps)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
    if not args.do_train:
        raise ValueError(
            'At least one of `do_train` or `do_eval` must be True.')
    args.output_dir = os.path.join(args.output_dir, 'finetune_coke-' + args
        .backbone + f'-{args.neighbor_hop}')
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir
        ) and args.do_train:
        raise ValueError(
            'Output directory ({}) already exists and is not empty.'.format
            (args.output_dir))
    os.makedirs(args.output_dir, exist_ok=True)
    processor = processors()
    label_list = None
    model_name = 'coke-' + args.backbone
    tokenizer = CokeBertTokenizer.from_pretrained(os.path.join(
        '../checkpoint', model_name), do_lower_case=args.do_lower_case)
    train_examples = None
    num_train_steps = None
    train_examples, label_list = processor.get_train_examples(args.data_dir)
    num_train_steps = int(len(train_examples) / args.train_batch_size /
        args.gradient_accumulation_steps * args.num_train_epochs)
    model = CokeBertForRelationClassification.from_pretrained(os.path.join(
        '../checkpoint', model_name), neighbor_hop=args.neighbor_hop,
        num_labels=num_labels)
    model.to(device)
    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                'Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.'
                )
        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)
    param_optimizer = list(model.named_parameters())
    no_grad = ['bert.encoder.layer.11.output.dense_ent',
        'bert.encoder.layer.11.output.LayerNorm_ent']
    param_optimizer = [(n, p) for n, p in param_optimizer if not any(nd in
        n for nd in no_grad)]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [{'params': [p for n, p in
        param_optimizer if not any(nd in n for nd in no_decay)],
        'weight_decay': 0.01}, {'params': [p for n, p in param_optimizer if
        any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    t_total = num_train_steps
    if args.local_rank != -1:
        t_total = t_total // torch.distributed.get_world_size()
    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError(
                'Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.'
                )
        optimizer = FusedAdam(optimizer_grouped_parameters, lr=args.
            learning_rate, bias_correction=False, max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.
                loss_scale)
    else:
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    logger.info('Loading KG ...')
    ent_neighbor, ent_r, ent_outORin = load_ent_emb_static('../data/pretrain')
    embed_ent, embed_r = load_kg_embedding('../data/pretrain')
    logger.info('Finish loading')
    global_step = 0
    train_features = convert_examples_to_features(train_examples,
        label_list, args.max_seq_length, tokenizer, args.threshold)
    eval_examples = processor.get_dev_examples(args.data_dir)
    eval_features = convert_examples_to_features(eval_examples, label_list,
        args.max_seq_length, tokenizer, args.threshold)
    logger.info('***** Running training *****')
    logger.info('  Num examples = %d', len(train_examples))
    logger.info('  Batch size = %d', args.train_batch_size)
    logger.info('  Num steps = %d', num_train_steps)
    all_input_ids = torch.tensor([f.input_ids for f in train_features],
        dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features],
        dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features],
        dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in train_features],
        dtype=torch.long)
    all_ent = torch.tensor([f.input_ent for f in train_features], dtype=
        torch.long)
    all_ent_masks = torch.tensor([f.ent_mask for f in train_features],
        dtype=torch.long)
    train_data = TensorDataset(all_input_ids, all_input_mask,
        all_segment_ids, all_ent, all_ent_masks, all_label_ids)
    if args.local_rank == -1:
        train_sampler = RandomSampler(train_data)
    else:
        train_sampler = DistributedSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler,
        batch_size=args.train_batch_size)
    output_loss_file = os.path.join(args.output_dir, 'loss')
    loss_fout = open(output_loss_file, 'w')
    model.train()
    all_input_ids = torch.tensor([f.input_ids for f in eval_features],
        dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features],
        dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features],
        dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype
        =torch.long)
    all_ent = torch.tensor([f.input_ent for f in eval_features], dtype=
        torch.long)
    all_ent_masks = torch.tensor([f.ent_mask for f in eval_features], dtype
        =torch.long)
    eval_data = TensorDataset(all_input_ids, all_input_mask,
        all_segment_ids, all_ent, all_ent_masks, all_label_ids)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler,
        batch_size=args.eval_batch_size)
    for epoch in trange(int(args.num_train_epochs), desc='Epoch'):
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(tqdm(train_dataloader, desc='Iteration')):
            batch = tuple(t.to(device) if i != 3 else t for i, t in
                enumerate(batch))
            (input_ids, input_mask, token_type_ids, input_ent, ent_mask,
                label_ids) = batch
            input_ent = input_ent + 1
            k_1, v_1, k_2, v_2 = load_k_v_queryR_small(input_ent,
                ent_neighbor, ent_r, ent_outORin, embed_ent, embed_r)
            loss = 0
            loss = model(input_ids, input_mask, token_type_ids, input_ent=
                input_ent.float(), ent_mask=ent_mask, labels=label_ids, k_1
                =k_1, v_1=v_1, k_2=k_2, v_2=v_2).loss
            if n_gpu > 1:
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            if args.fp16:
                optimizer.backward(loss)
            else:
                loss.backward()
            loss_fout.write('{}\n'.format(loss.item()))
            tr_loss += loss.item()
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1
            if (step + 1) % args.gradient_accumulation_steps == 0:
                lr_this_step = args.learning_rate * warmup_linear(
                    global_step / t_total, args.warmup_proportion)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_this_step
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
        eval_loss = 0
        eval_accuracy = 0
        nb_eval_steps = 0
        best_valid_acc = 0
        nb_eval_examples = 0
        with torch.no_grad():
            for input_ids, input_mask, segment_ids, input_ent, ent_mask, label_ids in eval_dataloader:
                input_ent = input_ent + 1
                input_ids = input_ids.to(device)
                input_mask = input_mask.to(device)
                segment_ids = segment_ids.to(device)
                input_ent = input_ent.to(device)
                ent_mask = ent_mask.to(device)
                label_ids = label_ids.to(device)
                k_1, v_1, k_2, v_2 = load_k_v_queryR_small(input_ent,
                    ent_neighbor, ent_r, ent_outORin, embed_ent, embed_r)
                outputs = model(input_ids, input_mask, segment_ids,
                    input_ent=input_ent, ent_mask=ent_mask, labels=
                    label_ids, k_1=k_1, v_1=v_1, k_2=k_2, v_2=v_2)
                tmp_eval_loss = outputs.loss
                logits = outputs.logits
                logits = logits.detach().cpu().numpy()
                label_ids = label_ids.to('cpu').numpy()
                tmp_eval_accuracy, pred = accuracy(logits, label_ids)
                eval_loss += tmp_eval_loss.mean().item()
                eval_accuracy += tmp_eval_accuracy
                nb_eval_examples += input_ids.size(0)
                nb_eval_steps += 1
            eval_loss = eval_loss / nb_eval_steps
            eval_accuracy = eval_accuracy / nb_eval_examples
            if eval_accuracy > best_valid_acc:
                best_valid_acc = eval_accuracy
                model_to_save = model.module if hasattr(model, 'module'
                    ) else model
                output_model_file = os.path.join(args.output_dir,
                    f'pytorch_model_{epoch}.bin')
                torch.save(model_to_save.state_dict(), output_model_file)
            result = {'epoch': epoch, 'eval_loss': eval_loss,
                'eval_accuracy': eval_accuracy}
            print(result)
