def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default=None, type=str, required=True,
        help=
        'The input data dir. Should contain the .tsv files (or other data files) for the task.'
        )
    parser.add_argument('--ernie_model', default=None, type=str, required=
        True, help='Ernie pre-trained model')
    parser.add_argument('--output_dir', default=None, type=str, required=
        True, help=
        'The output directory where the model predictions and checkpoints will be written.'
        )
    parser.add_argument('--max_seq_length', default=128, type=int, help=
        """The maximum total input sequence length after WordPiece tokenization. 
Sequences longer than this will be truncated, and sequences shorter 
than this will be padded."""
        )
    parser.add_argument('--do_train', default=False, action='store_true',
        help='Whether to run training.')
    parser.add_argument('--do_eval', default=False, action='store_true',
        help='Whether to run eval on the dev set.')
    parser.add_argument('--do_lower_case', default=False, action=
        'store_true', help='Set this flag if you are using an uncased model.')
    parser.add_argument('--train_batch_size', default=32, type=int, help=
        'Total batch size for training.')
    parser.add_argument('--learning_rate', default=5e-05, type=float, help=
        'The initial learning rate for Adam.')
    parser.add_argument('--num_train_epochs', default=3.0, type=float, help
        ='Total number of training epochs to perform.')
    parser.add_argument('--warmup_proportion', default=0.1, type=float,
        help=
        'Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.'
        )
    parser.add_argument('--no_cuda', default=False, action='store_true',
        help='Whether not to use CUDA when available')
    parser.add_argument('--local_rank', type=int, default=-1, help=
        'local_rank for distributed training on gpus')
    parser.add_argument('--seed', type=int, default=42, help=
        'random seed for initialization')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=
        1, help=
        'Number of updates steps to accumulate before performing a backward/update pass.'
        )
    parser.add_argument('--fp16', default=False, action='store_true', help=
        'Whether to use 16-bit float precision instead of 32-bit')
    parser.add_argument('--loss_scale', type=float, default=0, help=
        """Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.
0 (default value): dynamic loss scaling.
Positive power of 2: static loss scaling value.
"""
        )
    parser.add_argument('--threshold', type=float, default=0.3)
    parser.add_argument('--K_V_dim', type=int, default=100, help=
        'Key and Value dim == KG representation dim')
    parser.add_argument('--Q_dim', type=int, default=768, help=
        'Query dim == Bert six output layer representation dim')
    parser.add_argument('--graphsage', default=False, action='store_true',
        help='Whether to use Attention GraphSage instead of GAT')
    parser.add_argument('--self_att', default=True, action='store_true',
        help='Whether to use GAT')
    parser.add_argument('--weight_decay', default=0.0, type=float, help=
        'Weight decay if we apply some.')
    parser.add_argument('--adam_epsilon', default=1e-08, type=float, help=
        'Epsilon for Adam optimizer.')
    parser.add_argument('--max_grad_norm', default=1.0, type=float, help=
        'Max gradient norm.')
    parser.add_argument('--fp16_opt_level', type=str, default='O1', help=
        "For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3'].See details at https://nvidia.github.io/apex/amp.html"
        )
    parser.add_argument('--data_token', type=str, default='None', help=
        'Using token ids')
    args = parser.parse_args()
    processors = TacredProcessor
    num_labels_task = 80
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
    if not args.do_train and not args.do_eval:
        raise ValueError(
            'At least one of `do_train` or `do_eval` must be True.')
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir
        ) and args.do_train:
        raise ValueError(
            'Output directory ({}) already exists and is not empty.'.format
            (args.output_dir))
    os.makedirs(args.output_dir, exist_ok=True)
    processor = processors()
    label_list = None
    tokenizer = RobertaTokenizer.from_pretrained(args.ernie_model)
    train_examples = None
    num_train_steps = None
    train_examples, label_list = processor.get_train_examples(args.data_dir)
    num_labels = len(label_list)
    num_train_steps = int(len(train_examples) / args.train_batch_size /
        args.gradient_accumulation_steps * args.num_train_epochs)
    model, _ = BertForSequenceClassification.from_pretrained(args.
        ernie_model, cache_dir=PYTORCH_PRETRAINED_BERT_CACHE /
        'distributed_{}'.format(args.local_rank), num_labels=num_labels,
        args=args)
    """
    if args.fp16:
        model.half()
    model.to(device)

    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_grad = ['bert.encoder.layer.11.output.dense_ent', 'bert.encoder.layer.11.output.LayerNorm_ent']
    param_optimizer = [(n, p) for n, p in param_optimizer if not any(nd in n for nd in no_grad)]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    t_total = num_train_steps
    if args.local_rank != -1:
        t_total = t_total // torch.distributed.get_world_size()
    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)

    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=t_total)
    """
    t_total = num_train_steps
    if args.local_rank != -1:
        t_total = t_total // torch.distributed.get_world_size()
    model.to(device)
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    no_grad = ['bert.encoder.layer.11.output.dense_ent',
        'bert.encoder.layer.11.output.LayerNorm_ent']
    param_optimizer = [(n, p) for n, p in param_optimizer if not any(nd in
        n for nd in no_grad)]
    optimizer_grouped_parameters = [{'params': [p for n, p in
        param_optimizer if not any(nd in n for nd in no_decay)],
        'weight_decay': args.weight_decay}, {'params': [p for n, p in
        param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay':
        0.0}]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate,
        eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps
        =int(t_total * 0.1), num_training_steps=t_total)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError(
                'Please install apex from https://www.github.com/nvidia/apex to use fp16 training.'
                )
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.
            fp16_opt_level)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids
            =[args.local_rank], output_device=args.local_rank,
            find_unused_parameters=True)
    global_step = 0
    if args.do_train:
        train_features = convert_examples_to_features(train_examples,
            label_list, args.max_seq_length, tokenizer, args.threshold)
        logger.info('***** Running training *****')
        logger.info('  Num examples = %d', len(train_examples))
        logger.info('  Batch size = %d', args.train_batch_size)
        logger.info('  Num steps = %d', num_train_steps)
        all_input_ids = torch.tensor([f.input_ids for f in train_features],
            dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features
            ], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in
            train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features],
            dtype=torch.long)
        all_ent = torch.tensor([f.input_ent for f in train_features], dtype
            =torch.long)
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
        for _ in trange(int(args.num_train_epochs), desc='Epoch'):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc=
                'Iteration')):
                batch = tuple(t.to(device) if i != 3 else t for i, t in
                    enumerate(batch))
                (input_ids, input_mask, segment_ids, input_ent, ent_mask,
                    label_ids) = batch
                input_ent = input_ent + 1
                k_1, v_1, k_2, v_2 = load_k_v_queryR_small(input_ent)
                loss = model(input_ids, segment_ids, input_mask, input_ent.
                    float(), ent_mask, label_ids, k_1.half(), v_1.half(),
                    k_2.half(), v_2.half())
                if n_gpu > 1:
                    loss = loss.mean()
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                loss_fout.write('{}\n'.format(loss.item()))
                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    """
                    lr_this_step = args.learning_rate * warmup_linear(global_step/t_total, args.warmup_proportion)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1
                    """
                    if args.fp16:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(
                            optimizer), args.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(),
                            args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()
                    global_step += 1
            model_to_save = model.module if hasattr(model, 'module') else model
            output_model_file = os.path.join(args.output_dir,
                'pytorch_model.bin_{}'.format(global_step))
            torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save = model.module if hasattr(model, 'module') else model
        output_model_file = os.path.join(args.output_dir, 'pytorch_model.bin')
        torch.save(model_to_save.state_dict(), output_model_file)
