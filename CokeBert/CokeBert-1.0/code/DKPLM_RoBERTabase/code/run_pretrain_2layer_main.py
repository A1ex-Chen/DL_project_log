def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default=None, type=str, required=True,
        help=
        'The input data dir. Should contain the .tsv files (or other data files) for the task.'
        )
    parser.add_argument('--bert_model', default=None, type=str, required=
        True, help=
        'Bert pre-trained model selected in the list: bert-base-uncased, bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.'
        )
    parser.add_argument('--task_name', default=None, type=str, required=
        True, help='The name of the task to train.')
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
    parser.add_argument('--eval_batch_size', default=8, type=int, help=
        'Total batch size for eval.')
    parser.add_argument('--learning_rate', default=5e-05, type=float, help=
        'The initial learning rate for Adam.')
    parser.add_argument('--num_train_epochs', default=1.0, type=float, help
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
    parser.add_argument('--K_V_dim', type=int, default=100, help=
        'Key and Value dim == KG representation dim')
    parser.add_argument('--Q_dim', type=int, default=768, help=
        'Query dim == Bert six output layer representation dim')
    parser.add_argument('--graphsage', default=False, action='store_true',
        help='Whether to use Attention GraphSage instead of GAT')
    parser.add_argument('--self_att', default=True, action='store_true',
        help='Whether to use GAT')
    args = parser.parse_args()
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
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        raise ValueError(
            'Output directory ({}) already exists and is not empty.'.format
            (args.output_dir))
    os.makedirs(args.output_dir, exist_ok=True)
    task_name = args.task_name.lower()
    train_data = None
    num_train_steps = None
    if args.do_train:
        import indexed_dataset
        from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, BatchSampler
        import iterators
        train_data = indexed_dataset.IndexedDataset(args.data_dir,
            fix_lua_indexing=True)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_sampler = BatchSampler(train_sampler, args.train_batch_size, True
            )

        def collate_fn(x):
            x = torch.LongTensor([xx for xx in x])
            entity_idx = x[:, 3 * args.max_seq_length:4 * args.max_seq_length]
            uniq_idx = np.unique(entity_idx.numpy())
            ent_candidate = torch.LongTensor(uniq_idx + 1)
            ent_candidate = ent_candidate.repeat([n_gpu, 1])
            d = {}
            dd = []
            for i, idx in enumerate(uniq_idx):
                d[idx] = i
                dd.append(idx)
            """
            ent_size = len(uniq_idx)-1
            def map(x):
                if x == -1:
                    return -1
                else:
                    rnd = random.uniform(0, 1)
                    if rnd < 0.05:
                        return dd[random.randint(1, ent_size)]
                    elif rnd < 0.2:
                        return -1
                    else:
                        return x
            """
            ent_labels = entity_idx.clone()
            d[-1] = -1
            ent_labels = ent_labels.apply_(lambda x: d[x])
            """
            entity_idx.apply_(map)

            #ent_emb = embed(entity_idx+1)
            ent_emb = entity_idx+1 ##
            #! --> return entity+1 => input_ent

            mask = entity_idx.clone()
            mask.apply_(lambda x: 0 if x == -1 else 1)
            mask[:,0] = 1
            """
            mask = x[:, 4 * args.max_seq_length:5 * args.max_seq_length]
            mask[:, 0] = 1
            entity_idx = entity_idx * mask
            entity_idx[entity_idx == 0] = -1
            ent_emb = entity_idx + 1
            return x[:, :args.max_seq_length], x[:, args.max_seq_length:2 *
                args.max_seq_length], x[:, 2 * args.max_seq_length:3 * args
                .max_seq_length], ent_emb, mask, x[:, 5 * args.max_seq_length:
                ], ent_candidate, ent_labels
        train_iterator = iterators.EpochBatchIterator(train_data,
            collate_fn, train_sampler)
        num_train_steps = int(len(train_data) / args.train_batch_size /
            args.gradient_accumulation_steps * args.num_train_epochs)
    model, missing_keys = BertForPreTraining.from_pretrained(args.
        bert_model, cache_dir=PYTORCH_PRETRAINED_BERT_CACHE /
        'distributed_{}'.format(args.local_rank), args=args)
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
    no_linear = ['layer.2.output.dense_ent', 'layer.2.intermediate.dense_1',
        'bert.encoder.layer.2.intermediate.dense_1_ent',
        'layer.2.output.LayerNorm_ent']
    no_linear = [x.replace('2', '11') for x in no_linear]
    param_optimizer = [(n, p) for n, p in param_optimizer if not any(nl in
        n for nl in no_linear)]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight',
        'LayerNorm_ent.bias', 'LayerNorm_ent.weight']
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
        optimizer = BertAdam(optimizer_grouped_parameters, lr=args.
            learning_rate, warmup=args.warmup_proportion, t_total=t_total)
    global_step = 0
    if args.do_train:
        logger.info('***** Running training *****')
        logger.info('  Num examples = %d', len(train_data))
        logger.info('  Batch size = %d', args.train_batch_size)
        logger.info('  Num steps = %d', num_train_steps)
        model.train()
        import datetime
        fout = open(os.path.join(args.output_dir, 'loss.{}'.format(datetime
            .datetime.now())), 'w')
        more_than_one_2 = 0
        less_than_one_2 = 0
        for _ in trange(int(args.num_train_epochs), desc='Epoch'):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            """
            for step, batch in enumerate(tqdm(train_iterator.next_epoch_itr(), desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)

                input_ids, input_mask, segment_ids, masked_lm_labels, input_ent, ent_mask, next_sentence_label, ent_candidate, ent_labels = batch
            """
            if args.local_rank == 0 or args.local_rank == -1:
                iters = tqdm(train_iterator.next_epoch_itr(), desc='Iteration')
            else:
                iters = train_iterator.next_epoch_itr()
            for step, batch in enumerate(iters):
                batch = tuple(t.to(device) for t in batch)
                (input_ids, input_mask, masked_lm_labels, input_ent,
                    ent_mask, next_sentence_label, ent_candidate, ent_labels
                    ) = batch
                if len(input_ids[input_ids == 2]) != args.train_batch_size:
                    for i_th_1, input_id in enumerate(input_ids):
                        print(input_id[input_id == 2])
                        print(len(input_id[input_id == 2]))
                        if len(input_id[input_id == 2]) > 1:
                            for i_th_2, id in enumerate(input_id):
                                if id == 2:
                                    print('Befor:', input_id)
                                    input_ids[i_th_1][i_th_2] = 0
                                    more_than_one_2 += 1
                                    print('more_than_one_2:', more_than_one_2)
                                    print('After:', input_id)
                                    if len(input_id[input_id == 2] == 1):
                                        break
                        elif len(input_id[input_id == 2]) < 1:
                            print('Error!! Have no id=2 </s>')
                            less_than_one_2 += 1
                            print('less_than_one_2:', less_than_one_2)
                            print(input_id)
                            input_ids[i_th_1][-1] = 2
                        else:
                            print('ids_2 == 1')
                (k_1, v_1, k_2, v_2, k_cand_1, v_cand_1, k_cand_2, v_cand_2,
                    cand_pos_tensor) = load_k_v_queryR_small(input_ent,
                    ent_candidate)
                if args.fp16:
                    loss, original_loss = model(input_ids, None, input_mask,
                        masked_lm_labels, input_ent, ent_mask,
                        next_sentence_label, ent_candidate, ent_labels, k_1
                        .half(), v_1.half(), k_2.half(), v_2.half(),
                        k_cand_1.half(), v_cand_1.half(), k_cand_2.half(),
                        v_cand_2.half(), cand_pos_tensor)
                else:
                    loss, original_loss = model(input_ids, None, input_mask,
                        masked_lm_labels, input_ent, ent_mask,
                        next_sentence_label, ent_candidate, ent_labels, k_1,
                        v_1, k_2, v_2, k_cand_1, v_cand_1, k_cand_2,
                        v_cand_2, cand_pos_tensor)
                if n_gpu > 1:
                    loss = loss.mean()
                    original_loss = original_loss.mean()
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()
                end_time_4 = time.time()
                fout.write('{} {}\n'.format(loss.item() * args.
                    gradient_accumulation_steps, original_loss.item()))
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
                    if global_step % 100000 == 0:
                        model_to_save = model.module if hasattr(model, 'module'
                            ) else model
                        output_model_file = os.path.join(args.output_dir,
                            'pytorch_model.bin_{}'.format(global_step))
                        torch.save(model_to_save.state_dict(),
                            output_model_file)
        fout.close()
    model_to_save = model.module if hasattr(model, 'module') else model
    output_model_file = os.path.join(args.output_dir, 'pytorch_model.bin')
    torch.save(model_to_save.state_dict(), output_model_file)
