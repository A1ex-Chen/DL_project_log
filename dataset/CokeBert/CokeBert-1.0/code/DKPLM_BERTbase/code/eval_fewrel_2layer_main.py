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
    parser.add_argument('--eval_batch_size', default=8, type=int, help=
        'Total batch size for eval.')
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
    args = parser.parse_args()
    processors = FewrelProcessor
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
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
    if not args.do_train and not args.do_eval:
        raise ValueError(
            'At least one of `do_train` or `do_eval` must be True.')
    processor = processors()
    num_labels = num_labels_task
    label_list = None
    tokenizer = BertTokenizer.from_pretrained(args.ernie_model,
        do_lower_case=args.do_lower_case)
    train_examples = None
    num_train_steps = None
    train_examples, label_list = processor.get_train_examples(args.data_dir)
    """
    vecs = []
    vecs.append([0]*100)
    with open("kg_embed/entity2vec.vec", 'r') as fin:
        for line in fin:
            vec = line.strip().split('	')
            vec = [float(x) for x in vec]
            vecs.append(vec)
    embed = torch.FloatTensor(vecs)
    embed = torch.nn.Embedding.from_pretrained(embed)
    #embed = torch.nn.Embedding(5041175, 100)

    logger.info("Shape of entity embedding: "+str(embed.weight.size()))
    del vecs
    """
    filenames = os.listdir(args.output_dir)
    filenames = [x for x in filenames if 'pytorch_model.bin_' in x]
    filenames = [x for x in filenames if x in ['pytorch_model.bin_1750',
        'pytorch_model.bin_2000', 'pytorch_model.bin_2250',
        'pytorch_model.bin_2500', 'pytorch_model.bin_2750',
        'pytorch_model.bin_3000', 'pytorch_model.bin_3250',
        'pytorch_model.bin_3500', 'pytorch_model.bin_3750',
        'pytorch_model.bin_4000']]
    file_mark = []
    for x in filenames:
        file_mark.append([x, True])
        file_mark.append([x, False])
    eval_examples = processor.get_dev_examples(args.data_dir)
    dev = convert_examples_to_features(eval_examples, label_list, args.
        max_seq_length, tokenizer, args.threshold)
    eval_examples = processor.get_test_examples(args.data_dir)
    test = convert_examples_to_features(eval_examples, label_list, args.
        max_seq_length, tokenizer, args.threshold)
    for x, mark in file_mark:
        print(x, mark)
        output_model_file = os.path.join(args.output_dir, x)
        model_state_dict = torch.load(output_model_file)
        model, _ = BertForSequenceClassification.from_pretrained(args.
            ernie_model, state_dict=model_state_dict, num_labels=len(
            label_list), args=args)
        if args.fp16:
            model.half()
        model.to(device)
        if mark:
            eval_features = dev
        else:
            eval_features = test
        logger.info('***** Running evaluation *****')
        logger.info('  Num examples = %d', len(eval_examples))
        logger.info('  Batch size = %d', args.eval_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features],
            dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features],
            dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in
            eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in eval_features],
            dtype=torch.long)
        all_ent = torch.tensor([f.input_ent for f in eval_features], dtype=
            torch.long)
        all_ent_masks = torch.tensor([f.ent_mask for f in eval_features],
            dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask,
            all_segment_ids, all_ent, all_ent_masks, all_label_ids)
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler,
            batch_size=args.eval_batch_size)
        if mark:
            output_eval_file = os.path.join(args.output_dir,
                'eval_results_{}.txt'.format(x.split('_')[-1]))
            output_file_pred = os.path.join(args.output_dir,
                'eval_pred_{}.txt'.format(x.split('_')[-1]))
            output_file_glod = os.path.join(args.output_dir,
                'eval_gold_{}.txt'.format(x.split('_')[-1]))
        else:
            output_eval_file = os.path.join(args.output_dir,
                'test_results_{}.txt'.format(x.split('_')[-1]))
            output_file_pred = os.path.join(args.output_dir,
                'test_pred_{}.txt'.format(x.split('_')[-1]))
            output_file_glod = os.path.join(args.output_dir,
                'test_gold_{}.txt'.format(x.split('_')[-1]))
        fpred = open(output_file_pred, 'w')
        fgold = open(output_file_glod, 'w')
        model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        for input_ids, input_mask, segment_ids, input_ent, ent_mask, label_ids in eval_dataloader:
            input_ent = input_ent + 1
            """
            cordinate = np.array(torch.nonzero(input_ent))
            for x,y in cordinate:
                input_ent[x,y]=2
            """
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            input_ent = input_ent.to(device)
            ent_mask = ent_mask.to(device)
            label_ids = label_ids.to(device)
            k_1, v_1, k_2, v_2 = load_k_v_queryR_small(input_ent)
            with torch.no_grad():
                tmp_eval_loss = model(input_ids, segment_ids, input_mask,
                    input_ent, ent_mask, label_ids, k_1.half(), v_1.half(),
                    k_2.half(), v_2.half())
                logits = model(input_ids, segment_ids, input_mask,
                    input_ent, ent_mask, None, k_1.half(), v_1.half(), k_2.
                    half(), v_2.half())
            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()
            tmp_eval_accuracy, pred = accuracy(logits, label_ids)
            for a, b in zip(pred, label_ids):
                fgold.write('{}\n'.format(b))
                fpred.write('{}\n'.format(a))
            eval_loss += tmp_eval_loss.mean().item()
            eval_accuracy += tmp_eval_accuracy
            nb_eval_examples += input_ids.size(0)
            nb_eval_steps += 1
        eval_loss = eval_loss / nb_eval_steps
        eval_accuracy = eval_accuracy / nb_eval_examples
        result = {'eval_loss': eval_loss, 'eval_accuracy': eval_accuracy}
        with open(output_eval_file, 'w') as writer:
            logger.info('***** Eval results *****')
            for key in sorted(result.keys()):
                logger.info('  %s = %s', key, str(result[key]))
                writer.write('%s = %s\n' % (key, str(result[key])))
