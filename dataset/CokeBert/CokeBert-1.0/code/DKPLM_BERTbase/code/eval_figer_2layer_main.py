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
    processor = TypingProcessor()
    tokenizer_label = BertTokenizer_label.from_pretrained(args.ernie_model,
        do_lower_case=args.do_lower_case)
    tokenizer = BertTokenizer.from_pretrained(args.ernie_model,
        do_lower_case=args.do_lower_case)
    _, label_list, _ = processor.get_train_examples(args.data_dir)
    label_list = sorted(label_list)
    S = []
    for l in label_list:
        s = []
        for ll in label_list:
            if ll in l:
                s.append(1.0)
            else:
                s.append(0.0)
        S.append(s)
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
    logger.info("Shape of entity embedding: "+str(embed.weight.size()))
    del vecs
    """
    filenames = os.listdir(args.output_dir)
    filenames = [x for x in filenames if 'pytorch_model.bin_' in x]
    file_mark = []
    for x in filenames:
        file_mark.append([x, True])
        file_mark.append([x, False])
    for x, mark in file_mark:
        print(x, mark)
        output_model_file = os.path.join(args.output_dir, x)
        model_state_dict = torch.load(output_model_file)
        model, _ = BertForEntityTyping.from_pretrained(args.ernie_model,
            state_dict=model_state_dict, num_labels=len(label_list), args=args)
        model.to(device)
        if args.fp16:
            model.half()
        if mark:
            eval_examples = processor.get_dev_examples(args.data_dir)
        else:
            eval_examples = processor.get_test_examples(args.data_dir)
        eval_features = convert_examples_to_features(eval_examples,
            label_list, args.max_seq_length, tokenizer_label, tokenizer,
            args.threshold)
        logger.info('***** Running evaluation *****')
        logger.info('  Num examples = %d', len(eval_examples))
        logger.info('  Batch size = %d', args.eval_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features],
            dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features],
            dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in
            eval_features], dtype=torch.long)
        all_input_ent = torch.tensor([f.input_ent for f in eval_features],
            dtype=torch.long)
        all_ent_mask = torch.tensor([f.ent_mask for f in eval_features],
            dtype=torch.long)
        all_labels = torch.tensor([f.labels for f in eval_features], dtype=
            torch.float)
        eval_data = TensorDataset(all_input_ids, all_input_mask,
            all_segment_ids, all_input_ent, all_ent_mask, all_labels)
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler,
            batch_size=args.eval_batch_size)
        model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        pred = []
        true = []
        for input_ids, input_mask, segment_ids, input_ent, ent_mask, labels in eval_dataloader:
            input_ent = input_ent + 1
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            input_ent = input_ent.to(device)
            ent_mask = ent_mask.to(device)
            labels = labels.to(device)
            k_1, v_1, k_2, v_2 = load_k_v_queryR_small(input_ent)
            with torch.no_grad():
                tmp_eval_loss = model(input_ids, segment_ids, input_mask,
                    input_ent, ent_mask, labels.half(), k_1.half(), v_1.
                    half(), k_2.half(), v_2.half())
                logits = model(input_ids, segment_ids, input_mask,
                    input_ent, ent_mask, None, k_1.half(), v_1.half(), k_2.
                    half(), v_2.half())
            logits = logits.detach().cpu().numpy()
            labels = labels.to('cpu').numpy()
            tmp_eval_accuracy, tmp_pred, tmp_true = accuracy(logits, labels)
            pred.extend(tmp_pred)
            true.extend(tmp_true)
            eval_loss += tmp_eval_loss.mean().item()
            eval_accuracy += tmp_eval_accuracy
            nb_eval_examples += input_ids.size(0)
            nb_eval_steps += 1
        eval_loss = eval_loss / nb_eval_steps
        eval_accuracy = eval_accuracy / nb_eval_examples

        def f1(p, r):
            if r == 0.0:
                return 0.0
            return 2 * p * r / float(p + r)

        def loose_macro(true, pred):
            num_entities = len(true)
            p = 0.0
            r = 0.0
            for true_labels, predicted_labels in zip(true, pred):
                if len(predicted_labels) > 0:
                    p += len(set(predicted_labels).intersection(set(
                        true_labels))) / float(len(predicted_labels))
                if len(true_labels):
                    r += len(set(predicted_labels).intersection(set(
                        true_labels))) / float(len(true_labels))
            precision = p / num_entities
            recall = r / num_entities
            return precision, recall, f1(precision, recall)

        def loose_micro(true, pred):
            num_predicted_labels = 0.0
            num_true_labels = 0.0
            num_correct_labels = 0.0
            for true_labels, predicted_labels in zip(true, pred):
                num_predicted_labels += len(predicted_labels)
                num_true_labels += len(true_labels)
                num_correct_labels += len(set(predicted_labels).
                    intersection(set(true_labels)))
            if num_predicted_labels > 0:
                precision = num_correct_labels / num_predicted_labels
            else:
                precision = 0.0
            recall = num_correct_labels / num_true_labels
            return precision, recall, f1(precision, recall)
        if False:
            result = {'eval_loss': eval_loss, 'eval_accuracy':
                eval_accuracy, 'macro': loose_macro(true, pred), 'micro':
                loose_micro(true, pred)}
        else:
            result = {'eval_loss': eval_loss, 'eval_accuracy':
                eval_accuracy, 'macro': loose_macro(true, pred), 'micro':
                loose_micro(true, pred)}
        if mark:
            output_eval_file = os.path.join(args.output_dir,
                'eval_results_{}.txt'.format(x.split('_')[-1]))
        else:
            output_eval_file = os.path.join(args.output_dir,
                'test_results_{}.txt'.format(x.split('_')[-1]))
        with open(output_eval_file, 'w') as writer:
            logger.info('***** Eval results *****')
            for key in sorted(result.keys()):
                logger.info('  %s = %s', key, str(result[key]))
                writer.write('%s = %s\n' % (key, str(result[key])))
