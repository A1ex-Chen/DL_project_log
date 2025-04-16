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
    tokenizer = RobertaTokenizer.from_pretrained(args.ernie_model)
    train_examples = None
    num_train_steps = None
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
    filenames = [x for x in filenames if 'pytorch_model.bin' in x]
    file_mark = []
    for x in filenames:
        file_mark.append([x, False])
    """
    eval_examples = processor.get_dev_examples(args.data_dir)

    dev = convert_examples_to_features(
        eval_examples, label_list, args.max_seq_length, tokenizer, args.threshold)
    """
    eval_examples = processor.get_test_examples(args.data_dir)
    test = convert_examples_to_features(eval_examples, label_list, args.
        max_seq_length, tokenizer, args.threshold)
    for x, mark in file_mark:
        print(x, mark)
        output_model_file = os.path.join(args.output_dir, x)
        model_state_dict = torch.load(output_model_file)
        model, _ = BertForSequenceClassification.from_pretrained(args.
            ernie_model, state_dict=model_state_dict, num_labels=
            num_labels_task, args=args)
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
        all_label_ids = torch.tensor([(1) for f in eval_features], dtype=
            torch.long)
        all_ent = torch.tensor([f.input_ent for f in eval_features], dtype=
            torch.long)
        all_ent_masks = torch.tensor([f.ent_mask for f in eval_features],
            dtype=torch.long)
        output_label_map = dict()
        output_text_map = dict()
        output_ent_map = dict()
        output_ans_map = dict()
        for i, f in enumerate(eval_features):
            output_label_map[i] = f.label
            output_text_map[i] = f.text
            output_ent_map[i] = f.ent
            output_ans_map[i] = f.ans
        output_label_id = torch.tensor([f[0] for f in enumerate(
            eval_features)], dtype=torch.long)
        output_text_id = torch.tensor([f[0] for f in enumerate(
            eval_features)], dtype=torch.long)
        output_ent_id = torch.tensor([f[0] for f in enumerate(eval_features
            )], dtype=torch.long)
        output_ans_id = torch.tensor([f[0] for f in enumerate(eval_features
            )], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask,
            all_segment_ids, all_ent, all_ent_masks, all_label_ids,
            output_label_id, output_text_id, output_ent_id, output_ans_id)
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler,
            batch_size=args.eval_batch_size)
        if mark:
            output_file_pred = os.path.join(args.output_dir,
                'eval_pred_{}.txt'.format(x.split('_')[-1]))
        else:
            output_file_pred = os.path.join(args.output_dir,
                'test_pred_{}.txt'.format(x.split('_')[-1]))
        fpred = open(output_file_pred, 'w')
        model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        save_data_list = list()
        counter = 0
        acc_all = 0
        pre_all = 0
        f1_all = 0
        for input_ids, input_mask, segment_ids, input_ent, ent_mask, label_ids, output_label_id, output_text_id, output_ent_id, output_ans_id in eval_dataloader:
            input_ent = input_ent + 1
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            input_ent = input_ent.to(device)
            ent_mask = ent_mask.to(device)
            label_ids = label_ids.to(device)
            k_1, v_1, new_input_ent, input_ent_nb, input_ent_r = (
                load_k_v_queryR_small(input_ent))
            """
            ###
            ##################################
            print("==============================")
            print("==============================")
            #text = input_ids[input_ids!=0]
            #text = tokenizer.convert_ids_to_tokens(text.tolist())
            #text = " ".join(text)
            #print("!!!!!!!!!!!!!!!!!!!!!!")
            output_label = str(output_label_map[int(output_label_id)])
            output_text = str(output_text_map[int(output_text_id)])
            output_ent = output_ent_map[int(output_ent_id)]
            #print(output_label)
            #print(output_text)
            #print(output_ent)
            #print("!!!!!!!!!!!!!!!!!!!!!!")
            try:
                #print(text)
                print(output_label)
                print("--")
                print(output_text)
                print("--")
                print(output_ent)
            except:
                print("Error")
                continue
            print("=========")

            for i_th, batch_i in enumerate(new_input_ent.tolist()):
                for k_th, ent in enumerate(batch_i):
                    try:
                        #print("{}:  E1:{} ({})".format(k_th, q2ent[id2q[int(ent)]], ent) )
                        print("{}:  E1:{} ({})".format(k_th, q2ent[id2q[int(ent)]], id2q[int(ent)]) )
                        print("-------------")
                        for j_th, n in enumerate(input_ent_nb.tolist()[k_th]):
                            #print("!!!!")
                            #print(input_ent_r.tolist()[k_th])
                            m = input_ent_r.tolist()[k_th][j_th]
                            #print(m)
                            #print("!!!!")
                            #if n==0 or j_th==0:
                            if n==0 or j_th==0 or m==0:
                                continue
                            #print("{}:  {} ({}) ".format(j_th-1, q2ent[id2q[int(n)]], n))
                            #print("{}:  {} ({}) ".format(j_th-1, q2ent[id2q[int(n)]], id2q[int(n)]))
                            print("{}:  R:{} ({}) ; E:{} ({})".format(j_th-1, p2rel[id2p[int(m)]], id2p[int(m)], q2ent[id2q[int(n)]], id2q[int(n)]))
                        print("-------------")
                    except:
                        print("ascii ERROR")
                        continue
            #print("===============================")
            print("===============================")
            ##################################
            ###
            """
            with torch.no_grad():
                output_gen_ids = model(input_ids, segment_ids, input_mask,
                    input_ent, ent_mask, None, k_1.half(), v_1.half())
                output_ans = output_ans_map[int(output_ans_id)]
                if output_gen_ids == None:
                    print('None')
                    continue
                for i, ids_list_pre in enumerate(output_gen_ids):
                    ids_list_ans = output_ans[i]
                    if len(ids_list_pre) != len(ids_list_ans):
                        print('========')
                        print(ids_list_pre)
                        print(ids_list_ans)
                        print('========')
                        continue
                    counter += 1
                    acc = 0
                    pre = 0
                    f1 = 0
                    for j, ids in enumerate(ids_list_pre):
                        try:
                            if ids_list_ans[j] == ids:
                                acc += float(1 / len(ids_list_ans))
                                if ids_list_ans[j] == 1:
                                    pre += float(1 / ids_list_ans.count(1))
                                else:
                                    pre += 0.0
                                f1 = float(2.0 * pre * acc / (acc + pre))
                        except:
                            print('!!!')
                            print(ids_list_ans, j, 'wrong')
                            print('!!!')
                            exit()
                    acc_all += acc
                    pre_all += pre
                    f1_all += f1
        print('==============================')
        P = acc_all / counter
        R = pre_all / counter
        F1 = float(2.0 * P * R / (P + R))
        print('P:', P)
        print('R:', R)
        print('F1:', F1)
        print('==============================')
        with open(output_file_pred, 'w') as writer:
            logger.info('***** Results*****')
            fpred.write('P: {}\n'.format(P))
            fpred.write('R: {}\n'.format(R))
            fpred.write('F1: {}\n'.format(F1))
            """
            #if all 1 or all 0 don't choose!
            if_use = True
            try:
                for idxx, ids_list in enumerate(output_gen_ids):
                    j_0 = ids_list.count(0)
                    j_1 = ids_list.count(1)
                    if j_0==0 or j_0==len(ids_list) or j_1==0 or j_1==len(ids_list):
                        if_use = False
            except:
                continue

            if if_use:
                data={"label":output_label, "text":output_text, "ents":output_ent, "ans":output_gen_ids}
                save_data_list.append(data)
                counter+=1
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                print(counter)
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            else:
                continue

            if counter==100:
                with open('exp1/fewrel_100.json', 'w', encoding='utf-8') as f:
                    json.dump(save_data_list, f, ensure_ascii=False)
                    print("Done")
                    exit()
            """
            """
            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()
            tmp_eval_accuracy, pred = accuracy(logits, label_ids)
            for a, b in zip(pred, label_ids):
                fgold.write("{}
".format(b))
                fpred.write("{}
".format(a))

            eval_loss += tmp_eval_loss.mean().item()
            eval_accuracy += tmp_eval_accuracy

            nb_eval_examples += input_ids.size(0)
            nb_eval_steps += 1
            """
        """
        eval_loss = eval_loss / nb_eval_steps
        eval_accuracy = eval_accuracy / nb_eval_examples

        result = {'eval_loss': eval_loss,
                  'eval_accuracy': eval_accuracy
                  }


        #with open(output_eval_file+"_comb", "w") as writer: ###erine
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s
" % (key, str(result[key])))
        """
