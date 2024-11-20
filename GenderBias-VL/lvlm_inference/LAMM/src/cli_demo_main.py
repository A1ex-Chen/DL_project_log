def main(args):
    model = LAMMPEFTModel(**args.__dict__)
    if os.path.isfile(args.delta_ckpt_path):
        print('[!] Loading delta checkpoint: {}...'.format(args.
            delta_ckpt_path))
        delta_ckpt = torch.load(args.delta_ckpt_path, map_location=torch.
            device('cpu'))
        model.load_state_dict(delta_ckpt, strict=False)
    elif args.force_test:
        print('[!] Loading vicuna checkpoint: {}... while {} not found!'.
            format(args.llm_ckpt_path, args.delta_ckpt_path))
    else:
        raise ValueError('delta checkpoint not exists!')
    model = model.eval().half().cuda()
    print(f'[!] init the 13b model over ...')
    history = []
    if args.num_round > 0:
        print(
            f'>>>>>>>>>>>>>>>>>>>>>[!][!][!] start the conversation ...<<<<<<<<<<<<<<<<<<<<<<<<<'
            , flush=True)
        conversation = True
        while conversation:
            vision_paths = []
            print("Input file paths (type 'done' when finished): ", flush=True)
            while True:
                input_path = input('Vision path: ').strip()
                if input_path.lower() == 'done':
                    break
                if os.path.isfile(input_path):
                    vision_paths.append(input_path)
                    print(f'Added: {input_path}')
                else:
                    print(f'{input_path} not found!')
            if len(vision_paths) == 1:
                vision_path = vision_paths[0]
            elif len(vision_paths) == 0:
                print('No vision content provided!')
                continue
            else:
                vision_path = vision_paths
            input_dict = make_input_dict(args, vision_path)
            print('------------Conversation Begins-----------')
            history = []
            for i in range(args.num_round):
                print(f'[!] round {i + 1}')
                print("Input your content: (Say 'quit' to end / change PCL)",
                    flush=True)
                input_text = ''
                while len(input_text) == 0:
                    input_text = input('Human: ')
                    print(f'[!] ### Human: {input_text}', flush=True)
                if input_text == 'quit':
                    print('[!] ### Assistant: Bye!', flush=True)
                    break
                _, history, _, item_time = predict(args=args, model=model,
                    input=input_text, **input_dict, chatbot=[], max_length=
                    args.max_tgt_len, top_p=args.top_p, temperature=args.
                    temperature, history=history, modality_cache=[])
                print('[!] ### Assistant: {}'.format(history[-1][1][0].
                    split('\n##')[0]))
            print(
                '------------------------------------------------------------------'
                )
            print('Do you want to continue? (y/n)', flush=True)
            conversation = input('Human: ') == 'y'
    elif os.path.isfile(args.question_file):
        questions = [json.loads(q) for q in open(os.path.expanduser(args.
            question_file), 'r')]
        answer_file = open(os.path.expanduser(args.answer_file), 'w')
        answer_list = list()
        pbar = tqdm(total=len(questions))
        for q in tqdm(questions):
            question_id = q['question_id']
            input_text = q['text']
            if os.path.isdir(args.vision_root_path):
                vision_path = os.path.join(args.vision_root_path, q[args.
                    vision_type])
            else:
                vision_path = q[args.vision_type]
            if not os.path.isfile(vision_path):
                print(f'[!] Vision data path: {vision_path} is not exist!')
                continue
            input_dict = make_input_dict(args, vision_path)
            history = []
            chatbot, history, modality_cache, item_time = predict(args=args,
                model=model, input=input_text, **input_dict, chatbot=[],
                max_length=args.max_tgt_len, top_p=args.top_p, temperature=
                args.temperature, history=history, modality_cache=[],
                show_prompt=args.detail_log)
            response = history[-1][1]
            print(f'[!] Assistant ({item_time:3f}s): {history[-1][1]}')
            ans_dict = {'question_id': question_id, 'prompt': input,
                'response': response, 'model_id': args.delta_ckpt_path,
                f'{args.vision_type}': vision_path}
            answer_file.write(json.dumps(ans_dict) + '\n')
            answer_file.flush()
            answer_list.append(ans_dict)
            pbar.set_description(
                f'[!] question_id: {question_id}; Item time: {item_time:.3f}s')
            pbar.update(1)
        answer_file.close()
        with open(os.path.expanduser(os.path.splitext(args.answer_file)[0] +
            '.json'), 'w') as f:
            json.dump(answer_list, f, indent=4)
    else:
        print(
            'Please provide either a question file or a number of rounds to run the chatbot.'
            )
