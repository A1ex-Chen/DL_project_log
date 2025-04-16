def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='lamm_peft')
    parser.add_argument('--vision_type', type=str, default='pcl', choices=(
        'image', 'pcl'))
    parser.add_argument('--encoder_pretrain', type=str, default='epcl')
    parser.add_argument('--encoder_ckpt_path', type=str, help=
        'path of vision pretrained model; CLIP use default path in cache')
    parser.add_argument('--llm_ckpt_path', type=str, default=
        '../model_zoo/vicuna_ckpt/13b_v0')
    parser.add_argument('--delta_ckpt_path', type=str, default=
        '../model_zoo/pandagpt_ckpt/13b/pytorch_model.pt')
    parser.add_argument('--force_test', action='store_true', help=
        'whether to force test mode, ignore file missing')
    parser.add_argument('--stage', type=int, default=2, help=
        'has no function in testing')
    parser.add_argument('--lora_r', type=int, default=32)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--lora_dropout', type=float, default=0.1)
    parser.add_argument('--lora_target_modules', nargs='+', default=[
        'q_proj', 'k_proj', 'v_proj', 'o_proj'])
    parser.add_argument('--vision_feature_type', type=str, default='local',
        choices=('local', 'global'))
    parser.add_argument('--vision_output_layer', type=int, default=-1,
        choices=(-1, -2), help=
        'the layer to output visual features; -1 means global from last layer')
    parser.add_argument('--num_vision_token', type=int, default=256)
    parser.add_argument('--max_tgt_len', type=int, default=1024, help=
        'max length of generated tokens')
    parser.add_argument('--top_p', type=float, default=0.9)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--conv_mode', type=str, default='simple')
    parser.add_argument('--num_round', '-N', type=int, default=100, help=
        'number of rounds of conversation')
    parser.add_argument('--question_file', type=str, default='conv.txt',
        help='conversation file')
    parser.add_argument('--vision_root_path', type=str, default='', help=
        'image directory')
    parser.add_argument('--answer_file', type=str, default=
        '../answers/answer.txt', help='answer file')
    parser.add_argument('--detail_log', action='store_true', help=
        'whether to log detail conversation')
    args = parser.parse_args()
    if args.vision_feature_type == 'local':
        args.vision_output_layer = -2
        args.num_vision_token = 256
    elif args.vision_feature_type == 'global':
        args.vision_output_layer = -1
        args.num_vision_token = 1
    else:
        raise NotImplementedError('NOT implement vision feature type: {}'.
            format(args.vision_feature_type))
    assert len(args.vision_root_path) == 0 or os.path.exists(args.
        vision_root_path), 'vision root directory not exists!'
    assert os.path.exists(args.delta_ckpt_path
        ) or args.force_test, "delta checkpoint not exists and it's required!"
    assert os.path.exists(args.llm_ckpt_path), 'vicuna checkpoint not exists!'
    assert args.encoder_pretrain == 'clip' or os.path.exists(args.
        encoder_ckpt_path), 'vision checkpoint not exists!'
    args.max_tgt_len = args.max_tgt_len - 1 + args.num_vision_token
    if not os.path.isdir(os.path.dirname(args.answer_file)):
        os.makedirs(os.path.dirname(args.answer_file), exist_ok=True)
    args.max_tgt_len = max(args.max_tgt_len - 1 + args.num_vision_token, 2048)
    print(json.dumps(vars(args), indent=4, sort_keys=True))
    return args
