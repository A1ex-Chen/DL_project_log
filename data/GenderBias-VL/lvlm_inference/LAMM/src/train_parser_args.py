def parser_args():
    parser = argparse.ArgumentParser(description='train parameters for LAMM')
    parser.add_argument('--cfg', type=str, default='./config/train.yaml',
        help='config file')
    parser.add_argument('--data_path', type=str, help=
        'the path that stores the data JSON')
    parser.add_argument('--vision_root_path', type=str, help=
        'Root dir for images')
    parser.add_argument('--max_tgt_len', type=int, default=400, help=
        'max length of post-image texts in LLM input')
    parser.add_argument('--vision_type', type=str, default='image', choices
        =('image', 'pcl'), help='the type of vision data')
    parser.add_argument('--use_system', default=False, action='store_true',
        help='whether to use system messages')
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--save_path', type=str, help=
        'directory to save checkpoints')
    parser.add_argument('--log_path', type=str, help='directory to save logs')
    parser.add_argument('--model', type=str, default='lamm_peft', help=
        'Model class to use')
    parser.add_argument('--encoder_pretrain', type=str, default='clip',
        choices=('clip', 'epcl'), help='Vision Pretrain Model')
    parser.add_argument('--encoder_ckpt_path', type=str, help=
        'path of vision pretrained model; CLIP use default path in cache')
    parser.add_argument('--llm_ckpt_path', type=str, required=True, help=
        'path of LLM, default: Vicuna')
    parser.add_argument('--delta_ckpt_path', type=str, help=
        'path of delta parameters from previous stage; Only matter for stage 2'
        )
    parser.add_argument('--llm_proj_path', type=str, help=
        'path of LLM projection matrix; Only matter for stage 2')
    parser.add_argument('--gradient_checkpointing', default=False, action=
        'store_true', help=
        'whether to use gradient checkpointing to save memory')
    parser.add_argument('--vision_feature_type', type=str, default='local',
        choices=('local', 'global'), help='the type of vision features')
    parser.add_argument('--vision_output_layer', type=int, default=-1,
        choices=(-1, -2), help=
        'the layer to output visual features; -1 means global from last layer')
    parser.add_argument('--num_vision_token', type=int, default=1, help=
        'number of vision tokens')
    parser.add_argument('--conv_template', type=str, default='default',
        help='which conversation template to use')
    parser.add_argument('--stage', type=int, default=1, help=
        'number of training stage; 1 by default; 2 if delta ckpt specified')
    parser.add_argument('--use_color', default=False, action='store_true',
        help='whether to use color of point cloud')
    parser.add_argument('--use_height', default=False, action='store_true',
        help='whether to use height info of point cloud')
    parser.add_argument('--num_points', type=int, default=40000, help=
        'number of points in each point cloud')
    parser.add_argument('--use_flash_attn', default=False, action=
        'store_true', help='whether to use flash attention to speed up')
    parser.add_argument('--use_xformers', default=False, action=
        'store_true', help='whether to use xformers to speed up')
    args = parser.parse_args()
    assert not (args.use_flash_attn and args.use_xformers
        ), 'can only use one of flash attn and xformers.'
    if args.vision_feature_type == 'local':
        args.num_vision_token = 256
        args.vision_output_layer = -2
    elif args.vision_feature_type == 'global':
        args.num_vision_token = 1
        args.vision_output_layer = -1
    else:
        raise NotImplementedError('NOT implement vision feature type: {}'.
            format(args.vision_feature_type))
    print('Arguments: \n{}'.format(json.dumps(vars(parser.parse_args()),
        indent=4, sort_keys=True)))
    return args
