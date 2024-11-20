def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dump_path', required=False, default=None, type=str)
    parser.add_argument('--dump_path_stage_2', required=False, default=None,
        type=str)
    parser.add_argument('--dump_path_stage_3', required=False, default=None,
        type=str)
    parser.add_argument('--unet_config', required=False, default=None, type
        =str, help='Path to unet config file')
    parser.add_argument('--unet_checkpoint_path', required=False, default=
        None, type=str, help='Path to unet checkpoint file')
    parser.add_argument('--unet_checkpoint_path_stage_2', required=False,
        default=None, type=str, help='Path to stage 2 unet checkpoint file')
    parser.add_argument('--unet_checkpoint_path_stage_3', required=False,
        default=None, type=str, help='Path to stage 3 unet checkpoint file')
    parser.add_argument('--p_head_path', type=str, required=True)
    parser.add_argument('--w_head_path', type=str, required=True)
    args = parser.parse_args()
    return args
