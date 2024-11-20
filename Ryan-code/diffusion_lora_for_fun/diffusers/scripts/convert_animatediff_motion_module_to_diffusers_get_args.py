def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--use_motion_mid_block', action='store_true')
    parser.add_argument('--motion_max_seq_length', type=int, default=32)
    parser.add_argument('--block_out_channels', nargs='+', default=[320, 
        640, 1280, 1280], type=int)
    parser.add_argument('--save_fp16', action='store_true')
    return parser.parse_args()
