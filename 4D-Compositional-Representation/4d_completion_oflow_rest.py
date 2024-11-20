from lib import config, data
import torch
import torch.optim as optim
import numpy as np
import os
import argparse
from torch.utils.data import DataLoader
from torch.nn import functional as F
from tqdm import trange





if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser(
        description='Conduct backward optimization experiments.'
    )
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('--experiment', type=str, required=True,
                        help='Type of the backward experiment. temporal, spatial or future')
    parser.add_argument('--seq', type=str, default='50026_shake_arms',
                        help='Name of the sequence')
    parser.add_argument('--start_idx', type=int, default=30,
                        help='Start index of the sub-sequence')
    parser.add_argument('--seq_length', type=int, default=30,
                        help='Length of the sub-sequence,'
                             'we set it to 30 for 4D completion and 20 for future prediction.')
    parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')
    parser.add_argument('--g', type=str, default='0', help='gpu id')
    args = parser.parse_args()

    assert args.experiment in ['temporal', 'spatial', 'future']
    if args.experiment == 'future':
        args.seq_length = 20

    os.environ['CUDA_VISIBLE_DEVICES'] = args.g

    cfg = config.load_config(args.config, 'configs/default.yaml')
    is_cuda = (torch.cuda.is_available() and not args.no_cuda)
    device = torch.device("cuda" if is_cuda else "cpu")

    transf_pt = data.SubsamplePointsSeq(cfg['data']['n_training_points'],  random=True,
                                        spatial_completion=True if args.experiment == 'spatial' else False)
    fields = {
        'points': data.PointsSubseqField(
            cfg['data']['points_iou_seq_folder'], all_steps=True,
            seq_len=args.seq_length,
            unpackbits=cfg['data']['points_unpackbits'],
            transform=transf_pt,
            scale_type=cfg['data']['scale_type'],
            spatial_completion=True if args.experiment == 'spatial' else False),
        'idx': data.IndexField(),
    }

    specific_model = {'seq': args.seq,
                      'start_idx': args.start_idx}

    ################
    out_dir = cfg['training']['out_dir']
    mesh_out_folder = os.path.join(out_dir, args.experiment, args.seq)
    dataset_folder = cfg['data']['path']
    ################

    dataset = data.HumansDataset(dataset_folder, fields, 'test',
                                 specific_model=specific_model)

    test_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, num_workers=1,
        worker_init_fn=data.worker_init_fn,
        shuffle=False)

    model = config.get_model(cfg, device=device)
    model_dir = os.path.join(out_dir, cfg['test']['model_file'])
    print('Loading checkpoint from %s' % model_dir)
    load_dict = torch.load(model_dir)
    model.load_state_dict(load_dict['model'])

    cfg['generation']['n_time_steps'] = args.seq_length
    generator = config.get_generator(model, cfg, device=device)

    times = np.array([i / (args.seq_length - 1) for i in range(args.seq_length)], dtype=np.float32)
    if args.experiment == 'temporal':
        t_idx = np.random.choice(range(args.seq_length), size=args.seq_length // 2, replace=False)
        t_idx.sort()
    elif args.experiment == 'spatial':
        t_idx = np.arange(args.seq_length)
    else:
        t_idx = np.arange(args.seq_length // 2)

    back_optim(model, generator, test_loader, out_dir=mesh_out_folder,
               latent_size=cfg['model']['c_dim'],
               device=device, num_iterations=500,
               time_value=times[t_idx], t_idx=t_idx)