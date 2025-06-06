import os
import os.path as osp
import numpy as np
import pickle
from PIL import Image
import glob
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

import sys
sys.path.append('/Labs/Scripts/3DPC/exp_xmuda_journal')

from xmuda.data.semantic_kitti import splits

# prevent "RuntimeError: received 0 items of ancdata"
torch.multiprocessing.set_sharing_strategy('file_system')


class DummyDataset(Dataset):
    """Use torch dataloader for multiprocessing"""
    def __init__(self, root_dir, scenes):
        self.root_dir = root_dir
        self.data = []
        self.glob_frames(scenes)

    def glob_frames(self, scenes):
        for scene in scenes:
            glob_path = osp.join(self.root_dir, 'dataset', 'sequences', scene, 'image_2', '*.png')
            cam_paths = sorted(glob.glob(glob_path))
            # load calibration
            calib = self.read_calib(osp.join(self.root_dir, 'dataset', 'sequences', scene, 'calib.txt'))
            proj_matrix = calib['P2'] @ calib['Tr']
            proj_matrix = proj_matrix.astype(np.float32)

            for cam_path in cam_paths:
                basename = osp.basename(cam_path)
                frame_id = osp.splitext(basename)[0]
                assert frame_id.isdigit()
                data = {
                    'camera_path': cam_path,
                    'lidar_path': osp.join(self.root_dir, 'dataset', 'sequences', scene, 'velodyne',
                                           frame_id + '.bin'),
                    'label_path': osp.join(self.root_dir, 'dataset', 'sequences', scene, 'labels',
                                           frame_id + '.label'),
                    'proj_matrix': proj_matrix
                }
                for k in ['camera_path', 'lidar_path']:
                    if not osp.exists(data[k]):
                        raise IOError('File not found {}'.format(data[k]))
                self.data.append(data)

    @staticmethod
    def read_calib(calib_path):
        """
        :param calib_path: Path to a calibration text file.
        :return: dict with calibration matrices.
        """
        calib_all = {}
        with open(calib_path, 'r') as f:
            for line in f.readlines():
                if line == '\n':
                    break
                key, value = line.split(':', 1)
                calib_all[key] = np.array([float(x) for x in value.split()])

        # reshape matrices
        calib_out = {}
        calib_out['P2'] = calib_all['P2'].reshape(3, 4)  # 3x4 projection matrix for left camera
        calib_out['Tr'] = np.identity(4)  # 4x4 matrix
        calib_out['Tr'][:3, :4] = calib_all['Tr'].reshape(3, 4)
        return calib_out

    @staticmethod
    def select_points_in_frustum(points_2d, x1, y1, x2, y2):
        """
        Select points in a 2D frustum parametrized by x1, y1, x2, y2 in image coordinates
        :param points_2d: point cloud projected into 2D
        :param points_3d: point cloud
        :param x1: left bound
        :param y1: upper bound
        :param x2: right bound
        :param y2: lower bound
        :return: points (2D and 3D) that are in the frustum
        """
        keep_ind = (points_2d[:, 0] > x1) * \
                   (points_2d[:, 1] > y1) * \
                   (points_2d[:, 0] < x2) * \
                   (points_2d[:, 1] < y2)

        return keep_ind

    def __getitem__(self, index):
        data_dict = self.data[index].copy()
        scan = np.fromfile(data_dict['lidar_path'], dtype=np.float32)
        scan = scan.reshape((-1, 4))
        points = scan[:, :3]
        if osp.exists(data_dict['label_path']):
            label = np.fromfile(data_dict['label_path'], dtype=np.uint32)
            label = label.reshape((-1))
            label = label & 0xFFFF  # get lower half for semantics
        else:
            label = None

        # load image
        image = Image.open(data_dict['camera_path'])
        image_size = image.size

        # project points into image
        keep_idx = points[:, 0] > 0  # only keep point in front of the vehicle
        points_hcoords = np.concatenate([points[keep_idx], np.ones([keep_idx.sum(), 1], dtype=np.float32)], axis=1)
        img_points = (data_dict['proj_matrix'] @ points_hcoords.T).T
        img_points = img_points[:, :2] / np.expand_dims(img_points[:, 2], axis=1)  # scale 2D points
        keep_idx_img_pts = self.select_points_in_frustum(img_points, 0, 0, *image_size)
        keep_idx[keep_idx] = keep_idx_img_pts
        # fliplr so that indexing is row, col and not col, row
        img_points = np.fliplr(img_points)
        # debug
        # from xmuda.data.utils.visualize import draw_points_image, draw_bird_eye_view
        # draw_points_image(np.array(image), img_points[keep_idx_img_pts].astype(int), label[keep_idx],
        #                   color_palette_type='SemanticKITTI_long')

        if label is not None:
            data_dict['seg_label'] = label[keep_idx].astype(np.int16)
        data_dict['points'] = points[keep_idx]
        data_dict['points_img'] = img_points[keep_idx_img_pts]
        data_dict['image_size'] = np.array(image_size)

        return data_dict

    def __len__(self):
        return len(self.data)






    @staticmethod

    @staticmethod




def preprocess(split_name, root_dir, out_dir):
    pkl_data = []
    split = getattr(splits, split_name)

    dataloader = DataLoader(DummyDataset(root_dir, split), num_workers=8)

    num_skips = 0
    for i, data_dict in enumerate(dataloader):
        # data error leads to returning empty dict
        if not data_dict:
            print('empty dict, continue')
            num_skips += 1
            continue
        for k, v in data_dict.items():
            data_dict[k] = v[0]
        print('{}/{} {}'.format(i, len(dataloader), data_dict['lidar_path']))

        # convert to relative path
        lidar_path = data_dict['lidar_path'].replace(root_dir + '/', '')
        cam_path = data_dict['camera_path'].replace(root_dir + '/', '')

        seg_labels = data_dict.get('seg_label', None)
        if seg_labels is not None:
            seg_labels = seg_labels.numpy()

        # append data
        out_dict = {
            'points': data_dict['points'].numpy(),
            'seg_labels': seg_labels,
            'points_img': data_dict['points_img'].numpy(),  # row, col format, shape: (num_points, 2)
            'lidar_path': lidar_path,
            'camera_path': cam_path,
            'image_size': tuple(data_dict['image_size'].numpy())
        }
        pkl_data.append(out_dict)

    print('Skipped {} files'.format(num_skips))

    # save to pickle file
    save_dir = osp.join(out_dir, 'preprocess')
    os.makedirs(save_dir, exist_ok=True)
    save_path = osp.join(save_dir, '{}.pkl'.format(split_name))
    with open(save_path, 'wb') as f:
        pickle.dump(pkl_data, f)
        print('Wrote preprocessed data to ' + save_path)


def print_frames_per_scene(root_dir):
    from tabulate import tabulate
    import matplotlib.pyplot as plt
    glob_path = osp.join(root_dir, 'dataset', 'sequences', '*')
    seq_paths = sorted(glob.glob(glob_path))
    table = []
    for seq_path in seq_paths:
        frame_paths = sorted(glob.glob(osp.join(seq_path, 'image_2', '*.png')))
        cur_split = []
        for split_name in ['train', 'val', 'test', 'train_labeled', 'train_unlabeled1', 'train_unlabeled2']:
            if osp.basename(seq_path) in getattr(splits, split_name):
                cur_split.append(split_name)
        table.append([osp.basename(seq_path), cur_split, len(frame_paths)])
        # show first image of each sequence
        plt.imshow(np.array(Image.open(frame_paths[0])))
        plt.title(osp.basename(seq_path))
        plt.show()
    header = ['Seq', 'Split', '# Frames']
    print_table = tabulate(table, headers=header, tablefmt='psql')
    print(print_table)


if __name__ == '__main__':
    root_dir = '/Labs/Scripts/3DPC/Datasets/3DPC/SKITTI/semantic_kitti'
    out_dir = '/Labs/Scripts/3DPC/Datasets/3DPC/nuScenes_preprocess/semantickitti-preprocess'
    preprocess('val', root_dir, out_dir)
    preprocess('train', root_dir, out_dir)
    preprocess('test', root_dir, out_dir)

    # SSDA
    preprocess('train_labeled', root_dir, out_dir)
    # split into train_unlabeled1 and train_unlabeled2 to prevent segmentation fault in torch dataloader
    preprocess('train_unlabeled1', root_dir, out_dir)
    preprocess('train_unlabeled2', root_dir, out_dir)
    # merge train_unlabeled1 and train_unlabeled2
    data = []
    for curr_split in ['train_unlabeled1', 'train_unlabeled2']:
        with open(osp.join(out_dir, 'preprocess', curr_split + '.pkl'), 'rb') as f:
            data.extend(pickle.load(f))
    save_path = osp.join(out_dir, 'preprocess', 'train_unlabeled.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)
        print('Wrote preprocessed data to ' + save_path)
    for curr_split in ['train_unlabeled1', 'train_unlabeled2']:
        os.remove(osp.join(out_dir, 'preprocess', curr_split + '.pkl'))

    # print_frames_per_scene(root_dir)