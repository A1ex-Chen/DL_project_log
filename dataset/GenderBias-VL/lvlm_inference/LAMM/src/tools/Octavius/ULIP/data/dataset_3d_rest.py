'''
 * Copyright (c) 2023, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Le Xue
'''

import random

import torch
import glob
import numpy as np
import torch.utils.data as data

import yaml
from easydict import EasyDict

from utils.io import IO
from utils.build import DATASETS
from utils.logger import *
from utils.build import build_dataset_from_cfg
import json
from tqdm import tqdm
import pickle
from PIL import Image










import os, sys, h5py

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

@DATASETS.register_module()
class ModelNet(data.Dataset):
    def __init__(self, config):
        self.root = config.DATA_PATH
        self.npoints = config.npoints
        self.use_normals = config.USE_NORMALS
        self.num_category = config.NUM_CATEGORY
        self.process_data = True
        self.uniform = True
        self.generate_from_raw_data = False
        split = config.subset
        self.subset = config.subset

        if self.num_category == 10:
            self.catfile = os.path.join(self.root, 'modelnet10_shape_names.txt')
        else:
            self.catfile = os.path.join(self.root, 'modelnet40_shape_names.txt')

        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))

        shape_ids = {}
        if self.num_category == 10:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_test.txt'))]
        else:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_test.txt'))]

        assert (split == 'train' or split == 'test')
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
        self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i]) + '.txt') for i
                         in range(len(shape_ids[split]))]
        print_log('The size of %s data is %d' % (split, len(self.datapath)), logger='ModelNet')

        if self.uniform:
            self.save_path = os.path.join(self.root,
                                          'modelnet%d_%s_%dpts_fps.dat' % (self.num_category, split, self.npoints))
        else:
            self.save_path = os.path.join(self.root,
                                          'modelnet%d_%s_%dpts.dat' % (self.num_category, split, self.npoints))

        if self.process_data:
            if not os.path.exists(self.save_path):
                # make sure you have raw data in the path before you enable generate_from_raw_data=True.
                if self.generate_from_raw_data:
                    print_log('Processing data %s (only running in the first time)...' % self.save_path, logger='ModelNet')
                    self.list_of_points = [None] * len(self.datapath)
                    self.list_of_labels = [None] * len(self.datapath)

                    for index in tqdm(range(len(self.datapath)), total=len(self.datapath)):
                        fn = self.datapath[index]
                        cls = self.classes[self.datapath[index][0]]
                        cls = np.array([cls]).astype(np.int32)
                        point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)

                        if self.uniform:
                            point_set = farthest_point_sample(point_set, self.npoints)
                            print_log("uniformly sampled out {} points".format(self.npoints))
                        else:
                            point_set = point_set[0:self.npoints, :]

                        self.list_of_points[index] = point_set
                        self.list_of_labels[index] = cls

                    with open(self.save_path, 'wb') as f:
                        pickle.dump([self.list_of_points, self.list_of_labels], f)
                else:
                    # no pre-processed dataset found and no raw data found, then load 8192 points dataset then do fps after.
                    self.save_path = os.path.join(self.root,
                                                  'modelnet%d_%s_%dpts_fps.dat' % (
                                                  self.num_category, split, 8192))
                    print_log('Load processed data from %s...' % self.save_path, logger='ModelNet')
                    print_log('since no exact points pre-processed dataset found and no raw data found, load 8192 pointd dataset first, then do fps to {} after, the speed is excepted to be slower due to fps...'.format(self.npoints), logger='ModelNet')
                    with open(self.save_path, 'rb') as f:
                        self.list_of_points, self.list_of_labels = pickle.load(f)

            else:
                print_log('Load processed data from %s...' % self.save_path, logger='ModelNet')
                with open(self.save_path, 'rb') as f:
                    self.list_of_points, self.list_of_labels = pickle.load(f)

        self.shape_names_addr = os.path.join(self.root, 'modelnet40_shape_names.txt')
        with open(self.shape_names_addr) as file:
            lines = file.readlines()
            lines = [line.rstrip() for line in lines]
        self.shape_names = lines

        # TODO: disable for backbones except for PointNEXT!!!
        self.use_height = config.use_height

    def __len__(self):
        return len(self.list_of_labels)

    def _get_item(self, index):
        if self.process_data:
            point_set, label = self.list_of_points[index], self.list_of_labels[index]
        else:
            fn = self.datapath[index]
            cls = self.classes[self.datapath[index][0]]
            label = np.array([cls]).astype(np.int32)
            point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)

            if self.uniform:
                point_set = farthest_point_sample(point_set, self.npoints)
            else:
                point_set = point_set[0:self.npoints, :]

        if  self.npoints < point_set.shape[0]:
            point_set = farthest_point_sample(point_set, self.npoints)

        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        if not self.use_normals:
            point_set = point_set[:, 0:3]

        if self.use_height:
            self.gravity_dim = 1
            height_array = point_set[:, self.gravity_dim:self.gravity_dim + 1] - point_set[:,
                                                                            self.gravity_dim:self.gravity_dim + 1].min()
            point_set = np.concatenate((point_set, height_array), axis=1)

        return point_set, label[0]

    def __getitem__(self, index):
        points, label = self._get_item(index)
        pt_idxs = np.arange(0, points.shape[0])  # 2048
        if self.subset == 'train':
            np.random.shuffle(pt_idxs)
        current_points = points[pt_idxs].copy()
        current_points = torch.from_numpy(current_points).float()
        label_name = self.shape_names[int(label)]

        return current_points, label, label_name

@DATASETS.register_module()
class ShapeNet(data.Dataset):
    def __init__(self, config):

        self.data_root = config.DATA_PATH
        self.pc_path = config.PC_PATH
        self.subset = config.subset
        self.npoints = config.npoints
        self.tokenizer = config.tokenizer
        self.train_transform = config.train_transform
        self.id_map_addr = os.path.join(config.DATA_PATH, 'taxonomy.json')
        self.rendered_image_addr = config.IMAGE_PATH
        self.picked_image_type = ['', '_depth0001']
        self.picked_rotation_degrees = list(range(0, 360, 12))
        self.picked_rotation_degrees = [(3 - len(str(degree))) * '0' + str(degree) if len(str(degree)) < 3 else str(degree) for degree in self.picked_rotation_degrees]

        with open(self.id_map_addr, 'r') as f:
            self.id_map = json.load(f)

        self.prompt_template_addr = os.path.join('./data/templates.json')
        with open(self.prompt_template_addr) as f:
            self.templates = json.load(f)[config.pretrain_dataset_prompt]

        self.synset_id_map = {}
        for id_dict in self.id_map:
            synset_id = id_dict["synsetId"]
            self.synset_id_map[synset_id] = id_dict

        self.data_list_file = os.path.join(self.data_root, f'{self.subset}.txt')
        test_data_list_file = os.path.join(self.data_root, 'test.txt')

        self.sample_points_num = self.npoints
        self.whole = config.get('whole')

        print_log(f'[DATASET] sample out {self.sample_points_num} points', logger='ShapeNet-55')
        print_log(f'[DATASET] Open file {self.data_list_file}', logger='ShapeNet-55')
        with open(self.data_list_file, 'r') as f:
            lines = f.readlines()
        if self.whole:
            with open(test_data_list_file, 'r') as f:
                test_lines = f.readlines()
            print_log(f'[DATASET] Open file {test_data_list_file}', logger='ShapeNet-55')
            lines = test_lines + lines
        self.file_list = []
        for line in lines:
            line = line.strip()
            taxonomy_id = line.split('-')[0]
            model_id = line[len(taxonomy_id) + 1:].split('.')[0]
            self.file_list.append({
                'taxonomy_id': taxonomy_id,
                'model_id': model_id,
                'file_path': line
            })
        print_log(f'[DATASET] {len(self.file_list)} instances were loaded', logger='ShapeNet-55')

        self.permutation = np.arange(self.npoints)

        self.uniform = True
        self.augment = True
        self.use_caption_templates = False
        # =================================================
        # TODO: disable for backbones except for PointNEXT!!!
        self.use_height = config.use_height
        # =================================================

        if self.augment:
            print("using augmented point clouds.")

    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        pc = pc / m
        return pc

    def random_sample(self, pc, num):
        np.random.shuffle(self.permutation)
        pc = pc[self.permutation[:num]]
        return pc

    def __getitem__(self, idx):
        sample = self.file_list[idx]

        data = IO.get(os.path.join(self.pc_path, sample['file_path'])).astype(np.float32)
        print(data.shape)
        if self.uniform and self.sample_points_num < data.shape[0]:
            data = farthest_point_sample(data, self.sample_points_num)
        else:
            data = self.random_sample(data, self.sample_points_num)
        data = self.pc_norm(data)

        if self.augment:
            data = random_point_dropout(data[None, ...])
            data = random_scale_point_cloud(data)
            data = shift_point_cloud(data)
            data = rotate_perturbation_point_cloud(data)
            data = rotate_point_cloud(data)
            data = data.squeeze()

        if self.use_height:
            self.gravity_dim = 1
            height_array = data[:, self.gravity_dim:self.gravity_dim + 1] - data[:,
                                                                       self.gravity_dim:self.gravity_dim + 1].min()
            data = np.concatenate((data, height_array), axis=1)
            data = torch.from_numpy(data).float()
        else:
            data = torch.from_numpy(data).float()

        captions = self.synset_id_map[sample['taxonomy_id']]['name']
        captions = [caption.strip() for caption in captions.split(',') if caption.strip()]
        caption = random.choice(captions)
        captions = []
        tokenized_captions = []
        if self.use_caption_templates:
            for template in self.templates:
                caption = template.format(caption)
                captions.append(caption)
                tokenized_captions.append(self.tokenizer(caption))
        else:
            tokenized_captions.append(self.tokenizer(caption))

        tokenized_captions = torch.stack(tokenized_captions)

        picked_model_rendered_image_addr = self.rendered_image_addr + '/' +\
                                           sample['taxonomy_id'] + '-' + sample['model_id'] + '/'
        picked_image_name = sample['taxonomy_id'] + '-' + sample['model_id'] + '_r_' +\
                            str(random.choice(self.picked_rotation_degrees)) +\
                            random.choice(self.picked_image_type) + '.png'
        picked_image_addr = picked_model_rendered_image_addr + picked_image_name

        try:
            image = pil_loader(picked_image_addr)
            image = self.train_transform(image)
        except:
            raise ValueError("image is corrupted: {}".format(picked_image_addr))

        return sample['taxonomy_id'], sample['model_id'], tokenized_captions, data, image

    def __len__(self):
        return len(self.file_list)

           
@DATASETS.register_module()
class ScanRefer(data.Dataset):
    def __init__(self, config):
        
        self.config = config
        self.data_root = config.DATA_PATH
        self.pc_path = config.PC_PATH
        self.subset = config.subset
        self.npoints = config.npoints
        self.tokenizer = config.tokenizer
        self.train_transform = config.train_transform
        self.train_path = os.path.join(config.DATA_PATH, 'doc', 'ScanRefer_filtered_train.json')
        self.rendered_image_addr = config.IMAGE_PATH
        self.text_feature_addr = config.TEXT_PATH
        self.catfile = os.path.join(config.DATA_PATH, 'doc', 'scanrefer_261_sorted.txt')
        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))

        with open(self.train_path, 'r') as f:
            self.train_data = json.load(f)

        self.prompt_template_addr = os.path.join('./data/templates.json')
        with open(self.prompt_template_addr) as f:
            self.templates = json.load(f)[config.pretrain_dataset_prompt]

        self.sample_points_num = self.npoints
        dir_name = os.listdir(self.rendered_image_addr)
        self.dir_dict = {}
        for x in dir_name:
            self.dir_dict[os.path.join(self.rendered_image_addr, '_'.join(x.split('_')[:-1]))] = os.path.join(self.rendered_image_addr, x)

        print_log(f'[DATASET] sample out {self.sample_points_num} points', logger='ScanRefer')
        
        self.data_list = []
        for data in self.train_data:
            scene_id = data['scene_id']
            object_id = data['object_id']
            description = data['description']
            object_name = data['object_name']
            self.data_list.append({
                'scene_id': scene_id,
                'object_id': object_id,
                'description': description,
                'object_name': object_name
            })
        print_log(f'[DATASET] {len(self.data_list)} instances were loaded', logger='ScanRefer')

        self.permutation = np.arange(self.npoints)

        self.uniform = True
        self.augment = True
        self.use_caption_templates = True
        self.use_height = config.use_height

        if self.augment:
            print("using augmented point clouds.")
        
        # self.gen_memory_bank()

    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        pc = pc / m
        return pc

    def random_sample(self, pc, num):
        np.random.shuffle(self.permutation)
        pc = pc[self.permutation[:num]]
        return pc

    def __getitem__(self, idx):
        sample = self.data_list[idx]

        object_name = sample['object_name'].replace(' ','_')
        data = IO.get(os.path.join(self.pc_path, sample['scene_id'], sample['object_id']+'-'+object_name+'.npy')).astype(np.float32)
        
        if self.sample_points_num <= data.shape[0]:
            data = farthest_point_sample(data, self.sample_points_num)
        else:
            indexes = np.arange(data.shape[0])
            indexes = np.random.choice(indexes, self.sample_points_num-data.shape[0])
            data = np.concatenate((data, data[indexes]), axis=0)
        data_color = data[:,3:6]/255
        data = data[:,:3]
        data = self.pc_norm(data)

        if self.augment:
            data = random_point_dropout(data[None, ...])
            data = random_scale_point_cloud(data)
            data = shift_point_cloud(data)
            data = rotate_perturbation_point_cloud(data)
            data = rotate_point_cloud(data)
            data = data.squeeze()
        
        data = np.concatenate((data, data_color), axis=1)
        
        if self.use_height:
            self.gravity_dim = 1
            height_array = data[:, self.gravity_dim:self.gravity_dim + 1] - data[:,
                                                                       self.gravity_dim:self.gravity_dim + 1].min()
            data = np.concatenate((data, height_array), axis=1)
            data = torch.from_numpy(data).float()
        else:
            data = torch.from_numpy(data).float()

        tokenized_captions_addr = os.path.join(self.text_feature_addr, sample['scene_id'], sample['scene_id']+'_ins'+sample['object_id']+'.npy')
        tokenized_captions = torch.from_numpy(np.load(tokenized_captions_addr))
        
        picked_image_addr = self.dir_dict[os.path.join(self.rendered_image_addr, sample['scene_id']+'_'+sample['object_id'])]
        file = glob.glob(picked_image_addr + f'/0*.npy')[0]
        try:
            image = torch.from_numpy(np.load(file)).squeeze(0)
        except:
            raise ValueError("image is corrupted: {}".format(picked_image_addr))

        return '-', self.classes[sample['object_name']], tokenized_captions, data, image

    def __len__(self):
        return len(self.data_list)
    
    def gen_memory_bank(self):
        print_log(f'[DATASET] generate memory bank', logger='ScanRefer')

        bank_size = ['small', 'middle', 'big'][1]
        self.catfile = os.path.join('data/scanrefer', 'doc', 'scanrefer_261_sorted.txt')
        self.obj_classes = [line.rstrip() for line in open(self.catfile)]
        
        self.obj_class_memory_bank = {}
        self.text_memory_bank = {}
        
        if not os.path.exists(os.path.join(self.config['scannet_object_clip_root'], f'image_memory_bank_{bank_size}.pkl')):

            for i in self.obj_classes:
                self.obj_class_memory_bank[i] = []

            print_log(f'[DATASET] generate image memory bank', logger='ScanRefer')
            for sample in tqdm(self.data_list, desc='[DATASET] generate image memory bank'):
                picked_image_addr = self.dir_dict[os.path.join(self.rendered_image_addr, sample['scene_id']+'_'+sample['object_id'])]
                obj_name = sample['object_name']
                for cls in self.obj_classes:
                    if obj_name != cls:
                        for i in range(3):
                            file = glob.glob(picked_image_addr + f'/{i}*.npy')[0]
                            self.obj_class_memory_bank[cls].append(np.load(file).squeeze(0))
            if bank_size=='middle':
                for i in self.obj_classes:
                    random.shuffle(self.obj_class_memory_bank[i])
                    self.obj_class_memory_bank[i] = self.obj_class_memory_bank[i][:10000]
            if bank_size=='small':
                for i in self.obj_classes:
                    random.shuffle(self.obj_class_memory_bank[i])
                    self.obj_class_memory_bank[i] = self.obj_class_memory_bank[i][:2000]
            # save memory bank
            with open(os.path.join(self.config['scannet_object_clip_root'], f'image_memory_bank_{bank_size}.pkl'), 'wb') as f:
                pickle.dump(self.obj_class_memory_bank, f)
        
        if not os.path.exists(os.path.join(self.config['scannet_text_clip_root'], f'text_memory_bank_{bank_size}.pkl')):

            for i in self.obj_classes:
                self.text_memory_bank[i] = []

            print_log(f'[DATASET] generate text memory bank', logger='ScanRefer')

            for sample in tqdm(self.data_list, desc='[DATASET] generate text memory bank'):
                picked_text_addr = os.path.join(self.text_feature_addr, sample['scene_id'], sample['scene_id']+'_ins'+sample['object_id']+'.npy')
                obj_name = sample['object_name']
                for cls in self.obj_classes:
                    if obj_name != cls:
                        self.text_memory_bank[cls].append(np.load(picked_text_addr))
            if bank_size=='middle':
                for i in self.obj_classes:
                    random.shuffle(self.text_memory_bank[i])
                    self.text_memory_bank[i] = self.text_memory_bank[i][:10000]
            if bank_size=='small':
                for i in self.obj_classes:
                    random.shuffle(self.text_memory_bank[i])
                    self.text_memory_bank[i] = self.text_memory_bank[i][:2000]
            # save memory bank
            with open(os.path.join(self.config['scannet_text_clip_root'], f'text_memory_bank_{bank_size}.pkl'), 'wb') as f:
                pickle.dump(self.text_memory_bank, f)


@DATASETS.register_module()
class ScanReferValid(data.Dataset):
    def __init__(self, config):
        self.root = config.DATA_PATH
        self.npoints = config.npoints
        self.use_normals = config.USE_NORMALS
        self.num_category = config.NUM_CATEGORY
        self.uniform = False
        self.generate_from_raw_data = False
        split = config.subset
        self.subset = config.subset

        self.catfile = os.path.join(self.root, 'doc', 'scanrefer_261_sorted.txt')
        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))

        assert (split == 'test')
        self.datapath = os.path.join(self.root, 'doc', 'ScanRefer_filtered_val.json')
        with open(self.datapath, 'r') as f:
            self.val_data = json.load(f)
        val_data_len = len(self.val_data)
        print_log('The size of %s data is %d' % (split, val_data_len), logger='scanrefer_261_sorted')

        self.save_path = os.path.join(self.root, 'valid_data',
                                          'scanrefer%d_%s_%dpts.dat' % (self.num_category, split, self.npoints))

        if not os.path.exists(self.save_path):
            # make sure you have raw data in the path before you enable generate_from_raw_data=True.
            print_log('Processing data %s (only running in the first time)...' % self.save_path, logger='scanrefer_261_sorted')
            self.list_of_points = [None] * val_data_len
            self.list_of_labels = [None] * val_data_len

            for index in tqdm(range(val_data_len), total=val_data_len):
                data = self.val_data[index]
                cls = self.classes[data['object_name']]
                cls = np.array([cls]).astype(np.int32)
                # point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)
                point_set = IO.get(os.path.join(self.root, 'point_data', data['scene_id'], data['object_id']+'-'+data['object_name']+'.npy')).astype(np.float32)
                indexes = np.arange(point_set.shape[0])
                indexes = np.random.choice(indexes, self.npoints)
                point_set = point_set[indexes]
                
                self.list_of_points[index] = point_set
                self.list_of_labels[index] = cls

            with open(self.save_path, 'wb') as f:
                pickle.dump([self.list_of_points, self.list_of_labels], f)

        else:
            print_log('Load processed data from %s...' % self.save_path, logger='scanrefer_261_sorted')
            with open(self.save_path, 'rb') as f:
                self.list_of_points, self.list_of_labels = pickle.load(f)
        self.shape_names_addr = os.path.join(self.root, 'doc', 'scanrefer_261_sorted.txt')
        with open(self.shape_names_addr) as file:
            lines = file.readlines()
            lines = [line.rstrip() for line in lines]
        self.shape_names = lines
        self.use_height = config.use_height

    def __len__(self):
        return len(self.list_of_labels)

    def _get_item(self, index):
        point_set, label = self.list_of_points[index], self.list_of_labels[index]
        point_set = np.concatenate((pc_normalize(point_set[:,:3]), point_set[:,3:6]/255), axis=1)

        if self.use_height:
            self.gravity_dim = 1
            height_array = point_set[:, self.gravity_dim:self.gravity_dim + 1] - point_set[:,
                                                                            self.gravity_dim:self.gravity_dim + 1].min()
            point_set = np.concatenate((point_set, height_array), axis=1)

        return point_set, label[0]

    def __getitem__(self, index):
        points, label = self._get_item(index)
        pt_idxs = np.arange(0, points.shape[0])  # 2048
        if self.subset == 'train':
            np.random.shuffle(pt_idxs)
        current_points = points[pt_idxs].copy()
        current_points = torch.from_numpy(current_points).float()
        label_name = self.shape_names[int(label)]

        return current_points, label, label_name


import collections.abc as container_abcs
int_classes = int
from torch import string_classes

import re
default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}")
np_str_obj_array_pattern = re.compile(r'[SaUO]')








@DATASETS.register_module()
class ShapeNet(data.Dataset):





           
@DATASETS.register_module()
class ScanRefer(data.Dataset):
        
        # self.gen_memory_bank()




    


@DATASETS.register_module()
class ScanReferValid(data.Dataset):





import collections.abc as container_abcs
int_classes = int
from torch import string_classes

import re
default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}")
np_str_obj_array_pattern = re.compile(r'[SaUO]')

def customized_collate_fn(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    elem = batch[0]
    elem_type = type(elem)
    
    if isinstance(batch, list):
        batch = [example for example in batch if example[4] is not None]

    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return customized_collate_fn([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int_classes):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, container_abcs.Mapping):
        return {key: customized_collate_fn([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(customized_collate_fn(samples) for samples in zip(*batch)))
    elif isinstance(elem, container_abcs.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = zip(*batch)
        return [customized_collate_fn(samples) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))


def merge_new_config(config, new_config):
    for key, val in new_config.items():
        if not isinstance(val, dict):
            if key == '_base_':
                with open(new_config['_base_'], 'r') as f:
                    try:
                        val = yaml.load(f, Loader=yaml.FullLoader)
                    except:
                        val = yaml.load(f)
                config[key] = EasyDict()
                merge_new_config(config[key], val)
            else:
                config[key] = val
                continue
        if key not in config:
            config[key] = EasyDict()
        merge_new_config(config[key], val)
    return config

def cfg_from_yaml_file(cfg_file):
    config = EasyDict()
    with open(cfg_file, 'r') as f:
        new_config = yaml.load(f, Loader=yaml.FullLoader)
    merge_new_config(config=config, new_config=new_config)
    return config

class Dataset_3D():
