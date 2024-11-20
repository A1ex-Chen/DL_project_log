def gen_memory_bank(self):
    print_log(f'[DATASET] generate memory bank', logger='ScanRefer')
    bank_size = ['small', 'middle', 'big'][1]
    self.catfile = os.path.join('data/scanrefer', 'doc',
        'scanrefer_261_sorted.txt')
    self.obj_classes = [line.rstrip() for line in open(self.catfile)]
    self.obj_class_memory_bank = {}
    self.text_memory_bank = {}
    if not os.path.exists(os.path.join(self.config[
        'scannet_object_clip_root'], f'image_memory_bank_{bank_size}.pkl')):
        for i in self.obj_classes:
            self.obj_class_memory_bank[i] = []
        print_log(f'[DATASET] generate image memory bank', logger='ScanRefer')
        for sample in tqdm(self.data_list, desc=
            '[DATASET] generate image memory bank'):
            picked_image_addr = self.dir_dict[os.path.join(self.
                rendered_image_addr, sample['scene_id'] + '_' + sample[
                'object_id'])]
            obj_name = sample['object_name']
            for cls in self.obj_classes:
                if obj_name != cls:
                    for i in range(3):
                        file = glob.glob(picked_image_addr + f'/{i}*.npy')[0]
                        self.obj_class_memory_bank[cls].append(np.load(file
                            ).squeeze(0))
        if bank_size == 'middle':
            for i in self.obj_classes:
                random.shuffle(self.obj_class_memory_bank[i])
                self.obj_class_memory_bank[i] = self.obj_class_memory_bank[i][:
                    10000]
        if bank_size == 'small':
            for i in self.obj_classes:
                random.shuffle(self.obj_class_memory_bank[i])
                self.obj_class_memory_bank[i] = self.obj_class_memory_bank[i][:
                    2000]
        with open(os.path.join(self.config['scannet_object_clip_root'],
            f'image_memory_bank_{bank_size}.pkl'), 'wb') as f:
            pickle.dump(self.obj_class_memory_bank, f)
    if not os.path.exists(os.path.join(self.config['scannet_text_clip_root'
        ], f'text_memory_bank_{bank_size}.pkl')):
        for i in self.obj_classes:
            self.text_memory_bank[i] = []
        print_log(f'[DATASET] generate text memory bank', logger='ScanRefer')
        for sample in tqdm(self.data_list, desc=
            '[DATASET] generate text memory bank'):
            picked_text_addr = os.path.join(self.text_feature_addr, sample[
                'scene_id'], sample['scene_id'] + '_ins' + sample[
                'object_id'] + '.npy')
            obj_name = sample['object_name']
            for cls in self.obj_classes:
                if obj_name != cls:
                    self.text_memory_bank[cls].append(np.load(picked_text_addr)
                        )
        if bank_size == 'middle':
            for i in self.obj_classes:
                random.shuffle(self.text_memory_bank[i])
                self.text_memory_bank[i] = self.text_memory_bank[i][:10000]
        if bank_size == 'small':
            for i in self.obj_classes:
                random.shuffle(self.text_memory_bank[i])
                self.text_memory_bank[i] = self.text_memory_bank[i][:2000]
        with open(os.path.join(self.config['scannet_text_clip_root'],
            f'text_memory_bank_{bank_size}.pkl'), 'wb') as f:
            pickle.dump(self.text_memory_bank, f)
