def get_data_dict(self, motion, idx=0, id1=None, id2=None):
    category = motion['category']
    model = motion['model']
    start_idx = motion['start_idx']
    data = {}
    for field_name, field in self.fields.items():
        if self.mode == 'train':
            if field_name in ['points_ex', 'points_t_ex', 'points_iou_ex']:
                model_path = os.path.join(self.dataset_folder, 'train',
                    category, id2, model)
            else:
                model_path = os.path.join(self.dataset_folder, 'train',
                    category, id1, model)
        else:
            model_path = os.path.join(self.dataset_folder, 'test', category,
                model)
        field_data = field.load(model_path, idx, start_idx=start_idx,
            dataset_folder=self.dataset_folder)
        if isinstance(field_data, dict):
            for k, v in field_data.items():
                if k is None:
                    data[field_name] = v
                else:
                    data['%s.%s' % (field_name, k)] = v
        else:
            data[field_name] = field_data
    return data
