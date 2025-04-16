def __getitem__(self, idx):
    """ Returns an item of the dataset.

        Args:
            idx (int): ID of data point
        """
    category = self.models[idx]['category']
    model = self.models[idx]['model']
    c_idx = self.metadata[category]['idx']
    model_path = os.path.join(self.dataset_folder, category, model)
    data = {}
    for field_name, field in self.fields.items():
        try:
            field_data = field.load(model_path, idx, c_idx)
        except Exception:
            if self.no_except:
                logger.warn(
                    'Error occured when loading field %s of model %s' % (
                    field_name, model))
                return None
            else:
                raise
        if isinstance(field_data, dict):
            for k, v in field_data.items():
                if k is None:
                    data[field_name] = v
                else:
                    data['%s.%s' % (field_name, k)] = v
        else:
            data[field_name] = field_data
    if self.transform is not None:
        data = self.transform(data)
    return data
