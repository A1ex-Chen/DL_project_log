def __getitem__(self, idx):
    """ Returns an item of the dataset.

        Args:
            idx (int): ID of data point
        """
    data = {}
    for field_name, field in self.fields.items():
        field_data = field.load(idx)
        if isinstance(field_data, dict):
            for k, v in field_data.items():
                if k is None:
                    data[field_name] = v
                else:
                    data['%s.%s' % (field_name, k)] = v
        else:
            data[field_name] = field_data
    return data
