def dataframe(self):
    """Load the data as a pd.DataFrame"""
    data, labels = self.load_data()
    if isinstance(labels, dict):
        data_dict = {'data': data}
        for key, value in labels.items():
            data_dict[key] = value
    else:
        data_dict = {'data': data, 'labels': labels}
    return pd.DataFrame(data_dict)
