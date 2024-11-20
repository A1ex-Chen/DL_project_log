def init_dataset(self):
    for idx, data in enumerate(self.datalist):
        img_feature = self.extractor(data, return_tensors='pt')
        self.encode_dataset.append({'pixel_values': img_feature.
            pixel_values[0], 'metadata': {'id': idx, 'img': data}})
