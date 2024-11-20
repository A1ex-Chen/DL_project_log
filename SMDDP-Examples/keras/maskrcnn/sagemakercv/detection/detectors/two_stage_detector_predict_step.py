def predict_step(self, data_batch):
    model_outputs = self(data_batch['features'], data_batch.get('labels'),
        training=False)
    model_outputs.update({'source_ids': data_batch['features']['source_ids'
        ], 'image_info': data_batch['features']['image_info']})
    return model_outputs
