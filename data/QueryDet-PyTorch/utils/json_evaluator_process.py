def process(self, inputs, outputs):
    for input, output in zip(inputs, outputs):
        img_name = os.path.split(input['file_name'])[-1].split('.')[0]
        if 'instances' in output:
            prediction = {'img_name': img_name}
            instances = output['instances'].to(self._cpu_device)
            if self.class_add_1:
                instances.pred_classes += 1
            prediction['instances'] = instances_to_coco_json(instances,
                input['image_id'])
            self._predictions.append(prediction)
